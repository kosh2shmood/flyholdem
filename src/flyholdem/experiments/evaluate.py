"""Teacher-free frozen native poker evaluation on fixed, seat-swapped deals.

This baseline workflow measures returns without training or claiming a poker
gate passed. Each completed hand is fsynced before the next begins.
"""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import time
import numpy as np
import yaml
from flyholdem.connectome.registry import ROOT,digest
from flyholdem.interface.frozen import load_frozen
from flyholdem.interface.population import load_controller
from flyholdem.learning.curriculum import CurriculumHand,CURRICULA
from flyholdem.poker.opponents import Opponent,VERSIONS
from flyholdem.provenance import manifest,identity,assert_compatible
from flyholdem.neural.checkpoint import Checkpoints,atomic_json
from .journal import Journal


def frozen_hand(controller,curriculum,opponent,deal_seed,seat):
    if seat not in (0,1):raise ValueError('A heads-up seat is required')
    game=CurriculumHand(curriculum,deal_seed,button=0)
    controller.brain.reset_dynamics()
    other=Opponent(deal_seed+2000000,opponent)
    decisions=[];counts=np.zeros(5,dtype=np.int64)
    while not game.done:
        observation=game.observation()
        if game.actor==seat:
            decision=controller.decide(observation,temperature=0)
            expected=int(np.argmax(np.where(decision['legal_mask'],decision['scores'],-np.inf)))
            if decision['selected']!=expected or decision['decoder_fallback']:
                raise AssertionError('Frozen action differs from the actual legal neural argmax')
            action=game.act(decision['selected']);controller.commit_interval();counts[decision['selected']]+=1
            decisions.append({key:value for key,value in decision.items() if key not in ('counts','encoded')})
            decisions[-1]['counts_sha256']=hashlib.sha256(decision['counts'].tobytes()).hexdigest()
            decisions[-1]['committed_action']=action
        else:game.act(other.act(observation))
    table=game.view()
    if sum(table['payoffs'])!=0:raise AssertionError('PokerKit chip conservation failure')
    return {'deal_seed':deal_seed,'neural_seat':seat,'opponent':opponent,'opponent_version':VERSIONS[opponent],
        'curriculum':curriculum,'neural_return_bb':table['payoffs'][seat]/2,'action_counts':counts.tolist(),
        'neural_decisions':decisions,'public_terminal':table,'private_hand_checkpoint':game.serialize(),
        'learning':False,'teacher_connected':False,'reward_delivered':False}


def summarize(rows,config):
    grouped={kind:{} for kind in config['opponents']}
    for row in rows:
        pair=grouped[row['opponent']].setdefault(row['deal_seed'],{})
        if row['neural_seat'] in pair:raise ValueError('Duplicate evaluated deal/seat')
        pair[row['neural_seat']]=row
    summaries={}
    for i,kind in enumerate(config['opponents']):
        complete=[v for v in grouped[kind].values() if set(v)=={0,1}]
        if not complete:continue
        values=np.array([(v[0]['neural_return_bb']+v[1]['neural_return_bb'])/2 for v in complete])
        rng=np.random.default_rng(config['bootstrap_seed']+i)
        means=rng.choice(values,size=(config['bootstrap_repeats'],len(values)),replace=True).mean(axis=1)
        alpha=.05/len(config['opponents'])
        summaries[kind]={'opponent_version':VERSIONS[kind],'paired_deals':len(complete),
            'bb_per_hand':float(values.mean()),'bb_per_100_hands':100*float(values.mean()),
            'suite_adjusted_bootstrap_ci':np.quantile(means,[alpha/2,1-alpha/2]).tolist(),
            'paired_standard_error':float(values.std(ddof=1)/np.sqrt(len(values))) if len(values)>1 else None,
            'action_counts':np.sum([v[s]['action_counts'] for v in complete for s in (0,1)],axis=0).tolist(),
            'unpaired_completed_hands':sum(len(v) for v in grouped[kind].values())-2*len(complete)}
    return summaries


def evaluate(config,output,model=None,resume=False,stop_after=None):
    if config.get('schema')!='frozen-poker-evaluation-v1':raise ValueError('Registered frozen poker evaluation configuration required')
    if config['curriculum'] not in CURRICULA:raise ValueError('Unknown poker curriculum')
    if tuple(config['opponents'])!=tuple(VERSIONS):raise ValueError('Use the full fixed opponent suite in registered order')
    if type(config['paired_deals_per_opponent']) is not int or config['paired_deals_per_opponent']<2:
        raise ValueError('At least two paired deals per opponent are required')
    if config['bootstrap_repeats']<100 or config['status'] not in ('development','confirmatory-baseline'):
        raise ValueError('Explicit development/baseline scope and bootstrap required')
    graph=ROOT/'connectome_data/malecns_v1'/('prepared-'+config['mode'])
    if model:
        controller=load_frozen(model,graph);model_hash=digest(Path(model)/'manifest.json')
        learning_mode=controller.frozen_model['learning_mode']
    else:
        registration=ROOT/config['preregistration']
        if digest(registration)!=config['preregistration_sha256']:raise ValueError('Frozen registration checksum mismatch')
        controller=load_controller(graph,registration);model_hash=None;learning_mode='bio-plastic'
    brain=controller.brain;initial_hash=hashlib.sha256(brain.weights.tobytes()).hexdigest()
    curriculum_registration=CurriculumHand(config['curriculum'],config['deal_seed_start']).registration
    runtime=manifest({**config,'model_sha256':model_hash,'learning_mode':learning_mode,
        'curriculum_registration':curriculum_registration,'weights_sha256':initial_hash,'opponent_versions':VERSIONS},
        brain.graph_hash,identity(controller.registration['projection_indices']),identity(controller.registration['ensembles']),brain.build['binary_sha256'],model_hash)
    out=Path(output);out.mkdir(parents=True,exist_ok=resume)
    if resume:assert_compatible(json.loads((out/'manifest.json').read_text()),runtime)
    else:atomic_json(out/'manifest.json',runtime)
    journal=Journal(out/'hands.jsonl',resume);checkpoints=Checkpoints(out/'checkpoints')
    completed=0;stopped=False;in_hand=False;started=time.perf_counter();last_checkpoint=time.monotonic()
    if resume:
        arrays,extra,_=checkpoints.load(runtime);completed=extra['completed_hands']
        if completed>len(journal.rows) or extra['journal_head']!=(journal.rows[completed-1]['hash'] if completed else '0'*64):
            raise ValueError('Frozen evaluation checkpoint/journal mismatch')
        brain.restore_state({key:arrays['brain_'+key] for key in brain.state_names})
        controller.rng.bit_generator.state=extra['controller_rng']
        if hashlib.sha256(brain.weights.tobytes()).hexdigest()!=initial_hash:raise ValueError('Frozen checkpoint weights differ from the selected model')
    def checkpoint():
        nonlocal last_checkpoint
        if in_hand:return
        checkpoints.save({'brain_'+key:value for key,value in brain.state().items()},runtime,
            {'completed_hands':completed,'journal_head':journal.rows[completed-1]['hash'] if completed else '0'*64,
             'controller_rng':controller.rng.bit_generator.state,'private_hand':None,'boundary':'complete hand; next hand resets native dynamics'},completed)
        last_checkpoint=time.monotonic()
    def stop(signum,frame):
        nonlocal stopped
        stopped=True
    handlers={sig:signal.signal(sig,stop) for sig in (signal.SIGINT,signal.SIGTERM)}
    if not resume:checkpoint()
    schedule=[(kind,config['deal_seed_start']+deal,seat) for kind in config['opponents']
        for deal in range(config['paired_deals_per_opponent']) for seat in (0,1)]
    try:
        for index in range(completed,len(schedule)):
            kind,seed,seat=schedule[index];in_hand=True
            row=frozen_hand(controller,config['curriculum'],kind,seed,seat)
            journal.record(index,[kind,seed,seat],row);completed=index+1;in_hand=False
            if time.monotonic()-last_checkpoint>=300:checkpoint()
            if completed%config.get('progress_hands',32)==0:
                print(json.dumps({'hands':completed,'planned':len(schedule),'last_opponent':kind}),flush=True)
            if stopped or stop_after is not None and completed>=stop_after:break
        if hashlib.sha256(brain.weights.tobytes()).hexdigest()!=initial_hash:raise AssertionError('Frozen evaluation changed synaptic weights')
        rows=[r['value'] for r in journal.rows[:completed]]
        result={'schema':'frozen-poker-evaluation-result-v1','status':'baseline-complete' if completed==len(schedule) else 'interrupted',
            'mode':config['mode'],'learning_mode':learning_mode,'profile':config['status'],'curriculum':config['curriculum'],
            'model_sha256':model_hash,'hands_completed':completed,'planned_hands':len(schedule),
            'opponents':summarize(rows,config),'learning_claim':False,'teacher_connected':False,'weights_unchanged':True,
            'scope':'Frozen native poker baseline only; no training, policy selection or poker gate claim',
            'journal_head':journal.rows[completed-1]['hash'] if completed else '0'*64,
            'manifest_sha256':digest(out/'manifest.json'),'elapsed_seconds_this_invocation':time.perf_counter()-started}
        atomic_json(out/'result.json',result);return result
    finally:
        checkpoint();journal.close()
        for sig,handler in handlers.items():signal.signal(sig,handler)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--model')
    p.add_argument('--output',required=True);p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',type=int)
    a=p.parse_args();print(json.dumps(evaluate(yaml.safe_load(Path(a.config).read_text()),a.output,a.model,a.resume,a.stop_after),indent=2))


if __name__=='__main__':main()
