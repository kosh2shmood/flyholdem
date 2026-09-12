"""Held-out paired evaluation of the small tabular reference, never a fly."""
from collections import deque
import json
from pathlib import Path
import signal
import numpy as np
from flyholdem.learning.curriculum import CurriculumHand
from flyholdem.poker.opponents import Opponent,VERSIONS
from flyholdem.poker.observation import canonical_bytes
from flyholdem.connectome.registry import digest
from flyholdem.provenance import identity,manifest,assert_compatible
from flyholdem.neural.checkpoint import atomic_json
from flyholdem.experiments.journal import Journal
from flyholdem.experiments.reports import audit_journal
from .shove_fold import CURRICULUM,load_policy
from .evaluation import evaluation_summary


def boundary_check(policy):
    checked=0
    for seed in range(6300000,6300016):
        for node in (0,1):
            game=CurriculumHand(CURRICULUM,seed)
            if node:game.act(4)
            observation=game.observation();before=policy.probabilities(observation).tobytes()
            hand=game.hand;other=1-hand.state.actor_index
            hand.state.hole_cards[other][:]=list(hand.state.deck_cards)[:2]
            hand.state.deck_cards=deque(reversed(hand.state.deck_cards));hand.teacher_labels=[999]*5
            if canonical_bytes(observation)!=canonical_bytes(game.observation()) or before!=policy.probabilities(game.observation()).tobytes():
                raise AssertionError('Small teacher targets changed with private information')
            checked+=1
    return {'decisions_checked':checked,'hidden_hole_future_deck_and_teacher_label_invariance':True}


def paired_match(policy,kind,seed):
    returns=[];counts=np.zeros(5,dtype=np.int64)
    for seat in (0,1):
        game=CurriculumHand(CURRICULUM,seed);other=Opponent(seed+2000000,kind)
        rng=np.random.default_rng(seed+1000000)
        while not game.done:
            observation=game.observation()
            if game.actor==seat:
                action=int(rng.choice(5,p=policy.probabilities(observation)));counts[action]+=1
            else:action=other.act(observation)
            game.act(action)
        returns.append(game.view()['payoffs'][seat]/2)
    return {'deal_seed':seed,'seat_returns_bb':returns,'paired_bb_per_hand':float(np.mean(returns)),
            'action_counts':counts.tolist()}


def verify_run(run,config,policy_hash,profile):
    """Recompute the fixed schedule and summaries before accepting a prerequisite."""
    run=Path(run);result=json.loads((run/'result.json').read_text());runtime=json.loads((run/'manifest.json').read_text())
    expected={**config,'profile':profile,'policy_sha256':policy_hash}
    actual={k:v for k,v in runtime['config'].items() if k!='development_result_sha256'}
    if (result.get('schema')!='tabular-shove-fold-evaluation-v1' or actual!=expected
            or runtime.get('config_hash')!=identity(runtime['config'])
            or result.get('manifest_sha256')!=digest(run/'manifest.json') or result.get('policy_sha256')!=policy_hash
            or result.get('profile')!=profile or result.get('curriculum')!=CURRICULUM
            or result.get('allowed_as_teacher') is not False):raise ValueError('Small teacher evaluation identity mismatch')
    boundary=result.get('information_boundary',{})
    if boundary.get('decisions_checked')!=32 or boundary.get('hidden_hole_future_deck_and_teacher_label_invariance') is not True:
        raise ValueError('Small teacher information boundary is incomplete')
    selected=config['profiles'][profile];n=selected['paired_deals_per_opponent'];audit=audit_journal(run/'paired-deals.jsonl')
    if audit['rows']!=len(VERSIONS)*n or audit['head']!=result.get('journal_head'):raise ValueError('Small teacher paired journal incomplete')
    grouped={kind:[] for kind in config['opponents']}
    with (run/'paired-deals.jsonl').open() as stream:
        for index,line in enumerate(stream):
            row=json.loads(line);kind=config['opponents'][index//n];i=index%n;value=row['value']
            returns=np.asarray(value['seat_returns_bb'],dtype=float);counts=np.asarray(value['action_counts'])
            if (row['label']!=[kind,i] or value['deal_seed']!=selected['seed_start']+i or returns.shape!=(2,)
                    or not np.isfinite(returns).all() or np.any(np.abs(returns)>10)
                    or value['paired_bb_per_hand']!=float(returns.mean()) or counts.shape!=(5,)
                    or counts.dtype.kind not in 'iu' or np.any(counts<0)):
                raise ValueError('Small teacher paired schedule or returns mismatch')
            grouped[kind].append(value)
    summary=evaluation_summary(grouped,config)
    if any(result.get(key)!=summary[key] for key in ('opponents','passes_fixed_suite')):
        raise ValueError('Small teacher result differs from recorded paired returns')
    if result.get('allowed_as_small_game_teacher')!=(profile=='confirmatory' and summary['passes_fixed_suite']):
        raise ValueError('Small-game qualification flag mismatch')
    return summary


def evaluate(policy_path,config,output,profile='development',resume=False,development_reference=None,stop_after=None,training_run=None):
    if (config.get('schema')!='tabular-shove-fold-suite-v1' or config.get('curriculum')!=CURRICULUM
            or tuple(config.get('opponents',[]))!=tuple(VERSIONS) or profile not in ('development','confirmatory')
            or config['profiles'][profile]['paired_deals_per_opponent']<2 or config['bootstrap_repeats']<100):
        raise ValueError('Registered complete small-game suite required')
    policy,record=load_policy(policy_path);policy_hash=digest(Path(policy_path)/'manifest.json')
    if training_run is None:raise ValueError('The matching completed tabular training run is required')
    training_run=Path(training_run);training=json.loads((training_run/'manifest.json').read_text())
    trained=json.loads((training_run/'result.json').read_text());training_config=training['config']
    training_range=[training_config['deal_seed_start'],training_config['deal_seed_start']+training_config['iterations']]
    if (digest(training_run/'manifest.json')!=record['provenance'].get('training_manifest_sha256')
            or training['config_hash']!=identity(training_config) or training_range!=config.get('training_deal_range')
            or trained.get('status')!='trained-unvalidated' or trained.get('iterations_completed')!=training_config['iterations']
            or record['provenance'].get('iterations_completed')!=training_config['iterations']
            or trained.get('manifest_sha256')!=digest(training_run/'manifest.json')):
        raise ValueError('Tabular policy must match its completed registered training run')
    train_audit=audit_journal(training_run/'hands.jsonl')
    if train_audit['rows']!=training_config['iterations'] or train_audit['head']!=trained.get('journal_head'):
        raise ValueError('Tabular training journal is incomplete')
    ranges=[training_range]+[[v['seed_start'],v['seed_start']+v['paired_deals_per_opponent']] for v in config['profiles'].values()]+[[6300000,6300016]]
    if any(max(a[0],b[0])<min(a[1],b[1]) for i,a in enumerate(ranges) for b in ranges[i+1:]):
        raise ValueError('Training, development, confirmation and boundary-check deals must be disjoint')
    runtime_config={**config,'profile':profile,'policy_sha256':policy_hash}
    if profile=='confirmatory':
        if not development_reference:raise ValueError('Small teacher confirmation requires passing development evidence')
        verified=verify_run(development_reference,config,policy_hash,'development')
        if not verified['passes_fixed_suite']:raise ValueError('Small teacher development suite did not pass')
        runtime_config['development_result_sha256']=digest(Path(development_reference)/'result.json')
    runtime=manifest(runtime_config,'conventional-tabular-no-connectome',identity(CURRICULUM),policy_hash,'numpy-tabular')
    output=Path(output);output.mkdir(parents=True,exist_ok=resume)
    if resume:assert_compatible(json.loads((output/'manifest.json').read_text()),runtime)
    else:atomic_json(output/'manifest.json',runtime)
    boundary=boundary_check(policy);journal=Journal(output/'paired-deals.jsonl',resume);completed=0;stopped=False
    selected=config['profiles'][profile];grouped={kind:[] for kind in config['opponents']}
    def stop(signum,frame):
        nonlocal stopped
        stopped=True
    handlers={sig:signal.signal(sig,stop) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        for kind in config['opponents']:
            for repeat in range(selected['paired_deals_per_opponent']):
                label=[kind,repeat]
                if completed<len(journal.rows):
                    saved=journal.rows[completed]
                    if saved['label']!=label:raise ValueError('Small teacher evaluation plan changed')
                    row=saved['value']
                else:
                    row=paired_match(policy,kind,selected['seed_start']+repeat);journal.record(completed,label,row)
                grouped[kind].append(row);completed+=1
                if stopped or stop_after is not None and completed>=stop_after:
                    raise KeyboardInterrupt('Stopped at complete paired deal; use --resume')
            print(json.dumps({'opponent':kind,'paired_deals':len(grouped[kind]),
                'mean_bb_per_hand':float(np.mean([row['paired_bb_per_hand'] for row in grouped[kind]]))}),flush=True)
        if completed!=len(journal.rows):raise ValueError('Unexpected trailing small teacher evaluation rows')
        summary=evaluation_summary(grouped,config)
        result={'schema':'tabular-shove-fold-evaluation-v1',**summary,'mode':'conventional-tabular-control',
            'profile':profile,'curriculum':CURRICULUM,'policy_sha256':policy_hash,
            'information_boundary':boundary,'information_boundary_verified':True,
            'allowed_as_teacher':False,'allowed_as_small_game_teacher':profile=='confirmatory' and summary['passes_fixed_suite'],
            'scope':'Small 10 BB reference only; cannot qualify the registered full 20 BB teacher or any fly learning gate',
            'journal_head':journal.rows[-1]['hash'],'manifest_sha256':digest(output/'manifest.json')}
        atomic_json(output/'result.json',result);verify_run(output,config,policy_hash,profile);return result
    finally:
        journal.close()
        for sig,handler in handlers.items():signal.signal(sig,handler)
