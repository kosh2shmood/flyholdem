"""Resumable conventional population-response regret training; no fly module."""
import json
from pathlib import Path
import signal
import time
import resource
import sys
import numpy as np
from flyholdem.connectome.registry import ROOT,digest
from flyholdem.provenance import manifest,identity,assert_compatible
from flyholdem.neural.checkpoint import Checkpoints,atomic_json
from flyholdem.experiments.journal import Journal
from flyholdem.experiments.poker_evidence import read_journal
from flyholdem.poker.engine import Hand
from flyholdem.poker.opponents import VERSIONS
from .fast_opponents import TrainingOpponent
from .equity import backend_identity
from .regret import RegretTable

SCHEMA='external-sampling-fixed-population-teacher-v1'
AGGREGATION='own-reach-weighted-average-stationary-population-v1'


def runtime(config):
    if (config.get('schema')!=SCHEMA or type(config['iterations']) is not int or config['iterations']<1
            or tuple(config['opponents'])!=tuple(VERSIONS)
            or config['opponent_probabilities']!=[.25]*4 or config['aggregation']!=AGGREGATION
            or config['stack_bb']!=config['abstraction']['stack_bb']
            or type(config['sampling_seed']) is not int or type(config['deal_seed_start']) is not int):
        raise ValueError('Complete fixed-population teacher registration required')
    import yaml
    evaluation_path=ROOT/'configs/teacher_evaluation.yaml'
    evaluation=yaml.safe_load(evaluation_path.read_text())
    training=(config['deal_seed_start'],config['deal_seed_start']+config['iterations'])
    reserved=[(p['seed_start'],p['seed_start']+p['paired_deals_per_opponent']) for p in evaluation['profiles'].values()]+[(821001,821005)]
    if any(max(training[0],a)<min(training[1],b) for a,b in reserved):
        raise ValueError('Training must exclude the registered development, confirmation and boundary-check deals')
    backend=backend_identity()
    return manifest({**config,'opponent_versions':VERSIONS,'equity_backend':backend,'evaluation_protocol_sha256':digest(evaluation_path)},'conventional-no-connectome',
        identity(config['abstraction']),identity(AGGREGATION),identity(backend))


def train(config,output,resume=False,stop_after=None):
    table=RegretTable(config['abstraction']);expected=runtime(config);root=Path(output)
    root.mkdir(parents=True,exist_ok=resume)
    if resume:assert_compatible(json.loads((root/'manifest.json').read_text()),expected)
    else:atomic_json(root/'manifest.json',expected)
    checkpoints=Checkpoints(root/'checkpoints');journal=Journal(root/'traversals.jsonl',resume)
    rng=np.random.default_rng(config['sampling_seed']);completed=0;stopped=False;in_traversal=False
    started=time.perf_counter();last_checkpoint=time.monotonic()
    if resume:
        arrays,extra,_=checkpoints.load(expected);table.restore(arrays);rng.bit_generator.state=extra['sampling_rng'];completed=extra['iterations_completed']
        if completed>len(journal.rows) or extra['journal_head']!=(journal.rows[completed-1]['hash'] if completed else '0'*64):
            raise ValueError('Regret checkpoint/traversal journal mismatch')
        if completed==config['iterations'] and (root/'result.json').is_file():
            result=json.loads((root/'result.json').read_text())
            if (len(journal.rows)!=completed or result['status']!='trained-unvalidated'
                    or result['manifest_sha256']!=digest(root/'manifest.json') or result['journal_head']!=journal.rows[-1]['hash']
                    or result['iterations_completed']!=completed or result['information_sets']!=len(table.keys)):
                journal.close();raise ValueError('Completed conventional result differs from the restored final table')
            journal.close();return result
    def checkpoint():
        nonlocal last_checkpoint
        if in_traversal:return
        checkpoints.save(table.state(),expected,{'iterations_completed':completed,'sampling_rng':rng.bit_generator.state,
            'journal_head':journal.rows[completed-1]['hash'] if completed else '0'*64,'boundary':'complete sampled-chance traversal',
            'selection':'fixed final iteration, never a held-out-profit-selected checkpoint'},completed,best=completed==config['iterations'])
        last_checkpoint=time.monotonic()
    def stop(signum,frame):
        nonlocal stopped
        stopped=True
    handlers={sig:signal.signal(sig,stop) for sig in (signal.SIGINT,signal.SIGTERM)}
    if not resume:checkpoint()
    try:
        for index in range(completed,config['iterations']):
            kind=config['opponents'][int(rng.choice(4,p=config['opponent_probabilities']))]
            opponent_seed=int(rng.integers(0,2**63));seat=index%2;deal=config['deal_seed_start']+index
            in_traversal=True
            row=table.step(Hand(deal,button=0,stacks=(2*config['stack_bb'],)*2),seat,TrainingOpponent(opponent_seed,kind))
            row.update(opponent=kind,opponent_seed=opponent_seed,opponent_version=VERSIONS[kind],deal_seed=deal)
            journal.record(index,[deal,seat,kind],row);completed=index+1;in_traversal=False
            if time.monotonic()-last_checkpoint>=300:checkpoint()
            if completed%config.get('progress_iterations',100)==0:
                print(json.dumps({'iterations':completed,'planned':config['iterations'],'information_sets':len(table.keys),
                    'seconds':time.perf_counter()-started}),flush=True)
            if stopped or stop_after is not None and completed>=stop_after:break
        if completed==config['iterations'] and len(journal.rows)!=completed:raise ValueError('Unexpected trailing regret traversals')
        rows=[item['value'] for item in journal.rows[:completed]]
        result={'schema':'external-regret-training-result-v1','status':'trained-unvalidated' if completed==config['iterations'] else 'interrupted',
            'mode':'conventional-teacher-control','iterations_completed':completed,'planned_iterations':config['iterations'],
            'operations':completed,'information_sets':len(table.keys),'counterfactual_nodes':sum(row['nodes'] for row in rows),
            'terminal_branches':sum(row['terminal_branches'] for row in rows),'learning_claim':False,'allowed_as_teacher':False,
            'scope':'Tabular regret response to a stationary fixed population; neither Nash equilibrium nor fly learning claim',
            'manifest_sha256':digest(root/'manifest.json'),'journal_head':journal.rows[completed-1]['hash'] if completed else '0'*64,
            'elapsed_seconds_this_invocation':time.perf_counter()-started,
            'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)}
        atomic_json(root/'result.json',result);return result
    finally:
        checkpoint();journal.close()
        for sig,handler in handlers.items():signal.signal(sig,handler)


def completed_table(run):
    root=Path(run);saved=json.loads((root/'manifest.json').read_text());config={k:v for k,v in saved['config'].items() if k not in ('opponent_versions','equity_backend','evaluation_protocol_sha256')}
    expected=runtime(config);assert_compatible(saved,expected)
    result=json.loads((root/'result.json').read_text());journal=read_journal(root/'traversals.jsonl')
    if (result['status']!='trained-unvalidated' or result['iterations_completed']!=config['iterations']
            or len(journal)!=config['iterations'] or result['journal_head']!=journal[-1]['hash']
            or result['manifest_sha256']!=digest(root/'manifest.json') or result['allowed_as_teacher'] is not False):
        raise ValueError('Complete unselected final conventional training evidence required')
    rng=np.random.default_rng(config['sampling_seed'])
    for index,item in enumerate(journal):
        kind=config['opponents'][int(rng.choice(4,p=config['opponent_probabilities']))];opponent_seed=int(rng.integers(0,2**63))
        deal=config['deal_seed_start']+index;seat=index%2;row=item['value']
        if (item['label']!=[deal,seat,kind] or row['opponent_seed']!=opponent_seed or row['opponent']!=kind
                or row['deal_seed']!=deal or row['learning_seat']!=seat or row['opponent_version']!=VERSIONS[kind]):
            raise ValueError('Registered fixed population or deal sampling changed')
    arrays,extra,_=Checkpoints(root/'checkpoints').load(expected)
    if extra['iterations_completed']!=config['iterations'] or extra['journal_head']!=journal[-1]['hash'] or extra['sampling_rng']!=rng.bit_generator.state:
        raise ValueError('Final numeric checkpoint does not match completed conventional training')
    table=RegretTable(config['abstraction']);table.restore(arrays)
    if len(table.keys)!=result['information_sets']:raise ValueError('Final information-set count mismatch')
    return table,{'training_manifest_sha256':digest(root/'manifest.json'),'training_result_sha256':digest(root/'result.json'),
        'training_journal_sha256':digest(root/'traversals.jsonl'),'training_source_sha256':saved['source_hash'],
        'training_commit':saved['commit'],'iterations_completed':config['iterations'],'stack_bb':config['stack_bb'],
        'population':{'opponents':config['opponents'],'probabilities':config['opponent_probabilities']},
        'aggregation':AGGREGATION}
