"""Separate conventional candidate: positive regrets at the fixed final step.

No time averaging, greedy override, temperature selection or fly involvement.
This empirical extraction has no last-iterate convergence or equilibrium claim.
"""
import json
import os
from pathlib import Path
import tempfile
import numpy as np
from flyholdem.connectome.registry import digest
from flyholdem.provenance import identity, canonical, environment
from flyholdem.neural.checkpoint import Checkpoints, atomic_json, sync_dir
from flyholdem.experiments.poker_evidence import iter_journal
from flyholdem.poker.opponents import VERSIONS
from .regret import RegretTable, VERSION
from .regret_policy import FrozenRegretPolicy, implementation as shared_implementation
from .regret_training import SCHEMA as TRAINING_SCHEMA, AGGREGATION as TRAINING_AGGREGATION

SCHEMA='teacher-final-regret-policy-v1'
EXTRACTION='final-positive-regret-extraction-v1'
AGGREGATION='normalized-positive-final-regrets-uniform-if-nonpositive-v1'
PRIOR='uniform-over-legal-actions; includes untrained stack domains'


def implementation():
    return {'current_policy_sha256':digest(__file__),'shared_regret_runtime':shared_implementation()}


def probabilities_from_regrets(regrets,legal):
    regrets=np.asarray(regrets);legal=np.asarray(legal)
    if (regrets.dtype!=np.float64 or regrets.ndim!=2 or regrets.shape[1]!=5
            or legal.dtype!=np.bool_ or legal.shape!=regrets.shape or not legal.any(axis=1).all()
            or not np.isfinite(regrets).all() or np.any(regrets[~legal])):
        raise ValueError('Finite final numeric regrets and exact legal masks required')
    positive=np.maximum(regrets,0);total=positive.sum(axis=1,keepdims=True)
    if not np.isfinite(total).all():raise ValueError('Nonfinite positive regret total')
    return np.divide(positive,total,out=legal.astype(float)/legal.sum(axis=1,keepdims=True),where=total>0)


def completed_snapshot(config):
    """Verify a pinned historical final checkpoint without migrating its runtime.

    Unlike training resume, extraction does not execute that trainer. Exact
    source/manifest/config pins authenticate which numeric final state is read;
    the whole sampling journal, checkpoint chain and RNG must still agree.
    """
    if (config.get('schema')!=EXTRACTION or config.get('aggregation')!=AGGREGATION
            or config.get('status')!='development' or type(config.get('iterations')) is not int
            or config['iterations']<1):
        raise ValueError('Registered fixed-final regret extraction required')
    root=Path(config['training_run'])
    if digest(root/'manifest.json')!=config['training_manifest_sha256']:
        raise ValueError('Pinned training manifest mismatch')
    saved=json.loads((root/'manifest.json').read_text());parameters=saved['config']
    if (saved['source_hash']!=config['training_source_sha256']
            or saved['config_hash']!=config['training_config_sha256']
            or saved['config_hash']!=identity(parameters)
            or saved['environment']!=environment()):
        raise ValueError('Pinned training source, configuration or environment mismatch')
    if (parameters['schema']!=TRAINING_SCHEMA or parameters['aggregation']!=TRAINING_AGGREGATION
            or parameters['iterations']!=config['iterations']
            or parameters['stack_bb']!=parameters['abstraction']['stack_bb']
            or parameters['opponents']!=list(VERSIONS) or parameters['opponent_probabilities']!=[.25]*4
            or parameters['opponent_versions']!=VERSIONS):
        raise ValueError('Original fixed-population training definition required')
    result=json.loads((root/'result.json').read_text());count=config['iterations']
    if (result['schema']!='external-regret-training-result-v1' or result['status']!='trained-unvalidated'
            or result['iterations_completed']!=count or result['planned_iterations']!=count
            or result['operations']!=count or result['allowed_as_teacher'] is not False
            or result['manifest_sha256']!=config['training_manifest_sha256']):
        raise ValueError('Complete fixed-final training required before current-strategy extraction')
    rng=np.random.default_rng(parameters['sampling_seed']);nodes=leaves=rows=0;head='0'*64
    for index,item in enumerate(iter_journal(root/'traversals.jsonl')):
        kind=parameters['opponents'][int(rng.choice(4,p=parameters['opponent_probabilities']))]
        opponent_seed=int(rng.integers(0,2**63));deal=parameters['deal_seed_start']+index;seat=index%2
        row=item['value']
        if (item['label']!=[deal,seat,kind] or row['opponent_seed']!=opponent_seed
                or row['opponent']!=kind or row['deal_seed']!=deal or row['learning_seat']!=seat
                or row['opponent_version']!=VERSIONS[kind]):
            raise ValueError('Final strategy sampling journal mismatch')
        rows+=1;nodes+=row['nodes'];leaves+=row['terminal_branches'];head=item['hash']
    if (rows!=count or head!=result['journal_head'] or nodes!=result['counterfactual_nodes']
            or leaves!=result['terminal_branches']):
        raise ValueError('Incomplete final strategy journal or result totals')
    arrays,extra,meta=Checkpoints(root/'checkpoints').load(saved)
    if (meta['step']!=count or extra['iterations_completed']!=count or extra['journal_head']!=head
            or extra['sampling_rng']!=rng.bit_generator.state):
        raise ValueError('Final strategy checkpoint/RNG boundary mismatch')
    table=RegretTable(parameters['abstraction']);table.restore(arrays)
    if len(table.keys)!=result['information_sets']:raise ValueError('Final strategy information-set count mismatch')
    return table,{'training_manifest_sha256':digest(root/'manifest.json'),
        'training_result_sha256':digest(root/'result.json'),'training_journal_sha256':digest(root/'traversals.jsonl'),
        'training_source_sha256':saved['source_hash'],'training_commit':saved['commit'],
        'iterations_completed':count,'stack_bb':parameters['stack_bb'],
        'population':{'opponents':parameters['opponents'],'probabilities':parameters['opponent_probabilities']},
        'training_aggregation':TRAINING_AGGREGATION,'final_checkpoint_sha256':identity(meta),
        'final_regrets_npy_sha256':meta['arrays']['regrets']['sha256'],'extraction_config':config}


def export_current(config,output):
    output=Path(output)
    if output.exists():raise FileExistsError('Write a new final-regret policy directory')
    table,provenance=completed_snapshot(config);state=table.state()
    arrays={key:state[key] for key in ('key_offsets','key_bytes','regrets','legal')}
    arrays['probabilities']=probabilities_from_regrets(state['regrets'],state['legal'])
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=output.name+'-partial-',dir=output.parent) as temporary:
        root=Path(temporary)/'model';root.mkdir();files={}
        for name,value in arrays.items():
            path=root/(name+'.npy')
            with path.open('wb') as stream:
                np.save(stream,value,allow_pickle=False);stream.flush();os.fsync(stream.fileno())
            files[path.name]=digest(path)
        if files['regrets.npy']!=provenance['final_regrets_npy_sha256']:
            raise ValueError('Export must preserve exact final checkpoint regret bytes')
        runtime=implementation()
        record={'schema':SCHEMA,'mode':'conventional-teacher-control','aggregation':AGGREGATION,
            'feature_version':VERSION,'config':table.config,'information_sets':len(table.keys),
            'files':files,'provenance':provenance,'implementation':runtime,
            'inference_backend':identity(runtime['shared_regret_runtime']['equity_backend']),
            'allowed_as_teacher':False,'unseen_state_prior':PRIOR,
            'scope':'Fixed-final positive regret candidate; no equilibrium, last-iterate guarantee or fly-learning claim'}
        atomic_json(root/'manifest.json',record);sync_dir(root);root.rename(output);sync_dir(output.parent)
    return {'policy_sha256':digest(output/'manifest.json'),'allowed_as_teacher':False}


def load_policy(root):
    root=Path(root);record=json.loads((root/'manifest.json').read_text())
    if (record['schema']!=SCHEMA or record['aggregation']!=AGGREGATION or record['feature_version']!=VERSION
            or record['unseen_state_prior']!=PRIOR or record['implementation']!=implementation()):
        raise ValueError('Final-regret source/runtime or policy definition mismatch')
    RegretTable(record['config']);origin=record['provenance'];extraction=origin['extraction_config']
    if (record['config']['stack_bb']!=origin['stack_bb'] or record['config']['feature_version']!=VERSION
            or origin['iterations_completed']!=extraction['iterations'] or extraction['schema']!=EXTRACTION
            or extraction['aggregation']!=AGGREGATION or extraction['status']!='development'
            or origin['training_manifest_sha256']!=extraction['training_manifest_sha256']
            or origin['training_source_sha256']!=extraction['training_source_sha256']):
        raise ValueError('Final-regret extraction/provenance binding mismatch')
    names={'key_offsets','key_bytes','regrets','probabilities','legal'}
    if (set(record['files'])!={name+'.npy' for name in names}
            or any(digest(root/name)!=sha for name,sha in record['files'].items())
            or record['files']['regrets.npy']!=origin['final_regrets_npy_sha256']):
        raise ValueError('Final-regret numeric files changed')
    arrays={name:np.load(root/(name+'.npy'),allow_pickle=False) for name in names}
    offsets,data,p,legal,regrets=[arrays[k] for k in ('key_offsets','key_bytes','probabilities','legal','regrets')]
    n=record['information_sets']
    if (type(n) is not int or n<1 or offsets.dtype!=np.int64 or offsets.shape!=(n+1,)
            or offsets[0]!=0 or np.any(np.diff(offsets)<=0) or data.dtype!=np.uint8 or data.ndim!=1
            or offsets[-1]!=len(data) or p.dtype!=np.float64 or p.shape!=(n,5)
            or legal.shape!=(n,5) or regrets.shape!=(n,5)
            or not np.array_equal(p,probabilities_from_regrets(regrets,legal))):
        raise ValueError('Final-regret probabilities must exactly match the complete numeric state')
    keys=[data[a:b].tobytes() for a,b in zip(offsets[:-1],offsets[1:])]
    if len(set(keys))!=n:raise ValueError('Duplicate final-regret information key')
    for key,mask in zip(keys,legal):
        decoded=json.loads(key)
        if (canonical(decoded)!=key or decoded[0]!=VERSION or decoded[1]!=4*origin['stack_bb']
                or decoded[-1]!=mask.tolist()):raise ValueError('Final-regret information key binding changed')
    return FrozenRegretPolicy(record['config'],keys,p,legal),record
