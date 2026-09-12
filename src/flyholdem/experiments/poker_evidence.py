"""Recompute native poker experiment evidence from complete isolated hands.

This checks recorded execution consistency, not a cryptographic attestation or
an independent rerun of all historical native learning.
"""
import json
import hashlib
from pathlib import Path
import numpy as np
from flyholdem.connectome.registry import digest
from flyholdem.provenance import identity
from flyholdem.poker.opponents import Opponent, VERSIONS, visible_equity
from flyholdem.interface.encoder import encode_player_state, encoded_hash
from flyholdem.learning.curriculum import CurriculumHand
from .reports import audit_journal
from .evaluate import summarize
from .conditioning import paired_evidence


def iter_journal(path):
    """Yield audited records with bounded memory; consume fully before use.

    The final digest also binds the bytes consumed to the verified file, rather
    than reading an unchecked replacement after a separate successful audit.
    """
    checked=audit_journal(path);whole=hashlib.sha256();count=0
    with Path(path).open('rb') as stream:
        for line in stream:
            whole.update(line);count+=1
            yield json.loads(line)
    if whole.hexdigest()!=checked['sha256'] or count!=checked['rows'] or digest(path)!=checked['sha256']:
        raise ValueError('Journal changed while consuming audited records')


def read_journal(path):
    return list(iter_journal(path))


def audit_evaluation(root, expected_config=None, expected_model=None):
    """Verify isolation bindings, original opponent actions and full settlements."""
    root=Path(root);run=root/'evaluation';runtime=root/'runtime'
    contract=json.loads((root/'isolation.json').read_text());config=contract['config']
    isolated=json.loads((root/'isolation-result.json').read_text())
    result=json.loads((run/'result.json').read_text());manifest=json.loads((run/'manifest.json').read_text())
    if (contract['config_hash']!=identity(config) or expected_config is not None and config!=expected_config
            or expected_model is not None and contract['model_sha256']!=expected_model
            or result['status']!='baseline-complete' or result['teacher_connected'] is not False
            or result['weights_unchanged'] is not True or result['learning_claim'] is not False
            or isolated!={'schema':'disconnected-native-evaluation-v1','teacher_import_denied':True,
                'external_training_files_denied':True,'teacher_modules_loaded':False,'hands_completed':result['hands_completed'],
                'weights_unchanged':True,'status':'baseline-complete','learning_claim':False,
                **({'unavailable_training_paths_denied':contract['unavailable_training_paths']} if 'unavailable_training_paths' in contract else {})}
            or tuple(config['opponents'])!=tuple(VERSIONS)):
        raise ValueError('Complete teacher-disconnected evaluation required')
    if (identity(contract['source_files'])!=contract['source_files_hash']
            or (runtime/'src/flyholdem/teacher').exists() or list((runtime/'src').rglob('*.pyc'))
            or digest(runtime/'uv.lock')!=contract['lock_sha256']
            or digest(Path(contract['graph'])/'manifest.json')!=contract['graph_manifest_sha256']):
        raise ValueError('Isolated runtime source, graph, lock or teacher exclusion changed')
    binaries=[p for p in (runtime/'runs/build').iterdir() if p.suffix in ('.so','.dylib','.dll')]
    if len(binaries)!=1 or digest(binaries[0])!=contract['binary_sha256']:
        raise ValueError('Isolated native binary changed')
    for name,checksum in contract['source_files'].items():
        if digest(runtime/'src/flyholdem'/name)!=checksum:raise ValueError('Isolated evaluation source copy changed')
    model=runtime/'model';record=json.loads((model/'manifest.json').read_text())
    if digest(model/'manifest.json')!=contract['model_sha256'] or result['model_sha256']!=contract['model_sha256']:
        raise ValueError('Evaluated frozen model identity mismatch')
    for name,checksum in record['files'].items():
        if digest(model/name)!=checksum:raise ValueError('Evaluated frozen model bytes changed')
    registration=json.loads((model/'preregistration.json').read_text())
    weights_hash=frozen_weights_hash(model,Path(contract['graph']))
    if (weights_hash!=manifest['config']['weights_sha256'] or record['binary_sha256']!=contract['binary_sha256']
            or record['graph_hash']!=registration['graph_hash'] or record['teacher_connected'] is not False):
        raise ValueError('Frozen evaluation weights differ from the actual numeric model')
    if (manifest['config_hash']!=identity(manifest['config']) or result['manifest_sha256']!=digest(run/'manifest.json')
            or any(manifest['config'].get(k)!=v for k,v in config.items())
            or manifest['config']['model_sha256']!=contract['model_sha256']
            or manifest['config']['opponent_versions']!=VERSIONS):
        raise ValueError('Frozen evaluation manifest mismatch')
    journal=read_journal(run/'hands.jsonl');n=config['paired_deals_per_opponent'];planned=2*n*len(VERSIONS)
    if (len(journal)!=planned or result['hands_completed']!=planned or result['planned_hands']!=planned
            or result['journal_head']!=journal[-1]['hash']):raise ValueError('Incomplete paired evaluation journal')
    rows=[]
    for index,item in enumerate(journal):
        kind=config['opponents'][index//(2*n)];seed=config['deal_seed_start']+(index//2)%n;seat=index%2;row=item['value']
        if (item['label']!=[kind,seed,seat] or row['opponent']!=kind or row['opponent_version']!=VERSIONS[kind]
                or row['deal_seed']!=seed or row['neural_seat']!=seat or row['curriculum']!=config['curriculum']
                or any(row[k] is not False for k in ('learning','teacher_connected','reward_delivered'))):
            raise ValueError('Paired evaluation schedule or inference boundary changed')
        game=CurriculumHand(config['curriculum'],seed,button=0);other=Opponent(seed+2000000,kind)
        decisions=iter(row['neural_decisions']);counts=np.zeros(5,dtype=np.int64)
        while not game.done:
            observation=game.observation()
            if game.actor!=seat:game.act(other.act(observation));continue
            decision=next(decisions,None)
            if decision is None:raise ValueError('Missing native decision')
            legal=np.asarray(observation['legal_mask'],dtype=bool);scores=np.asarray(decision['scores'],dtype=float)
            rates=np.asarray(decision['raw_rates_hz'],dtype=float)
            expected=(rates-np.asarray(registration['baseline_hz']))/registration['config']['score_scale_hz']
            if (scores.shape!=(5,) or rates.shape!=(5,) or not np.isfinite(scores).all() or not np.isfinite(rates).all()
                    or np.any(rates<0) or not np.array_equal(scores,expected)
                    or decision['observation']!=observation or decision['legal_mask']!=legal.tolist()
                    or decision['encoded_hash']!=encoded_hash(encode_player_state(observation))
                    or decision['temperature']!=0 or decision['decoder_fallback'] is not False
                    or decision['score_source']!='MaleCNS-native-LIF-spikes'
                    or decision['selected']!=int(np.argmax(np.where(legal,scores,-np.inf)))):
                raise ValueError('Recorded action differs from its visible native legal argmax')
            action=game.act(decision['selected']);counts[decision['selected']]+=1
            if action!=decision['committed_action']:raise ValueError('Recorded PokerKit action mismatch')
        if (next(decisions,None) is not None or counts.tolist()!=row['action_counts'] or game.view()!=row['public_terminal']
                or game.serialize()!=row['private_hand_checkpoint'] or row['neural_return_bb']!=game.view()['payoffs'][seat]/2):
            raise ValueError('Recorded native hand differs from actual PokerKit settlement')
        rows.append(row)
    if summarize(rows,config)!=result['opponents']:raise ValueError('Frozen summary differs from its complete paired hands')
    return {'rows':rows,'config':config,'weights_sha256':weights_hash,'model_sha256':contract['model_sha256'],
        'evidence':{name:digest(root/name) for name in ('isolation.json','isolation-result.json','evaluation/manifest.json',
            'evaluation/result.json','evaluation/hands.jsonl')}}



def frozen_weights_hash(model,graph):
    """Check actual numeric edge bytes without allocating a native worker."""
    from flyholdem.connectome.prepare import load_graph
    from flyholdem.interface.population import scaled_weights
    model=Path(model);arrays,prepared=load_graph(graph)
    record=json.loads((model/'manifest.json').read_text());registration=json.loads((model/'preregistration.json').read_text())
    if prepared['graph_hash']!=registration.get('base_graph_hash',registration['graph_hash']):
        raise ValueError('Frozen model base graph mismatch')
    weights=scaled_weights(arrays['weight'],registration['config'].get('global_weight_scale',1))
    h=hashlib.sha256()
    for array in (arrays['ptr'],arrays['post'],weights):h.update(memoryview(array))
    if h.hexdigest()!=record['graph_hash']:raise ValueError('Frozen model scaled graph mismatch')
    indices=np.load(model/'edge_indices.npy',allow_pickle=False);values=np.load(model/'edge_weights.npy',allow_pickle=False)
    if (indices.dtype!=np.int64 or indices.ndim!=1 or not len(indices) or len(np.unique(indices))!=len(indices)
            or np.any(indices<0) or np.any(indices>=len(weights)) or values.dtype!=np.float32
            or values.shape!=indices.shape or not np.isfinite(values).all() or np.any(weights[indices]==0)):
        raise ValueError('Invalid frozen numeric edges')
    lower,upper=record['weight_bounds'];ratios=values.astype(float)/weights[indices]
    tolerance=np.finfo(np.float32).eps*max(1,upper)*2
    if not 0<lower<=1<=upper or np.any(ratios<lower-tolerance) or np.any(ratios>upper+tolerance):
        raise ValueError('Frozen model changed edge bounds')
    weights[indices]=values
    return hashlib.sha256(weights.tobytes()).hexdigest()



def compact_rows(rows):
    """Keep endpoint fields and a fingerprint of every full native hand byte.

    Full journals remain on disk. This avoids retaining all private poker
    trajectories for every seed/control in the parent process at once.
    """
    return [{**{key:row[key] for key in ('opponent','deal_seed','neural_seat','neural_return_bb','action_counts')},
        'complete_record_sha256':identity(row),
        'neural_decisions':[{'selected':d['selected'],'observation':{key:d['observation'][key] for key in ('hole','board','history')}}
            for d in row['neural_decisions']] if row['curriculum']=='shove-fold-10bb-v1' else []} for row in rows]

def aggregate_endpoints(seed_phases, config):
    """One sampling unit per independent training seed, never per poker hand."""
    seeds=config['seeds'];controls=config['controls'];phases=['initial','plastic','retention','erased',*controls]
    if list(seed_phases)!=seeds:raise ValueError('Every registered independent training seed is required in order')
    values={phase:[] for phase in phases};held={phase:[] for phase in phases};actions=np.zeros(5,dtype=np.int64)
    buckets=np.zeros((4,2),dtype=np.int64);equities={};retained=True;erased=True
    for seed in seeds:
        item=seed_phases[seed]
        if set(item)!=set(phases):raise ValueError('Every registered native control and retention/erasure phase is required')
        reference=[(r['opponent'],r['deal_seed'],r['neural_seat']) for r in item['initial']]
        if not reference or len(set(reference))!=len(reference):raise ValueError('Distinct complete held-out hands required')
        for phase in phases:
            rows=item[phase]
            if [(r['opponent'],r['deal_seed'],r['neural_seat']) for r in rows]!=reference:
                raise ValueError('Trained/control phases must use identical paired deals, seats and opponents')
            returns=np.asarray([r['neural_return_bb'] for r in rows],dtype=float)
            if not np.isfinite(returns).all():raise ValueError('Finite settled returns required')
            values[phase].append(float(returns.mean()*100))
            selected=[r['neural_return_bb'] for r in rows if r['opponent']==config['held_out_opponent']]
            if not selected:raise ValueError('The registered held-out opponent is missing')
            held[phase].append(float(np.mean(selected)*100))
        # Fresh-rest frozen inference must be bit-for-bit stable after an idle
        # interval; erasure restores the stage's starting weights, not a new seed.
        retained &= item['plastic']==item['retention'];erased &= item['initial']==item['erased']
        for row in item['plastic']:
            actions+=row['action_counts']
            if config['curriculum']!='shove-fold-10bb-v1' or row['neural_seat']!=0:continue
            for decision in row['neural_decisions']:
                observation=decision['observation']
                if observation['board'] or observation['history']:continue
                cards=sorted(observation['hole'],key=lambda c:'23456789TJQKA'.index(c[0]),reverse=True)
                hole=[cards[0][0]+'c',cards[1][0]+('c' if cards[0][1]==cards[1][1] else 'd')]
                key=''.join(hole)
                if key not in equities:equities[key]=visible_equity({'hole':hole,'board':[]},samples=config['strength_samples'])
                bucket=int(np.searchsorted([.4,.5,.6],equities[key],side='right'))
                buckets[bucket,0]+=1;buckets[bucket,1]+=int(decision['selected']==4)
    evidence={}
    for index,control in enumerate(['initial',*controls,'erased']):
        differences=np.asarray(values['plastic'])-values[control]
        evidence[control]=paired_evidence(differences,config['bootstrap_seed']+index,config['bootstrap_repeats'])
    evidence['held-out-opponent']=paired_evidence(np.asarray(held['plastic'])-held['initial'],
        config['bootstrap_seed']+100,config['bootstrap_repeats'])
    evidence['retained-improvement']=paired_evidence(np.asarray(values['retention'])-values['initial'],
        config['bootstrap_seed']+101,config['bootstrap_repeats'])
    positive=all(e['paired_seed_bootstrap_95'][0]>0 for e in evidence.values())
    significant=all(e['one_sided_sign_flip_p']<=.05 for e in evidence.values())
    frequency=np.divide(buckets[:,1],buckets[:,0],out=np.zeros(4,dtype=float),where=buckets[:,0]>0)
    monotonic=(bool(np.all(buckets[:,0]>=config['minimum_bucket_decisions']) and np.all(np.diff(frequency)>=0)
        and frequency[-1]-frequency[0]>=config['minimum_shove_span']))
    distribution=actions/max(1,int(actions.sum()))
    noncollapse=bool(np.count_nonzero(distribution>=config['minimum_action_frequency'])>=3 and distribution.max()<=config['maximum_action_frequency'])
    behavior=monotonic if config['curriculum']=='shove-fold-10bb-v1' else noncollapse
    passed=bool(positive and retained and erased and behavior and (config['profile']!='confirmatory' or significant))
    return {'schema':'poker-curriculum-endpoints-v1','unit':'BB/100 hands; resample independent training seeds',
        'seed_returns':values,'held_out_opponent_returns':held,'paired_evidence':evidence,
        'retention_decisions_exact':bool(retained),'erasure_decisions_exact':bool(erased),
        'action_counts':actions.tolist(),'action_frequencies':distribution.tolist(),'noncollapsed':noncollapse,
        'opening_shove':{'equity_method':'visible canonical hand-class Monte Carlo, scoring process only',
            'samples':config['strength_samples'],'bucket_boundaries':[.4,.5,.6],'counts':buckets[:,0].tolist(),
            'shoves':buckets[:,1].tolist(),'frequencies':frequency.tolist(),'monotonic':monotonic},
        'criteria_met':passed,'gate_passed':passed and config['profile']=='confirmatory',
        'native_training_reexecuted':False}
