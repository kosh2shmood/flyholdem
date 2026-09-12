"""Resumable, sequential native curriculum phases and their matched controls.

The public entry point in curriculum_protocol verifies prerequisites first.
The private executor permits small numerical CI fixtures, never a gate bypass.
"""
import hashlib
import json
from pathlib import Path
import signal
import tempfile
import time
import resource
import sys
import numpy as np
from flyholdem.connectome.registry import digest
from flyholdem.provenance import identity, manifest, assert_compatible
from flyholdem.neural.checkpoint import atomic_json
from flyholdem.neural.kernel.build import LIBRARY
from flyholdem.interface.frozen import export_frozen, verify_changed_edges
from flyholdem.learning.poker_controls import shuffled_rewards
from .poker_training import _train_arm
from .disconnected_evaluation import evaluate_disconnected
from .poker_evidence import read_journal, audit_evaluation, aggregate_endpoints, compact_rows
from .journal import Journal


def _model(player,path,reference):
    """Persist inherited modifications separately from the currently trainable set."""
    path=Path(path);brain=player.brain
    changed=np.flatnonzero(brain.weights!=brain.initial)
    edges=np.union1d(player.eligible.edges,changed).astype(np.int64)
    bounds=[player.eligible.lower,player.eligible.upper]
    verify_changed_edges(brain,edges,bounds)
    reference={**reference,'weights_sha256':hashlib.sha256(brain.weights.tobytes()).hexdigest(),'trainable_edge_indices_hash':identity(player.eligible.edges.tolist()),
        'trainable_edge_count':len(player.eligible.edges),'persisted_edge_count':len(edges),
        'persisted_edges_include_inherited_changes':True,'selection':'registered final boundary; no profit selection'}
    if path.exists():
        record=json.loads((path/'manifest.json').read_text())
        if (record['training_reference']!=reference or record['learning_mode']!=player.mode or record['weight_bounds']!=bounds
                or record['graph_hash']!=brain.graph_hash or record['binary_sha256']!=brain.build['binary_sha256']
                or json.loads((path/'preregistration.json').read_text())!=player.controller.registration
                or any(digest(path/name)!=checksum for name,checksum in record['files'].items())
                or not np.array_equal(np.load(path/'edge_indices.npy',allow_pickle=False),edges)
                or not np.array_equal(np.load(path/'edge_weights.npy',allow_pickle=False),brain.weights[edges])):
            raise ValueError('Existing curriculum model differs from its exact registered phase')
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='model-partial-',dir=path.parent) as tmp:
            export_frozen(player.controller,edges,bounds,player.mode,Path(tmp)/'model',reference,
                {'status':'unevaluated','learning_claim':False})
            (Path(tmp)/'model').rename(path)
    return digest(path/'manifest.json')


def _training_rows(path,plan,seed,arm):
    path=Path(path);result=json.loads((path/'result.json').read_text());runtime=json.loads((path/'manifest.json').read_text())
    rows=read_journal(path/'hands.jsonl');config=runtime['config']
    expected={**plan['training'],'schema':'poker-training-arm-v1','arm':arm,'curriculum':plan['curriculum'],
        'seed':seed,'deal_seed_start':plan['training_starts'][str(seed)],
        'teacher_split':'train' if plan['optimization']!='terminal-local-eligibility' else None}
    if (any(config.get(key)!=value for key,value in expected.items())
            or config['initial_rng']!=np.random.default_rng(seed).bit_generator.state
            or 'player_config' in plan and config['player_config']!=plan['player_config']
            or 'plasticity' in plan and config['plasticity_registration']['config']!=plan['plasticity']):
        raise ValueError('Training implementation differs from its preregistered parameters')
    from flyholdem.learning.dopamine import PastBaseline
    baseline=PastBaseline();control_rewards=None
    if arm=='shuffled-reward':
        plastic=_training_rows(path.parent.parent/'plastic/training',plan,seed,'plastic')
        control_rewards=shuffled_rewards([r['neural_return_bb'] for r in plastic],
            [int(r['neural_seat']==0) for r in plastic],int(identity([seed,'reward-control'])[:16],16))
        if config['control_rewards_sha256']!=hashlib.sha256(control_rewards.tobytes()).hexdigest():
            raise ValueError('Shuffled rewards differ from the matched plastic-arm returns')
    if (result['status']!='training-complete' or result['hands_completed']!=plan['training']['hands']
            or len(rows)!=plan['training']['hands'] or result['journal_head']!=rows[-1]['hash']
            or result['manifest_sha256']!=digest(path/'manifest.json') or runtime['config_hash']!=identity(config)
            or config['seed']!=seed or config['arm']!=arm or result['learning_claim'] is not False):
        raise ValueError('Complete registered training-arm evidence required')
    for index,item in enumerate(rows):
        row=item['value'];kind=plan['training']['opponent_cycle'][(index//2)%len(plan['training']['opponent_cycle'])]
        deal=plan['training_starts'][str(seed)]+index//2;seat=index%2
        if (item['label']!=[kind,deal,seat] or row['neural_seat']!=seat or row['deal_seed']!=deal
                or row['curriculum']!=plan['curriculum'] or row['phase']!='training' or row['learning']!=(arm!='frozen')
                or row['learning_mode']!=plan['learning_mode'] or row['optimization']!=plan['optimization']
                or row['opponent']!=kind):
            raise ValueError('Training-arm matched schedule changed')
        from flyholdem.learning.curriculum import CurriculumHand
        game=CurriculumHand.restore(row['private_hand_checkpoint'])
        if (not game.done or game.view()!=row['public_terminal']
                or row['neural_return_bb']!=game.view()['payoffs'][seat]/2):
            raise ValueError('Training return differs from the actual settled PokerKit hand')
        if index==0 and row['weights_before_sha256']!=result['initial_weights_sha256']:
            raise ValueError('Training did not start with its registered initial weights')
        if index and row['weights_before_sha256']!=rows[index-1]['value']['weights_after_sha256']:
            raise ValueError('Training weights changed between registered hands')
        if index==len(rows)-1 and row['weights_after_sha256']!=result['final_weights_sha256']:
            raise ValueError('Final training weights do not match the last complete hand')
        if arm=='frozen' and row['weights_before_sha256']!=row['weights_after_sha256']:
            raise ValueError('Frozen training changed weights')
        if plan['optimization']=='terminal-local-eligibility' and arm!='frozen':
            raw=row['neural_return_bb'] if control_rewards is None else float(control_rewards[index])
            reward=baseline.event(raw,'position-'+str(int(seat==0)),config['player_config']['reward_scale_bb'])
            terminal=row['terminal_reinforcement']
            if (terminal['reinforcement']!=reward or terminal['raw_net_bb']!=row['neural_return_bb']
                    or not terminal['reward_delivered'] or terminal['reward_shuffled']!=(control_rewards is not None)):
                raise ValueError('Terminal training used a different reward or non-past baseline')
        if plan['optimization']=='terminal-local-eligibility' or arm=='frozen':
            if row['teacher_connected'] or any(d['original_teacher_target'] is not None for d in row['neural_decisions']):
                raise ValueError('A terminal or frozen arm received teacher labels')
        else:
            from flyholdem.teacher.corpus import split_for_id
            if row['teacher_split']!='train':raise ValueError('Online targets must exclude validation and test information IDs')
            for decision in row['neural_decisions']:
                from flyholdem.poker.infoset import canonical_information_id
                info=canonical_information_id(decision['observation'])
                if info!=decision['teacher_input_id'] or decision['teacher_target_held_out']!=(split_for_id(info)!='train'):
                    raise ValueError('Held-out teacher target exclusion mismatch')
    return [item['value'] for item in rows]


def _execute_curriculum(plan,output,factory,*,teacher=None,teacher_sha256=None,frozen_opponents=None,
                        authorization=None,reference_policy=None,resume=False,stop_after=None):
    """Factory returns a fresh (PokerLearningPlayer, prepared_graph_path)."""
    started=time.perf_counter()
    root=Path(output);root.mkdir(parents=True,exist_ok=resume)
    protocol={'schema':'native-poker-curriculum-run-v1','plan':plan,'authorization':authorization,
        'scope':'gated-full-connectome' if authorization else 'numerical-engineering-fixture-only'}
    runtime=manifest(protocol,plan['graph_hash'],plan['encoder_hash'],plan['decoder_hash'],digest(LIBRARY))
    if resume:assert_compatible(json.loads((root/'manifest.json').read_text()),runtime)
    else:atomic_json(root/'manifest.json',runtime)
    journal=Journal(root/'phases.jsonl',resume);phases={};operation=0;stopped=False
    def stop(signum,frame):
        nonlocal stopped
        stopped=True
    handlers={sig:signal.signal(sig,stop) for sig in (signal.SIGINT,signal.SIGTERM)}
    def phase(seed,name,player,graph,training=None,idle=None):
        nonlocal operation
        folder=root/str(seed)/name
        reference={'plan_sha256':identity(plan),'seed':seed,'phase':name,
            'training':None if training is None else {key:digest(training/key) for key in ('manifest.json','result.json','hands.jsonl')},
            'retention':idle}
        model_hash=_model(player,folder/'model',reference)
        config={**plan['evaluation'],'mode':player.controller.registration['mode'],
            'unavailable_training_paths':[*plan['evaluation'].get('unavailable_training_paths',[]),str(root.resolve())]}
        evaluated=folder/'evaluation'
        # A phase already in the outer journal is re-audited without rewriting
        # its result. Partial children resume their complete-hand checkpoints.
        if not (evaluated/'evaluation/result.json').exists() or json.loads((evaluated/'evaluation/result.json').read_text())['status']!='baseline-complete':
            result=evaluate_disconnected(config,evaluated,folder/'model',graph,resume=evaluated.exists())
            if result['result']['status']!='baseline-complete':return False
        audit=audit_evaluation(evaluated,config,model_hash)
        evidence={'model_sha256':model_hash,'graph_manifest_sha256':digest(Path(graph)/'manifest.json'),
            'evaluation':audit['evidence'],'training':reference['training'],'retention':idle}
        journal.record(operation,[seed,name],evidence);operation+=1
        phases.setdefault(seed,{})[name]=compact_rows(audit['rows'])
        atomic_json(root/'progress.json',{'completed_phases':operation,'seed':seed,'phase':name,'learning_claim':False})
        print(json.dumps({'curriculum_phase':[seed,name],'completed_phases':operation}),flush=True)
        return not stopped and not (stop_after is not None and operation>=stop_after)
    def interrupted():
        result={'schema':'native-poker-curriculum-result-v1','status':'interrupted','completed_phases':operation,
            'manifest_sha256':digest(root/'manifest.json'),'learning_claim':False,'gate_passed':False}
        atomic_json(root/'result.json',result);return result
    try:
        reference_evidence=None
        if authorization is not None:
            if reference_policy is None:raise ValueError('A separately qualified conventional reference is required')
            from flyholdem.teacher.curriculum_reference import run_reference,verify_reference
            reference_root=root/'conventional-reference'
            value=run_reference(reference_policy,plan['reference_teacher'],plan['evaluation'],reference_root,resume=reference_root.exists())
            if value['status']!='complete':return interrupted()
            reference_evidence=verify_reference(reference_policy,plan['reference_teacher'],plan['evaluation'],reference_root)
        for seed in plan['seeds']:
            player,graph=factory(seed,'plastic',root/str(seed)/'null-graph')
            initial=player.brain.weights.copy()
            if not phase(seed,'initial',player,graph):return interrupted()
            # Each child verifies its original initial weights before restoring
            # a prior complete-hand checkpoint. No evaluation state is reused.
            del player
            plastic_rows=None
            for arm in ['plastic',*plan['controls']]:
                player,graph=factory(seed,arm,root/str(seed)/'null-graph')
                train_path=root/str(seed)/arm/'training'
                cfg={**plan['training'],'schema':'poker-training-arm-v1','arm':arm,'curriculum':plan['curriculum'],
                    'seed':seed,'deal_seed_start':plan['training_starts'][str(seed)],'teacher_split':'train' if player.requires_teacher else None}
                reward=None
                if arm=='shuffled-reward':
                    reward=shuffled_rewards([r['neural_return_bb'] for r in plastic_rows],
                        [int(r['neural_seat']==0) for r in plastic_rows],int(identity([seed,'reward-control'])[:16],16))
                teaching=player.requires_teacher and arm!='frozen'
                result=_train_arm(player,cfg,train_path,teacher=teacher if teaching else None,
                    teacher_sha256=teacher_sha256 if teaching else None,shuffled_returns=reward,
                    frozen_opponents=frozen_opponents,resume=train_path.exists())
                if result['status']!='training-complete':return interrupted()
                rows=_training_rows(train_path,plan,seed,arm)
                if arm=='plastic':plastic_rows=rows
                if not phase(seed,arm,player,graph,train_path):return interrupted()
                if arm=='plastic':
                    before=hashlib.sha256(player.brain.weights.tobytes()).hexdigest()
                    spikes=player.brain.advance(np.zeros(player.brain.n,dtype=np.float32),plan['retention_ms'])
                    after=hashlib.sha256(player.brain.weights.tobytes()).hexdigest()
                    if before!=after:raise AssertionError('Unreinforced retention interval changed weights')
                    idle={'duration_ms':plan['retention_ms'],'before_weights_sha256':before,'after_weights_sha256':after,
                        'native_counts_sha256':hashlib.sha256(spikes.tobytes()).hexdigest(),'learning':False,'teacher_connected':False}
                    if not phase(seed,'retention',player,graph,train_path,idle):return interrupted()
                    player.brain.weights[:]=initial;player.brain.reset_dynamics()
                    if not phase(seed,'erased',player,graph):return interrupted()
                del player
            del initial
        if len(journal.rows)!=operation:raise ValueError('Unexpected trailing curriculum phases')
        endpoints=aggregate_endpoints(phases,plan)
        result={'schema':'native-poker-curriculum-result-v1','status':'complete','completed_phases':operation,
            'manifest_sha256':digest(root/'manifest.json'),'journal_head':journal.rows[-1]['hash'],
            'endpoints':endpoints,'conventional_reference':reference_evidence,'gate_passed':bool(authorization and endpoints['gate_passed']),
            'learning_claim':bool(authorization and endpoints['gate_passed']),
            'scope':protocol['scope'],'elapsed_seconds_this_invocation':time.perf_counter()-started,
            'resource_measurement':{'parent_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
                'largest_child_peak_rss_bytes':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
                'scope':'Separate process peaks, not a measurement of simultaneous system memory'},
            'seed_models':{str(seed):{'path':str((root/str(seed)/'plastic/model').resolve()),
                'sha256':digest(root/str(seed)/'plastic/model/manifest.json')} for seed in plan['seeds']}}
        atomic_json(root/'result.json',result);return result
    finally:
        journal.close()
        for sig,handler in handlers.items():signal.signal(sig,handler)
