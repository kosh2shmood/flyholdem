"""Prerequisite-enforced entry point for the four registered poker stages."""
import json
from pathlib import Path
import numpy as np
import yaml
from flyholdem.connectome.registry import ROOT,digest
from flyholdem.provenance import identity
from flyholdem.interface.population import load_controller
from flyholdem.interface.frozen import load_frozen
from flyholdem.learning.eligibility import Eligibility
from flyholdem.learning.dopamine import annotated_populations
from flyholdem.learning.poker_player import PokerLearningPlayer
from flyholdem.learning.poker_controls import shuffled_encoder,shuffled_connectome,export_control_graph
from flyholdem.learning.frozen_opponent import FrozenNeuralOpponent
from .gates import verify_gate2a
from .poker_curriculum import _execute_curriculum,_training_rows
from .poker_evidence import read_journal,audit_evaluation,aggregate_endpoints,frozen_weights_hash,compact_rows


def _json(path):return json.loads(Path(path).read_text())
def _yaml(path):return yaml.safe_load(Path(path).read_text())

def _registered(config):
    if config!=_yaml(ROOT/'configs/poker_curricula.yaml') or config.get('schema')!='gated-native-poker-curricula-v1':
        raise ValueError('Use the exact source-registered complete curriculum protocol')
    seeds=[s for item in config['profiles'].values() for s in item['seeds']]
    if len(set(seeds))!=len(seeds) or len(config['profiles']['confirmatory']['seeds'])<5 or len(config['profiles']['development']['seeds'])<3:
        raise ValueError('Disjoint development and at least five confirmatory training seeds required')
    ranges=[]
    for stage in config['stages'].values():
        for profile in config['profiles'].values():
            if stage['hands']%2 or stage['hands']<2:raise ValueError('Paired training hands required')
            for i in range(len(profile['seeds'])):
                start=stage['train_deal_start']+profile['deal_offset']+i*config['training_seed_stride']
                ranges.append((start,start+stage['hands']//2))
            start=stage['evaluation_deal_start']+profile['deal_offset']
            ranges.append((start,start+profile['paired_deals_per_opponent']))
    if any(max(a[0],b[0])<min(a[1],b[1]) for i,a in enumerate(ranges) for b in ranges[i+1:]):
        raise ValueError('All curriculum training, development and confirmation deal ranges must be disjoint')


def _compile(config,stage,profile,learning_mode,gate2a,previous=None,development=None,seen=None):
    _registered(config)
    if stage not in config['stages'] or profile not in config['profiles'] or learning_mode not in ('bio-plastic','distilled-connectome'):
        raise ValueError('Registered stage, independent-seed profile and core learning mode required')
    # Deliberately first, before prepared graph/model loading or output creation.
    prerequisite=verify_gate2a(gate2a)
    authorization={'gate2a':{'path':str(Path(gate2a).resolve()),'sha256':digest(gate2a)},'previous':None,'development':None}
    selected=config['stages'][stage];prev=None
    if selected['previous'] is not None:
        if previous is None:raise ValueError('The preceding confirmed curriculum must pass first')
        prev=verify_curriculum(previous,require_pass=True,seen=seen)
        p=prev['plan']
        if p['stage']!=selected['previous'] or p['learning_mode']!=learning_mode or p['protocol_hash']!=identity(config):
            raise ValueError('Prior curriculum stage, mode or registered protocol mismatch')
        authorization['previous']={'path':str(Path(previous).resolve()),'sha256':digest(Path(previous)/'result.json')}
    elif previous is not None:raise ValueError('The first poker curriculum starts from its fixed original registration')
    if profile=='confirmatory':
        if development is None:raise ValueError('Confirmation requires a passing development experiment')
        dev=verify_curriculum(development,seen=seen)
        p=dev['plan']
        if (p['stage']!=stage or p['profile']!='development' or p['learning_mode']!=learning_mode
                or p['protocol_hash']!=identity(config) or not dev['result']['endpoints']['criteria_met']
                or dev['authorization']['gate2a']!=authorization['gate2a']
                or dev['authorization']['previous']!=authorization['previous']):
            raise ValueError('Matching development endpoint criteria must pass before confirmation')
        authorization['development']={'path':str(Path(development).resolve()),'sha256':digest(Path(development)/'result.json')}
    elif development is not None:raise ValueError('Development cannot reuse a confirmation/development selection reference')
    conditioning=_yaml(ROOT/'configs/conditioning.yaml');surrogate=_yaml(ROOT/'configs/exact_transfer_surrogate.yaml')
    gate_config=_yaml(ROOT/'configs/gate2a.yaml')
    conditioning_result=_json(ROOT/gate_config['conditioning']['run']/'result.json')
    transfer_result=_json(ROOT/gate_config['transfer']['run']/'result.json')
    registration_path=ROOT/conditioning['preregistration'];registration=_json(registration_path)
    if digest(registration_path)!=conditioning['preregistration_sha256'] or registration['mode']!='full':
        raise ValueError('The full conditioning-approved fixed registration is required')
    method='direct-readout-rate-surrogate-v1' if learning_mode=='distilled-connectome' and stage in ('3','4') else 'terminal-local-eligibility'
    player_config={'learning_mode':learning_mode,'optimization':method,'reward_scale_bb':selected['stack_bb'],
        'dopamine':{'pulse_ms':conditioning['dopamine']['pulse_ms'],'gain':conditioning['dopamine']['gain']},
        'surrogate':{**surrogate['surrogate'],'learning_rate':transfer_result['learning_rate']}}
    plasticity={**conditioning['plasticity'],'learning_rate':conditioning_result['learning_rate']}
    seeds=config['profiles'][profile]['seeds'];initial_models={};opponent_models=[]
    if prev:
        prior=[prev['result']['seed_models'][str(seed)] for seed in prev['plan']['seeds']]
        # Fixed ordinal transfer, all confirmatory seeds, never best-profit choice.
        initial_models={str(seed):prior[i] for i,seed in enumerate(seeds)}
        if stage in ('5','6'):opponent_models=prior
    teacher=None
    if stage=='3':
        ref=config['small_teacher'];policy=ROOT/ref['policy'];confirmation=ROOT/ref['confirmation']
        if digest(policy/'manifest.json')!=ref['policy_sha256'] or digest(confirmation/'result.json')!=ref['result_sha256']:
            raise ValueError('Registered small-game teacher changed')
        from flyholdem.teacher.shove_fold import load_policy
        from flyholdem.teacher.shove_fold_evaluation import verify_run,boundary_check
        small,_=load_policy(policy)
        evidence=verify_run(confirmation,_yaml(ROOT/'configs/shove_fold_teacher_evaluation.yaml'),ref['policy_sha256'],'confirmatory')
        if not evidence['passes_fixed_suite']:raise ValueError('The small-game teacher has not passed its own confirmation')
        boundary_check(small)
        reference_teacher={'kind':'small-tabular','path':str(policy),'sha256':ref['policy_sha256']}
    else:
        path=Path(prerequisite['inputs']['policy'])
        reference_teacher={'kind':'full-qualified','path':str(path),'sha256':digest(path/'manifest.json')}
    if method!='terminal-local-eligibility':teacher=reference_teacher
    control='shuffled-reward' if method=='terminal-local-eligibility' else 'shuffled-teacher'
    chosen=config['profiles'][profile];graph=ROOT/'connectome_data/malecns_v1/prepared-full'
    plan={**config['endpoints'],'protocol_hash':identity(config),'stage':stage,'profile':profile,'curriculum':selected['curriculum'],
        'mode':'full','learning_mode':learning_mode,'optimization':method,'seeds':seeds,
        'controls':['frozen',control,'shuffled-encoder','shuffled-connectome'],'held_out_opponent':'equity-bucket',
        'retention_ms':conditioning['retention_ms'],'player_config':player_config,'plasticity':plasticity,
        'graph':str(graph),'graph_manifest_sha256':digest(graph/'manifest.json'),'graph_hash':registration['graph_hash'],
        'registration':str(registration_path),'registration_sha256':digest(registration_path),
        'encoder_hash':identity(registration['projection_indices']),'decoder_hash':identity(registration['ensembles']),
        'initial_models':initial_models,'opponent_models':opponent_models,'teacher':teacher,'reference_teacher':reference_teacher,
        'training_starts':{str(seed):selected['train_deal_start']+chosen['deal_offset']+i*config['training_seed_stride'] for i,seed in enumerate(seeds)},
        'training':{'hands':selected['hands'],'opponent_cycle':['random','calling-station','tight-aggressive']+
            ['frozen-native:'+ref['sha256'] for ref in opponent_models],
            'temperature':{**config['temperature'],'decay_hands':selected['hands']},'progress_hands':128},
        'evaluation':{'schema':'frozen-poker-evaluation-v1','status':'development' if profile=='development' else 'confirmatory-baseline',
            'unavailable_training_paths':list(prerequisite['inputs'].values())+[reference_teacher['path']]+([str(Path(previous).resolve())] if previous else []),
            'mode':'full','curriculum':selected['curriculum'],'opponents':['random','calling-station','tight-aggressive','equity-bucket'],
            'paired_deals_per_opponent':chosen['paired_deals_per_opponent'],'deal_seed_start':selected['evaluation_deal_start']+chosen['deal_offset'],
            'bootstrap_seed':config['endpoints']['bootstrap_seed'],'bootstrap_repeats':config['endpoints']['bootstrap_repeats'],'progress_hands':128}}
    return plan,authorization


def _factory(plan):
    from pyarrow import feather
    graph=Path(plan['graph']);nodes=feather.read_table(graph/'neurons.feather').to_pandas()
    inputs=np.flatnonzero(nodes['class'].eq('Kenyon_Cell').to_numpy());outputs=np.flatnonzero(nodes['class'].eq('MBON').to_numpy())
    populations=annotated_populations(nodes)
    def make(seed,arm,control_graph):
        initial=plan['initial_models'].get(str(seed))
        controller=load_frozen(initial['path'],graph,initial['sha256'],seed) if initial else load_controller(graph,plan['registration'],seed)
        actual_graph=graph;control_seed=int(identity([seed,arm,'fixed-control-v1'])[:16],16)
        if arm=='shuffled-encoder':controller=shuffled_encoder(controller,control_seed)
        if arm=='shuffled-connectome':
            controller=export_control_graph(shuffled_connectome(controller,control_seed),graph,control_graph);actual_graph=control_graph
        eligibility=Eligibility(controller.brain,inputs,outputs,plan['plasticity'])
        return PokerLearningPlayer(controller,eligibility,populations,plan['player_config'],seed),actual_graph
    return make


def run_curriculum(config,output,stage,profile,learning_mode,gate2a,*,previous=None,development=None,resume=False,stop_after=None):
    plan,authorization=_compile(config,stage,profile,learning_mode,gate2a,previous,development)
    if plan['reference_teacher']['kind']=='small-tabular':from flyholdem.teacher.shove_fold import load_policy
    else:from flyholdem.teacher.loaders import load_policy
    reference_policy,_=load_policy(plan['reference_teacher']['path'])
    teacher=reference_policy if plan['teacher'] else None
    opponents={}
    for reference in plan['opponent_models']:
        opponent=FrozenNeuralOpponent.load(reference['path'],plan['graph'],reference['sha256']);opponents[opponent.name]=opponent
    return _execute_curriculum(plan,output,_factory(plan),teacher=teacher,
        teacher_sha256=plan['teacher']['sha256'] if teacher else None,frozen_opponents=opponents,
        authorization=authorization,reference_policy=reference_policy,resume=resume,stop_after=stop_after)


def verify_curriculum(root,require_pass=False,seen=None):
    root=Path(root).resolve();seen=set() if seen is None else set(seen)
    if root in seen:raise ValueError('Cyclic curriculum prerequisite reference')
    seen.add(root)
    runtime=_json(root/'manifest.json');protocol=runtime['config'];plan=protocol['plan'];result=_json(root/'result.json')
    auth=protocol.get('authorization')
    if auth is None or protocol['scope']!='gated-full-connectome':raise ValueError('Numerical engineering fixtures cannot qualify a curriculum')
    for reference in auth.values():
        if reference is not None:
            path=Path(reference['path']);path=path/'result.json' if path.is_dir() else path
            if digest(path)!=reference['sha256']:raise ValueError('Curriculum prerequisite artifact changed')
    config=_yaml(ROOT/'configs/poker_curricula.yaml')
    expected,authorization=_compile(config,plan['stage'],plan['profile'],plan['learning_mode'],auth['gate2a']['path'],
        auth['previous']['path'] if auth['previous'] else None,auth['development']['path'] if auth['development'] else None,seen)
    if (plan!=expected or authorization!=auth or runtime['config_hash']!=identity(protocol)
            or result['status']!='complete' or result['manifest_sha256']!=digest(root/'manifest.json')):
        raise ValueError('Registered curriculum plan or complete result mismatch')
    if plan['reference_teacher']['kind']=='small-tabular':from flyholdem.teacher.shove_fold import load_policy
    else:from flyholdem.teacher.loaders import load_policy
    reference_policy,_=load_policy(plan['reference_teacher']['path'])
    from flyholdem.teacher.curriculum_reference import verify_reference
    reference_evidence=verify_reference(reference_policy,plan['reference_teacher'],plan['evaluation'],root/'conventional-reference')
    if result['conventional_reference']!=reference_evidence:raise ValueError('Conventional reference artifact changed')
    journal=read_journal(root/'phases.jsonl');phases={seed:{} for seed in plan['seeds']};index=0
    order=['initial','plastic','retention','erased',*plan['controls']]
    for seed in plan['seeds']:
        initial_model=plan['initial_models'].get(str(seed))
        if initial_model:
            initial_hash=frozen_weights_hash(initial_model['path'],plan['graph'])
        else:
            import hashlib
            from flyholdem.connectome.prepare import load_graph
            from flyholdem.interface.population import scaled_weights
            arrays,_=load_graph(plan['graph']);registration=_json(plan['registration'])
            initial_hash=hashlib.sha256(scaled_weights(arrays['weight'],registration['config'].get('global_weight_scale',1)).tobytes()).hexdigest()
        for phase in order:
            folder=root/str(seed)/phase;model=_json(folder/'model/manifest.json');reference=model['training_reference']
            graph=Path(plan['graph']) if phase!='shuffled-connectome' else root/str(seed)/'null-graph'
            evaluation={**plan['evaluation'],'mode':model['mode'],
                'unavailable_training_paths':[*plan['evaluation'].get('unavailable_training_paths',[]),str(root)]}
            audit=audit_evaluation(folder/'evaluation',evaluation,digest(folder/'model/manifest.json'))
            training=folder/'training' if phase in ['plastic',*plan['controls']] else root/str(seed)/'plastic/training' if phase=='retention' else None
            hashes=None
            if training:
                _training_rows(training,plan,seed,'plastic' if phase=='retention' else phase)
                hashes={key:digest(training/key) for key in ('manifest.json','result.json','hands.jsonl')}
            if (reference['plan_sha256']!=identity(plan) or reference['seed']!=seed or reference['phase']!=phase
                    or reference['training']!=hashes or reference['weights_sha256']!=audit['weights_sha256']):
                raise ValueError('Curriculum model/training binding mismatch')
            if training and audit['weights_sha256']!=_json(training/'result.json')['final_weights_sha256']:
                raise ValueError('Evaluated model differs from the registered final training weights')
            if phase in ('initial','erased','frozen') and audit['weights_sha256']!=initial_hash:
                raise ValueError('Initial, erased or frozen phase differs from the registered starting weights')
            if training:
                if _json(training/'result.json')['initial_weights_sha256']!=initial_hash:
                    raise ValueError('A training/control arm started from different learned weights')
                if _json(training/'manifest.json')['config']['controller_registration']!=_json(folder/'model/preregistration.json'):
                    raise ValueError('Evaluation changed the training encoder, readouts or connectivity')
            if phase=='retention':
                idle=reference['retention']
                if (idle['duration_ms']!=plan['retention_ms'] or idle['before_weights_sha256']!=audit['weights_sha256']
                        or idle['after_weights_sha256']!=audit['weights_sha256'] or idle['learning'] or idle['teacher_connected']):
                    raise ValueError('Retention interval or unchanged-weight evidence mismatch')
            evidence={'model_sha256':audit['model_sha256'],'graph_manifest_sha256':digest(graph/'manifest.json'),
                'evaluation':audit['evidence'],'training':hashes,'retention':reference['retention']}
            if index>=len(journal) or journal[index]['label']!=[seed,phase] or journal[index]['value']!=evidence:
                raise ValueError('Curriculum phase journal differs from child evidence')
            phases[seed][phase]=compact_rows(audit['rows']);index+=1
    endpoints=aggregate_endpoints(phases,plan)
    models={str(seed):{'path':str(root/str(seed)/'plastic/model'),'sha256':digest(root/str(seed)/'plastic/model/manifest.json')} for seed in plan['seeds']}
    if (len(journal)!=index or result['completed_phases']!=index or result['journal_head']!=journal[-1]['hash']
            or result['endpoints']!=endpoints or result['gate_passed']!=endpoints['gate_passed']
            or result['learning_claim']!=endpoints['gate_passed'] or result['seed_models']!=models):
        raise ValueError('Curriculum endpoint differs from reverified complete phases')
    if require_pass and not result['gate_passed']:raise ValueError('The required confirmed curriculum did not pass')
    return {'plan':plan,'authorization':auth,'result':result,'result_sha256':digest(root/'result.json'),'historical_native_training_reexecuted':False}
