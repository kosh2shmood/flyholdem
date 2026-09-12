"""Real native/PokerKit phase execution; fixture evidence cannot qualify gates."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from flyholdem.connectome.registry import ROOT,digest
from flyholdem.provenance import identity
from flyholdem.interface.frozen import load_frozen
from flyholdem.learning.eligibility import Eligibility
from flyholdem.learning.poker_player import PokerLearningPlayer
from flyholdem.learning.poker_controls import shuffled_encoder,shuffled_connectome,export_control_graph
from flyholdem.experiments.poker_curriculum import _execute_curriculum,_model
from flyholdem.experiments.poker_evidence import audit_evaluation,aggregate_endpoints
from flyholdem.experiments.curriculum_protocol import run_curriculum,verify_curriculum,_registered
from test_disconnected_evaluation import native_model


def fixture(tmp_path,teaching=False):
    evaluation,model,graph=native_model(tmp_path)
    evaluation['curriculum']='hu-20bb-v1';evaluation['progress_hands']=1000
    registration=json.loads((model/'preregistration.json').read_text())
    plasticity=dict(pre_tau_ms=20,eligibility_tau_ms=5000,coincidence_scale=.001,bin_ms=5,learning_rate=.3,weight_bounds=[.1,2])
    config={'learning_mode':'bio-plastic','optimization':'terminal-local-eligibility','reward_scale_bb':20,
        'dopamine':{'pulse_ms':200,'gain':12},'surrogate':{'learning_rate':.1,'temperature':1.,'gradient_norm_cap':1.,'synaptic_tau_ms':5.,'threshold_distance_mv':7.}}
    if teaching:config.update(learning_mode='distilled-connectome',optimization='direct-readout-rate-surrogate-v1')
    plan={'seeds':[51,52],'controls':['frozen','shuffled-reward','shuffled-encoder','shuffled-connectome'],
        'curriculum':'hu-20bb-v1','profile':'development','stage':'5','mode':'fixture-native',
        'learning_mode':'bio-plastic','optimization':'terminal-local-eligibility',
        'graph_manifest_sha256':digest(graph/'manifest.json'),'registration_sha256':digest(model/'preregistration.json'),
        'graph_hash':registration['graph_hash'],'encoder_hash':identity(registration['projection_indices']),
        'decoder_hash':identity(registration['ensembles']),
        'training_starts':{'51':17800,'52':17900},'retention_ms':50,
        'training':{'hands':4,'opponent_cycle':['random','calling-station','tight-aggressive'],
            'temperature':{'start':.2,'end':.1,'decay_hands':4},'progress_hands':1000},
        'evaluation':evaluation,'held_out_opponent':'equity-bucket','bootstrap_seed':5,'bootstrap_repeats':100,
        'strength_samples':16,'minimum_bucket_decisions':2,'minimum_shove_span':.1,
        'minimum_action_frequency':.01,'maximum_action_frequency':.95}
    if teaching:
        plan.update(learning_mode=config['learning_mode'],optimization=config['optimization'])
        plan['controls'][1]='shuffled-teacher'
    def factory(seed,arm,null):
        controller=load_frozen(model,graph,seed=seed);actual=graph
        if arm=='shuffled-encoder':controller=shuffled_encoder(controller,seed)
        if arm=='shuffled-connectome':controller=export_control_graph(shuffled_connectome(controller,seed),graph,null);actual=null
        eligible=Eligibility(controller.brain,[0,1],[2,3,4,5,6],plasticity)
        return PokerLearningPlayer(controller,eligible,(np.array([0]),np.array([1])),config,seed),actual
    return plan,factory,graph


def test_actual_multiseed_curriculum_all_controls_recover_exactly(tmp_path):
    plan,factory,graph=fixture(tmp_path)
    full=_execute_curriculum(plan,tmp_path/'whole',factory)
    assert full['status']=='complete' and full['completed_phases']==16 and not full['learning_claim'] and not full['gate_passed']
    from flyholdem.experiments.reports import evidence,write_report
    report=evidence(tmp_path/'whole')
    assert report['mode']=='fixture-native' and report['learning_mode']=='bio-plastic' and len(report['summaries'])==8
    assert all(r['unit']=='BB/100 hands' for r in report['summaries'])
    write_report(tmp_path/'whole',tmp_path/'report')
    assert '100' in (tmp_path/'report/REPORT.md').read_text()
    assert full['endpoints']['retention_decisions_exact'] and full['endpoints']['erasure_decisions_exact']
    part=_execute_curriculum(plan,tmp_path/'resume',factory,stop_after=3)
    assert part['status']=='interrupted' and part['completed_phases']==3
    before=(tmp_path/'resume/51/plastic/training/result.json').read_bytes()
    done=_execute_curriculum(plan,tmp_path/'resume',factory,resume=True)
    assert done['status']=='complete' and done['endpoints']==full['endpoints']
    assert before==(tmp_path/'resume/51/plastic/training/result.json').read_bytes()
    for seed in plan['seeds']:
        for phase in ('initial','plastic','retention','erased',*plan['controls']):
            a=tmp_path/'whole'/str(seed)/phase/'evaluation/evaluation/hands.jsonl'
            b=tmp_path/'resume'/str(seed)/phase/'evaluation/evaluation/hands.jsonl'
            assert a.read_bytes()==b.read_bytes()
        for arm in ['plastic',*plan['controls']]:
            a=tmp_path/'whole'/str(seed)/arm/'training/hands.jsonl';b=tmp_path/'resume'/str(seed)/arm/'training/hands.jsonl'
            assert a.read_bytes()==b.read_bytes()
    with pytest.raises(ValueError,match='fixtures cannot qualify'):verify_curriculum(tmp_path/'whole')
    # Rehashed forged neural scores must still fail the independent visible
    # observation / score reconstruction check before any endpoint is accepted.
    root=tmp_path/'resume/51/initial/evaluation';journal=root/'evaluation/hands.jsonl'
    rows=[json.loads(line) for line in journal.read_text().splitlines()]
    decision=next(row['value']['neural_decisions'][0] for row in rows if row['value']['neural_decisions'])
    decision['scores'][0]+=1
    previous='0'*64
    for row in rows:
        row['previous']=previous;row['hash']=identity({k:v for k,v in row.items() if k!='hash'});previous=row['hash']
    journal.write_text(''.join(json.dumps(row,sort_keys=True)+'\n' for row in rows))
    path=root/'evaluation/result.json';result=json.loads(path.read_text());result['journal_head']=previous;path.write_text(json.dumps(result))
    with pytest.raises(ValueError,match='legal argmax'):audit_evaluation(root)


def test_endpoint_resamples_seeds_and_requires_each_control_and_retention():
    plan={'seeds':[1,2,3,4,5],'controls':['frozen','shuffled-reward','shuffled-encoder','shuffled-connectome'],
        'curriculum':'hu-20bb-v1','profile':'confirmatory','held_out_opponent':'equity-bucket',
        'bootstrap_seed':3,'bootstrap_repeats':100,'strength_samples':16,'minimum_bucket_decisions':2,
        'minimum_shove_span':.1,'minimum_action_frequency':.01,'maximum_action_frequency':.95}
    def row(value):return {'opponent':'equity-bucket','deal_seed':100,'neural_seat':0,'neural_return_bb':value,
        'action_counts':[1,2,3,2,2],'neural_decisions':[]}
    phases={seed:{name:[row(1 if name in ('plastic','retention') else 0)]
        for name in ['initial','plastic','retention','erased',*plan['controls']]} for seed in plan['seeds']}
    value=aggregate_endpoints(phases,plan)
    assert value['gate_passed'] and value['paired_evidence']['initial']['seed_differences']==[100]*5
    assert value['paired_evidence']['initial']['one_sided_sign_flip_p']==1/32
    bad=copy.deepcopy(phases)
    for seed in plan['seeds']:bad[seed]['shuffled-reward']=[row(1)]
    assert not aggregate_endpoints(bad,plan)['gate_passed']
    bad=copy.deepcopy(phases);bad[1]['retention'][0]['action_counts']=[2,1,3,2,2]
    assert not aggregate_endpoints(bad,plan)['gate_passed']
    bad=copy.deepcopy(phases);bad[1]['frozen'][0]['deal_seed']=101
    with pytest.raises(ValueError,match='identical paired'):aggregate_endpoints(bad,plan)


def test_public_curriculum_rejects_missing_gate_before_native_factory_or_output(tmp_path,monkeypatch):
    import yaml
    import flyholdem.experiments.curriculum_protocol as protocol
    config=yaml.safe_load((ROOT/'configs/poker_curricula.yaml').read_text());_registered(config)
    gate=tmp_path/'gate.json';gate.write_text(json.dumps({'schema':'verified-gate2a-certificate-v1','gate':'2A','passed':False}))
    monkeypatch.setattr(protocol,'_factory',lambda *args:pytest.fail('Loaded native player before verifying prerequisite'))
    with pytest.raises(ValueError,match='full verified Gate 2A'):
        run_curriculum(config,tmp_path/'no-run','3','development','bio-plastic',gate)
    assert not (tmp_path/'no-run').exists()
    changed=copy.deepcopy(config);changed['profiles']['confirmatory']['seeds']=[1]
    with pytest.raises(ValueError,match='exact source-registered'):_registered(changed)


def test_model_export_distinguishes_inherited_edges_from_current_plastic_subset(tmp_path):
    from flyholdem.neural.sparse import SparseBrain
    from flyholdem.interface.population import NeuralController
    from test_poker_rollout import make
    controller,player=make();brain=SparseBrain([0,2,4,4,4,4,4,4],[0,2,3,1],[1,2,3,4])
    registration=copy.deepcopy(controller.registration);registration['graph_hash']=brain.graph_hash
    controller=NeuralController(brain,registration)
    eligible=Eligibility(brain,[0,1],[2,3,4,5,6],player.eligible.config)
    brain.weights[0]*=1.5 # An inherited change outside the new control's eligible subset.
    player=PokerLearningPlayer(controller,eligible,(np.array([0]),np.array([1])),player.config)
    _model(player,tmp_path/'model',{'scope':'numerical inherited-edge persistence only'})
    record=json.loads((tmp_path/'model/manifest.json').read_text())['training_reference']
    assert record['trainable_edge_count']==2 and record['persisted_edge_count']==3
    assert np.load(tmp_path/'model/edge_indices.npy',allow_pickle=False).tolist()==[0,1,2]


def test_complete_distilled_fixture_uses_only_train_targets_and_isolated_inference(tmp_path):
    from test_poker_rollout import VisibleTeacher
    plan,factory,graph=fixture(tmp_path,teaching=True);plan['seeds']=[51];plan['training_starts']={'51':17800}
    result=_execute_curriculum(plan,tmp_path/'distilled',factory,teacher=VisibleTeacher(),teacher_sha256='a'*64)
    assert result['status']=='complete' and not result['learning_claim']
    rows=[json.loads(line)['value'] for line in (tmp_path/'distilled/51/plastic/training/hands.jsonl').read_text().splitlines()]
    assert sum(row['teacher_targets_delivered'] for row in rows)>0
    assert all(row['teacher_split']=='train' for row in rows)
    for phase in ('plastic','shuffled-teacher'):
        assert not (tmp_path/'distilled/51'/phase/'evaluation/runtime/src/flyholdem/teacher').exists()


def test_conventional_reference_replays_same_paired_deals_without_native_substitution(tmp_path):
    from test_poker_rollout import VisibleTeacher
    from flyholdem.teacher.curriculum_reference import run_reference,verify_reference
    evaluation,model,graph=native_model(tmp_path);reference={'kind':'numerical-fixture-only','sha256':'a'*64}
    policy=VisibleTeacher();root=tmp_path/'reference'
    result=run_reference(policy,reference,evaluation,root)
    assert result['status']=='complete' and result['hands_completed']==16 and not result['learning_claim']
    hashes=verify_reference(policy,reference,evaluation,root)
    assert result==run_reference(policy,reference,evaluation,root,resume=True)
    assert hashes==verify_reference(policy,reference,evaluation,root)
    class Changed(VisibleTeacher):
        def probabilities(self,observation):
            legal=np.asarray(observation['legal_mask'],dtype=float);return legal/legal.sum()
    with pytest.raises(ValueError,match='exact policy/PokerKit replay'):
        verify_reference(Changed(),reference,evaluation,root)


@pytest.mark.parametrize('stage,profile,message',[('4','development','preceding confirmed'),('3','confirmatory','passing development')])
def test_curriculum_order_and_confirmation_cannot_skip_prerequisites(tmp_path,monkeypatch,stage,profile,message):
    import yaml
    import flyholdem.experiments.curriculum_protocol as protocol
    config=yaml.safe_load((ROOT/'configs/poker_curricula.yaml').read_text())
    gate=tmp_path/'gate';gate.write_text('fixture stub for prerequisite-order unit test')
    monkeypatch.setattr(protocol,'verify_gate2a',lambda path:{'inputs':{}})
    with pytest.raises(ValueError,match=message):
        protocol._compile(config,stage,profile,'bio-plastic',gate)
