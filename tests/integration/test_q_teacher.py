import copy
import json
from pathlib import Path
import numpy as np
import pytest
torch=pytest.importorskip('torch',reason='Optional conventional teacher')
from flyholdem.connectome.registry import ROOT,digest
from flyholdem.provenance import source_identity
from flyholdem.poker.engine import Hand
from flyholdem.poker.opponents import VERSIONS
from flyholdem.teacher.training import train
from flyholdem.teacher.q_policy import BestResponsePolicy,export_components,ALGORITHM,AGGREGATION
from flyholdem.teacher.loaders import load_policy
from flyholdem.teacher.evaluation import evaluate,verify_information_boundary
from flyholdem.teacher.validation import verify_evaluation
from flyholdem.neural.checkpoint import Checkpoints
from flyholdem.teacher.serialization import unpack


def tiny_training(root):
    config=dict(schema='engineering-fixture',algorithm='NFSP-fixed-policy-prior-v1',stack_bb=20,
        torch_threads=1,seed=81,hands=16,deal_seed_start=81200,epsilon_start=.9,epsilon_end=.1,
        epsilon_decay_hands=10,updates_per_hand=1,progress_hands=16,
        agent=dict(value_learning='double-dqn',feature_version='canonical-visible-card-structure-v2',hidden=8,
            q_lr=.001,average_lr=.001,replay_capacity=100,reservoir_capacity=100,anticipatory=.5,
            batch_size=4,warmup_transitions=4,discount=1,gradient_clip=5,target_update_steps=4),
        population=dict(cycle=['self-play']*4+list(VERSIONS),fixed_hand_update='active-agent-only',fixed_opponent_backend='exact-v1-batched-ranks'))
    run=root/'training';train(config,run)
    extraction={'schema':'teacher-best-response-extraction-v1','algorithm':ALGORITHM,'aggregation':AGGREGATION,
        'status':'development','training_run':str(run),'training_manifest_sha256':digest(run/'manifest.json'),
        'training_source':str(ROOT),'training_source_sha256':source_identity(),'completed_hands':16}
    return run,extraction


def test_best_response_mixture_uses_each_component_greedy_legal_action():
    class Fixed(torch.nn.Module):
        def __init__(self,values):super().__init__();self.values=torch.tensor(values,dtype=torch.float32)
        def forward(self,x):return self.values
    policy=BestResponsePolicy([Fixed([1,0,99,2,3]),Fixed([0,5,99,0,1])],'canonical-visible-card-structure-v2')
    obs=Hand(1).observation();obs['legal_mask']=[True,True,False,False,True]
    assert np.array_equal(policy.probabilities(obs),[0,.5,0,0,.5])
    policy.models[0].values[0]=float('nan')
    with pytest.raises(ValueError,match='Finite'):policy.probabilities(obs)


def test_numeric_q_export_is_the_actual_completed_q_head_and_privately_invariant(tmp_path):
    run,config=tiny_training(tmp_path);path=tmp_path/'q';export_components(config,path)
    policy,record=load_policy(path)
    assert record['aggregation']==AGGREGATION and not record['allowed_as_teacher']
    saved=json.loads((run/'manifest.json').read_text());arrays,extra,_=Checkpoints(run/'checkpoints').load(saved)
    for seat in range(2):
        state=unpack(extra['agents'][seat]['tree'],arrays)
        for name,value in state['q'].items():
            assert np.array_equal(np.load(path/f'agent{seat}-{name}.npy',allow_pickle=False),value.numpy())
        assert any(not torch.equal(state['q'][name],state['average'][name]) for name in state['q'])
    assert verify_information_boundary(policy)['decisions_checked']==32
    changed=copy.deepcopy(config);changed['completed_hands']=15
    with pytest.raises(ValueError,match='matching completed'):export_components(changed,tmp_path/'bad')
    assert not (tmp_path/'bad').exists()
    tensor=next(path.glob('*.npy'));tensor.write_bytes(tensor.read_bytes()+b'x')
    with pytest.raises(ValueError,match='checksum'):load_policy(path)


def test_q_candidate_uses_the_original_complete_evaluation_and_recomputed_sampling_label(tmp_path):
    run,config=tiny_training(tmp_path);path=tmp_path/'q';export_components(config,path)
    suite={'schema':'conventional-teacher-suite-v1','stack_bb':20,'opponents':list(VERSIONS),
        'bootstrap_seed':42,'bootstrap_repeats':100,
        'profiles':{'development':{'seed_start':82200,'paired_deals_per_opponent':2},
                    'confirmatory':{'seed_start':83200,'paired_deals_per_opponent':2}}}
    result=evaluate(path,suite,tmp_path/'evaluation')
    assert result['sampling']=='frozen-equal-mixture-of-legal-greedy-Q-policies' and not result['allowed_as_teacher']
    verify_evaluation(tmp_path/'evaluation',path,suite,require_confirmatory=False)
    assert set(result['opponents'])==set(VERSIONS)
