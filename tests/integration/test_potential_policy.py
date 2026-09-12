import copy
import json
import numpy as np
import pytest
torch=pytest.importorskip('torch')
import yaml
from flyholdem.connectome.registry import ROOT,digest
from flyholdem.provenance import source_identity
from flyholdem.teacher.training import train
from flyholdem.teacher.q_policy import export_components,ALGORITHM as Q_ALGORITHM,AGGREGATION as Q_AGGREGATION
from flyholdem.teacher.potential_policy import PotentialBoundaryPolicy,export_boundary,ALGORITHM,AGGREGATION
from flyholdem.teacher.loaders import load_policy
from flyholdem.teacher.evaluation import evaluate,verify_information_boundary
from flyholdem.teacher.validation import verify_evaluation
from flyholdem.poker.engine import Hand
from flyholdem.poker.opponents import VERSIONS


def source(root):
    config=yaml.safe_load((ROOT/'configs/teacher_nfsp_potential_v7.yaml').read_text())
    config.update(hands=16,progress_hands=16)
    config['agent'].update(hidden=8,replay_capacity=100,reservoir_capacity=100,batch_size=4,warmup_transitions=4,target_update_steps=4)
    run=root/'training';train(config,run)
    extraction={'schema':'teacher-best-response-extraction-v1','algorithm':Q_ALGORITHM,'aggregation':Q_AGGREGATION,'status':'development',
        'training_run':str(run),'training_manifest_sha256':digest(run/'manifest.json'),'training_source':str(ROOT),
        'training_source_sha256':source_identity(),'completed_hands':16}
    q=root/'q';export_components(extraction,q)
    return q,{'schema':'teacher-potential-boundary-extraction-v1','algorithm':ALGORITHM,'aggregation':AGGREGATION,
        'reward_parameterization':'own-stack-potential-v1','status':'development','source_policy':str(q),'source_policy_sha256':digest(q/'manifest.json'),
        'training_run':str(run),'training_manifest_sha256':digest(run/'manifest.json')}


def test_exact_terminal_fold_value_is_applied_before_legal_greedy_selection():
    class Fixed(torch.nn.Module):
        def __init__(self,values):super().__init__();self.values=torch.tensor(values,dtype=torch.float32)
        def forward(self,x):return self.values
    models=[Fixed([-2,-.1,99,-.3,-.2]),Fixed([.3,-.2,99,-.3,.1])]
    policy=PotentialBoundaryPolicy(models,'canonical-visible-card-structure-v2')
    obs=Hand(1).observation();obs['legal_mask']=[True,True,False,False,True]
    assert np.array_equal(policy.probabilities(obs),[.5,0,0,0,.5])
    assert models[0].values[0]==-2 and models[1].values[0]==pytest.approx(.3)
    obs['legal_mask'][0]=False
    assert np.array_equal(policy.probabilities(obs),[0,.5,0,0,.5])
    models[0].values[0]=float('nan')
    with pytest.raises(ValueError,match='Finite'):policy.probabilities(obs)


def test_boundary_export_preserves_exact_q_tensors_and_requires_the_matching_reward_definition(tmp_path):
    q,config=source(tmp_path);out=tmp_path/'boundary';export_boundary(config,out)
    policy,record=load_policy(out)
    for name in record['files']:assert (out/name).read_bytes()==(q/name).read_bytes()
    assert verify_information_boundary(policy)['decisions_checked']==32
    changed=copy.deepcopy(config);changed['reward_parameterization']='other'
    with pytest.raises(ValueError,match='Registered'):export_boundary(changed,tmp_path/'bad')
    run=tmp_path/'training/manifest.json';saved=json.loads(run.read_text());saved['config']['reward_parameterization']=None
    run.write_text(json.dumps(saved));changed=copy.deepcopy(config);changed['training_manifest_sha256']=digest(run)
    with pytest.raises(ValueError,match='Zero fold'):export_boundary(changed,tmp_path/'bad-reward')
    record['boundary']['value']=1;(out/'manifest.json').write_text(json.dumps(record))
    with pytest.raises(ValueError,match='exact payoff'):load_policy(out)


def test_boundary_candidate_uses_the_complete_original_evaluator_and_qualification_math(tmp_path):
    q,config=source(tmp_path);out=tmp_path/'boundary';export_boundary(config,out)
    suite={'schema':'conventional-teacher-suite-v1','stack_bb':20,'opponents':list(VERSIONS),'bootstrap_seed':42,'bootstrap_repeats':100,
        'profiles':{'development':{'seed_start':18100,'paired_deals_per_opponent':2},'confirmatory':{'seed_start':18200,'paired_deals_per_opponent':2}}}
    result=evaluate(out,suite,tmp_path/'evaluation')
    assert result['sampling']=='frozen-Q-mixture-with-exact-stack-potential-fold-boundary' and not result['allowed_as_teacher']
    verify_evaluation(tmp_path/'evaluation',out,suite,require_confirmatory=False)
