"""Fixed-final extraction remains numeric, legal, source-bound and unqualified."""
import copy
import json
import numpy as np
import pytest
from flyholdem.connectome.registry import digest
from flyholdem.neural.checkpoint import atomic_json
from flyholdem.poker.engine import Hand
from flyholdem.poker.opponents import VERSIONS
from flyholdem.teacher.regret import abstraction, regret_matching
from flyholdem.teacher.regret_training import train, completed_table
from flyholdem.teacher.regret_current_policy import (
    EXTRACTION, AGGREGATION, export_current, load_policy, probabilities_from_regrets,
)
from flyholdem.teacher.evaluation import verify_information_boundary, evaluate
from flyholdem.teacher.validation import verify_evaluation
from test_external_regret import config as training_config


def setup(root,stop_after=None):
    cfg=training_config();cfg['iterations']=4
    run=root/'training';train(cfg,run,stop_after=stop_after)
    manifest=json.loads((run/'manifest.json').read_text())
    return run,{'schema':EXTRACTION,'status':'development','aggregation':AGGREGATION,
        'training_run':str(run),'iterations':4,'training_manifest_sha256':digest(run/'manifest.json'),
        'training_source_sha256':manifest['source_hash'],'training_config_sha256':manifest['config_hash']}


def test_current_strategy_exact_normalization_keeps_all_legal_actions():
    regrets=np.array([[-1,2,3,4,5],[-2,-1,0,-4,-5],[0,2,0,0,6]],dtype=np.float64)
    legal=np.array([[1]*5,[1]*5,[0,1,0,0,1]],dtype=bool)
    result=probabilities_from_regrets(regrets,legal)
    np.testing.assert_array_equal(result[0],np.array([0,2,3,4,5])/14)
    np.testing.assert_array_equal(result[1],np.ones(5)/5)
    np.testing.assert_array_equal(result[2],[0,.25,0,0,.75])
    assert result[0,1:4].sum()>0  # Check/call and both raise sizes remain represented.
    for values,mask in [(regrets.astype(np.float32),legal),(regrets,np.zeros_like(legal))]:
        with pytest.raises(ValueError):probabilities_from_regrets(values,mask)
    bad=regrets.copy();bad[2,0]=1
    with pytest.raises(ValueError):probabilities_from_regrets(bad,legal)


def test_complete_extraction_matches_actual_final_table_and_is_self_contained(tmp_path):
    run,cfg=setup(tmp_path);table,_=completed_table(run)
    output=tmp_path/'policy';before={p.name:p.read_bytes() for p in run.glob('*.json*')}
    result=export_current(cfg,output);policy,record=load_policy(output)
    assert result['policy_sha256']==digest(output/'manifest.json') and not result['allowed_as_teacher']
    assert record['aggregation']==AGGREGATION and record['provenance']['iterations_completed']==4
    assert np.array_equal(policy.values,np.array([regret_matching(r,l) for r,l in zip(table.regrets[:len(table.keys)],table.legal[:len(table.keys)])]))
    assert any(not np.array_equal(p,a/a.sum()) for p,a in zip(policy.values,table.averages) if a.sum()>0)
    assert {p.name:p.read_bytes() for p in run.glob('*.json*')}==before
    observations=[]
    for seed in range(991000000,991000004):
        hand=Hand(seed)
        while not hand.done:
            obs=hand.observation();key,legal=abstraction(obs,table.config)
            if key in table.index:
                assert np.array_equal(policy.probabilities(obs),regret_matching(table.regrets[table.index[key]],legal))
            observations.append((obs,policy.probabilities(obs).tobytes()));hand.act(1)
    run.rename(tmp_path/'offline-training')
    policy,_=load_policy(output)
    assert all(policy.probabilities(obs).tobytes()==value for obs,value in observations)
    assert verify_information_boundary(policy)['decisions_checked']==32
    untrained=Hand(1,stacks=(200,200)).observation()
    np.testing.assert_array_equal(policy.probabilities(untrained),np.array(untrained['legal_mask'])/sum(untrained['legal_mask']))
    with pytest.raises(FileExistsError):export_current(cfg,output)


@pytest.mark.parametrize('pin',['training_manifest_sha256','training_source_sha256','training_config_sha256'])
def test_current_extraction_refuses_other_training_identities_before_output(tmp_path,pin):
    _,cfg=setup(tmp_path);cfg[pin]='0'*64
    with pytest.raises(ValueError,match='[Pp]inned'):export_current(cfg,tmp_path/'forbidden')
    assert not (tmp_path/'forbidden').exists()


def test_current_extraction_refuses_live_partial_and_corrupt_complete_evidence(tmp_path):
    run,cfg=setup(tmp_path,stop_after=2)
    with pytest.raises(ValueError,match='Complete fixed-final'):export_current(cfg,tmp_path/'partial')
    assert not (tmp_path/'partial').exists()
    full=training_config();full['iterations']=4;train(full,run,resume=True)
    result=json.loads((run/'result.json').read_text());result['counterfactual_nodes']+=1
    atomic_json(run/'result.json',result)
    with pytest.raises(ValueError,match='result totals'):export_current(cfg,tmp_path/'wrong-totals')
    assert not (tmp_path/'wrong-totals').exists()


def test_current_extraction_rechecks_sampling_and_checkpoint_rng(tmp_path):
    from flyholdem.experiments.journal import Journal
    run,cfg=setup(tmp_path)
    journal=run/'traversals.jsonl';original=journal.read_bytes();rows=[json.loads(line) for line in original.splitlines()]
    rows[-1]['value']['opponent_seed']+=1
    journal.unlink();writer=Journal(journal)
    for i,row in enumerate(rows):writer.record(i,row['label'],row['value'])
    writer.close()
    with pytest.raises(ValueError,match='sampling journal'):export_current(cfg,tmp_path/'wrong-sample')
    journal.write_bytes(original)
    pointer_path=run/'checkpoints/latest.json';pointer=json.loads(pointer_path.read_text())
    meta_path=run/'checkpoints'/pointer['generation']/'manifest.json';meta=json.loads(meta_path.read_text())
    meta['extra']['sampling_rng']['state']['state']+=1;atomic_json(meta_path,meta)
    pointer['manifest_sha256']=digest(meta_path);atomic_json(pointer_path,pointer)
    with pytest.raises(ValueError,match='checkpoint/RNG'):export_current(cfg,tmp_path/'wrong-rng')


def test_current_loader_rejects_rehashed_strategy_override_and_source_change(tmp_path):
    _,cfg=setup(tmp_path);output=tmp_path/'policy';export_current(cfg,output)
    meta_path=output/'manifest.json';meta=json.loads(meta_path.read_text())
    saved=copy.deepcopy(meta);meta['implementation']['current_policy_sha256']='0'*64;atomic_json(meta_path,meta)
    with pytest.raises(ValueError,match='source/runtime'):load_policy(output)
    meta=saved;path=output/'probabilities.npy';p=np.load(path);legal=np.load(output/'legal.npy')
    i=next(i for i,m in enumerate(legal) if m.sum()>1)
    actions=np.flatnonzero(legal[i]);p[i]=0;p[i,actions[0]]=1
    if np.array_equal(p[i],probabilities_from_regrets(np.load(output/'regrets.npy'),legal)[i]):
        p[i]=0;p[i,actions[1]]=1
    np.save(path,p,allow_pickle=False);meta['files']['probabilities.npy']=digest(path);atomic_json(meta_path,meta)
    with pytest.raises(ValueError,match='exactly match'):load_policy(output)


def test_current_candidate_uses_original_qualification_and_no_torch(tmp_path,monkeypatch):
    import builtins
    original=builtins.__import__
    def no_torch(name,*args,**kwargs):
        if name=='torch' or name.startswith('torch.'):raise AssertionError('Numeric policy must not import Torch')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',no_torch)
    _,cfg=setup(tmp_path);output=tmp_path/'policy';export_current(cfg,output)
    suite={'stack_bb':20,'opponents':list(VERSIONS),'bootstrap_seed':91900,'bootstrap_repeats':100,
        'profiles':{'development':{'paired_deals_per_opponent':2,'seed_start':992000000},
                    'confirmatory':{'paired_deals_per_opponent':2,'seed_start':993000000}}}
    result=evaluate(output,suite,tmp_path/'evaluation')
    assert result['sampling']=='frozen-normalized-positive-final-regrets'
    evidence=verify_evaluation(tmp_path/'evaluation',output,suite,require_confirmatory=False)
    assert evidence['summary_recomputed'] and not evidence['allowed_as_teacher']
    with pytest.raises(ValueError,match='Confirmation requires'):
        evaluate(output,suite,tmp_path/'forbidden-confirmation',profile='confirmatory')
    assert not (tmp_path/'forbidden-confirmation').exists()
