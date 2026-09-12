"""Numerical artifact fixtures only; these are not measured poker results."""
from pathlib import Path
import copy
import json
import numpy as np
import pytest
from flyholdem.connectome.registry import digest
from flyholdem.experiments.journal import Journal
from flyholdem.provenance import identity
from flyholdem.poker.opponents import VERSIONS
from flyholdem.teacher.evaluation import evaluation_summary,evaluate
from flyholdem.teacher.loaders import sampling
from flyholdem.teacher.validation import verify_evaluation,verify_passing_development


def fixture(root,profile='confirmatory',positive=True,policy=None):
    root.mkdir(parents=True,exist_ok=True)
    run=root/'evaluation';run.mkdir()
    if policy is None:
        policy=root/'policy';policy.mkdir()
        np.save(policy/'fixture.npy',np.array([1],dtype=np.float32),allow_pickle=False)
        (policy/'manifest.json').write_text(json.dumps({'schema':'teacher-average-policy-v1',
        'provenance':{'stack_bb':20,'scope':'synthetic numerical evidence fixture'},
        'files':{'fixture.npy':digest(policy/'fixture.npy')}}))
    policy_hash=digest(policy/'manifest.json')
    config={'schema':'conventional-teacher-suite-v1','stack_bb':20,'opponents':list(VERSIONS),
        'bootstrap_seed':42,'bootstrap_repeats':100,
        'profiles':{'development':{'seed_start':100,'paired_deals_per_opponent':3},
                    'confirmatory':{'seed_start':200,'paired_deals_per_opponent':3}}}
    runtime_config={**config,'profile':profile,'policy_sha256':policy_hash}
    if profile=='confirmatory':
        development,_,_=fixture(root/'development','development',True,policy)
        runtime_config['development_reference']=verify_passing_development(development,policy,config)
    (run/'manifest.json').write_text(json.dumps({'config':runtime_config,'config_hash':identity(runtime_config)}))
    grouped={kind:[] for kind in VERSIONS};journal=Journal(run/'paired-deals.jsonl')
    for kind in VERSIONS:
        for i in range(3):
            value=float(i+1)*(1 if positive else -1)
            row={'deal_seed':config['profiles'][profile]['seed_start']+i,'seat_returns_bb':[value,value],
                'paired_bb_per_hand':value,'action_counts':[1,2,0,0,1]}
            grouped[kind].append(row);journal.record(len(journal.rows),[kind,i],row)
    result={'schema':'teacher-evaluation-v1',**evaluation_summary(grouped,config),'profile':profile,
        'policy_sha256':policy_hash,'stack_bb':20,'information_boundary_verified':True,
        'information_boundary':{'hidden_hole_future_deck_and_teacher_label_invariance':True,'decisions_checked':32},
        'allowed_as_teacher':profile=='confirmatory' and positive,'journal_head':journal.rows[-1]['hash'],
        'manifest_sha256':digest(run/'manifest.json')}
    result['sampling']=sampling(json.loads((policy/'manifest.json').read_text()))
    journal.close();(run/'result.json').write_text(json.dumps(result))
    return run,policy,config


def mutate(path,fn):
    value=json.loads(path.read_text());fn(value);path.write_text(json.dumps(value))


def test_recomputation_distinguishes_confirmation_from_development_and_negative_results(tmp_path):
    for profile,positive in [('confirmatory',True),('development',True),('confirmatory',False)]:
        root=tmp_path/(profile+str(positive));root.mkdir();run,policy,config=fixture(root,profile,positive)
        found=verify_evaluation(run,policy,config,require_confirmatory=False)
        assert found['allowed_as_teacher']==(profile=='confirmatory' and positive)
        assert found['passes_fixed_suite']==positive and found['paired_deals']==12
        if not found['allowed_as_teacher']:
            with pytest.raises(ValueError):verify_evaluation(run,policy,config)
    # The public project registration cannot be replaced by a tiny fixture suite.
    with pytest.raises(ValueError,match='registered suite'):verify_evaluation(run,policy)


@pytest.mark.parametrize('mutation',[
    lambda r:r['opponents']['random'].update(bb_per_hand=99),
    lambda r:r['opponents']['equity-bucket'].update(suite_adjusted_bootstrap_ci=[1,100]),
    lambda r:r.update(allowed_as_teacher=False),
    lambda r:r['information_boundary'].update(decisions_checked=0),
])
def test_passing_flags_or_intervals_cannot_replace_actual_evidence(tmp_path,mutation):
    run,policy,config=fixture(tmp_path);mutate(run/'result.json',mutation)
    with pytest.raises(ValueError):verify_evaluation(run,policy,config)


def test_rehashed_wrong_deal_schedule_and_missing_opponent_are_rejected(tmp_path):
    run,policy,config=fixture(tmp_path)
    rows=[json.loads(line) for line in (run/'paired-deals.jsonl').read_text().splitlines()]
    (run/'paired-deals.jsonl').unlink();rows[0]['value']['deal_seed']+=1
    journal=Journal(run/'paired-deals.jsonl')
    for row in rows:journal.record(row['index'],row['label'],row['value'])
    head=journal.rows[-1]['hash'];journal.close()
    mutate(run/'result.json',lambda r:r.update(journal_head=head))
    with pytest.raises(ValueError,match='schedule'):verify_evaluation(run,policy,config)
    altered=copy.deepcopy(config);altered['opponents']=altered['opponents'][:-1]
    with pytest.raises(ValueError,match='registered suite'):verify_evaluation(run,policy,altered)


def test_tensor_mutation_and_truncated_journal_cannot_qualify(tmp_path):
    run,policy,config=fixture(tmp_path);original=(policy/'fixture.npy').read_bytes()
    (policy/'fixture.npy').write_bytes(original+b'x')
    with pytest.raises(ValueError,match='tensor checksum'):verify_evaluation(run,policy,config)
    (policy/'fixture.npy').write_bytes(original)
    path=run/'paired-deals.jsonl';path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError,match='Truncated'):verify_evaluation(run,policy,config)


def test_confirmation_refuses_missing_or_failed_development_before_loading_or_output(tmp_path,monkeypatch,capsys):
    development,policy,config=fixture(tmp_path/'negative','development',False)
    def forbidden(*args,**kwargs):raise AssertionError('A failed prerequisite reached the policy or poker evaluator')
    monkeypatch.setattr('flyholdem.teacher.loaders.load_policy',forbidden)
    monkeypatch.setattr('flyholdem.teacher.evaluation.paired_match',forbidden)
    for reference,message in [(None,'requires verified'),(development,'has not passed')]:
        with pytest.raises(ValueError,match=message):
            evaluate(policy,config,tmp_path/'forbidden','confirmatory',development_reference=reference)
        assert not (tmp_path/'forbidden').exists()
    from flyholdem.cli import main
    protocol=tmp_path/'suite.yaml'
    import yaml
    protocol.write_text(yaml.safe_dump(config))
    with pytest.raises(SystemExit) as stopped:
        main(['teacher','evaluate','--policy',str(policy),'--config',str(protocol),'--profile','confirmatory',
            '--development-reference',str(development),'--output',str(tmp_path/'cli-forbidden')])
    assert stopped.value.code==2 and 'has not passed' in capsys.readouterr().err
    assert not (tmp_path/'cli-forbidden').exists()


def test_confirmation_reverifies_development_hashes_and_rejects_cycles(tmp_path):
    run,policy,config=fixture(tmp_path)
    assert verify_evaluation(run,policy,config)['development_reference_verified']
    with pytest.raises(ValueError,match='must be a development'):
        verify_passing_development(run,policy,config)
    dependency=json.loads((run/'manifest.json').read_text())['config']['development_reference']
    development=Path(dependency['path'])
    # Even a harmless extra field changes the exact bound development evidence.
    mutate(development/'result.json',lambda value:value.update(extra_note='changed after confirmation'))
    assert verify_evaluation(development,policy,config,False)['passes_fixed_suite']
    with pytest.raises(ValueError,match='registered suite'):
        verify_evaluation(run,policy,config)


def test_actual_tabular_confirmation_and_recovery_bind_the_prerequisite(tmp_path):
    # Actual model export and PokerKit evaluation. Only this tiny, separately
    # registered fixture's prerequisite returns are synthetic; public project
    # qualification must reject this fixture suite.
    from flyholdem.teacher.regret_training import train,SCHEMA,AGGREGATION
    from flyholdem.teacher.regret import VERSION
    from flyholdem.teacher.regret_policy import export_policy
    config={'schema':SCHEMA,'stack_bb':20,'iterations':2,'sampling_seed':91200,'deal_seed_start':993000000,
        'opponents':list(VERSIONS),'opponent_probabilities':[.25]*4,'aggregation':AGGREGATION,
        'abstraction':{'feature_version':VERSION,'stack_bb':20,'equity_buckets':8,'equity_samples':16,
            'max_information_sets':20000,'max_nodes_per_traversal':100000}}
    train(config,tmp_path/'training');export_policy(tmp_path/'training',tmp_path/'policy')
    development,policy,suite=fixture(tmp_path/'synthetic-development','development',True,tmp_path/'policy')
    run=tmp_path/'actual-confirmation'
    result=evaluate(policy,suite,run,'confirmatory',development_reference=development)
    journal=(run/'paired-deals.jsonl').read_bytes()
    restored=evaluate(policy,suite,run,'confirmatory',resume=True,development_reference=development)
    assert result==restored and (run/'paired-deals.jsonl').read_bytes()==journal
    found=verify_evaluation(run,policy,suite,require_confirmatory=False)
    assert found['development_reference_verified'] and found['summary_recomputed']
    with pytest.raises(ValueError,match='registered suite'):
        verify_evaluation(run,policy,require_confirmatory=False)
    mutate(development/'result.json',lambda value:value.update(passes_fixed_suite=False))
    with pytest.raises(ValueError,match='summary differs'):
        evaluate(policy,suite,run,'confirmatory',resume=True,development_reference=development)
    assert (run/'paired-deals.jsonl').read_bytes()==journal
