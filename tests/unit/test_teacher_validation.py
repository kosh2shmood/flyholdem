"""Numerical artifact fixtures only; these are not measured poker results."""
import copy
import json
import numpy as np
import pytest
from flyholdem.connectome.registry import digest
from flyholdem.experiments.journal import Journal
from flyholdem.provenance import identity
from flyholdem.poker.opponents import VERSIONS
from flyholdem.teacher.evaluation import evaluation_summary
from flyholdem.teacher.validation import verify_evaluation


def fixture(root,profile='confirmatory',positive=True):
    run=root/'evaluation';run.mkdir();policy=root/'policy';policy.mkdir()
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
