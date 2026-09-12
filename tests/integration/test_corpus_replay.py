"""Real collection with an untrained numeric policy; no scientific qualification."""
import copy
import json
import numpy as np
import pytest
torch=pytest.importorskip('torch')
from flyholdem.connectome.registry import digest
from flyholdem.provenance import identity
from flyholdem.experiments.journal import Journal
from flyholdem.teacher.corpus import export_corpus,verify_corpus
from flyholdem.teacher.corpus_validation import replay_corpus,verify_qualified_corpus
from flyholdem.teacher.nfsp import NFSPAgent
from flyholdem.teacher.policy import export_policy,load_policy


def fixture(root,monkeypatch):
    torch.set_num_threads(1);torch.manual_seed(99)
    settings=dict(hidden=8,q_lr=.001,average_lr=.001,replay_capacity=32,reservoir_capacity=32,
        anticipatory=.5,batch_size=4,warmup_transitions=4,discount=1,gradient_clip=5,target_update_steps=10)
    agents=[NFSPAgent(settings,20+i) for i in range(2)];policy=root/'policy'
    export_policy(agents,policy,{'stack_bb':20,'status':'untrained engineering fixture, not qualified'})
    validation=root/'validation';validation.mkdir();result=validation/'result.json'
    result.write_text(json.dumps({'schema':'teacher-evaluation-v1','profile':'confirmatory','passes_fixed_suite':True,
        'information_boundary_verified':True,'allowed_as_teacher':True,'policy_sha256':digest(policy/'manifest.json'),'stack_bb':20}))
    config=dict(schema='canonical-teacher-corpus-v1',stack_bb=20,seed_start=20100,maximum_hands=200,
        minimum_rows=80,minimum_per_street=4,maximum_per_stratum=16,
        calling_station_collection_probability=.35,random_collection_probability=.35)
    corpus=root/'corpus'
    # Isolate collection testing from scientific teacher qualification. The real
    # policy, information-boundary test and complete PokerKit collection execute.
    with monkeypatch.context() as patch:
        patch.setattr('flyholdem.teacher.validation.verify_evaluation',lambda *args,**kwargs:{'engineering_fixture':True})
        export_corpus(policy,result,config,corpus)
    return corpus,load_policy(policy)[0],config,policy,validation


def rewrite_record(corpus,fn):
    path=corpus/'manifest.json';record=json.loads(path.read_text());fn(record);path.write_text(json.dumps(record))


def test_replay_matches_actual_frozen_teacher_and_complete_collection(tmp_path,monkeypatch):
    corpus,policy,config,policy_path,validation=fixture(tmp_path,monkeypatch)
    result=replay_corpus(corpus,policy,config)
    assert result['complete_collection_reproduced'] and result['teacher_targets_reproduced']
    assert result['unique_information_sets']>=80 and not result['allowed_as_teacher']
    # The fixture's flags cannot pass the real qualification boundary.
    with pytest.raises((ValueError,FileNotFoundError)):
        verify_qualified_corpus(corpus,policy_path,validation)
    changed=copy.deepcopy(config);changed['minimum_rows']-=1
    with pytest.raises(ValueError,match='registered'):replay_corpus(corpus,policy,changed)


@pytest.mark.parametrize('change',['targets','coverage','collection'])
def test_rehashed_semantic_corruption_cannot_pass_corpus_replay(tmp_path,monkeypatch,change):
    corpus,policy,config,_,_=fixture(tmp_path,monkeypatch)
    if change=='targets':
        path=corpus/'train.jsonl';rows=[json.loads(line) for line in path.read_text().splitlines()]
        row=rows[0];legal=np.asarray(row['observation']['legal_mask'],dtype=bool)
        target=np.zeros(5);target[legal]=1/legal.sum();row['teacher_probabilities']=target.tolist()
        from flyholdem.poker.observation import canonical_bytes
        path.write_bytes(b''.join(canonical_bytes(row)+b'\n' for row in rows))
        rewrite_record(corpus,lambda record:record['files'].update({'train.jsonl':digest(path)}))
        assert verify_corpus(corpus)['canonical_ids_disjoint']
    elif change=='coverage':
        rewrite_record(corpus,lambda record:record.update(street_counts=[999]*4))
    else:
        path=corpus/'collection.jsonl';rows=[json.loads(line) for line in path.read_text().splitlines()];path.unlink()
        rows[0]['value']['visible_states'][0]['probabilities']=[0,1,0,0,0]
        journal=Journal(path)
        for row in rows:journal.record(row['index'],row['label'],row['value'])
        journal.close();rewrite_record(corpus,lambda record:record.update(collection_sha256=digest(path)))
    with pytest.raises(ValueError,match='split differs|coverage metadata|collection trajectory'):
        replay_corpus(corpus,policy,config)
