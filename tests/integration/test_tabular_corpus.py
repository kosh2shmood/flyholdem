"""Actual numeric tabular collection/replay; qualification flags are test-only."""
import builtins
import json
from pathlib import Path
import numpy as np
import pytest


def test_tabular_corpus_exports_replays_and_resumes_without_torch(tmp_path,monkeypatch):
    original_import=builtins.__import__
    def without_torch(name,*args,**kwargs):
        if name=='torch' or name.startswith('torch.'):
            raise AssertionError('Numeric tabular corpus work must not import optional Torch')
        return original_import(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',without_torch)
    import yaml
    from flyholdem.connectome.registry import ROOT,digest
    from flyholdem.neural.checkpoint import atomic_json
    from flyholdem.poker.observation import canonical_bytes
    from flyholdem.teacher.regret_training import train
    from flyholdem.teacher.regret_policy import export_policy,load_policy
    from flyholdem.teacher.corpus import export_corpus,verify_corpus
    from flyholdem.teacher.corpus_validation import replay_corpus,verify_qualified_corpus
    settings=yaml.safe_load((ROOT/'configs/teacher_external_regret_v11.yaml').read_text())
    settings.update(iterations=4,deal_seed_start=991700000,sampling_seed=97700)
    settings['abstraction']['equity_samples']=16
    training=tmp_path/'training';policy_path=tmp_path/'policy'
    train(settings,training);export_policy(training,policy_path)
    policy,policy_record=load_policy(policy_path)
    assert policy_record['allowed_as_teacher'] is False
    validation=tmp_path/'validation';validation.mkdir();result=validation/'result.json'
    atomic_json(result,{'schema':'teacher-evaluation-v1','profile':'confirmatory','passes_fixed_suite':True,
        'information_boundary_verified':True,'allowed_as_teacher':True,'policy_sha256':digest(policy_path/'manifest.json'),
        'stack_bb':20,'scope':'Artificial flags for isolated formatter tests; not qualified evidence'})
    config=dict(schema='canonical-teacher-corpus-v1',stack_bb=20,seed_start=991800000,maximum_hands=200,
        minimum_rows=80,minimum_per_street=4,maximum_per_stratum=16,
        calling_station_collection_probability=.35,random_collection_probability=.35)
    corpus=tmp_path/'corpus'
    # Only bypass the prerequisite in this test scope. Collection, actual policy
    # execution, leakage checks, stratification, hashing and replay remain real.
    with monkeypatch.context() as isolated:
        isolated.setattr('flyholdem.teacher.validation.verify_evaluation',lambda *args,**kwargs:{'engineering_fixture':True})
        manifest=export_corpus(policy_path,result,config,corpus)
        runtime=json.loads((corpus/'run-manifest.json').read_text())
        assert runtime['binary_hash']==policy_record['inference_backend']
        assert runtime['binary_hash']!='pytorch-cpu'
        before={p.name:p.read_bytes() for p in corpus.iterdir() if p.is_file()}
        assert export_corpus(policy_path,result,config,corpus,resume=True)==manifest
        assert before=={p.name:p.read_bytes() for p in corpus.iterdir() if p.is_file()}
    verified=replay_corpus(corpus,policy,config)
    assert verified['teacher_targets_reproduced'] and verified['complete_collection_reproduced']
    assert verified['canonical_ids_disjoint'] and not verified['allowed_as_teacher']
    # Restored real qualification must reject these artificial flags.
    with pytest.raises((ValueError,FileNotFoundError)):
        verify_qualified_corpus(corpus,policy_path,validation)
    refused=tmp_path/'refused'
    with pytest.raises((ValueError,FileNotFoundError)):
        export_corpus(policy_path,result,config,refused)
    assert not refused.exists()
    # Rehashing modified targets cannot turn them into this frozen policy's corpus.
    path=corpus/'train.jsonl';rows=[json.loads(line) for line in path.read_text().splitlines()]
    row=next(r for r in rows if sum(r['observation']['legal_mask'])>1)
    legal=np.flatnonzero(row['observation']['legal_mask']);target=np.zeros(5)
    chosen=int(legal[0] if row['teacher_probabilities'][legal[0]]!=1 else legal[1]);target[chosen]=1
    row['teacher_probabilities']=target.tolist();path.write_bytes(b''.join(canonical_bytes(r)+b'\n' for r in rows))
    manifest['files']['train.jsonl']=digest(path);atomic_json(corpus/'manifest.json',manifest)
    assert verify_corpus(corpus)['canonical_ids_disjoint']
    with pytest.raises(ValueError,match='split differs'):replay_corpus(corpus,policy,config)
