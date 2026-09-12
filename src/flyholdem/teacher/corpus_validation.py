"""Reproduce a qualified corpus from its fixed teacher and complete PokerKit hands.

Checksums alone cannot show that targets were produced by the named policy.
This audit repeats collection, stratification and stopping, then compares bytes.
"""
import json
from pathlib import Path
import yaml
from flyholdem.connectome.registry import ROOT,digest
from flyholdem.poker.infoset import canonical_information_id
from flyholdem.poker.observation import canonical_bytes
from flyholdem.provenance import identity
from flyholdem.experiments.reports import audit_journal
from .corpus import collection_hand,validate_targets,verify_corpus,stratum,split_for_id


def replay_corpus(output,policy,registered_config):
    """Audit collection/targets; this alone does not qualify the policy as a teacher."""
    output=Path(output);record=json.loads((output/'manifest.json').read_text())
    config=record['config']
    if config!=registered_config or record['config_hash']!=identity(config):
        raise ValueError('Corpus differs from its registered coverage and collection protocol')
    verify_corpus(output)
    runtime=json.loads((output/'run-manifest.json').read_text())
    expected={**config,'teacher_sha256':record['teacher_policy_sha256'],'validation_sha256':record['teacher_validation_sha256']}
    if runtime['config']!=expected or runtime['config_hash']!=identity(expected):
        raise ValueError('Corpus run manifest binding mismatch')
    audit=audit_journal(output/'collection.jsonl')
    if audit['rows']!=record['collection_hands'] or not 1<=audit['rows']<=config['maximum_hands']:
        raise ValueError('Corpus collection hand count mismatch')
    rows={};occurrences={};strata={};streets=[0]*4
    with (output/'collection.jsonl').open() as stream:
        for hand_index,line in enumerate(stream):
            saved=json.loads(line)
            if saved['label']!=['collection-hand',hand_index] or set(saved['value'])!={'visible_states'}:
                raise ValueError('Corpus collection schedule mismatch')
            actual=collection_hand(policy,config,hand_index)
            if canonical_bytes(actual)!=canonical_bytes(saved['value']['visible_states']):
                raise ValueError('Corpus targets or complete PokerKit collection trajectory differ from the frozen teacher')
            for item in actual:
                observation,probabilities=validate_targets(item['observation'],item['probabilities'])
                info=canonical_information_id(observation);value=(canonical_bytes(observation),probabilities.tobytes())
                if info in occurrences and occurrences[info]!=value:
                    raise ValueError('Repeated information set has inconsistent targets')
                occurrences[info]=value;bucket=stratum(observation);key=identity(bucket)
                if info not in rows and strata.get(key,0)<config['maximum_per_stratum']:
                    rows[info]={'schema':'distillation-row-v1','information_id':info,'observation':observation,
                        'teacher_probabilities':probabilities.tolist(),'split':split_for_id(info),'stratum':bucket}
                    strata[key]=strata.get(key,0)+1;streets[observation['street']]+=1
            sufficient=len(rows)>=config['minimum_rows'] and min(streets)>=config['minimum_per_street']
            if sufficient and hand_index+1!=audit['rows']:
                raise ValueError('Corpus continued after its registered first coverage boundary')
    if not sufficient:raise ValueError('Corpus did not reach registered coverage')
    counts={}
    for split in ('train','validation','test'):
        values=sorted((row for row in rows.values() if row['split']==split),key=lambda row:row['information_id'])
        expected=b''.join(canonical_bytes(row)+b'\n' for row in values)
        if not values or (output/(split+'.jsonl')).read_bytes()!=expected:
            raise ValueError('Corpus split differs from reproduced stratified rows')
        counts[split]=len(values)
    checks={'split_counts':counts,'street_counts':streets,'strata':len(strata),'unique_information_sets':len(rows),
        'collection_hands':audit['rows'],'hidden_information':False,'strength_bucket':'visible-strength-rule-v1; coarse proxy'}
    if any(record.get(key)!=value for key,value in checks.items()):
        raise ValueError('Corpus coverage metadata differs from actual collected states')
    if digest(output/'collection.jsonl')!=audit['sha256']:raise ValueError('Corpus changed during verification')
    return {'schema':'replayed-corpus-audit-v1','canonical_ids_disjoint':True,'complete_collection_reproduced':True,
        'teacher_targets_reproduced':True,'corpus_sha256':digest(output/'manifest.json'),
        'collection_sha256':audit['sha256'],**checks,'allowed_as_teacher':False}


def verify_qualified_corpus(output,policy_path,validation_run):
    from .validation import verify_evaluation
    from .loaders import load_policy
    from .evaluation import verify_information_boundary
    qualification=verify_evaluation(validation_run,policy_path)
    output=Path(output);record=json.loads((output/'manifest.json').read_text())
    if (record['teacher_policy_sha256']!=digest(Path(policy_path)/'manifest.json')
            or record['teacher_validation_sha256']!=digest(Path(validation_run)/'result.json')):
        raise ValueError('Corpus does not name this qualified teacher and confirmation')
    policy,_=load_policy(policy_path);boundary=verify_information_boundary(policy)
    registered=yaml.safe_load((ROOT/'configs/corpus.yaml').read_text())
    result=replay_corpus(output,policy,registered)
    return {**result,'schema':'qualified-corpus-audit-v1','qualified_teacher':qualification,
        'information_boundary':boundary,'qualified_for_full_20bb_transfer':True}
