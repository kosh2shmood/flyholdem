"""Gate 2A authorization from reverified full teacher, corpus and native evidence.

A certificate is a reproducible evidence bundle, not a signature or permission
to omit checks. Loading it repeats every dependency and actual teacher removal.
"""
import json
from pathlib import Path
import yaml
from flyholdem.connectome.registry import ROOT,digest
from flyholdem.provenance import identity
from flyholdem.neural.checkpoint import atomic_json


def _reference(item):
    run=ROOT/item['run'];protocol=yaml.safe_load((ROOT/item['protocol']).read_text())
    if digest(run/'result.json')!=item['result_sha256'] or digest(run/'manifest.json')!=item['manifest_sha256']:
        raise ValueError('Registered prerequisite evidence checksum mismatch')
    registration=ROOT/protocol['preregistration']
    if digest(registration)!=protocol['preregistration_sha256']:
        raise ValueError('Prerequisite controllability registration changed')
    return run,protocol


def _gate2a_evidence(policy,confirmation,corpus):
    from flyholdem.teacher.corpus_validation import verify_qualified_corpus
    from .cue_validation import verify_cue_evidence
    from .teacher_removal import verify_cue_teacher_removal
    config=yaml.safe_load((ROOT/'configs/gate2a.yaml').read_text())
    if (config.get('schema')!='full-teacher-transfer-gate2a-v1'
            or digest(ROOT/'configs/teacher_evaluation.yaml')!=config['teacher_evaluation_yaml_sha256']):
        raise ValueError('Full teacher transfer gate registration changed')
    # Reject an unqualified full teacher before loading any biological model.
    corpus_evidence=verify_qualified_corpus(corpus,policy,confirmation)
    conditioning,conditioning_protocol=_reference(config['conditioning'])
    transferred,transfer_protocol=_reference(config['transfer'])
    failed,failed_protocol=_reference(config['failed_local'])
    conditioning_evidence=verify_cue_evidence(conditioning,conditioning_protocol)
    transfer_evidence=verify_cue_evidence(transferred,transfer_protocol)
    failure_evidence=verify_cue_evidence(failed,failed_protocol,require_pass=False)
    if (conditioning_evidence['mode']!='full' or transfer_evidence['mode'] not in ('fixture-native','circuit')
            or failure_evidence['passed'] or transfer_protocol['failed_local_reference_sha256']!=failure_evidence['result_sha256']
            or transfer_protocol['conditioning_reference_sha256']!=conditioning_evidence['result_sha256']):
        raise ValueError('Gate 2A prerequisite scope or local-before-surrogate sequence mismatch')
    model=ROOT/config['transfer']['model']
    if digest(model/'manifest.json')!=config['transfer']['model_sha256']:
        raise ValueError('Registered transferred model checksum mismatch')
    removal=verify_cue_teacher_removal(model,ROOT/config['transfer']['graph'],transferred)
    return config,{'conditioning':conditioning_evidence,'failed_local':failure_evidence,
        'transfer':transfer_evidence,'qualified_corpus':corpus_evidence,'teacher_removal':removal}


def certify_gate2a(policy,confirmation,corpus,output):
    output=Path(output)
    if output.exists():raise FileExistsError('Write a new immutable gate certificate path')
    inputs={key:str(Path(value).resolve()) for key,value in {'policy':policy,'confirmation':confirmation,'corpus':corpus}.items()}
    config,evidence=_gate2a_evidence(**inputs)
    record={'schema':'verified-gate2a-certificate-v1','gate':'2A','passed':True,
        'scope':config['scope'],'protocol_hash':identity(config),'inputs':inputs,'evidence':evidence,
        'evidence_hash':identity(evidence),'poker_learning_claim':False}
    output.parent.mkdir(parents=True,exist_ok=True);atomic_json(output,record)
    return {'certificate':str(output.resolve()),'sha256':digest(output),'gate':'2A','passed':True,'poker_learning_claim':False}


def verify_gate2a(certificate,expected_sha256=None):
    path=Path(certificate)
    if expected_sha256 is not None and digest(path)!=expected_sha256:raise ValueError('Pinned Gate 2A certificate changed')
    record=json.loads(path.read_text())
    if record.get('schema')!='verified-gate2a-certificate-v1' or record.get('gate')!='2A' or record.get('passed') is not True:
        raise ValueError('A full verified Gate 2A certificate is required')
    if set(record['inputs'])!={'policy','confirmation','corpus'}:raise ValueError('Gate evidence inputs changed')
    config,evidence=_gate2a_evidence(**record['inputs'])
    if (record['protocol_hash']!=identity(config) or record['scope']!=config['scope']
            or record['evidence_hash']!=identity(evidence) or record['evidence']!=evidence
            or record.get('poker_learning_claim') is not False):
        raise ValueError('Gate certificate differs from recomputed dependency evidence')
    if expected_sha256 is not None and digest(path)!=expected_sha256:raise ValueError('Gate certificate changed during verification')
    return record
