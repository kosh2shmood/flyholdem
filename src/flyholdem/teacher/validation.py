"""Recompute teacher qualification from the registered paired-deal evidence.

These checks authenticate local artifact consistency, not an external signature.
A passing flag alone cannot authorize corpus generation or student training.
"""
from pathlib import Path
import numpy as np
import yaml
from flyholdem.connectome.registry import ROOT, digest
from flyholdem.experiments.reports import _json, audit_journal
from flyholdem.poker.opponents import VERSIONS
from flyholdem.provenance import identity
from .evaluation import evaluation_summary
from .loaders import sampling


def registered_suite():
    return yaml.safe_load((ROOT/'configs/teacher_evaluation.yaml').read_text())


def verify_evaluation(run, policy_path, registered_config=None, require_confirmatory=True):
    """Recompute all intervals and enforce the exact registered schedule.

    An explicit configuration supports separately preregistered experiments and
    small numerical fixtures. Corpus export always uses the project registration.
    No teacher is executed here; its runtime boundary is checked when loaded.
    """
    run=Path(run);policy_path=Path(policy_path)
    expected=registered_suite() if registered_config is None else registered_config
    result=_json((run/'result.json').read_text())
    runtime=_json((run/'manifest.json').read_text())
    policy=_json((policy_path/'manifest.json').read_text())
    config=runtime.get('config',{});profile=result.get('profile')
    policy_hash=digest(policy_path/'manifest.json')
    if (result.get('schema')!='teacher-evaluation-v1' or profile not in ('development','confirmatory')
            or require_confirmatory and profile!='confirmatory'):
        raise ValueError('A confirmatory validated teacher evaluation is required')
    if (tuple(expected['opponents'])!=tuple(VERSIONS) or config!={**expected,'profile':profile,'policy_sha256':policy_hash}
            or runtime.get('config_hash')!=identity(config)):
        raise ValueError('Teacher evaluation differs from its registered suite or configuration')
    if (result.get('manifest_sha256')!=digest(run/'manifest.json')
            or result.get('policy_sha256')!=policy_hash
            or result.get('stack_bb')!=config['stack_bb']
            or policy.get('schema') not in ('teacher-average-policy-v1','teacher-best-response-policy-v1','teacher-potential-boundary-policy-v1','teacher-external-regret-policy-v1')
            or policy.get('provenance',{}).get('stack_bb')!=config['stack_bb']):
        raise ValueError('Teacher evaluation policy, stack or manifest mismatch')
    # Tensor identities are checked offline as well as by load_policy later.
    files=policy.get('files',{})
    if not files or any(Path(name).name!=name or not name.endswith('.npy') for name in files):
        raise ValueError('Teacher policy must identify its local numeric tensors')
    if any(digest(policy_path/name)!=checksum for name,checksum in files.items()):
        raise ValueError('Teacher tensor checksum mismatch')
    boundary=result.get('information_boundary',{})
    if (result.get('information_boundary_verified') is not True
            or boundary.get('hidden_hole_future_deck_and_teacher_label_invariance') is not True
            or type(boundary.get('decisions_checked')) is not int or boundary['decisions_checked']<32):
        raise ValueError('The recorded teacher information-boundary check is incomplete')
    selected=config['profiles'][profile];count=selected['paired_deals_per_opponent']
    if type(count) is not int or count<2:raise ValueError('At least two paired deals are required')
    journal=run/'paired-deals.jsonl';audit=audit_journal(journal)
    if audit['rows']!=count*len(VERSIONS) or result.get('journal_head')!=audit['head']:
        raise ValueError('Teacher evaluation is incomplete or has trailing paired deals')
    grouped={kind:[] for kind in config['opponents']}
    with journal.open() as stream:
        for index,line in enumerate(stream):
            entry=_json(line);kind=config['opponents'][index//count];repeat=index%count
            row=entry['value'];returns=np.asarray(row.get('seat_returns_bb'),dtype=float)
            actions=np.asarray(row.get('action_counts'))
            if (entry.get('label')!=[kind,repeat] or row.get('deal_seed')!=selected['seed_start']+repeat
                    or returns.shape!=(2,) or not np.isfinite(returns).all()
                    or np.any(np.abs(returns)>config['stack_bb'])
                    or row.get('paired_bb_per_hand')!=float(returns.mean())
                    or actions.shape!=(5,) or actions.dtype.kind not in 'iu' or np.any(actions<0)):
                raise ValueError('Invalid teacher paired-deal schedule, returns or action counts')
            grouped[kind].append(row)
    recalculated=evaluation_summary(grouped,config)
    recalculated["sampling"]=sampling(policy)
    for key,value in recalculated.items():
        if result.get(key)!=value:raise ValueError('Teacher evaluation summary differs from its paired-deal evidence: '+key)
    allowed=profile=='confirmatory' and recalculated['passes_fixed_suite']
    if result.get('allowed_as_teacher') is not allowed:
        raise ValueError('Teacher qualification flag disagrees with recomputed confirmation')
    if require_confirmatory and not allowed:raise ValueError('Teacher has not passed its complete registered confirmation suite')
    if audit_journal(journal)!=audit:raise ValueError('Teacher evaluation changed during verification')
    return {'schema':'verified-teacher-evaluation-v1','profile':profile,'allowed_as_teacher':allowed,
        'passes_fixed_suite':recalculated['passes_fixed_suite'],'policy_sha256':policy_hash,
        'evaluation_result_sha256':digest(run/'result.json'),'evaluation_manifest_sha256':digest(run/'manifest.json'),
        'paired_deals_sha256':audit['sha256'],'paired_deals':audit['rows'],
        'summary_recomputed':True,'registered_schedule_verified':True}
