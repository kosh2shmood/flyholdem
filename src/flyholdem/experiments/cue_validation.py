"""Recompute registered cue gate statistics from every recorded operation.

This is evidence consistency validation, not a rerun of native training and not
an assertion that a hand-written log is an independently measured experiment.
"""
import json
from pathlib import Path
import numpy as np
from flyholdem.connectome.registry import digest
from flyholdem.provenance import identity
from .conditioning import paired_evidence
from .reports import audit_journal


def verify_cue_evidence(run,protocol,require_pass=True):
    run=Path(run);result=json.loads((run/'result.json').read_text())
    manifest=json.loads((run/'manifest.json').read_text());config=manifest['config']
    transfer=result.get('schema')=='exact-transfer-result-v1'
    if not transfer and result.get('schema')!='conditioning-result-v1':raise ValueError('Unsupported cue evidence')
    expected_mode='distilled-connectome' if transfer else 'bio-plastic'
    if (result.get('profile')!='confirmatory' or config.get('profile')!='confirmatory'
            or result.get('learning_mode')!=expected_mode or protocol.get('learning_mode')!=expected_mode
            or result.get('mode')!=protocol['mode'] or result.get('protocol_hash')!=identity(protocol)
            or manifest.get('config_hash')!=identity(config) or any(config.get(k)!=v for k,v in protocol.items())
            or result.get('manifest_sha256')!=digest(run/'manifest.json')
            or result.get('learning_rate')!=config.get('selected_learning_rate')
            or result['learning_rate'] not in protocol['learning_rates']):
        raise ValueError('Cue evidence does not match its complete registered confirmation protocol')
    if (transfer and result.get('gate_component')!='2A-small-exact-transfer'
            or not transfer and result.get('gate')!=2):raise ValueError('Cue gate scope mismatch')
    cues=json.loads((run/'cues.json').read_text())
    if identity(cues)!=result['cue_hash'] or identity(cues)!=config['cue_hash'] or cues['targets']!=protocol['cue']['actions']:
        raise ValueError('Cue sensory registration mismatch')
    if result['plasticity']!=config['plasticity_registration']:raise ValueError('Cue plasticity binding mismatch')
    if transfer and (result['teacher_registration']!=config['teacher_registration']
            or result['teacher_connected_at_evaluation'] is not False):
        raise ValueError('Transfer teacher registration mismatch')
    surrogate=transfer and protocol.get('optimization')=='direct-readout-rate-surrogate-v1'
    if surrogate and result['surrogate_registration']!=config['surrogate_registration']:
        raise ValueError('Surrogate edge registration mismatch')
    shuffled='shuffled-teacher' if surrogate else 'shuffled-reward'
    selected=protocol['profiles']['confirmatory'];seeds=selected['seeds']
    if len(seeds)<5 or len(set(seeds))!=len(seeds):raise ValueError('At least five independent confirmation seeds required')
    other={seed for name,p in protocol['profiles'].items() if name!='confirmatory' for seed in p['seeds']}
    if other.intersection(seeds):raise ValueError('Confirmation seed overlap')
    audit=audit_journal(run/'trials.jsonl')
    if audit['head']!=result['journal_head'] or audit['rows']!=result['operations']:
        raise ValueError('Cue journal binding mismatch')
    source=iter(json.loads(line) for line in (run/'trials.jsonl').read_text().splitlines());consumed=0
    def take(label):
        nonlocal consumed
        try:row=next(source)
        except StopIteration:raise ValueError('Incomplete registered cue schedule') from None
        if row['label']!=label:raise ValueError('Cue operation schedule differs from registration')
        consumed+=1;return row['value']
    summaries=[];actions=cues['targets'];legal=np.array([a in actions for a in range(5)])
    for seed in seeds:
        schedule=np.random.default_rng(seed).permutation(np.arange(selected['training_trials'])%2)
        measures={};decisions={}
        for arm in ('plastic','frozen',shuffled):
            initial=take([seed,arm,'initialize'])
            if initial!={'weights':'original','neural_state':'rest','baseline':'empty'}:raise ValueError('Cue initial-state contract changed')
            phases=('pre','train','post','retention','erased') if arm=='plastic' else ('train','post')
            for phase in phases:
                if phase=='retention':
                    retained=take([seed,arm,'retention-interval'])
                    if retained.get('interval_ms')!=protocol['retention_ms'] or retained.get('weights_unchanged') is not True:
                        raise ValueError('Invalid no-learning retention interval')
                if phase=='erased':
                    erased=take([seed,arm,'restore-original-weights'])
                    if erased.get('restored_existing_edges')!=result['plasticity']['edge_count']:
                        raise ValueError('Erasure must restore the registered existing edges')
                sequence=schedule if phase=='train' else np.arange(2*selected['evaluation_trials_per_cue'])%2
                chosen=[];correct=[]
                for repeat,cue in enumerate(sequence):
                    row=take([seed,arm,phase,repeat]);scores=np.asarray(row['scores'],dtype=float)
                    expected_seed=int(identity([seed,'train' if phase=='train' else 'evaluation',repeat])[:16],16)
                    if (scores.shape!=(5,) or not np.isfinite(scores).all() or row['cue']!=int(cue)
                            or row['target']!=actions[cue] or row['stimulus_seed']!=expected_seed
                            or type(row['selected']) is not int or row['selected']!=int(np.argmax(np.where(legal,scores,-np.inf)))
                            or row['correct'] is not (row['selected']==actions[cue])):
                        raise ValueError('Cue decision, native-score argmax, stimulus or correctness mismatch')
                    if transfer and row.get('teacher_connected') is not (phase=='train'):
                        raise ValueError('Teacher connection differs from the registered phase')
                    if phase!='train' and any(key in row for key in ('reinforcement','plasticity','dopamine_pulse','supervision')):
                        raise ValueError('Cue evaluation contains a learning event')
                    chosen.append(row['selected']);correct.append(row['correct'])
                if phase!='train':
                    key=arm+'/'+phase;measures[key]=float(np.mean(correct));decisions[key]=chosen
        if decisions['plastic/pre']!=decisions['plastic/erased'] or decisions['plastic/pre']!=decisions['frozen/post']:
            raise ValueError('Erased or frozen decisions differ from initial weights')
        summaries.append({'seed':seed,**measures,'restored_decisions_equal_initial':True,'frozen_decisions_equal_initial':True})
    if consumed!=audit['rows']:raise ValueError('Trailing cue operations')
    criteria=protocol['criterion'];evidence={}
    for arm in ('frozen/post',shuffled+'/post','plastic/erased'):
        evidence[arm]=paired_evidence([row['plastic/post']-row[arm] for row in summaries],criteria['bootstrap_seed'],criteria['bootstrap_repeats'])
    accuracies=[row['plastic/post'] for row in summaries]
    retained=all(row['plastic/post']-row['plastic/retention']<=criteria['maximum_retention_drop'] for row in summaries)
    passed=bool(np.mean(accuracies)>=criteria['mean_accuracy'] and min(accuracies)>=criteria['minimum_seed_accuracy'] and retained
        and all(value['mean']>=criteria['mean_improvement'] and value['one_sided_sign_flip_p']<=criteria['one_sided_p'] for value in evidence.values()))
    expected={'seed_results':summaries,'paired_evidence':evidence,'retention_criterion_met':retained,
        'status':'pass' if passed else 'fail','learning_claim':passed,'development_criteria_met':False}
    if any(result.get(key)!=value for key,value in expected.items()):raise ValueError('Cue gate summaries differ from complete operation evidence')
    if require_pass and not passed:raise ValueError('Cue confirmation failed its registered criteria')
    if audit_journal(run/'trials.jsonl')!=audit:raise ValueError('Cue evidence changed during verification')
    return {'schema':'verified-cue-evidence-v1','scope':'Small exact transfer' if transfer else 'Conditioning',
        'profile':'confirmatory','passed':passed,'mode':protocol['mode'],'optimization':protocol.get('optimization','teacher-advantage-local-eligibility' if transfer else 'terminal-local-eligibility'),
        'result_sha256':digest(run/'result.json'),'manifest_sha256':digest(run/'manifest.json'),
        'journal_sha256':audit['sha256'],'operations':consumed,'independent_seeds':len(seeds),
        'mean_accuracy':float(np.mean(accuracies)),'complete_schedule_verified':True,'statistics_recomputed':True,
        'native_training_reexecuted':False,'whole_gate2a_passed':False}
