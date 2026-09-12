"""Exact-table teacher-advantage transfer into existing circuit synapses.

This independent two-cue curriculum is a Gate 2A component. It is not poker
learning and does not qualify the separately trained conventional poker teacher.
"""
import argparse
import json
from pathlib import Path
import resource
import signal
import time
import numpy as np
import yaml
from flyholdem.connectome.registry import ROOT, digest
from flyholdem.interface.population import load_controller
from flyholdem.learning.eligibility import Eligibility
from flyholdem.learning.dopamine import PastBaseline, annotated_populations, pulse
from flyholdem.neural.checkpoint import Checkpoints, atomic_json
from flyholdem.provenance import identity, manifest, assert_compatible
from .journal import Journal

from .conditioning import cue_registration, paired_evidence
from flyholdem.interface.cue_player import CuePlayer
from flyholdem.teacher.exact_cue import ExactCueTeacher

def run(config, output, profile, learning_rate, resume=False, development_reference=None,
        stop_after=None):
    from pyarrow import feather
    if learning_rate not in config['learning_rates']:
        raise ValueError('Learning rate is outside the registered candidate sequence')
    if profile not in config['profiles']:
        raise ValueError('Unknown registered profile')
    prerequisite = ROOT / config['conditioning_reference']
    if digest(prerequisite) != config['conditioning_reference_sha256']:
        raise ValueError('Conditioning prerequisite checksum mismatch')
    evidence = json.loads(prerequisite.read_text())
    if evidence.get('gate') != 2 or evidence.get('status') != 'pass' or evidence.get('profile') != 'confirmatory':
        raise ValueError('Passing registered conditioning confirmation required before transfer')
    optimization = config.get('optimization', 'teacher-advantage-local-eligibility')
    if optimization not in ('teacher-advantage-local-eligibility', 'direct-readout-rate-surrogate-v1'):
        raise ValueError('Unknown registered exact-transfer optimizer')
    use_surrogate = optimization == 'direct-readout-rate-surrogate-v1'
    if use_surrogate:
        failed_path = ROOT / config['failed_local_reference']
        failed = json.loads(failed_path.read_text())
        if (digest(failed_path) != config['failed_local_reference_sha256'] or failed.get('status') != 'fail'
                or failed.get('profile') != 'confirmatory' or failed.get('gate_component') != '2A-small-exact-transfer'):
            raise ValueError('Preserved failed local transfer confirmation is required')
    base = ROOT / config['preregistration']
    if digest(base) != config['preregistration_sha256']:
        raise ValueError('Frozen controllability artifact changed')
    graph_path = ROOT / 'connectome_data/malecns_v1' / ('prepared-' + config['mode'])
    controller = load_controller(graph_path, base)
    brain, registration = controller.brain, controller.registration
    nodes = feather.read_table(graph_path / 'neurons.feather').to_pandas()
    plastic_config = {**config['plasticity'], 'learning_rate': learning_rate}
    eligible = Eligibility(brain, np.flatnonzero(nodes['class'].eq('Kenyon_Cell').to_numpy()),
                           np.flatnonzero(nodes['class'].eq('MBON').to_numpy()), plastic_config)
    surrogate = None
    if use_surrogate:
        from flyholdem.learning.surrogate import ReadoutSurrogate
        surrogate = ReadoutSurrogate(eligible, controller.ensembles, {**config['surrogate'], 'learning_rate': learning_rate})
    dopamine_populations = annotated_populations(nodes)
    cues = cue_registration(brain, registration, config['cue'])
    if use_surrogate and (failed['cue_hash'] != identity(cues) or failed['mode'] != config['mode']):
        raise ValueError('Surrogate must retain the failed local cue/mode registration')
    teacher = ExactCueTeacher(config['cue']['actions'])
    sensory = {'schema': 'native-cue-input-v1', 'cells': cues['indices'], 'actions': config['cue']['actions'],
               'amplitude_jitter': config['cue']['amplitude_jitter'], 'bin_ms': config['plasticity']['bin_ms']}
    player = CuePlayer(controller, sensory)
    selected = config['profiles'][profile]
    runtime_config = {**config, 'profile': profile, 'selected_learning_rate': learning_rate,
                      'cue_hash': identity(cues), 'plasticity_registration': eligible.registration,
                      'dopamine_indices': [p.tolist() for p in dopamine_populations],
                      'teacher_registration': teacher.registration, 'sensory_protocol': sensory}
    if surrogate:
        runtime_config['surrogate_registration'] = surrogate.registration
    if profile == 'confirmatory':
        if not development_reference:
            raise ValueError('Confirmation requires a passing development result')
        reference_path = Path(development_reference)
        reference = json.loads(reference_path.read_text())
        if (reference['profile'] != 'development' or not reference['development_criteria_met']
                or reference['protocol_hash'] != identity(config) or reference['learning_rate'] != learning_rate
                or reference['cue_hash'] != identity(cues)):
            raise ValueError('Incompatible or failed development reference')
        runtime_config['development_result_sha256'] = digest(reference_path)
    runtime = manifest(runtime_config, brain.graph_hash, identity(cues), identity(registration['ensembles']),
                       brain.build['binary_sha256'])
    output = Path(output); output.mkdir(parents=True, exist_ok=resume)
    if resume:
        assert_compatible(json.loads((output / 'manifest.json').read_text()), runtime)
    else:
        atomic_json(output / 'manifest.json', runtime)
        atomic_json(output / 'cues.json', cues)
        atomic_json(output / 'teacher.json', teacher.registration)
    baseline = PastBaseline()
    checkpoints = Checkpoints(output / 'checkpoints')
    journal = Journal(output / 'trials.jsonl', resume)
    checkpoint_index = 0
    best_validation = {'accuracy': -1.0, 'seed': None, 'scope': 'exact-cue transfer checkpoint; not poker seed selection'}
    if resume:
        arrays, extra, _ = checkpoints.load(runtime)
        checkpoint_index = extra['completed_operations']
        if checkpoint_index > len(journal.rows):
            raise ValueError('Checkpoint is ahead of journal')
        expected_head = journal.rows[checkpoint_index - 1]['hash'] if checkpoint_index else '0' * 64
        if extra['journal_head'] != expected_head:
            raise ValueError('Checkpoint/journal prefix mismatch')
        brain.restore_state({key: arrays['brain_' + key] for key in brain.state_names})
        eligible.restore_state({key: value for key, value in arrays.items() if key.startswith('eligibility_')})
        baseline.restore_state(extra['baseline'])
        best_validation = extra['best_validation']
    index = 0; last_checkpoint = time.monotonic(); started = time.perf_counter()
    stopped = False
    in_operation = False

    def checkpoint(best=False):
        nonlocal last_checkpoint
        if index < checkpoint_index or in_operation:
            return  # Preserve the previous valid boundary on an unexpected failure.
        arrays = {'brain_' + key: value for key, value in brain.state().items()}
        arrays.update(eligible.state())
        checkpoints.save(arrays, runtime, {'completed_operations': index,
            'journal_head': journal.rows[index - 1]['hash'] if index else '0' * 64,
            'baseline': baseline.state(), 'best_validation': best_validation, 'rng': 'Independent SHA-256-derived seed per stimulus; schedules reconstructed from recorded seed',
            'next_operation_may_continue_weights': True}, index, best=best)
        last_checkpoint = time.monotonic()

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    prior_handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    if not resume:
        checkpoint()

    def execute(label, operation):
        nonlocal index, in_operation
        if index < checkpoint_index:
            row = journal.rows[index]
            if row['label'] != label:
                raise ValueError('Checkpoint operation plan changed')
            value = row['value']
        else:
            in_operation = True
            value = journal.record(index, label, operation())
            in_operation = False
        index += 1
        if time.monotonic() - last_checkpoint >= 300:
            checkpoint()
        if index >= checkpoint_index and (stopped or stop_after is not None and index >= stop_after):
            raise KeyboardInterrupt('Stopped at complete exact-transfer operation; resume with --resume')
        return value

    def initialize():
        brain.weights[:] = brain.initial
        brain.reset_dynamics(); eligible.clear_traces(); baseline.contexts.clear()
        return {'weights': 'original', 'neural_state': 'rest', 'baseline': 'empty'}

    def trial(seed, cue, repeat, phase, learning=False, override_reward=None):
        random_seed = int(identity([seed, 'train' if phase == 'train' else 'evaluation', repeat])[:16], 16)
        row = player.decide(cue, random_seed, eligible)
        # Evaluation targets are used only by the scorer after a committed neural
        # action. The native CuePlayer accepts no target or teacher object.
        row.update(target=cues['targets'][cue], correct=row['selected'] == cues['targets'][cue],
                   teacher_connected=phase == 'train')
        if phase == 'train':
            advantage = teacher.advantage(cue, row['selected'])
            row['teacher_probabilities'] = teacher.probabilities(cue).tolist()
            row['teacher_advantage'] = advantage
            if surrogate:
                teaching_cue = cue if override_reward is None else int(override_reward)
                target = teacher.probabilities(teaching_cue)
                row['supervision'] = {'target': target.tolist(), 'teacher_cue': teaching_cue,
                                      'shuffled': override_reward is not None}
                row['plasticity'] = surrogate.step(row['scores'], [a in cues['targets'] for a in range(5)], target, learning)
                row['dopamine_pulse'] = {'delivered': False, 'reason': 'nonbiological surrogate optimization'}
                return row
            delivered = advantage if override_reward is None else float(override_reward)
            reward = baseline.event(delivered, 'exact-cue-teacher-advantage', transform='identity')
            update = eligible.reinforce(reward['dopamine'], learning)
            activity = pulse(brain, reward['dopamine'], dopamine_populations,
                             config['dopamine']['pulse_ms'], config['dopamine']['gain'])
            row.update({'reinforcement': reward, 'plasticity': update, 'dopamine_pulse': activity,
                        'reward_shuffled': override_reward is not None})
        return row

    shuffled_name = 'shuffled-teacher' if surrogate else 'shuffled-reward'
    summaries = []
    try:
        for seed in selected['seeds']:
            schedule = np.random.default_rng(seed).permutation(np.arange(selected['training_trials']) % 2)
            earned_rewards = []; measures = {}
            for control in ('plastic', 'frozen', shuffled_name):
                execute([seed, control, 'initialize'], initialize)
                phases = ['pre', 'train', 'post', 'retention', 'erased'] if control == 'plastic' else ['train', 'post']
                shuffled = (np.random.default_rng(seed + 30000).permutation(schedule if surrogate else earned_rewards)
                            if control == shuffled_name else None)
                for phase in phases:
                    if phase == 'retention':
                        def retain():
                            before = brain.weights[eligible.edges].copy()
                            counts = brain.advance(controller.blank, config['retention_ms'])
                            if not np.array_equal(before, brain.weights[eligible.edges]):
                                raise AssertionError('No-training interval changed weights')
                            return {'interval_ms': config['retention_ms'], 'spikes': int(counts.sum()), 'weights_unchanged': True}
                        execute([seed, control, 'retention-interval'], retain)
                    if phase == 'erased':
                        def erase():
                            brain.weights[eligible.edges] = eligible.initial
                            return {'restored_existing_edges': len(eligible.edges)}
                        execute([seed, control, 'restore-original-weights'], erase)
                    sequence = schedule if phase == 'train' else np.arange(2 * selected['evaluation_trials_per_cue']) % 2
                    rows = []
                    for repeat, cue in enumerate(sequence):
                        reward = int(shuffled[repeat]) if shuffled is not None and phase == 'train' else None
                        row = execute([seed, control, phase, repeat], lambda: trial(seed, int(cue), repeat, phase,
                                      learning=control != 'frozen', override_reward=reward))
                        rows.append(row)
                        if control == 'plastic' and phase == 'train':
                            earned_rewards.append(row['teacher_advantage'])
                    if phase != 'train':
                        measures[control + '/' + phase] = {'accuracy': float(np.mean([r['correct'] for r in rows])),
                            'selected': [r['selected'] for r in rows]}
                    if control == 'plastic' and phase == 'post' and index > checkpoint_index:
                        score = measures['plastic/post']['accuracy']
                        if score > best_validation['accuracy']:
                            best_validation.update(accuracy=score, seed=seed)
                            checkpoint(best=True)
                    print(json.dumps({'seed': seed, 'control': control, 'phase': phase,
                                      'accuracy': float(np.mean([r['correct'] for r in rows]))}), flush=True)
            pre = measures['plastic/pre']['selected']
            if pre != measures['plastic/erased']['selected'] or pre != measures['frozen/post']['selected']:
                raise AssertionError('Restored or frozen decisions differ from matched initial decisions')
            summary = {'seed': seed, **{name: value['accuracy'] for name, value in measures.items()},
                       'restored_decisions_equal_initial': True, 'frozen_decisions_equal_initial': True}
            summaries.append(summary)
        if index != len(journal.rows):
            raise ValueError('Journal has unexpected trailing operations')
        evidence = {}
        criteria = config['criterion']
        for control in ('frozen/post', shuffled_name + '/post', 'plastic/erased'):
            differences = [s['plastic/post'] - s[control] for s in summaries]
            evidence[control] = paired_evidence(differences, criteria['bootstrap_seed'], criteria['bootstrap_repeats'])
        accuracy = [s['plastic/post'] for s in summaries]
        retained = all(s['plastic/post'] - s['plastic/retention'] <= criteria['maximum_retention_drop'] for s in summaries)
        common = (np.mean(accuracy) >= criteria['mean_accuracy'] and min(accuracy) >= criteria['minimum_seed_accuracy']
                  and retained and all(e['mean'] >= criteria['mean_improvement'] for e in evidence.values()))
        confirm = common and all(e['one_sided_sign_flip_p'] <= criteria['one_sided_p'] for e in evidence.values())
        result = {'schema': 'exact-transfer-result-v1', 'gate_component': '2A-small-exact-transfer', 'mode': config['mode'], 'learning_mode': 'distilled-connectome',
                  'profile': profile, 'status': ('pass' if confirm else 'fail') if profile == 'confirmatory' else profile,
                  'development_criteria_met': bool(common) if profile == 'development' else False,
                  'protocol_hash': identity(config), 'cue_hash': identity(cues), 'learning_rate': learning_rate,
                  'plasticity': eligible.registration, 'seed_results': summaries, 'paired_evidence': evidence,
                  'retention_criterion_met': bool(retained), 'learning_claim': bool(profile == 'confirmatory' and confirm),
                  'journal_head': journal.rows[-1]['hash'], 'operations': index,
                  'elapsed_seconds_this_invocation': time.perf_counter() - started,
                  'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if __import__('sys').platform == 'darwin' else 1024),
                  'scope': 'Exact-table two-cue transfer in circuit; no poker claim or complete Gate 2A pass',
                  'optimization': optimization,
                  'surrogate_registration': surrogate.registration if surrogate else None,
                  'teacher_registration': teacher.registration, 'teacher_connected_at_evaluation': False,
                  'manifest_sha256': digest(output / 'manifest.json')}
        atomic_json(output / 'result.json', result)
        return result
    finally:
        checkpoint()
        journal.close()
        for sig, handler in prior_handlers.items():
            signal.signal(sig, handler)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/exact_transfer.yaml')
    p.add_argument('--output', required=True)
    p.add_argument('--profile', choices=['smoke', 'development', 'confirmatory'], default='development')
    p.add_argument('--learning-rate', type=float, required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--development-reference')
    p.add_argument('--stop-after', type=int, help='Deterministic recovery exercise at an operation boundary')
    args = p.parse_args()
    result = run(yaml.safe_load(Path(args.config).read_text()), args.output, args.profile, args.learning_rate,
                 args.resume, args.development_reference, args.stop_after)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
