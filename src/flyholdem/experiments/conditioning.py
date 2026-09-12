"""Registered two-cue conditioning, matched controls, retention and erasure.

Each cue trial starts with fresh membrane/queue/trace state, preserving weights.
Stimuli and targets contain no poker information. Both cue groups have direct
paths to both frozen readout ensembles. Neural argmax is the only policy.
"""
import argparse
import itertools
import json
from pathlib import Path
import resource
import signal
import time
import numpy as np
import yaml
from flyholdem.connectome.registry import ROOT, digest
from flyholdem.interface.population import load_controller, neural_scores, select_action
from flyholdem.learning.eligibility import Eligibility
from flyholdem.learning.dopamine import PastBaseline, annotated_populations, pulse
from flyholdem.neural.checkpoint import Checkpoints, atomic_json
from flyholdem.provenance import identity, manifest, assert_compatible
from .journal import Journal


def paired_evidence(differences, seed, repeats):
    values = np.asarray(differences, dtype=np.float64)
    observed = float(values.mean())
    null = [np.mean(values * signs) for signs in itertools.product((-1, 1), repeat=len(values))]
    probability = sum(value >= observed - 1e-15 for value in null) / len(null)
    means = np.random.default_rng(seed).choice(values, size=(repeats, len(values)), replace=True).mean(axis=1)
    return {'mean': observed, 'seed_differences': values.tolist(), 'one_sided_sign_flip_p': probability,
            'paired_seed_bootstrap_95': np.quantile(means, [.025, .975]).tolist()}


def cue_registration(brain, registration, config):
    inputs = np.asarray(registration['input_indices'], dtype=np.int32)
    actions = config['actions']
    if len(actions) != 2 or len(set(actions)) != 2 or any(a not in range(5) for a in actions):
        raise ValueError('Two distinct frozen actions required')
    groups = [np.asarray(registration['ensembles'][a]['indices']) for a in actions]
    # Uniform gain leaves contact ranking unchanged. Use absolute original strength
    # so signs do not turn an existing anatomical path into a deleted connection.
    profiles = np.zeros((len(inputs), 2), dtype=np.float64)
    for row, node in enumerate(inputs):
        start, end = brain.ptr[node:node + 2]
        for action, group in enumerate(groups):
            profiles[row, action] = np.abs(brain.initial[start:end][np.isin(brain.post[start:end], group)]).sum()
    shared = np.flatnonzero(np.all(profiles > 0, axis=1))
    count = config['cells_per_cue']
    if len(shared) < count * 2:
        raise ValueError('Insufficient KC inputs with paths to both frozen readouts')
    order = shared[np.lexsort((inputs[shared], -profiles[shared].min(axis=1)))]
    chosen = np.random.default_rng(config['seed']).permutation(order[:count * 2])
    cues = inputs[chosen].reshape(2, count)
    return {'schema': 'conditioning-cues-v1', 'selection': config['selection'],
            'shared_candidates': len(shared), 'indices': cues.tolist(), 'targets': actions,
            'selected_strengths': profiles[chosen].tolist(), 'selection_seed': config['seed'],
            'no_poker_information_used': True}


def run(config, output, profile, learning_rate, resume=False, development_reference=None,
        stop_after=None):
    from pyarrow import feather
    if learning_rate not in config['learning_rates']:
        raise ValueError('Learning rate is outside the registered candidate sequence')
    if profile not in config['profiles']:
        raise ValueError('Unknown registered profile')
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
    dopamine_populations = annotated_populations(nodes)
    cues = cue_registration(brain, registration, config['cue'])
    selected = config['profiles'][profile]
    runtime_config = {**config, 'profile': profile, 'selected_learning_rate': learning_rate,
                      'cue_hash': identity(cues), 'plasticity_registration': eligible.registration,
                      'dopamine_indices': [p.tolist() for p in dopamine_populations]}
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
    baseline = PastBaseline()
    checkpoints = Checkpoints(output / 'checkpoints')
    journal = Journal(output / 'trials.jsonl', resume)
    checkpoint_index = 0
    best_validation = {'accuracy': -1.0, 'seed': None, 'scope': 'conditioning checkpoint, not seed selection for poker'}
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

    prior_handler = signal.signal(signal.SIGTERM, stop)
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
            raise KeyboardInterrupt('Stopped at complete conditioning operation; resume with --resume')
        return value

    def initialize():
        brain.weights[:] = brain.initial
        brain.reset_dynamics(); eligible.clear_traces(); baseline.contexts.clear()
        return {'weights': 'original', 'neural_state': 'rest', 'baseline': 'empty'}

    def trial(seed, cue, repeat, phase, learning=False, override_reward=None):
        brain.reset_dynamics(); eligible.clear_traces()
        random_seed = int(identity([seed, 'train' if phase == 'train' else 'evaluation', repeat])[:16], 16)
        rng = np.random.default_rng(random_seed)
        drive = np.zeros(brain.n, dtype=np.float32)
        cells = np.asarray(cues['indices'][cue])
        jitter = config['cue']['amplitude_jitter']
        drive[cells] = registration['selected_gain'] * rng.uniform(1 - jitter, 1 + jitter, len(cells))
        timing = registration['config']['decision_ms']
        eligible.advance(controller.blank, timing['baseline'])
        eligible.advance(drive, timing['stimulus'])
        counts = eligible.advance(drive, timing['readout'])
        rates, scores = neural_scores(counts, controller.ensembles, timing['readout'], registration['baseline_hz'],
                                     registration['config']['score_scale_hz'])
        legal = [a in cues['targets'] for a in range(5)]
        action = select_action(scores, legal, rng)
        correct = action == cues['targets'][cue]
        row = {'cue': int(cue), 'target': cues['targets'][cue], 'selected': action, 'correct': bool(correct),
               'rates_hz': rates.tolist(), 'scores': scores.tolist(), 'readout_spikes': int(counts.sum()),
               'stimulus_seed': random_seed, 'earned_reward': 1 if correct else -1}
        if phase == 'train':
            delivered = row['earned_reward'] if override_reward is None else int(override_reward)
            reward = baseline.event(delivered, 'conditioning-all-cues', transform='identity')
            update = eligible.reinforce(reward['dopamine'], learning)
            activity = pulse(brain, reward['dopamine'], dopamine_populations,
                             config['dopamine']['pulse_ms'], config['dopamine']['gain'])
            row.update({'reinforcement': reward, 'plasticity': update, 'dopamine_pulse': activity,
                        'reward_shuffled': override_reward is not None})
        return row

    summaries = []
    try:
        for seed in selected['seeds']:
            schedule = np.random.default_rng(seed).permutation(np.arange(selected['training_trials']) % 2)
            earned_rewards = []; measures = {}
            for control in ('plastic', 'frozen', 'shuffled-reward'):
                execute([seed, control, 'initialize'], initialize)
                phases = ['pre', 'train', 'post', 'retention', 'erased'] if control == 'plastic' else ['train', 'post']
                shuffled = (np.random.default_rng(seed + 30000).permutation(earned_rewards)
                            if control == 'shuffled-reward' else None)
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
                            earned_rewards.append(row['earned_reward'])
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
        for control in ('frozen/post', 'shuffled-reward/post', 'plastic/erased'):
            differences = [s['plastic/post'] - s[control] for s in summaries]
            evidence[control] = paired_evidence(differences, criteria['bootstrap_seed'], criteria['bootstrap_repeats'])
        accuracy = [s['plastic/post'] for s in summaries]
        retained = all(s['plastic/post'] - s['plastic/retention'] <= criteria['maximum_retention_drop'] for s in summaries)
        common = (np.mean(accuracy) >= criteria['mean_accuracy'] and min(accuracy) >= criteria['minimum_seed_accuracy']
                  and retained and all(e['mean'] >= criteria['mean_improvement'] for e in evidence.values()))
        confirm = common and all(e['one_sided_sign_flip_p'] <= criteria['one_sided_p'] for e in evidence.values())
        result = {'schema': 'conditioning-result-v1', 'gate': 2, 'mode': config['mode'], 'learning_mode': 'bio-plastic',
                  'profile': profile, 'status': ('pass' if confirm else 'fail') if profile == 'confirmatory' else profile,
                  'development_criteria_met': bool(common) if profile == 'development' else False,
                  'protocol_hash': identity(config), 'cue_hash': identity(cues), 'learning_rate': learning_rate,
                  'plasticity': eligible.registration, 'seed_results': summaries, 'paired_evidence': evidence,
                  'retention_criterion_met': bool(retained), 'learning_claim': bool(profile == 'confirmatory' and confirm),
                  'journal_head': journal.rows[-1]['hash'], 'operations': index,
                  'elapsed_seconds_this_invocation': time.perf_counter() - started,
                  'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if __import__('sys').platform == 'darwin' else 1024),
                  'scope': 'Two engineered cues; not a poker learning result',
                  'manifest_sha256': digest(output / 'manifest.json')}
        atomic_json(output / 'result.json', result)
        return result
    finally:
        checkpoint()
        journal.close()
        signal.signal(signal.SIGTERM, prior_handler)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/conditioning.yaml')
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
