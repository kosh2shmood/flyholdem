"""Fixed-final, resumable two-player self-play; no biological controller."""
import copy
import json
from pathlib import Path
import resource
import signal
import sys
import time

import numpy as np
import yaml

from flyholdem.connectome.registry import ROOT, digest
from flyholdem.experiments.journal import Journal
from flyholdem.experiments.poker_evidence import iter_journal
from flyholdem.neural.checkpoint import Checkpoints, atomic_json
from flyholdem.poker.engine import Hand
from flyholdem.provenance import assert_compatible, identity, manifest
from .equity import backend_identity
from .self_play_regret import AGGREGATION, ROLE_NAMES, SCHEMA, SelfPlayRegretTable

RESULT_SCHEMA = 'self-play-regret-training-result-v1'
METHOD = {'public_roles': list(ROLE_NAMES), 'passes_per_iteration': 2,
          'chance_schedule': 'two-consecutive-deal-seeds-per-iteration-fixed-button-zero',
          'strategy_updates': 'synchronous-after-both-complete-passes',
          'sampling': 'PCG64-explicit-probability-choice-one-float64-per-new-opponent-infoset'}
_DERIVED = {'self_play_method', 'equity_backend', 'evaluation_protocol_sha256'}


def runtime(config):
    if (config.get('schema') != SCHEMA or config.get('aggregation') != AGGREGATION
            or type(config.get('iterations')) is not int or not 0 < config['iterations'] < 10**12
            or type(config.get('stack_bb')) is not int or config['stack_bb'] < 2
            or config['stack_bb'] != config['abstraction']['stack_bb']
            or type(config.get('sampling_seed')) is not int or config['sampling_seed'] < 0
            or type(config.get('deal_seed_start')) is not int or config['deal_seed_start'] < 0
            or type(config.get('progress_iterations', 100)) is not int or config.get('progress_iterations', 100) < 1
            or _DERIVED.intersection(config) or {'opponents', 'opponent_probabilities'}.intersection(config)
            or config.get('mode', 'conventional-teacher-control') != 'conventional-teacher-control'):
        raise ValueError('Complete fixed two-player self-play registration required')
    SelfPlayRegretTable(config['abstraction'])
    evaluation_path = ROOT / 'configs/teacher_evaluation.yaml'
    evaluation = yaml.safe_load(evaluation_path.read_text())
    interval = (config['deal_seed_start'], config['deal_seed_start'] + 2 * config['iterations'])
    reserved = [(p['seed_start'], p['seed_start'] + p['paired_deals_per_opponent'])
                for p in evaluation['profiles'].values()] + [(821001, 821005)]
    if any(max(interval[0], start) < min(interval[1], end) for start, end in reserved):
        raise ValueError('Both self-play chance deals must exclude registered evaluation and boundary ranges')
    backend = backend_identity()
    return manifest({**config, 'self_play_method': METHOD, 'equity_backend': backend,
                     'evaluation_protocol_sha256': digest(evaluation_path)},
                    'conventional-no-connectome', identity(config['abstraction']),
                    identity(AGGREGATION), identity(backend))


def _assert_runtime(saved, expected):
    if (identity(saved.get('config')) != saved.get('config_hash')
            or saved.get('config') != expected['config']):
        raise ValueError('Self-play stored configuration and derived runtime binding changed')
    assert_compatible(saved, expected)


def _deal_seeds(config, index):
    first = config['deal_seed_start'] + 2 * index
    return [first, first + 1]


def _draws(row):
    values = [item['sampled_action_draws'] for item in row['traversals']]
    if (len(values) != 2 or any(type(value) is not int or value < 0 for value in values)
            or type(row.get('sampled_action_draws')) is not int
            or row['sampled_action_draws'] != sum(values)):
        raise ValueError('Self-play sampling-draw evidence changed')
    return sum(values)


def _core_row(row):
    return {key: value for key, value in row.items()
            if key not in ('deal_seeds', 'sampling_rng_before', 'sampling_rng_after', 'iteration_trace_sha256')}


def _advance_rng(rng, draws):
    # Explicit-p Generator.choice consumes one float64 draw per call in the
    # locked runtime. Independent core tests and every actual iteration check
    # this contract, including memoized information sets consuming no draw.
    while draws:
        block = min(draws, 65536)
        rng.random(block)
        draws -= block


def verify_journal(run, config, *, count=None, through=None):
    """Stream the complete chain, deal schedule and original sampler state."""
    rng = np.random.default_rng(config['sampling_seed'])
    rows = nodes = leaves = draws = 0
    head = '0' * 64
    counts = [0, 0]
    for index, item in enumerate(iter_journal(Path(run) / 'iterations.jsonl')):
        if through is not None and index >= through:
            # Consume the complete file so iter_journal also checks the final
            # byte digest, while retaining the requested checkpoint prefix.
            continue
        row = item['value']
        seeds = _deal_seeds(config, index)
        passes = row.get('traversals', [])
        if (item['index'] != index or item['label'] != seeds
                or row.get('schema') != 'self-play-regret-iteration-v1' or row.get('iteration') != index + 1
                or row.get('deal_seeds') != seeds or len(passes) != 2
                or [p['updating_role'] for p in passes] != [0, 1]
                or row.get('iteration_trace_sha256') != identity(_core_row(row))
                or row.get('sampling_rng_before') != rng.bit_generator.state):
            raise ValueError('Self-play journal iteration, source trace or sampling schedule changed')
        for role, (seed, item_pass) in enumerate(zip(seeds, passes)):
            expected = Hand(seed, button=0, stacks=(2 * config['stack_bb'],) * 2).serialize()
            tally_names = ('nodes', 'terminal_branches', 'regret_information_sets',
                           'average_information_sets', 'average_encounters', 'sampled_action_draws')
            valid_tallies = all(type(item_pass.get(name)) is int and item_pass[name] >= 0 for name in tally_names)
            if (item_pass['root_private_checkpoint'] != expected
                    or item_pass.get('public_role') != ROLE_NAMES[role] or not valid_tallies
                    or not 1 <= item_pass['nodes'] <= config['abstraction']['max_nodes_per_traversal']
                    or not 1 <= item_pass['terminal_branches'] <= item_pass['nodes']
                    or item_pass['sampled_action_draws'] != item_pass['average_information_sets']
                    or item_pass['average_information_sets'] > item_pass['average_encounters']
                    or item_pass['nodes'] != item_pass['terminal_branches'] + item_pass['regret_information_sets'] + item_pass['average_encounters']):
                raise ValueError('Self-play private deal or complete traversal evidence changed')
        step_draws = _draws(row)
        _advance_rng(rng, step_draws)
        current_counts = row['information_sets_by_role']
        if (row.get('sampling_rng_after') != rng.bit_generator.state
                or row['nodes'] != sum(p['nodes'] for p in passes)
                or row['terminal_branches'] != sum(p['terminal_branches'] for p in passes)
                or len(current_counts) != 2 or any(type(n) is not int or n < old for n, old in zip(current_counts, counts))
                or row['information_sets'] != sum(current_counts)
                or sum(current_counts) > config['abstraction']['max_information_sets']):
            raise ValueError('Self-play complete counts or RNG advancement changed')
        rows += 1
        nodes += row['nodes']
        leaves += row['terminal_branches']
        draws += step_draws
        counts = current_counts
        head = item['hash']
    if rows > config['iterations'] or count is not None and rows != count:
        raise ValueError('Incomplete or trailing self-play iterations')
    return {'iterations_completed': rows, 'counterfactual_nodes': nodes, 'terminal_branches': leaves,
            'sampled_action_draws': draws, 'information_sets_by_role': counts,
            'information_sets': sum(counts), 'journal_head': head,
            'sampling_rng': copy.deepcopy(rng.bit_generator.state)}


def train(config, output, resume=False, stop_after=None):
    if stop_after is not None and (type(stop_after) is not int or stop_after < 1):
        raise ValueError('Stop only at a positive complete iteration boundary')
    expected = runtime(config)
    table = SelfPlayRegretTable(config['abstraction'])
    root = Path(output)
    root.mkdir(parents=True, exist_ok=resume)
    if resume:
        _assert_runtime(json.loads((root / 'manifest.json').read_text()), expected)
    else:
        atomic_json(root / 'manifest.json', expected)
    checkpoints = Checkpoints(root / 'checkpoints')
    journal = Journal(root / 'iterations.jsonl', resume)
    rng = np.random.default_rng(config['sampling_seed'])
    completed = 0
    stopped = in_iteration = False
    started = time.perf_counter()
    last_checkpoint = time.monotonic()
    last_saved_completed = None
    checkpoint_failed = False
    if resume:
        try:
            arrays, extra, metadata = checkpoints.load(expected)
            _assert_runtime(metadata['identity'], expected)
            table.restore(arrays)
            completed = table.iterations_completed
            prefix = verify_journal(root, config, count=completed, through=completed)
            if (metadata['step'] != completed or extra['iterations_completed'] != completed
                    or not completed <= len(journal.rows) <= config['iterations']
                    or extra['journal_head'] != prefix['journal_head']
                    or extra['sampling_rng'] != prefix['sampling_rng']
                    or [len(t.keys) for t in table.tables] != prefix['information_sets_by_role']):
                raise ValueError('Self-play checkpoint/journal/RNG boundary mismatch')
            rng.bit_generator.state = extra['sampling_rng']
            last_saved_completed = completed
            if completed == config['iterations'] and (root / 'result.json').is_file():
                previous_result = json.loads((root / 'result.json').read_text())
                if previous_result.get('status') == 'trained-unvalidated':
                    completed_table(root)
                    journal.close()
                    return previous_result
        except BaseException:
            journal.close()
            raise

    def checkpoint():
        nonlocal last_checkpoint, last_saved_completed, checkpoint_failed
        if in_iteration or last_saved_completed == completed:
            return
        try:
            checkpoints.save(table.state(), expected,
                             {'iterations_completed': completed, 'sampling_rng': rng.bit_generator.state,
                              'journal_head': journal.rows[completed - 1]['hash'] if completed else '0' * 64,
                              'boundary': 'complete synchronous pair of external-sampling passes',
                              'selection': 'fixed final iteration, never held-out-profit-selected'},
                             completed, best=completed == config['iterations'])
        except BaseException:
            checkpoint_failed = True
            raise
        last_checkpoint = time.monotonic()
        last_saved_completed = completed

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        if not resume:
            checkpoint()
        for index in range(completed, config['iterations']):
            seeds = _deal_seeds(config, index)
            hands = [Hand(seed, button=0, stacks=(2 * config['stack_bb'],) * 2) for seed in seeds]
            in_iteration = True
            before = copy.deepcopy(rng.bit_generator.state)
            row = table.step(*hands, rng)
            trace = identity(row)
            after = copy.deepcopy(rng.bit_generator.state)
            audit_rng = np.random.default_rng()
            audit_rng.bit_generator.state = before
            step_draws = _draws(row)
            if not step_draws <= row['nodes'] <= 2 * config['abstraction']['max_nodes_per_traversal']:
                raise ValueError('Self-play sampling draw count exceeds its complete traversal')
            _advance_rng(audit_rng, step_draws)
            if audit_rng.bit_generator.state != after:
                raise ValueError('Locked explicit-probability choice RNG contract changed')
            row.update(deal_seeds=seeds, sampling_rng_before=before,
                       sampling_rng_after=after, iteration_trace_sha256=trace)
            journal.record(index, seeds, row)
            completed = index + 1
            in_iteration = False
            if time.monotonic() - last_checkpoint >= 300:
                checkpoint()
            if completed % config.get('progress_iterations', 100) == 0:
                print(json.dumps({'iterations': completed, 'planned': config['iterations'],
                                  'traversals': 2 * completed,
                                  'information_sets': sum(len(t.keys) for t in table.tables),
                                  'seconds': time.perf_counter() - started}), flush=True)
            if stopped or stop_after is not None and completed >= stop_after:
                break
        if len(journal.rows) != completed:
            raise ValueError('Trailing self-play journal records remain at this boundary')
        values = [entry['value'] for entry in journal.rows[:completed]]
        result = {'schema': RESULT_SCHEMA,
                  'status': 'trained-unvalidated' if completed == config['iterations'] else 'interrupted',
                  'mode': 'conventional-teacher-control', 'iterations_completed': completed,
                  'planned_iterations': config['iterations'], 'operations': 2 * completed,
                  'information_sets_by_role': [len(t.keys) for t in table.tables],
                  'information_sets': sum(len(t.keys) for t in table.tables),
                  'counterfactual_nodes': sum(row['nodes'] for row in values),
                  'terminal_branches': sum(row['terminal_branches'] for row in values),
                  'sampled_action_draws': sum(row['sampled_action_draws'] for row in values),
                  'learning_claim': False, 'allowed_as_teacher': False,
                  'scope': 'Conventional two-player abstract-game self-play; no finite-run equilibrium or fly-learning claim',
                  'manifest_sha256': digest(root / 'manifest.json'),
                  'journal_head': journal.rows[-1]['hash'] if completed else '0' * 64,
                  'elapsed_seconds_this_invocation': time.perf_counter() - started,
                  'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)}
        checkpoint()
        atomic_json(root / 'result.json', result)
        return result
    finally:
        try:
            if not checkpoint_failed and not in_iteration and table.iterations_completed == completed:
                checkpoint()
        finally:
            try:
                journal.close()
            finally:
                for sig, handler in handlers.items():
                    signal.signal(sig, handler)


def completed_table(run):
    root = Path(run)
    saved = json.loads((root / 'manifest.json').read_text())
    config = {key: value for key, value in saved['config'].items() if key not in _DERIVED}
    expected = runtime(config)
    _assert_runtime(saved, expected)
    result = json.loads((root / 'result.json').read_text())
    if (result['schema'] != RESULT_SCHEMA or result['status'] != 'trained-unvalidated'
            or result['iterations_completed'] != config['iterations']
            or result['planned_iterations'] != config['iterations']
            or result['operations'] != 2 * config['iterations'] or result['allowed_as_teacher'] is not False
            or result['manifest_sha256'] != digest(root / 'manifest.json')):
        raise ValueError('Complete fixed-final unvalidated self-play required before export')
    checked = verify_journal(root, config, count=config['iterations'])
    for key in ('iterations_completed', 'information_sets', 'information_sets_by_role',
                'counterfactual_nodes', 'terminal_branches', 'sampled_action_draws', 'journal_head'):
        if result[key] != checked[key]:
            raise ValueError('Self-play final result differs from complete journal: ' + key)
    arrays, extra, metadata = Checkpoints(root / 'checkpoints').load(expected)
    _assert_runtime(metadata['identity'], expected)
    if (metadata['step'] != config['iterations'] or extra['iterations_completed'] != config['iterations']
            or extra['journal_head'] != checked['journal_head'] or extra['sampling_rng'] != checked['sampling_rng']):
        raise ValueError('Self-play final checkpoint/RNG boundary mismatch')
    table = SelfPlayRegretTable(config['abstraction'])
    table.restore(arrays)
    if (table.iterations_completed != config['iterations']
            or [len(t.keys) for t in table.tables] != checked['information_sets_by_role']):
        raise ValueError('Self-play final numeric table differs from completed evidence')
    return table, {'training_manifest_sha256': digest(root / 'manifest.json'),
                   'training_result_sha256': digest(root / 'result.json'),
                   'training_journal_sha256': digest(root / 'iterations.jsonl'),
                   'training_source_sha256': saved['source_hash'], 'training_commit': saved['commit'],
                   'iterations_completed': config['iterations'], 'stack_bb': config['stack_bb'],
                   'aggregation': AGGREGATION, 'abstraction_sha256': saved['encoder_hash'],
                   'final_checkpoint_sha256': identity(metadata),
                   'final_checkpoint_arrays_sha256': {name + '.npy': item['sha256']
                                                     for name, item in metadata['arrays'].items()}}
