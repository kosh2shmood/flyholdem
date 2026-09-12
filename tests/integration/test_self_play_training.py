"""Tiny actual self-play journals, recovery and failure boundaries; no strength tests."""
import copy
import json
from pathlib import Path
import signal

import numpy as np
import pytest

from flyholdem.connectome.registry import digest
from flyholdem.neural.checkpoint import Checkpoints, atomic_json
from flyholdem.poker.engine import Hand
from flyholdem.provenance import canonical, identity
from flyholdem.teacher import self_play_regret as core
from flyholdem.teacher import self_play_training as training
from flyholdem.teacher.regret import VERSION


def config(iterations=3):
    return {'schema': core.SCHEMA, 'mode': 'conventional-teacher-control',
            'aggregation': core.AGGREGATION, 'iterations': iterations,
            'stack_bb': 2, 'sampling_seed': 913000, 'deal_seed_start': 991800000,
            'progress_iterations': 100,
            'abstraction': {'feature_version': VERSION, 'stack_bb': 2,
                            'equity_buckets': 8, 'equity_samples': 2,
                            'max_information_sets': 10000, 'max_nodes_per_traversal': 10000}}


def arrays_equal(first, second):
    assert first.keys() == second.keys()
    for name in first:
        np.testing.assert_array_equal(first[name], second[name], err_msg=name)


def files(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in Path(root).rglob('*') if path.is_file()}


def rows(root):
    return [json.loads(line) for line in (root / 'iterations.jsonl').read_text().splitlines()]


def rewrite_chain(root, entries):
    """Repair every nonsemantic hash, so tests exercise meaning as well as bytes."""
    previous = '0' * 64
    for index, entry in enumerate(entries):
        entry['value']['iteration_trace_sha256'] = identity(training._core_row(entry['value']))
        entry['index'], entry['previous'] = index, previous
        entry['hash'] = identity({key: value for key, value in entry.items() if key != 'hash'})
        previous = entry['hash']
    (root / 'iterations.jsonl').write_text(''.join(json.dumps(entry, sort_keys=True) + '\n' for entry in entries))


def latest(root):
    saved = json.loads((root / 'manifest.json').read_text())
    return Checkpoints(root / 'checkpoints').load(saved)


@pytest.fixture
def tracked_journals(monkeypatch):
    opened = []
    base = training.Journal

    class TrackedJournal(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            opened.append(self)

    monkeypatch.setattr(training, 'Journal', TrackedJournal)
    return opened


def handlers():
    return {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}


def test_actual_continuous_interrupted_and_completed_resume_are_exact(tmp_path):
    cfg = config(4)
    whole, interrupted = tmp_path / 'whole', tmp_path / 'interrupted'
    complete = training.train(cfg, whole)
    partial = training.train(cfg, interrupted, stop_after=2)
    assert partial['status'] == 'interrupted' and partial['operations'] == 4
    from flyholdem.teacher.self_play_policy import export_policy, load_policy
    with pytest.raises(ValueError, match='fixed-final'):
        export_policy(interrupted, tmp_path / 'too-early')
    assert not (tmp_path / 'too-early').exists()
    resumed = training.train(cfg, interrupted, resume=True)
    assert complete['status'] == resumed['status'] == 'trained-unvalidated'
    assert resumed['operations'] == 8 and resumed['allowed_as_teacher'] is False
    assert (whole / 'iterations.jsonl').read_bytes() == (interrupted / 'iterations.jsonl').read_bytes()
    a, a_origin = training.completed_table(whole)
    b, b_origin = training.completed_table(interrupted)
    arrays_equal(a.state(), b.state())
    assert a_origin['training_journal_sha256'] == b_origin['training_journal_sha256']
    before = files(interrupted)
    assert training.train(cfg, interrupted, resume=True) == resumed
    assert files(interrupted) == before

    # One complete real trainer -> immutable export -> loader path. Other
    # policy tests separately exercise tampering and teacher disconnection.
    checksum = export_policy(interrupted, tmp_path / 'policy')
    policy, record = load_policy(tmp_path / 'policy')
    assert checksum == digest(tmp_path / 'policy/manifest.json')
    assert record['allowed_as_teacher'] is False
    assert record['provenance']['iterations_completed'] == 4
    for seed in range(cfg['deal_seed_start'], cfg['deal_seed_start'] + 2):
        hand = Hand(seed, stacks=(4, 4))
        while not hand.done:
            observation = hand.observation()
            np.testing.assert_array_equal(policy.probabilities(observation), b.probabilities(observation))
            hand.act(1)


def test_older_checkpoint_reexecutes_only_exact_existing_journal_tail(tmp_path, monkeypatch):
    cfg, root = config(4), tmp_path / 'tail'
    training.train(cfg, root, stop_after=2)
    old_pointer = (root / 'checkpoints/latest.json').read_bytes()
    training.train(cfg, root, resume=True)
    expected, _ = training.completed_table(root)
    journal = (root / 'iterations.jsonl').read_bytes()
    (root / 'checkpoints/latest.json').write_bytes(old_pointer)
    calls = []
    original = core.SelfPlayRegretTable.step

    def recorded(self, first, second, rng):
        calls.append([first.seed, second.seed])
        return original(self, first, second, rng)

    monkeypatch.setattr(core.SelfPlayRegretTable, 'step', recorded)
    training.train(cfg, root, resume=True)
    assert calls == [[991800004, 991800005], [991800006, 991800007]]
    assert (root / 'iterations.jsonl').read_bytes() == journal
    actual, _ = training.completed_table(root)
    arrays_equal(actual.state(), expected.state())


@pytest.mark.parametrize('prior_result', ['missing', 'stale-interrupted'])
def test_final_checkpoint_recovers_result_without_another_pair(tmp_path, monkeypatch, prior_result):
    cfg, root = config(), tmp_path / 'final'
    training.train(cfg, root, stop_after=1)
    stale = (root / 'result.json').read_bytes()
    training.train(cfg, root, resume=True)
    expected, _ = training.completed_table(root)
    pointers = {name: (root / 'checkpoints' / name).read_bytes() for name in ('latest.json', 'best.json')}
    journal = (root / 'iterations.jsonl').read_bytes()
    if prior_result == 'missing':
        (root / 'result.json').unlink()
    else:
        (root / 'result.json').write_bytes(stale)

    def forbidden(*args, **kwargs):
        raise AssertionError('A complete final checkpoint must not run another pair')

    monkeypatch.setattr(core.SelfPlayRegretTable, 'step', forbidden)
    result = training.train(cfg, root, resume=True)
    assert result['status'] == 'trained-unvalidated' and result['iterations_completed'] == 3
    assert (root / 'iterations.jsonl').read_bytes() == journal
    assert {name: (root / 'checkpoints' / name).read_bytes() for name in pointers} == pointers
    restored, _ = training.completed_table(root)
    arrays_equal(restored.state(), expected.state())


@pytest.mark.parametrize('change', ['deal', 'rng', 'draws', 'role', 'node-tally', 'average-tally'])
def test_coherently_rehashed_journal_semantics_are_rejected(tmp_path, change):
    cfg, root = config(2), tmp_path / 'changed'
    training.train(cfg, root)
    entries = rows(root)
    row = entries[-1]['value']
    first = row['traversals'][0]
    if change == 'deal':
        row['deal_seeds'][1] += 100
        entries[-1]['label'] = row['deal_seeds']
        row['traversals'][1]['root_private_checkpoint'] = Hand(row['deal_seeds'][1], stacks=(4, 4)).serialize()
    elif change == 'rng':
        rng = np.random.default_rng()
        rng.bit_generator.state = row['sampling_rng_after']
        rng.random()
        row['sampling_rng_after'] = rng.bit_generator.state
    elif change == 'draws':
        # Repair total and after-RNG too; the per-pass distinct information-set
        # count must still expose the invented extra sampled draw.
        first['sampled_action_draws'] += 1
        row['sampled_action_draws'] += 1
        rng = np.random.default_rng()
        rng.bit_generator.state = row['sampling_rng_before']
        rng.random(row['sampled_action_draws'])
        row['sampling_rng_after'] = rng.bit_generator.state
    elif change == 'role':
        first['public_role'] = core.ROLE_NAMES[1]
    elif change == 'node-tally':
        first['nodes'] += 1
        row['nodes'] += 1
    else:
        first['average_information_sets'] = first['average_encounters'] + 1
        first['sampled_action_draws'] = first['average_information_sets']
        row['sampled_action_draws'] = sum(p['sampled_action_draws'] for p in row['traversals'])
    rewrite_chain(root, entries)
    with pytest.raises(ValueError, match='Self-play'):
        training.verify_journal(root, cfg, count=2)


def test_checkpoint_rng_with_repaired_manifest_checksum_is_rejected_before_work(tmp_path, monkeypatch, tracked_journals):
    cfg, root = config(), tmp_path / 'checkpoint'
    training.train(cfg, root, stop_after=1)
    pointer_path = root / 'checkpoints/latest.json'
    pointer = json.loads(pointer_path.read_text())
    metadata_path = root / 'checkpoints' / pointer['generation'] / 'manifest.json'
    metadata = json.loads(metadata_path.read_text())
    metadata['extra']['sampling_rng'] = np.random.default_rng(111).bit_generator.state
    atomic_json(metadata_path, metadata)
    pointer['manifest_sha256'] = digest(metadata_path)
    atomic_json(pointer_path, pointer)
    latest(root)  # Generic checksum validation succeeds; semantic binding must fail.
    before_handlers = handlers()
    monkeypatch.setattr(core.SelfPlayRegretTable, 'step', lambda *args: pytest.fail('checkpoint RNG must be checked first'))
    with pytest.raises(ValueError, match='checkpoint/journal/RNG'):
        training.train(cfg, root, resume=True)
    assert all(journal.stream.closed for journal in tracked_journals)
    assert handlers() == before_handlers


def test_coherently_changed_checkpoint_tail_fails_actual_reexecution(tmp_path, tracked_journals):
    cfg, root = config(), tmp_path / 'changed-tail'
    training.train(cfg, root, stop_after=1)
    pointer = (root / 'checkpoints/latest.json').read_bytes()
    training.train(cfg, root, resume=True)
    (root / 'checkpoints/latest.json').write_bytes(pointer)
    entries = rows(root)
    entries[1]['value']['traversals'][0]['complete_traversal_sha256'] = '0' * 64
    rewrite_chain(root, entries)
    with pytest.raises(ValueError, match='Checkpoint-tail re-execution differs'):
        training.train(cfg, root, resume=True)
    assert (root / 'checkpoints/latest.json').read_bytes() == pointer
    assert all(journal.stream.closed for journal in tracked_journals)


@pytest.mark.parametrize('boundary', [821001, 2000000, 3000000])
@pytest.mark.parametrize('offset', [0, -1])
def test_either_chance_deal_overlap_is_refused_before_output(tmp_path, boundary, offset):
    cfg = config(1)
    cfg['deal_seed_start'] = boundary + offset
    with pytest.raises(ValueError, match='Both self-play chance deals'):
        training.train(cfg, tmp_path / 'forbidden')
    assert not (tmp_path / 'forbidden').exists()


@pytest.mark.parametrize('extra', [{'opponents': ['random']}, {'opponent_probabilities': [.25] * 4}, {'mode': 'native'}])
def test_stationary_population_and_nonconventional_modes_are_refused(tmp_path, extra):
    cfg = {**config(), **extra}
    with pytest.raises(ValueError, match='fixed two-player'):
        training.train(cfg, tmp_path / 'forbidden')
    assert not (tmp_path / 'forbidden').exists()


def test_prefix_audit_still_consumes_full_digest_checked_stream(tmp_path, monkeypatch):
    cfg, root = config(), tmp_path / 'prefix'
    training.train(cfg, root)
    original = training.iter_journal

    def changed_while_consuming(path):
        for index, entry in enumerate(original(path)):
            if index == 1:
                data = Path(path).read_bytes()
                replacement = Path(path).with_suffix('.replacement')
                replacement.write_bytes(data.replace(b'\n', b' \n', 1))
                replacement.replace(path)  # Existing reader retains its valid old inode.
            yield entry

    monkeypatch.setattr(training, 'iter_journal', changed_while_consuming)
    with pytest.raises(ValueError, match='Journal changed'):
        training.verify_journal(root, cfg, count=1, through=1)


def test_resume_rejects_a_coherent_tail_beyond_registered_horizon(tmp_path, monkeypatch, tracked_journals):
    cfg, root = config(2), tmp_path / 'long-tail'
    training.train(cfg, root, stop_after=1)
    pointer = (root / 'checkpoints/latest.json').read_bytes()
    training.train(cfg, root, resume=True)
    (root / 'checkpoints/latest.json').write_bytes(pointer)
    entries = rows(root)
    extra = copy.deepcopy(entries[-1])
    row = extra['value']
    row['iteration'] = 3
    row['deal_seeds'] = [991800004, 991800005]
    extra['label'] = row['deal_seeds']
    for part, seed in zip(row['traversals'], row['deal_seeds']):
        part['root_private_checkpoint'] = Hand(seed, stacks=(4, 4)).serialize()
    row['sampling_rng_before'] = entries[-1]['value']['sampling_rng_after']
    rng = np.random.default_rng()
    rng.bit_generator.state = row['sampling_rng_before']
    rng.random(row['sampled_action_draws'])
    row['sampling_rng_after'] = rng.bit_generator.state
    entries.append(extra)
    rewrite_chain(root, entries)
    monkeypatch.setattr(core.SelfPlayRegretTable, 'step', lambda *args: pytest.fail('Reject trailing work before another pair'))
    with pytest.raises(ValueError, match='checkpoint/journal/RNG'):
        training.train(cfg, root, resume=True)
    assert all(journal.stream.closed for journal in tracked_journals)


def test_second_pass_failure_keeps_checkpoint_at_journal_boundary_and_cleans_up(tmp_path, monkeypatch, tracked_journals):
    cfg, root = config(1), tmp_path / 'pass-error'
    before_handlers = handlers()
    original = core.external_sampling_traversal

    def fail_second(tree, role, strategy, rng, **kwargs):
        if role == 1:
            rng.random()
            raise OSError('second-pass failure')
        return original(tree, role, strategy, rng, **kwargs)

    monkeypatch.setattr(core, 'external_sampling_traversal', fail_second)
    with pytest.raises(OSError, match='second-pass'):
        training.train(cfg, root)
    arrays, extra, metadata = latest(root)
    assert arrays['iterations_completed'].tolist() == [0]
    assert extra['iterations_completed'] == metadata['step'] == len(rows(root)) == 0
    assert not (root / 'result.json').exists()
    assert all(journal.stream.closed for journal in tracked_journals)
    assert handlers() == before_handlers
    monkeypatch.setattr(core, 'external_sampling_traversal', original)
    assert training.train(cfg, root, resume=True)['status'] == 'trained-unvalidated'


@pytest.mark.parametrize('failure', ['before-write', 'after-complete-write', 'after-partial-write'])
def test_append_failures_never_publish_ahead_of_the_durable_journal(tmp_path, monkeypatch, tracked_journals, failure):
    cfg, root = config(1), tmp_path / 'append-error'
    before_handlers = handlers()
    base = training.Journal

    class FailingJournal(base):
        def record(self, index, label, value):
            if failure == 'after-complete-write':
                body = {'index': index, 'previous': '0' * 64, 'label': label, 'value': value}
                self.stream.write(json.dumps({**body, 'hash': identity(body)}, sort_keys=True) + '\n')
                self.stream.flush()
            elif failure == 'after-partial-write':
                self.stream.write('{"index":')
                self.stream.flush()
            raise OSError('injected append failure')

    monkeypatch.setattr(training, 'Journal', FailingJournal)
    with pytest.raises(OSError, match='append failure'):
        training.train(cfg, root)
    arrays, extra, metadata = latest(root)
    assert arrays['iterations_completed'].tolist() == [0]
    assert extra['iterations_completed'] == metadata['step'] == 0
    assert not (root / 'result.json').exists()
    assert all(journal.stream.closed for journal in tracked_journals)
    assert handlers() == before_handlers
    monkeypatch.setattr(training, 'Journal', base)
    if failure == 'after-partial-write':
        broken = (root / 'iterations.jsonl').read_bytes()
        with pytest.raises(ValueError, match='Truncated journal'):
            training.train(cfg, root, resume=True)
        assert (root / 'iterations.jsonl').read_bytes() == broken
    else:
        assert training.train(cfg, root, resume=True)['status'] == 'trained-unvalidated'


@pytest.mark.parametrize('failed_step', [0, 1])
def test_initial_or_final_checkpoint_io_failure_closes_streams_restores_handlers_and_is_not_retried(tmp_path, monkeypatch, tracked_journals, failed_step):
    cfg, root = config(1), tmp_path / 'checkpoint-error'
    before_handlers = handlers()
    original = training.Checkpoints.save
    calls = []

    def fail(self, arrays, manifest, extra, step, best=False):
        calls.append(step)
        if step == failed_step:
            raise OSError('injected checkpoint write failure')
        return original(self, arrays, manifest, extra, step, best=best)

    monkeypatch.setattr(training.Checkpoints, 'save', fail)
    with pytest.raises(OSError, match='checkpoint write failure'):
        training.train(cfg, root)
    assert calls.count(failed_step) == 1  # Cleanup must not retry the failed save.
    assert all(journal.stream.closed for journal in tracked_journals)
    assert handlers() == before_handlers
    assert not (root / 'result.json').exists()
    if failed_step == 0:
        assert not (root / 'checkpoints/latest.json').exists()
        assert len(rows(root)) == 0
    else:
        arrays, extra, metadata = latest(root)
        assert arrays['iterations_completed'].tolist() == [0]
        assert metadata['step'] == extra['iterations_completed'] == 0 < len(rows(root)) == 1
        monkeypatch.setattr(training.Checkpoints, 'save', original)
        assert training.train(cfg, root, resume=True)['status'] == 'trained-unvalidated'


@pytest.mark.parametrize('field', ['self_play_method', 'equity_backend', 'evaluation_protocol_sha256'])
@pytest.mark.parametrize('location', ['run-manifest', 'checkpoint'])
@pytest.mark.parametrize('repair_config_hash', [False, True], ids=['outer-hash-only', 'coherent-config-hash'])
def test_derived_runtime_fields_cannot_change_under_repaired_artifact_hashes(
        tmp_path, monkeypatch, tracked_journals, field, location, repair_config_hash):
    cfg, root = config(1), tmp_path / 'changed-runtime'
    training.train(cfg, root)
    from flyholdem.teacher.self_play_policy import export_policy
    manifest_path = root / 'manifest.json'
    pointer_path = root / 'checkpoints/latest.json'
    pointer = json.loads(pointer_path.read_text())
    checkpoint_path = root / 'checkpoints' / pointer['generation'] / 'manifest.json'
    path = manifest_path if location == 'run-manifest' else checkpoint_path
    artifact = json.loads(path.read_text())
    runtime = artifact if location == 'run-manifest' else artifact['identity']
    original_hash = runtime['config_hash']
    if field == 'self_play_method':
        runtime['config'][field]['strategy_updates'] = 'sequential-between-passes'
    elif field == 'equity_backend':
        runtime['config'][field]['binary_sha256'] = '0' * 64
    else:
        runtime['config'][field] = '0' * 64
    actual_hash = identity(runtime['config'])
    assert actual_hash != original_hash
    if repair_config_hash:
        runtime['config_hash'] = actual_hash
    atomic_json(path, artifact)

    # Repair the containing artifact's checksums. Rejection must come from the
    # declared method/backend/evaluation definition, not a damaged outer file.
    if location == 'run-manifest':
        result_path = root / 'result.json'
        result = json.loads(result_path.read_text())
        result['manifest_sha256'] = digest(manifest_path)
        atomic_json(result_path, result)
    else:
        for name in ('latest.json', 'best.json'):
            binding_path = root / 'checkpoints' / name
            binding = json.loads(binding_path.read_text())
            assert binding['generation'] == pointer['generation']
            binding['manifest_sha256'] = digest(checkpoint_path)
            atomic_json(binding_path, binding)
    if not repair_config_hash:
        # The shared compatibility helper compares stored hash strings only.
        # Both the arrays and every outer pointer remain checksum-valid.
        latest(root)

    def forbidden(*args, **kwargs):
        pytest.fail('Changed runtime evidence reached another self-play pair')

    monkeypatch.setattr(core.SelfPlayRegretTable, 'step', forbidden)
    before, previous_handlers = files(root), handlers()
    with pytest.raises(ValueError, match='[Cc]onfig|[Rr]untime'):
        training.train(cfg, root, resume=True)
    assert files(root) == before
    assert handlers() == previous_handlers
    assert all(journal.stream.closed for journal in tracked_journals)
    output = tmp_path / 'forbidden-policy'
    with pytest.raises(ValueError, match='[Cc]onfig|[Rr]untime'):
        export_policy(root, output)
    assert not output.exists()
    assert files(root) == before
