"""Numeric self-play average exports, roles and original-checkpoint bindings."""
import builtins
import copy
import json
import sys
import types

import numpy as np
import pytest

from flyholdem.connectome.registry import digest
from flyholdem.neural.checkpoint import Checkpoints, atomic_json
from flyholdem.poker.engine import Hand
from flyholdem.provenance import identity, manifest
from flyholdem.teacher.regret import VERSION, abstraction
from flyholdem.teacher.self_play_regret import SelfPlayRegretTable
from flyholdem.teacher.self_play_policy import (
    AGGREGATION, CHECKPOINT_ARRAYS, SCHEMA, export_policy, load_policy,
    probabilities_from_averages,
)


@pytest.fixture
def numeric_core(tmp_path, monkeypatch):
    config = {'feature_version': VERSION, 'stack_bb': 20, 'equity_buckets': 8,
              'equity_samples': 16, 'max_information_sets': 20000,
              'max_nodes_per_traversal': 100000}
    table = SelfPlayRegretTable(config)
    rng = np.random.default_rng(991941100)
    # Repeat two fixed engineering deals to revisit keys and produce actual
    # nonuniform learned average masses for serialization/rebinding checks.
    for _ in range(3):
        table.step(Hand(991940000), Hand(991940001), rng)
    root = tmp_path / 'numeric-checkpoint'
    root.mkdir()
    recorded = manifest({'scope': 'small serialization fixture', 'abstraction': config},
                        'conventional-no-connectome', identity(config), identity(AGGREGATION))
    atomic_json(root / 'manifest.json', recorded)
    atomic_json(root / 'result.json', {'scope': 'small serialization fixture, not qualification'})
    (root / 'iterations.jsonl').write_bytes(b'')
    checkpoints = Checkpoints(root / 'checkpoints')
    checkpoints.save(table.state(), recorded, {}, table.iterations_completed)
    state, _, checkpoint = checkpoints.load(recorded)
    assert all(np.array_equal(state[name], value) for name, value in table.state().items())
    provenance = {'training_manifest_sha256': digest(root / 'manifest.json'),
                  'training_result_sha256': digest(root / 'result.json'),
                  'training_journal_sha256': digest(root / 'iterations.jsonl'),
                  'training_source_sha256': recorded['source_hash'],
                  'training_commit': recorded['commit'],
                  'iterations_completed': table.iterations_completed,
                  'stack_bb': config['stack_bb'], 'aggregation': AGGREGATION,
                  'abstraction_sha256': recorded['encoder_hash'],
                  'final_checkpoint_sha256': identity(checkpoint),
                  'final_checkpoint_arrays_sha256': {
                      name + '.npy': entry['sha256'] for name, entry in checkpoint['arrays'].items()}}
    # Only the completed-training guard is substituted to exercise serialization
    # of an actual small core/checkpoint. No experiment or qualification is forged.
    training = types.ModuleType('flyholdem.teacher.self_play_training')
    training.completed_table = lambda run: (table, copy.deepcopy(provenance))
    monkeypatch.setitem(sys.modules, training.__name__, training)
    return table, root, provenance


def test_average_normalization_is_legal_and_uniform_only_without_mass():
    averages = np.array([[0, 2, 3, 5, 0], [0, 0, 0, 0, 0]], dtype=np.float64)
    legal = np.array([[True] * 5, [False, True, False, True, False]])
    probabilities = probabilities_from_averages(averages, legal)
    np.testing.assert_array_equal(probabilities, [[0, .2, .3, .5, 0], [0, .5, 0, .5, 0]])
    for bad in (averages.astype(np.float32), np.full_like(averages, -1),
                np.full_like(averages, np.nan), np.full_like(averages, np.finfo(float).max)):
        with pytest.raises(ValueError):
            probabilities_from_averages(bad, np.ones_like(legal))
    bad = averages.copy()
    bad[1, 0] = 1
    with pytest.raises(ValueError):
        probabilities_from_averages(bad, legal)


def test_real_core_average_roundtrip_role_remapping_and_self_contained_inference(numeric_core, tmp_path, monkeypatch):
    table, root, provenance = numeric_core
    output = tmp_path / 'policy'
    policy_sha = export_policy(root, output)
    policy, record = load_policy(output)
    assert policy_sha == digest(output / 'manifest.json')
    assert record['schema'] == SCHEMA and not record['allowed_as_teacher']
    assert record['information_sets_by_role'] == [len(role.keys) for role in table.tables]
    for role, original in enumerate(table.tables):
        expected = probabilities_from_averages(original.averages[:len(original.keys)],
                                               original.legal[:len(original.keys)])
        np.testing.assert_array_equal(policy.policies[role].values, expected)
        for name in CHECKPOINT_ARRAYS:
            file = f'role{role}_{name}.npy'
            assert record['files'][file] == provenance['final_checkpoint_arrays_sha256'][file]
    observations = []
    seen_roles = set()
    for seed in range(991940000, 991940006):
        traces = []
        for button in (0, 1):
            game = Hand(seed, button=button)
            trace = []
            while not game.done:
                observation = game.observation()
                expected = table.probabilities(observation)
                actual = policy.probabilities(observation)
                np.testing.assert_array_equal(actual, expected)
                key, _ = abstraction(observation, table.config)
                if key in table.tables[observation['position']].index:
                    seen_roles.add(observation['position'])
                observations.append((observation, expected.tobytes()))
                trace.append(actual.tobytes())
                actual[:] = 0  # Callers receive a copy, never mutate the frozen rows.
                assert policy.probabilities(observation).tobytes() == expected.tobytes()
                game.act(1)
            traces.append(trace)
        assert traces[0] == traces[1]
    assert seen_roles == {0, 1}
    root.rename(tmp_path / 'training-offline')
    sys.modules.pop('flyholdem.teacher.self_play_training')
    original_import = builtins.__import__
    def no_training(name, *args, **kwargs):
        if name == 'torch' or name.startswith('torch.') or 'self_play_training' in name or 'self_play_regret' in name:
            raise AssertionError('Frozen inference must not import a trainer or Torch')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', no_training)
    frozen, _ = load_policy(output)
    assert all(frozen.probabilities(obs).tobytes() == expected for obs, expected in observations)
    other = Hand(991950000, stacks=(200, 200))
    for _ in range(2):
        observation = other.observation()
        np.testing.assert_array_equal(frozen.probabilities(observation),
            np.array(observation['legal_mask'], dtype=float) / sum(observation['legal_mask']))
        other.act(1)
    with pytest.raises(FileExistsError):
        export_policy(root, output)


def _rewrite_keys(output, record, role, keys, bind=False):
    values = {'key_offsets': np.concatenate((np.zeros(1, dtype=np.int64),
        np.cumsum(np.array([len(key) for key in keys], dtype=np.int64)))),
        'key_bytes': np.frombuffer(b''.join(keys), dtype=np.uint8).copy()}
    for name, value in values.items():
        file = output / f'role{role}_{name}.npy'
        np.save(file, value, allow_pickle=False)
        record['files'][file.name] = digest(file)
        if bind:
            record['provenance']['final_checkpoint_arrays_sha256'][file.name] = digest(file)
    atomic_json(output / 'manifest.json', record)


def test_rehashed_key_rebinding_cannot_substitute_another_average(numeric_core, tmp_path):
    table, root, _ = numeric_core
    output = tmp_path / 'policy'
    export_policy(root, output)
    policy, record = load_policy(output)
    role, i, j = next((role, i, j) for role in (0, 1)
        for i in range(len(table.tables[role].keys)) for j in range(i + 1, len(table.tables[role].keys))
        if np.array_equal(policy.policies[role].legal[i], policy.policies[role].legal[j])
        and not np.array_equal(policy.policies[role].values[i], policy.policies[role].values[j]))
    keys = list(table.tables[role].keys)
    keys[i], keys[j] = keys[j], keys[i]
    _rewrite_keys(output, record, role, keys)
    with pytest.raises(ValueError, match='checkpoint numeric binding'):
        load_policy(output)


def test_wrong_public_role_is_rejected_even_after_rehashing_numeric_bindings(numeric_core, tmp_path):
    table, root, _ = numeric_core
    output = tmp_path / 'policy'
    export_policy(root, output)
    _, record = load_policy(output)
    keys = list(table.tables[0].keys)
    from flyholdem.provenance import canonical
    decoded = json.loads(keys[0])
    decoded[3][0] = 1
    keys[0] = canonical(decoded)
    _rewrite_keys(output, record, 0, keys, bind=True)
    with pytest.raises(ValueError, match='key role'):
        load_policy(output)


def test_probabilities_must_be_rederived_from_original_average_masses(numeric_core, tmp_path):
    _, root, _ = numeric_core
    output = tmp_path / 'policy'
    export_policy(root, output)
    policy, record = load_policy(output)
    role = 1
    probabilities = policy.policies[role].values.copy()
    legal = policy.policies[role].legal
    index = next(i for i, mask in enumerate(legal) if mask.sum() > 1)
    actions = np.flatnonzero(legal[index])
    chosen = next(action for action in actions if probabilities[index, action] != 1)
    probabilities[index] = 0
    probabilities[index, chosen] = 1
    file = output / f'role{role}_probabilities.npy'
    np.save(file, probabilities, allow_pickle=False)
    record['files'][file.name] = digest(file)
    atomic_json(output / 'manifest.json', record)
    with pytest.raises(ValueError, match='complete numeric average'):
        load_policy(output)


def test_original_average_bytes_and_complete_bindings_are_required(numeric_core, tmp_path):
    table, root, _ = numeric_core
    output = tmp_path / 'policy'
    export_policy(root, output)
    _, original = load_policy(output)
    for role in (0, 1):
        for name in CHECKPOINT_ARRAYS:
            record = copy.deepcopy(original)
            record['provenance']['final_checkpoint_arrays_sha256'].pop(f'role{role}_{name}.npy')
            atomic_json(output / 'manifest.json', record)
            with pytest.raises(ValueError, match='Complete self-play checkpoint'):
                load_policy(output)
    record = copy.deepcopy(original)
    record['provenance'].pop('final_checkpoint_sha256')
    atomic_json(output / 'manifest.json', record)
    with pytest.raises(ValueError, match='Complete self-play training/checkpoint provenance'):
        load_policy(output)
    table.tables[0].averages[0] += table.tables[0].legal[0]
    with pytest.raises(ValueError, match='exact final self-play checkpoint'):
        export_policy(root, tmp_path / 'changed-average')
    assert not (tmp_path / 'changed-average').exists()


def test_source_configuration_and_role_metadata_are_bound(numeric_core, tmp_path):
    _, root, _ = numeric_core
    output = tmp_path / 'policy'
    export_policy(root, output)
    _, original = load_policy(output)
    for change in ('source', 'config', 'roles', 'mode', 'backend'):
        record = copy.deepcopy(original)
        if change == 'source':
            record['implementation']['self_play_policy_sha256'] = '0' * 64
        elif change == 'config':
            record['config']['equity_buckets'] = 7
        elif change == 'roles':
            record['roles']['role0']['position'] = 1
        elif change == 'mode':
            record['allowed_as_teacher'] = True
        else:
            record['inference_backend'] = 'pytorch-cpu'
        atomic_json(output / 'manifest.json', record)
        with pytest.raises(ValueError, match='binding|source/runtime'):
            load_policy(output)
