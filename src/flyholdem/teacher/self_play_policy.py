"""Frozen numeric self-play averages; no trainer, teacher or fly at inference."""
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from flyholdem import provenance as provenance_runtime
from flyholdem.connectome.registry import digest
from flyholdem.neural.checkpoint import atomic_json, sync_dir
from flyholdem.provenance import canonical, identity
from .regret import RegretTable, VERSION
from .regret_policy import FrozenRegretPolicy, implementation as shared_implementation

SCHEMA = 'teacher-self-play-regret-policy-v1'
AGGREGATION = 'opponent-node-sampled-average-synchronous-profile-v1'
PRIOR = 'uniform-over-legal-actions; includes untrained stack domains'
ROLES = {'role0': {'position': 0, 'label': 'nonbutton-big-blind'},
         'role1': {'position': 1, 'label': 'button-small-blind'}}
CHECKPOINT_ARRAYS = ('key_offsets', 'key_bytes', 'legal', 'averages')


def implementation():
    return {'self_play_policy_sha256': digest(__file__),
            'canonical_runtime_sha256': digest(provenance_runtime.__file__),
            'shared_regret_runtime': shared_implementation()}


def probabilities_from_averages(averages, legal):
    averages, legal = np.asarray(averages), np.asarray(legal)
    if (averages.dtype != np.float64 or averages.ndim != 2 or averages.shape[1] != 5
            or legal.dtype != np.bool_ or legal.shape != averages.shape
            or not legal.any(axis=1).all() or not np.isfinite(averages).all()
            or np.any(averages < 0) or np.any(averages[~legal])):
        raise ValueError('Finite nonnegative legal self-play average masses required')
    with np.errstate(over='ignore'):
        total = averages.sum(axis=1, keepdims=True)
    if not np.isfinite(total).all():
        raise ValueError('Nonfinite self-play average total')
    return np.divide(averages, total, out=legal.astype(float) / legal.sum(axis=1, keepdims=True),
                     where=total > 0)


def _checkpoint_names():
    return {f'role{role}_{name}.npy' for role in (0, 1) for name in CHECKPOINT_ARRAYS}


def _origin(config, provenance):
    RegretTable(config)
    hash_fields = ('training_manifest_sha256', 'training_result_sha256',
                   'training_journal_sha256', 'training_source_sha256',
                   'abstraction_sha256', 'final_checkpoint_sha256')
    def valid_hash(value):
        return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)
    if (any(not valid_hash(provenance.get(name)) for name in hash_fields)
            or not isinstance(provenance.get('training_commit'), str) or not provenance['training_commit']):
        raise ValueError('Complete self-play training/checkpoint provenance required')
    if (provenance['aggregation'] != AGGREGATION
            or provenance['stack_bb'] != config['stack_bb']
            or provenance['abstraction_sha256'] != identity(config)
            or type(provenance['iterations_completed']) is not int
            or provenance['iterations_completed'] < 1):
        raise ValueError('Self-play average training/configuration binding mismatch')
    hashes = provenance.get('final_checkpoint_arrays_sha256')
    if (not isinstance(hashes, dict) or not _checkpoint_names() <= set(hashes)
            or any(not valid_hash(hashes[name]) for name in _checkpoint_names())):
        raise ValueError('Complete self-play checkpoint array bindings required')
    return hashes


def _role_policy(config, role, arrays, count):
    offsets, data, legal, averages, probabilities = [arrays[name] for name in
        ('key_offsets', 'key_bytes', 'legal', 'averages', 'probabilities')]
    if (type(count) is not int or count < 1 or offsets.dtype != np.int64
            or offsets.shape != (count + 1,) or offsets[0] != 0
            or np.any(np.diff(offsets) <= 0) or data.dtype != np.uint8 or data.ndim != 1
            or offsets[-1] != len(data) or legal.shape != (count, 5)
            or averages.shape != (count, 5) or probabilities.dtype != np.float64
            or probabilities.shape != (count, 5)
            or not np.array_equal(probabilities, probabilities_from_averages(averages, legal))):
        raise ValueError('Self-play probabilities must match the complete numeric average state')
    keys = [data[a:b].tobytes() for a, b in zip(offsets[:-1], offsets[1:])]
    if len(set(keys)) != count:
        raise ValueError('Duplicate self-play information key')
    for key, mask in zip(keys, legal):
        try:
            value = json.loads(key)
            valid = (canonical(value) == key and isinstance(value, list) and len(value) == 6
                     and value[0] == VERSION and value[1] == 4 * config['stack_bb']
                     and isinstance(value[3], list) and len(value[3]) == 9
                     and type(value[3][0]) is int and value[3][0] == role
                     and value[-1] == mask.tolist())
        except (ValueError, TypeError, IndexError, KeyError):
            valid = False
        if not valid:
            raise ValueError('Self-play information key role, stack, version or legality changed')
    return FrozenRegretPolicy(config, keys, probabilities, legal)


class FrozenSelfPlayPolicy:
    def __init__(self, config, policies):
        self.config = dict(config)
        self.policies = tuple(policies)

    def probabilities(self, observation):
        role = observation.get('position')
        if type(role) is not int or role not in (0, 1):
            raise ValueError('The acting player public position must select a self-play role')
        return self.policies[role].probabilities(observation)


def export_policy(run, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Write a new frozen self-play policy directory')
    # Loading the immutable policy never imports this training dependency.
    from .self_play_training import completed_table
    table, provenance = completed_table(run)
    checkpoint_hashes = _origin(table.config, provenance)
    if table.iterations_completed != provenance['iterations_completed']:
        raise ValueError('Self-play completed iteration binding mismatch')
    state = table.state()
    arrays = {f'role{role}_{name}': state[f'role{role}_{name}']
              for role in (0, 1) for name in CHECKPOINT_ARRAYS}
    counts = []
    for role in (0, 1):
        prefix = f'role{role}_'
        arrays[prefix + 'probabilities'] = probabilities_from_averages(
            arrays[prefix + 'averages'], arrays[prefix + 'legal'])
        count = len(arrays[prefix + 'key_offsets']) - 1
        _role_policy(table.config, role,
                     {name: arrays[prefix + name] for name in (*CHECKPOINT_ARRAYS, 'probabilities')}, count)
        counts.append(count)
    if sum(counts) > table.config['max_information_sets']:
        raise ValueError('Self-play policy exceeds the registered combined information-set capacity')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=output.name + '-partial-', dir=output.parent) as temporary:
        root = Path(temporary) / 'model'
        root.mkdir()
        files = {}
        for name, value in arrays.items():
            path = root / (name + '.npy')
            with path.open('wb') as stream:
                np.save(stream, value, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            files[path.name] = digest(path)
        if any(files[name] != checkpoint_hashes[name] for name in _checkpoint_names()):
            raise ValueError('Export must preserve exact final self-play checkpoint array bytes')
        runtime = implementation()
        record = {'schema': SCHEMA, 'mode': 'conventional-teacher-control',
                  'aggregation': AGGREGATION, 'feature_version': VERSION,
                  'config': table.config, 'roles': ROLES, 'information_sets': sum(counts),
                  'information_sets_by_role': counts, 'files': files, 'provenance': provenance,
                  'implementation': runtime,
                  'inference_backend': identity(runtime['shared_regret_runtime']['equity_backend']),
                  'allowed_as_teacher': False, 'unseen_state_prior': PRIOR,
                  'scope': 'Fixed-final sampled self-play average; no finite-run equilibrium or fly-strength claim'}
        atomic_json(root / 'manifest.json', record)
        sync_dir(root)
        root.rename(output)
        sync_dir(output.parent)
    return digest(output / 'manifest.json')


def load_policy(root):
    root = Path(root)
    record = json.loads((root / 'manifest.json').read_text())
    runtime = implementation()
    if (record['schema'] != SCHEMA or record['mode'] != 'conventional-teacher-control'
            or record['aggregation'] != AGGREGATION or record['feature_version'] != VERSION
            or record['roles'] != ROLES or record['unseen_state_prior'] != PRIOR
            or record['allowed_as_teacher'] is not False or record['implementation'] != runtime
            or record['inference_backend'] != identity(runtime['shared_regret_runtime']['equity_backend'])):
        raise ValueError('Self-play source/runtime, roles or policy definition mismatch')
    hashes = _origin(record['config'], record['provenance'])
    names = _checkpoint_names() | {'role0_probabilities.npy', 'role1_probabilities.npy'}
    if set(record['files']) != names or any(digest(root / name) != record['files'][name] for name in names):
        raise ValueError('Self-play numeric files changed')
    if any(record['files'][name] != hashes[name] for name in _checkpoint_names()):
        raise ValueError('Self-play checkpoint numeric binding changed')
    counts = record['information_sets_by_role']
    if (not isinstance(counts, list) or len(counts) != 2
            or any(type(value) is not int or value < 1 for value in counts)
            or type(record['information_sets']) is not int or sum(counts) != record['information_sets']
            or sum(counts) > record['config']['max_information_sets']):
        raise ValueError('Invalid self-play role information-set counts')
    policies = []
    for role, count in enumerate(counts):
        arrays = {name: np.load(root / f'role{role}_{name}.npy', allow_pickle=False)
                  for name in (*CHECKPOINT_ARRAYS, 'probabilities')}
        policies.append(_role_policy(record['config'], role, arrays, count))
    return FrozenSelfPlayPolicy(record['config'], policies), record
