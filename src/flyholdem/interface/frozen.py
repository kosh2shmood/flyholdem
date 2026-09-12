"""Teacher-disconnected frozen synaptic models over the verified native graph."""
import json
from pathlib import Path
import numpy as np
from flyholdem.connectome.registry import digest
from flyholdem.inference_identity import frozen_runtime_identity, assert_frozen_runtime
from flyholdem.neural.checkpoint import atomic_json, sync_dir
from flyholdem.provenance import identity
from .population import load_controller


def verify_changed_edges(brain, edges, bounds):
    edges = np.asarray(edges)
    if (edges.ndim != 1 or edges.dtype.kind not in 'iu' or not len(edges)
            or len(np.unique(edges)) != len(edges) or np.any(edges < 0) or np.any(edges >= len(brain.weights))):
        raise ValueError('Unique existing edge indices required')
    lower, upper = bounds
    if not 0 < lower <= 1 <= upper:
        raise ValueError('Sign-preserving multiplicative bounds required')
    if np.any(brain.initial[edges] == 0):
        raise ValueError('Cannot learn a new zero-strength connection')
    ratios = brain.weights[edges].astype(np.float64) / brain.initial[edges]
    # Account only for float32 storage rounding at the declared endpoints.
    tolerance = np.finfo(np.float32).eps * max(upper, 1) * 2
    if not np.isfinite(ratios).all() or np.any(ratios < lower - tolerance) or np.any(ratios > upper + tolerance):
        raise ValueError('Frozen edge strengths exceed registered bounds')
    restored = brain.weights.copy(); restored[edges] = brain.initial[edges]
    if not np.array_equal(restored, brain.initial):
        raise ValueError('Weights outside the registered plastic subset changed')
    return edges.astype(np.int64)


def export_frozen(controller, edges, bounds, learning_mode, output, training_reference, validation_reference):
    if learning_mode not in ('bio-plastic', 'distilled-connectome'):
        raise ValueError('Explicit core learning mode required')
    brain = controller.brain
    indices = verify_changed_edges(brain, edges, bounds)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    for name, value in [('edge_indices', indices), ('edge_weights', brain.weights[indices])]:
        with (output / (name + '.npy')).open('wb') as stream:
            np.save(stream, value, allow_pickle=False)
            stream.flush(); __import__('os').fsync(stream.fileno())
    atomic_json(output / 'preregistration.json', controller.registration)
    entries = {name: digest(output / name) for name in ('edge_indices.npy', 'edge_weights.npy', 'preregistration.json')}
    record = {'schema': 'frozen-connectome-policy-v1', 'mode': controller.registration['mode'],
              'learning_mode': learning_mode, 'strategy_location': 'bounded-existing-connectome-edges',
              'teacher_connected': False, 'files': entries, 'weight_bounds': list(bounds),
              'graph_hash': brain.graph_hash, 'binary_sha256': brain.build['binary_sha256'],
              'runtime': frozen_runtime_identity(), 'start_state': 'fresh-neural-rest',
              'training_reference': training_reference, 'validation_reference': validation_reference,
              'learning_claim': False}
    atomic_json(output / 'manifest.json', record); sync_dir(output)
    return digest(output / 'manifest.json')


def load_frozen(output, graph_path, expected_sha256=None, seed=0):
    output = Path(output)
    if expected_sha256 is not None and digest(output / 'manifest.json') != expected_sha256:
        raise ValueError('Selected frozen model checksum mismatch')
    record = json.loads((output / 'manifest.json').read_text())
    if (record['schema'] != 'frozen-connectome-policy-v1' or record['teacher_connected'] is not False
            or record['learning_mode'] not in ('bio-plastic', 'distilled-connectome')
            or record['strategy_location'] != 'bounded-existing-connectome-edges'
            or record['start_state'] != 'fresh-neural-rest'):
        raise ValueError('Unsupported frozen connectome policy')
    if set(record['files']) != {'edge_indices.npy', 'edge_weights.npy', 'preregistration.json'}:
        raise ValueError('Unexpected frozen model files')
    assert_frozen_runtime(record['runtime'])
    for name, checksum in record['files'].items():
        if digest(output / name) != checksum:
            raise ValueError('Frozen policy file checksum mismatch')
    controller = load_controller(graph_path, output / 'preregistration.json', seed)
    brain = controller.brain
    if (brain.graph_hash != record['graph_hash'] or brain.build['binary_sha256'] != record['binary_sha256']
            or controller.registration['mode'] != record['mode']):
        raise ValueError('Frozen policy graph/native binary mismatch')
    edges = np.load(output / 'edge_indices.npy', allow_pickle=False)
    weights = np.load(output / 'edge_weights.npy', allow_pickle=False)
    if (edges.dtype != np.int64 or edges.ndim != 1 or weights.dtype != np.float32 or weights.shape != edges.shape
            or not np.isfinite(weights).all() or np.any(edges < 0) or np.any(edges >= len(brain.weights))):
        raise ValueError('Invalid frozen synaptic arrays')
    brain.weights[edges] = weights
    verify_changed_edges(brain, edges, record['weight_bounds'])
    controller.frozen_model = {'sha256': digest(output / 'manifest.json'), 'learning_mode': record['learning_mode'],
                               'teacher_connected': False, 'mode': record['mode']}
    return controller
