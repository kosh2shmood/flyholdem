"""Deterministic, explicitly labeled controls for gated poker experiments."""
import copy
import hashlib
import numpy as np
from flyholdem.neural.sparse import SparseBrain
from flyholdem.interface.population import NeuralController


def shuffled_connectome(controller,seed):
    """Permute postsynaptic endpoints; retain each source/sign/weight and degrees.

    This is a synthetic null graph, not a retained MaleCNS connectivity result.
    Its separate graph hash prevents export as the original registered model.
    """
    if type(seed) is not int:raise ValueError('A registered integer topology-control seed is required')
    original=controller.brain
    post=np.random.default_rng(seed).permutation(original.post)
    if np.array_equal(post,original.post):raise ValueError('This seed did not produce a distinct topology control')
    brain=SparseBrain(original.ptr,post,original.initial)
    brain.weights[:]=original.weights
    registration=copy.deepcopy(controller.registration)
    registration['graph_hash']=brain.graph_hash
    registration['control']={'kind':'shuffled-connectome','method':'global-postsynaptic-endpoint-permutation-v1',
        'seed':seed,'original_graph_hash':original.graph_hash,
        'endpoints_sha256':hashlib.sha256(post.tobytes()).hexdigest(),
        'starting_weights_sha256':hashlib.sha256(original.weights.tobytes()).hexdigest(),
        'scope':'Synthetic rewired null graph; preserves source degree, destination degree, signs and original edge weights'}
    if (not np.array_equal(original.ptr,brain.ptr) or not np.array_equal(original.initial,brain.initial)
            or not np.array_equal(np.bincount(original.post,minlength=original.n),np.bincount(post,minlength=original.n))):
        raise AssertionError('Topology control failed its registered invariants')
    return NeuralController(brain,registration,seed)


def shuffled_rewards(returns,positions,seed):
    """Permute completed plastic-arm rewards independently within each position."""
    values=np.asarray(returns,dtype=float);positions=np.asarray(positions)
    if (values.ndim!=1 or values.shape!=positions.shape or not np.isfinite(values).all()
            or positions.dtype.kind not in 'iu' or np.any((positions!=0)&(positions!=1)) or type(seed) is not int):
        raise ValueError('Finite matched rewards, binary positions and a registered seed required')
    shuffled=values.copy();rng=np.random.default_rng(seed)
    for position in (0,1):
        mask=positions==position;shuffled[mask]=rng.permutation(values[mask])
    return shuffled
