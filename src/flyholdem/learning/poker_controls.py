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



def shuffled_encoder(controller,seed):
    """Permute the fixed channel-to-cell rows; retain all gains and readouts."""
    if type(seed) is not int:raise ValueError('A registered encoder-control seed is required')
    registration=copy.deepcopy(controller.registration)
    original=np.asarray(registration['projection_indices'],dtype=np.int32)
    permutation=np.random.default_rng(seed).permutation(len(original));changed=original[permutation]
    if np.array_equal(changed,original):raise ValueError('This seed did not produce a distinct channel control')
    registration['projection_indices']=changed.tolist()
    registration['control']={'kind':'shuffled-encoder','seed':seed,'method':'fixed-channel-row-permutation-v1',
        'original_projection_sha256':hashlib.sha256(original.tobytes()).hexdigest(),'permutation':permutation.tolist(),
        'scope':'Channel assignment null control; unchanged neural scores, readouts and connectivity'}
    return NeuralController(controller.brain,registration,seed)


def export_control_graph(controller,source_graph,output):
    """Persist the synthetic topology for honest standalone frozen evaluation.

    Initial weights already contain the declared global scale. The exported
    control therefore uses scale one and never presents its wiring as MaleCNS.
    """
    import json,shutil,tempfile
    from pathlib import Path
    from flyholdem.connectome.prepare import load_graph
    from flyholdem.connectome.registry import digest
    from flyholdem.interface.population import scaled_weights
    from flyholdem.neural.checkpoint import atomic_json,sync_dir
    control=controller.registration.get('control',{})
    if control.get('kind')!='shuffled-connectome':raise ValueError('Only an explicit synthetic topology control may be exported here')
    source_graph=Path(source_graph);output=Path(output);original,source=load_graph(source_graph)
    weights=scaled_weights(original['weight'],controller.registration['config'].get('global_weight_scale',1))
    h=hashlib.sha256()
    for value in (original['ptr'],original['post'],weights):h.update(memoryview(value))
    if h.hexdigest()!=control['original_graph_hash'] or not np.array_equal(weights,controller.brain.initial):
        raise ValueError('Synthetic control does not match its original graph and global scale')
    registration=copy.deepcopy(controller.registration)
    registration['mode']=controller.registration['mode']+'-shuffled-control'
    registration['base_graph_hash']=controller.brain.graph_hash;registration['config']['global_weight_scale']=1
    arrays={**original,'ptr':controller.brain.ptr,'post':controller.brain.post,'weight':controller.brain.initial}
    expected={'schema':'synthetic-connectome-control-v1','mode':registration['mode'],'graph_hash':controller.brain.graph_hash,
        'source_graph_manifest_sha256':digest(source_graph/'manifest.json'),'control':control,
        'retained_malecns_graph':False,'global_scale_already_applied':True}
    if output.exists():
        _,record=load_graph(output)
        if any(record.get(key)!=value for key,value in expected.items()):raise ValueError('Existing synthetic graph identity mismatch')
    else:
        output.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=output.name+'-partial-',dir=output.parent) as temporary:
            target=Path(temporary);files={}
            for name,array in arrays.items():
                path=target/(name+'.npy')
                with path.open('wb') as stream:np.save(stream,array,allow_pickle=False);stream.flush();__import__('os').fsync(stream.fileno())
                files[path.name]=digest(path)
            if (source_graph/'neurons.feather').is_file():
                shutil.copy2(source_graph/'neurons.feather',target/'neurons.feather');files['neurons.feather']=digest(target/'neurons.feather')
            atomic_json(target/'manifest.json',{**expected,'files':files});sync_dir(target);target.rename(output);sync_dir(output.parent)
    return NeuralController(controller.brain,registration,0)
