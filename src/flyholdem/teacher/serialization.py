"""Safe numeric tensor-tree checkpoints; no pickle or executable objects."""
import math
import numpy as np
import torch


def pack(value, arrays, prefix='state'):
    if isinstance(value, torch.Tensor):
        name = f'{prefix}_{len(arrays)}'
        array = value.detach().cpu().numpy().copy()
        if array.dtype.kind not in 'biuf' or not np.isfinite(array).all():
            raise ValueError('Finite numeric CPU tensor required')
        arrays[name] = array
        return {'kind': 'tensor', 'array': name, 'shape': list(array.shape)}
    if isinstance(value, dict):
        if any(type(key) not in (str, int) for key in value):
            raise ValueError('Unsupported optimizer dictionary key')
        return {'kind': 'dict', 'items': [[key, pack(item, arrays, prefix)] for key, item in value.items()]}
    if isinstance(value, (list, tuple)):
        return {'kind': 'tuple' if isinstance(value, tuple) else 'list',
                'items': [pack(item, arrays, prefix) for item in value]}
    if value is None or type(value) in (str, bool, int) or type(value) is float and math.isfinite(value):
        return {'kind': 'value', 'value': value}
    raise ValueError('Unsupported checkpoint object')


def unpack(tree, arrays):
    kind = tree['kind']
    if kind == 'tensor':
        return torch.from_numpy(arrays[tree['array']].reshape(tree['shape']).copy())
    if kind == 'dict':
        return {key: unpack(item, arrays) for key, item in tree['items']}
    if kind in ('tuple', 'list'):
        values = [unpack(item, arrays) for item in tree['items']]
        return tuple(values) if kind == 'tuple' else values
    if kind == 'value':
        return tree['value']
    raise ValueError('Unknown checkpoint tree node')
