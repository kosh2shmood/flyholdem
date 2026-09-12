"""Numeric ring replay and uniform reservoir memory, with complete RNG state."""
import numpy as np


class Memory:
    def __init__(self, capacity, fields, seed, reservoir=False):
        if type(capacity) is not int or capacity < 1:
            raise ValueError('Positive memory capacity required')
        self.capacity = capacity
        self.arrays = {name: np.zeros((capacity, *shape), dtype=dtype) for name, (shape, dtype) in fields.items()}
        self.seen = 0
        self.size = 0
        self.reservoir = bool(reservoir)
        self.rng = np.random.default_rng(seed)

    def add(self, **values):
        if set(values) != set(self.arrays):
            raise ValueError('Incomplete replay transition')
        for name, target in self.arrays.items():
            value = np.asarray(values[name])
            if value.shape != target.shape[1:] or not np.isfinite(value).all():
                raise ValueError('Invalid replay shape or numeric value')
        index = (self.seen if self.seen < self.capacity else int(self.rng.integers(self.seen + 1))) if self.reservoir else self.seen % self.capacity
        if index < self.capacity:
            for name, value in values.items():
                self.arrays[name][index] = value
        self.seen += 1
        self.size = min(self.seen, self.capacity)

    def sample(self, count):
        if self.size < count or count < 1:
            raise ValueError('Insufficient replay samples')
        indices = self.rng.choice(self.size, count, replace=False)
        return {name: values[indices] for name, values in self.arrays.items()}

    def state(self):
        return ({name: values[:self.size].copy() for name, values in self.arrays.items()},
                {'capacity': self.capacity, 'size': self.size, 'seen': self.seen,
                 'reservoir': self.reservoir, 'rng': self.rng.bit_generator.state})

    def restore_state(self, arrays, metadata):
        if (metadata['capacity'] != self.capacity or metadata['reservoir'] != self.reservoir
                or set(arrays) != set(self.arrays) or type(metadata['size']) is not int
                or type(metadata['seen']) is not int or not 0 <= metadata['size'] <= self.capacity
                or metadata['size'] != min(metadata['seen'], self.capacity)):
            raise ValueError('Incompatible memory checkpoint')
        for name, target in self.arrays.items():
            value = arrays[name]
            if value.shape != (metadata['size'], *target.shape[1:]) or value.dtype != target.dtype or not np.isfinite(value).all():
                raise ValueError('Invalid replay checkpoint arrays')
        rng = np.random.default_rng(); rng.bit_generator.state = metadata['rng']
        for name, target in self.arrays.items():
            target.fill(0); target[:metadata['size']] = arrays[name]
        self.size, self.seen, self.rng = metadata['size'], metadata['seen'], rng
