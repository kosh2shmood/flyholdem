"""Local, binned three-factor plasticity on existing KC-to-MBON edges.

The LIF solver remains at 0.1 ms. Eligibility uses aggregate spike counts in
explicitly registered bins, rather than claiming exact spike-timing plasticity.
No action, observation, target or teacher is accepted by this module.
"""
import numpy as np
from flyholdem.provenance import identity


def existing_edges(ptr, post, inputs, outputs):
    """Select CSR edge indices without allocating a full-graph presynaptic array."""
    inputs = np.asarray(inputs, dtype=np.int32)
    outputs = np.asarray(outputs, dtype=np.int32)
    n = len(ptr) - 1
    if (len(np.unique(inputs)) != len(inputs) or len(np.unique(outputs)) != len(outputs)
            or np.any(inputs < 0) or np.any(inputs >= n)
            or np.any(outputs < 0) or np.any(outputs >= n)):
        raise ValueError('Unique, in-range populations required')
    lookup = np.full(n, -1, dtype=np.int32)
    lookup[outputs] = np.arange(len(outputs))
    edges, pres, posts = [], [], []
    for local_pre, node in enumerate(inputs):
        start, end = map(int, ptr[node:node + 2])
        selected = np.flatnonzero(lookup[post[start:end]] >= 0)
        edges.extend((start + selected).tolist())
        pres.extend([local_pre] * len(selected))
        posts.extend(lookup[post[start:end][selected]].tolist())
    return (np.asarray(edges, dtype=np.int64), np.asarray(pres, dtype=np.int32),
            np.asarray(posts, dtype=np.int32))


class Eligibility:
    def __init__(self, brain, inputs, outputs, config):
        self.brain = brain
        self.inputs = np.asarray(inputs, dtype=np.int32)
        self.outputs = np.asarray(outputs, dtype=np.int32)
        self.edges, self.pre, self.post = existing_edges(brain.ptr, brain.post, inputs, outputs)
        if not len(self.edges):
            raise ValueError('No existing edges in plastic population')
        self.initial = brain.initial[self.edges].copy()
        if np.any(self.initial == 0):
            raise ValueError('Plastic multipliers require nonzero original edges')
        self.config = dict(config)
        for key in ('pre_tau_ms', 'eligibility_tau_ms', 'coincidence_scale', 'bin_ms', 'learning_rate'):
            if not np.isfinite(config[key]) or config[key] <= 0:
                raise ValueError(f'Positive finite plasticity parameter required: {key}')
        self.lower, self.upper = map(float, config['weight_bounds'])
        if not 0 < self.lower <= 1 <= self.upper:
            raise ValueError('Weight bounds must preserve sign and contain the original')
        self.pretrace = np.zeros(len(inputs), dtype=np.float64)
        self.trace = np.zeros(len(self.edges), dtype=np.float64)

    @property
    def registration(self):
        return {'rule': 'binned-pretrace-postcount-rpe-v1', 'edge_count': len(self.edges),
                'edge_indices_hash': identity(self.edges.tolist()),
                'input_indices_hash': identity(self.inputs.tolist()),
                'output_indices_hash': identity(self.outputs.tolist()), 'config': self.config}

    def clear_traces(self):
        self.pretrace.fill(0)
        self.trace.fill(0)

    def observe(self, counts, duration_ms):
        counts = np.asarray(counts)
        if counts.shape != (self.brain.n,) or not np.isfinite(counts).all() or np.any(counts < 0):
            raise ValueError('Nonnegative finite per-neuron counts required')
        if not np.isfinite(duration_ms) or duration_ms <= 0:
            raise ValueError('Positive time interval required')
        self.pretrace *= np.exp(-duration_ms / self.config['pre_tau_ms'])
        self.pretrace += counts[self.inputs]
        self.trace *= np.exp(-duration_ms / self.config['eligibility_tau_ms'])
        self.trace += (self.config['coincidence_scale'] * self.pretrace[self.pre]
                       * counts[self.outputs][self.post])

    def advance(self, drive, duration_ms):
        ticks = round(duration_ms * 10)
        bin_ticks = round(self.config['bin_ms'] * 10)
        if ticks < 1 or bin_ticks < 1 or not np.isclose(ticks / 10, duration_ms):
            raise ValueError('Positive 0.1 ms durations required')
        counts = np.zeros(self.brain.n, dtype=np.int64)
        while ticks:
            step = min(ticks, bin_ticks)
            current = self.brain.advance(drive, step / 10)
            self.observe(current, step / 10)
            counts += current
            ticks -= step
        return counts

    def reinforce(self, rpe, learning=True):
        if not np.isfinite(rpe):
            raise ValueError('Finite reward prediction error required')
        old = self.brain.weights[self.edges].copy()
        ratios = old.astype(np.float64) / self.initial
        proposed = ratios + self.config['learning_rate'] * rpe * self.trace
        bounded = np.clip(proposed, self.lower, self.upper)
        if learning:
            self.brain.weights[self.edges] = (self.initial * bounded).astype(np.float32)
        delta = self.brain.weights[self.edges] - old
        actual = self.brain.weights[self.edges].astype(np.float64) / self.initial
        return {'learning': bool(learning), 'dopamine': float(rpe),
                'eligible_synapses': int(np.count_nonzero(self.trace)),
                'changed_synapses': int(np.count_nonzero(delta)),
                'absolute_update': float(np.abs(delta.astype(np.float64)).sum()),
                'clipped_updates': int(np.count_nonzero(proposed != bounded)) if learning else 0,
                'eligibility_mean': float(self.trace.mean()),
                'eligibility_max': float(self.trace.max()),
                'weight_ratio_range': [float(actual.min()), float(actual.max())]}

    def state(self):
        return {'eligibility_pretrace': self.pretrace.copy(), 'eligibility_trace': self.trace.copy()}

    def restore_state(self, arrays):
        expected = self.state()
        if set(arrays) != set(expected):
            raise ValueError('Incomplete eligibility state')
        for key, old in expected.items():
            new = arrays[key]
            if (new.shape != old.shape or new.dtype != old.dtype or not np.isfinite(new).all()
                    or np.any(new < 0)):
                raise ValueError('Invalid eligibility state')
        self.pretrace[:] = arrays['eligibility_pretrace']
        self.trace[:] = arrays['eligibility_trace']
