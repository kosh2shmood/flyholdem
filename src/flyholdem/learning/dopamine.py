"""Engineered scalar reinforcement gate and separately recorded DAN stimulation.

PAM/PPL1 labels identify stimulation proxies, not verified valence or synaptic
receptor physiology. The scalar gate is an explicit modeling assumption.
"""
import math
import numpy as np


class PastBaseline:
    def __init__(self):
        self.contexts = {}

    def event(self, raw_reward, context, scale=1.0, transform='tanh'):
        if not math.isfinite(raw_reward) or not math.isfinite(scale) or scale <= 0:
            raise ValueError('Finite reward and positive scale required')
        if transform not in ('tanh', 'identity'):
            raise ValueError('Unknown reward transform')
        reward = math.tanh(raw_reward / scale) if transform == 'tanh' else float(raw_reward)
        past = self.contexts.get(str(context), {'count': 0, 'mean': 0.0})
        count, baseline = past['count'], past['mean']
        self.contexts[str(context)] = {'count': count + 1, 'mean': baseline + (reward - baseline) / (count + 1)}
        return {'raw_reward': float(raw_reward), 'normalized_reward': reward, 'baseline': baseline,
                'past_count': count, 'context': str(context), 'dopamine': reward - baseline,
                'transform': transform, 'scale': scale}

    def state(self):
        return {key: dict(value) for key, value in self.contexts.items()}

    def restore_state(self, state):
        for value in state.values():
            if (set(value) != {'count', 'mean'} or type(value['count']) is not int
                    or value['count'] < 0 or not math.isfinite(value['mean'])):
                raise ValueError('Invalid past-only reward baseline')
        self.contexts = {str(key): dict(value) for key, value in state.items()}


def annotated_populations(nodes):
    dan = nodes['class'].eq('DAN')
    types = nodes['type'].fillna('')
    positive = np.flatnonzero((dan & types.str.startswith('PAM')).to_numpy()).astype(np.int32)
    negative = np.flatnonzero((dan & types.str.startswith('PPL1')).to_numpy()).astype(np.int32)
    if not len(positive) or not len(negative):
        raise ValueError('Both annotated dopamine proxy populations are required')
    return positive, negative


def pulse(brain, rpe, populations, duration_ms=200, gain=12):
    if not np.isfinite(rpe) or gain <= 0:
        raise ValueError('Finite dose and positive gain required')
    positive, negative = populations
    selected = positive if rpe >= 0 else negative
    drive = np.zeros(brain.n, dtype=np.float32)
    drive[selected] = gain * min(abs(rpe), 2.0)
    counts = brain.advance(drive, duration_ms)
    return {'proxy_population': 'PAM' if rpe >= 0 else 'PPL1', 'cell_count': len(selected),
            'pulse_ms': duration_ms, 'per_cell_drive': float(gain * min(abs(rpe), 2.0)),
            'population_spikes': int(counts[selected].sum()), 'total_spikes': int(counts.sum()),
            'valence_status': 'engineered-annotation-proxy'}
