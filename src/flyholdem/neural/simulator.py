"""Synthetic fixture LIF circuit, not measured MaleCNS cells or dynamics.

One millisecond Euler steps; tau_m=20ms, tau_syn=5ms; reset=0,
threshold=1; two full refractory ticks. Spikes deliver next tick. Incoming
current is discarded while refractory. The V0 solver is Python/NumPy.
"""
import hashlib
import numpy as np
from flyholdem.interface.encoder import projection, stimulus, encode, encoded_hash, CHANNELS
from flyholdem.interface.decoder import decode


class FixtureBrain:
    def __init__(self, seed=1729):
        self.n = 126
        rng = np.random.default_rng(1729)  # topology is fixed independently of run seed
        self.pre = np.repeat(np.arange(96), 20)
        self.post = np.tile(np.arange(96, 116), 96)
        self.initial = rng.uniform(.08, .24, len(self.pre))
        self.weights = self.initial.copy()
        self.plastic = np.ones(len(self.pre), dtype=bool)  # all existing synthetic KC→MBON edges
        self.mapping = projection()
        self.ensembles = [np.arange(96 + 4*i, 100 + 4*i) for i in range(5)]
        self.v = np.zeros(self.n)
        self.current = np.zeros(self.n)
        self.refractory = np.zeros(self.n, dtype=np.int32)
        self.pending = np.zeros(self.n)
        self.pretrace = np.zeros(self.n)
        self.eligibility = np.zeros(len(self.pre))
        self.rng = np.random.default_rng(seed)
        self.time_ms = 0
        self.baselines = [0.0, 0.0]
        self.baseline_counts = [0, 0]
        self.last_counts = np.zeros(self.n, dtype=np.int64)

    @property
    def graph_hash(self):
        return hashlib.sha256(self.pre.tobytes()+self.post.tobytes()+self.initial.tobytes()).hexdigest()

    def advance(self, drive, duration_ms):
        counts = np.zeros(self.n, dtype=np.int64)
        decay = np.exp(-1 / 5000)
        for _ in range(duration_ms):
            blocked = self.refractory > 0
            self.refractory[blocked] -= 1
            self.current[~blocked] += self.pending[~blocked]
            self.current[blocked] = 0
            self.v[~blocked] += (-self.v[~blocked] + drive[~blocked] + self.current[~blocked]) / 20
            self.current *= .8
            spikes = (self.v >= 1) & ~blocked
            counts += spikes
            self.v[spikes] = 0
            self.refractory[spikes] = 2
            self.pending = np.bincount(self.post, weights=self.weights * spikes[self.pre], minlength=self.n)
            self.pretrace = self.pretrace * np.exp(-1 / 20) + spikes
            self.eligibility *= decay
            self.eligibility += .001 * self.pretrace[self.pre] * spikes[self.post]
            self.time_ms += 1
        self.last_counts = counts
        return counts

    def decide(self, observation, temperature=.65):
        x = encode(observation)
        drive = stimulus(x, self.mapping, self.n)
        self.advance(np.zeros(self.n), 50)
        self.advance(drive, 300)
        # Maintain the state stimulus during the 100ms measurement window.
        counts = self.advance(drive, 100)
        decision = decode(counts, self.ensembles, observation['legal_mask'], self.rng, temperature)
        result = {**decision, 'encoded_hash': encoded_hash(x),
            'encoded': [[CHANNELS[i], round(float(x[i]), 8)] for i in np.flatnonzero(x)],
            'activity': counts.tolist(), 'virtual_time_ms': self.time_ms,
            'observation': observation, 'score_source': 'fixture-LIF-spikes'}
        self.advance(np.zeros(self.n), 50)
        return result

    def reinforce(self, net_bb, position, learning=True):
        reward = float(np.tanh(net_bb / 10))
        baseline = self.baselines[position]
        rpe = reward - baseline
        old = self.weights.copy()
        proposed = .015 * rpe * self.eligibility * self.initial
        if learning:
            self.weights[self.plastic] = np.clip((self.weights + proposed)[self.plastic],
                .1*self.initial[self.plastic], 2*self.initial[self.plastic])
        delta = self.weights - old
        self.baseline_counts[position] += 1
        self.baselines[position] += (reward - baseline) / self.baseline_counts[position]
        drive = np.zeros(self.n)
        drive[116:121 if rpe >= 0 else 116] = 3 * abs(rpe)
        if rpe < 0:
            drive[121:126] = 3 * abs(rpe)
        counts = self.advance(drive, 200)
        ratios = self.weights / self.initial
        return {'raw_net_bb': net_bb, 'normalized_reward': reward, 'baseline': baseline,
                'dopamine': rpe, 'learning': learning, 'eligible_synapses': int(np.count_nonzero(self.eligibility)),
                'changed_synapses': int(np.count_nonzero(delta)), 'absolute_update': float(np.abs(delta).sum()),
                'clipped_updates': int(np.count_nonzero((old+proposed != self.weights) & self.plastic)) if learning else 0,
                'eligibility_mean': float(np.mean(self.eligibility)), 'weight_ratio_range': [float(ratios.min()), float(ratios.max())],
                'weight_histogram': np.histogram(ratios, bins=np.linspace(.1, 2, 20))[0].tolist(),
                'largest_changes': [{'edge': int(i), 'delta': float(delta[i])} for i in np.argsort(np.abs(delta))[-5:][::-1]],
                'activity': counts.tolist(), 'virtual_time_ms': self.time_ms}

    def graph_view(self):
        nodes=[]
        for i in range(self.n):
            group = 'KC' if i < 96 else f'A{(i-96)//4}' if i < 116 else 'DAN+' if i < 121 else 'DAN−'
            angle = i * 2.3999632297
            radius = np.sqrt((i % 96 + 1)/96)
            nodes.append({'id': f'fixture:{i}', 'group': group,
                'x': float(np.cos(angle)*radius*.68 if i < 96 else -.8+(i-96)%10*.18),
                'y': float(np.sin(angle)*radius*.72 if i < 96 else .87+(i-96)//10*.17)})
        return {'nodes': nodes, 'edge_count': len(self.pre),
                'sample_edges': [[int(self.pre[i]), int(self.post[i])] for i in range(0, len(self.pre), 16)],
                'hash': self.graph_hash, 'layout': 'synthetic schematic; not anatomical coordinates'}
