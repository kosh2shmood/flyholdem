"""Native two-cue player: sensory cells and neural counts, with no teacher.

This small curriculum uses the registered controllability readouts. Its cue
inputs are not the poker encoder and must never be called poker performance.
"""
import json
from pathlib import Path
import numpy as np
from flyholdem.connectome.registry import digest
from .population import neural_scores, select_action


class CuePlayer:
    def __init__(self, controller, sensory_protocol):
        self.controller = controller
        self.protocol = dict(sensory_protocol)
        if set(self.protocol) != {'schema', 'cells', 'actions', 'amplitude_jitter', 'bin_ms'}:
            raise ValueError('Unexpected cue sensory protocol fields')
        if self.protocol['schema'] != 'native-cue-input-v1':
            raise ValueError('Unsupported cue sensory protocol')
        self.cells = np.asarray(self.protocol['cells'], dtype=np.int32)
        actions = self.protocol['actions']
        if (self.cells.ndim != 2 or self.cells.shape[0] != 2 or self.cells.shape[1] < 1
                or len(np.unique(self.cells)) != self.cells.size
                or not np.isin(self.cells, controller.registration['input_indices']).all()
                or len(actions) != 2 or len(set(actions)) != 2
                or any(type(a) is not int or a not in range(5) for a in actions)):
            raise ValueError('Two disjoint registered input cues and two legal actions required')
        if not 0 <= self.protocol['amplitude_jitter'] < 1:
            raise ValueError('Cue jitter must preserve positive gain')
        self.bin_ticks = round(self.protocol['bin_ms'] * 10)
        if self.bin_ticks <= 0 or not np.isclose(self.bin_ticks / 10, self.protocol['bin_ms']):
            raise ValueError('Cue integration bins must be positive multiples of 0.1 ms')

    def decide(self, cue, stimulus_seed, eligibility=None):
        if type(cue) is not int or cue not in (0, 1):
            raise ValueError('Only the visible cue token 0 or 1 is accepted')
        c = self.controller; brain = c.brain
        brain.reset_dynamics()
        if eligibility is not None:
            if eligibility.brain is not brain or eligibility.config['bin_ms'] != self.protocol['bin_ms']:
                raise ValueError('Learning observer must use the same native brain and time bins')
            eligibility.clear_traces()
        rng = np.random.default_rng(stimulus_seed)
        jitter = self.protocol['amplitude_jitter']
        drive = np.zeros(brain.n, dtype=np.float32)
        cells = self.cells[cue]
        drive[cells] = c.registration['selected_gain'] * rng.uniform(1-jitter, 1+jitter, len(cells))

        def advance(value, duration):
            if eligibility is not None:
                return eligibility.advance(value, duration)
            ticks = round(duration * 10)
            counts = np.zeros(brain.n, dtype=np.int64)
            while ticks:
                step = min(ticks, self.bin_ticks)
                counts += brain.advance(value, step / 10)
                ticks -= step
            return counts

        timing = c.registration['config']['decision_ms']
        advance(c.blank, timing['baseline'])
        advance(drive, timing['stimulus'])
        counts = advance(drive, timing['readout'])
        rates, scores = neural_scores(counts, c.ensembles, timing['readout'], c.registration['baseline_hz'],
                                     c.registration['config']['score_scale_hz'])
        legal = [a in self.protocol['actions'] for a in range(5)]
        action = select_action(scores, legal, rng)
        return {'cue': cue, 'selected': action, 'rates_hz': rates.tolist(), 'scores': scores.tolist(),
                'readout_spikes': int(counts.sum()), 'stimulus_seed': int(stimulus_seed),
                'all_legal_readouts_silent': bool(np.all(rates[legal] == 0))}


def load_cue_player(model, graph, expected_sha256=None):
    from .frozen import load_frozen
    controller = load_frozen(model, graph, expected_sha256)
    record = json.loads((Path(model) / 'manifest.json').read_text())
    reference = record['training_reference']['cue_player']
    if reference['implementation_sha256'] != digest(Path(__file__)):
        raise ValueError('Cue inference implementation checksum mismatch')
    return CuePlayer(controller, reference['sensory_protocol'])
