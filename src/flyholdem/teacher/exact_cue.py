"""Exact table for a non-poker, two-cue/two-action distillation curriculum."""
import numpy as np
from flyholdem.provenance import identity


class ExactCueTeacher:
    def __init__(self, actions):
        if len(actions) != 2 or len(set(actions)) != 2 or any(type(a) is not int or a not in range(5) for a in actions):
            raise ValueError('Two distinct abstract actions required')
        self.actions = list(actions)
        self.table = np.eye(5, dtype=np.float64)[actions]

    @property
    def registration(self):
        return {'schema': 'exact-cue-teacher-v1', 'scope': 'engineered two-cue task; no poker claim',
                'targets': self.table.tolist(), 'actions': self.actions,
                'table_sha256': identity(self.table.tolist()), 'exhaustively_verified_states': 2,
                'input': 'visible cue integer only',
                'advantage': '2 * (target probability of selected action - uniform-legal mean)'}

    def probabilities(self, cue):
        if type(cue) is not int or cue not in (0, 1):
            raise ValueError('Exact teacher accepts only visible cue token 0 or 1')
        return self.table[cue].copy()

    def advantage(self, cue, action):
        if type(action) is not int or action not in self.actions:
            raise ValueError('Legal neural action required')
        target = self.probabilities(cue)
        return float(2 * (target[action] - target[self.actions].mean()))
