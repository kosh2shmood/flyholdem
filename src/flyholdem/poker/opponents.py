"""Labeled opponent controls. Never imported by the neural policy."""
import random


class Opponent:
    def __init__(self, seed=1, kind='calling-station'):
        self.rng, self.kind = random.Random(seed), kind

    def act(self, observation):
        legal = [i for i, ok in enumerate(observation['legal_mask']) if ok]
        if self.kind == 'random':
            return self.rng.choice(legal)
        if self.kind == 'calling-station':
            return 1 if 1 in legal else self.rng.choice(legal)
        raise ValueError(f'Unknown opponent {self.kind}')
