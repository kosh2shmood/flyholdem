"""Locally trained neural fictitious self-play, following Heinrich/Silver 2016.

Two independent agents retain Q replay, a uniform reservoir of their own
best-response behavior and an average policy. The mixture is sampled once per
hand. No hidden cards, full game object or equity labels enter these networks.
"""
import numpy as np
import torch
from torch import nn
from .features import features, feature_names, V1
from .memory import Memory
from .serialization import pack, unpack


def network(hidden, dimension=None):
    return nn.Sequential(nn.Linear(len(feature_names(V1)) if dimension is None else dimension, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 5))


class NFSPAgent:
    def __init__(self, config, seed):
        self.config = dict(config)
        self.feature_version = config.get('feature_version', V1)
        self.dimension = len(feature_names(self.feature_version))
        torch.manual_seed(seed)
        self.q = network(config['hidden'], self.dimension)
        self.target = network(config['hidden'], self.dimension)
        self.target.load_state_dict(self.q.state_dict())
        self.target.requires_grad_(False)
        self.average = network(config['hidden'], self.dimension)
        self.q_optimizer = torch.optim.Adam(self.q.parameters(), lr=config['q_lr'])
        self.average_optimizer = torch.optim.Adam(self.average.parameters(), lr=config['average_lr'])
        self.rng = np.random.default_rng(seed + 1)
        dimension = self.dimension
        self.replay = Memory(config['replay_capacity'], {
            'observation': ((dimension,), np.float32), 'action': ((), np.int64),
            'reward': ((), np.float32), 'next_observation': ((dimension,), np.float32),
            'next_legal': ((5,), np.bool_), 'terminal': ((), np.bool_)}, seed + 2)
        self.reservoir = Memory(config['reservoir_capacity'], {
            'observation': ((dimension,), np.float32), 'legal': ((5,), np.bool_),
            'action': ((), np.int64)}, seed + 3, reservoir=True)
        self.updates = 0
        self.episode_br = False

    def begin_hand(self):
        self.episode_br = bool(self.rng.random() < self.config['anticipatory'])

    @torch.no_grad()
    def probabilities(self, observation):
        x = features(observation, self.feature_version)
        legal = np.asarray(observation['legal_mask'], dtype=bool)
        logits = self.average(torch.from_numpy(x))
        logits = logits.masked_fill(~torch.from_numpy(legal), -torch.inf)
        return torch.softmax(logits, dim=-1).numpy().astype(np.float64)

    @torch.no_grad()
    def act(self, observation, epsilon, record=True):
        if not 0 <= epsilon <= 1:
            raise ValueError('Exploration probability must be in [0,1]')
        x = features(observation, self.feature_version)
        legal = np.asarray(observation['legal_mask'], dtype=bool)
        if self.episode_br:
            if self.rng.random() < epsilon:
                action = int(self.rng.choice(np.flatnonzero(legal)))
            else:
                q = self.q(torch.from_numpy(x)).numpy()
                action = int(np.argmax(np.where(legal, q, -np.inf)))
            if record:
                self.reservoir.add(observation=x, legal=legal, action=action)
        else:
            probabilities = self.probabilities(observation)
            probabilities /= probabilities.sum()  # float32 softmax -> exact NumPy sampling normalization
            action = int(self.rng.choice(5, p=probabilities))
        return action, x

    def transition(self, previous, reward, next_observation=None):
        x, action = previous
        terminal = next_observation is None
        next_x = np.zeros(self.dimension, dtype=np.float32) if terminal else features(next_observation, self.feature_version)
        next_legal = np.zeros(5, dtype=bool) if terminal else np.asarray(next_observation['legal_mask'], dtype=bool)
        self.replay.add(observation=x, action=action, reward=reward, next_observation=next_x,
                        next_legal=next_legal, terminal=terminal)

    def train_step(self):
        batch_size = self.config['batch_size']
        metrics = {}
        if self.replay.size >= max(batch_size, self.config['warmup_transitions']):
            sample = {key: torch.from_numpy(value) for key, value in self.replay.sample(batch_size).items()}
            with torch.no_grad():
                legal = sample['next_legal'].clone()
                # Terminal rows need a finite placeholder before multiplication by zero.
                legal[sample['terminal'], 0] = True
                q_next = self.target(sample['next_observation']).masked_fill(~legal, -torch.inf).max(dim=1).values
                target = sample['reward'] + self.config['discount'] * q_next * (~sample['terminal'])
            predicted = self.q(sample['observation']).gather(1, sample['action'][:, None]).squeeze(1)
            loss = nn.functional.smooth_l1_loss(predicted, target)
            self.q_optimizer.zero_grad(set_to_none=True)
            loss.backward(); nn.utils.clip_grad_norm_(self.q.parameters(), self.config['gradient_clip'])
            self.q_optimizer.step()
            self.updates += 1
            if self.updates % self.config['target_update_steps'] == 0:
                self.target.load_state_dict(self.q.state_dict())
            metrics['q_loss'] = float(loss.detach())
        if self.reservoir.size >= batch_size:
            sample = {key: torch.from_numpy(value) for key, value in self.reservoir.sample(batch_size).items()}
            logits = self.average(sample['observation']).masked_fill(~sample['legal'], -torch.inf)
            loss = nn.functional.cross_entropy(logits, sample['action'])
            self.average_optimizer.zero_grad(set_to_none=True)
            loss.backward(); nn.utils.clip_grad_norm_(self.average.parameters(), self.config['gradient_clip'])
            self.average_optimizer.step()
            metrics['average_loss'] = float(loss.detach())
        if not all(np.isfinite(value) for value in metrics.values()):
            raise FloatingPointError('Nonfinite teacher optimization loss')
        return metrics

    def state(self, prefix):
        arrays = {}
        tree = pack({'q': self.q.state_dict(), 'target': self.target.state_dict(),
                     'average': self.average.state_dict(), 'q_optimizer': self.q_optimizer.state_dict(),
                     'average_optimizer': self.average_optimizer.state_dict()}, arrays, prefix)
        metadata = {'config': self.config, 'tree': tree, 'rng': self.rng.bit_generator.state,
                    'updates': self.updates, 'episode_br': self.episode_br, 'memories': {}}
        for name, memory in (('replay', self.replay), ('reservoir', self.reservoir)):
            values, meta = memory.state()
            arrays.update({f'{prefix}_{name}_{key}': value for key, value in values.items()})
            metadata['memories'][name] = meta
        return arrays, metadata

    def restore_state(self, arrays, metadata, prefix):
        if metadata['config'] != self.config:
            raise ValueError('Teacher configuration mismatch')
        state = unpack(metadata['tree'], arrays)
        for name in ('q', 'target', 'average', 'q_optimizer', 'average_optimizer'):
            getattr(self, name).load_state_dict(state[name])
        for name, memory in (('replay', self.replay), ('reservoir', self.reservoir)):
            values = {key: arrays[f'{prefix}_{name}_{key}'] for key in memory.arrays}
            memory.restore_state(values, metadata['memories'][name])
        self.rng.bit_generator.state = metadata['rng']
        self.updates, self.episode_br = metadata['updates'], metadata['episode_br']
