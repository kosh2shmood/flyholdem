"""Synchronous two-player external-sampling regret, separate from every fly.

Each iteration traverses one actual chance deal for each updating public role.
Both passes read the same boundary regrets. Only a complete pair commits its
sparse deltas. Average strategy mass is accumulated at sampled opponent nodes;
this is the two-player rule, not the stationary-population averaging rule.
"""
from dataclasses import dataclass, field
import copy
import hashlib
import json

import numpy as np

from flyholdem.poker.engine import Hand
from flyholdem.provenance import canonical
from .poker_tree import counterfactual_root, fork_hand
from .regret import VERSION, RegretTable, abstraction, regret_matching


ROLE_NAMES = ('nonbutton-big-blind', 'button-small-blind')
AGGREGATION = 'opponent-node-sampled-average-synchronous-profile-v1'
SCHEMA = 'synchronous-two-player-external-sampling-regret-v1'
_TABLE_FIELDS = ('key_offsets', 'key_bytes', 'regrets', 'averages', 'legal',
                 'visits', 'average_visits')
_ARRAY_FIELDS = ('regrets', 'averages', 'legal', 'visits', 'average_visits')


@dataclass
class TraversalResult:
    """Uncommitted numeric updates; dictionary keys are (public role, bytes)."""
    regrets: dict = field(default_factory=dict)
    averages: dict = field(default_factory=dict)
    regret_visits: dict = field(default_factory=dict)
    average_visits: dict = field(default_factory=dict)
    legal: dict = field(default_factory=dict)
    sampled_actions: dict = field(default_factory=dict)
    value_bb: float = 0.
    nodes: int = 0
    terminal_branches: int = 0
    sampled_action_draws: int = 0
    trace_sha256: str = ''


def _decision(role, key, legal):
    if type(role) is not int or role not in (0, 1) or not isinstance(key, bytes) or not key:
        raise ValueError('An explicit public role and nonempty information key are required')
    legal = np.asarray(legal)
    if legal.dtype != np.bool_ or legal.shape != (5,) or not legal.any():
        raise ValueError('Five boolean canonical actions with a nonempty legal set are required')
    return (role, key), legal.copy()


def external_sampling_traversal(root, updating_role, strategy_for, rng, *, max_nodes=100000):
    """Return a complete pass's staged deltas on an already sampled chance tree.

    Node protocol: is_terminal(), terminal_returns_bb() in role order,
    information_set() -> (role, key bytes, bool[5]), and child(action) returning
    an independent child. A caller testing exact expectations can enumerate
    fixed-chance roots and pure opponent maps outside this function. The
    strategy callback must read a frozen profile and return legal float[5].
    The helper consumes rng; its caller owns rollback and commits no partial
    result. PokerKitNode below provides the actual rules/settlement adapter.
    """
    if type(updating_role) is not int or updating_role not in (0, 1):
        raise ValueError('Updating public role must be zero or one')
    if type(max_nodes) is not int or max_nodes < 1:
        raise ValueError('A positive complete-traversal node ceiling is required')
    result = TraversalResult()
    trace = hashlib.sha256()
    updating_seen = set()
    # Repeated sampled-opponent keys can occur across hidden-action branches
    # in a general imperfect-information game. One pure strategy is sampled.
    policies = {}

    def record(value):
        trace.update(canonical(value) + b'\n')

    def walk(node):
        result.nodes += 1
        if result.nodes > max_nodes:
            raise RuntimeError('Complete self-play traversal node ceiling exceeded')
        if node.is_terminal():
            returns = np.asarray(node.terminal_returns_bb(), dtype=np.float64)
            if (returns.shape != (2,) or not np.isfinite(returns).all()
                    or not np.isclose(returns.sum(), 0., rtol=0., atol=1e-12)):
                raise ValueError('Finite zero-sum terminal returns in public-role BB are required')
            result.terminal_branches += 1
            record({'terminal_returns_bb': returns.tolist()})
            return float(returns[updating_role])

        role, key, raw_legal = node.information_set()
        token, legal = _decision(role, key, raw_legal)
        if token in result.legal and not np.array_equal(result.legal[token], legal):
            raise ValueError('A self-play information set merged different legal actions')
        result.legal[token] = legal
        if token not in policies:
            policy = np.asarray(strategy_for(role, key, legal.copy()), dtype=np.float64).copy()
            if (policy.shape != (5,) or not np.isfinite(policy).all()
                    or np.any(policy < 0) or np.any(policy[~legal])
                    or not np.isclose(policy.sum(), 1., rtol=0., atol=1e-12)):
                raise ValueError('The boundary strategy must be finite, normalized and legal')
            policies[token] = policy
        policy = policies[token]
        label = {'role': role, 'information_key_sha256': hashlib.sha256(key).hexdigest(),
                 'legal': legal.tolist(), 'boundary_strategy': policy.tolist()}

        if role == updating_role:
            if token in updating_seen:
                raise ValueError('An updating information set repeated in a sampled perfect-recall tree')
            updating_seen.add(token)
            values = np.zeros(5, dtype=np.float64)
            for action in np.flatnonzero(legal):
                values[action] = walk(node.child(int(action)))
            value = float(np.dot(policy, values))
            delta = np.where(legal, values - value, 0.)
            if not np.isfinite(delta).all() or not np.isfinite(value):
                raise FloatingPointError('Nonfinite self-play counterfactual values')
            result.regrets[token] = delta
            result.regret_visits[token] = 1
            record({**label, 'updating_role': updating_role,
                    'action_values_bb': values.tolist(), 'regret_delta': delta.tolist()})
            return value

        reused = token in result.sampled_actions
        if not reused:
            choice = rng.choice(5, p=policy.copy())
            result.sampled_action_draws += 1
            if not isinstance(choice, (int, np.integer)) or isinstance(choice, (bool, np.bool_)):
                raise ValueError('The sampler must return a canonical action index')
            choice = int(choice)
            if choice not in range(5) or not legal[choice] or policy[choice] <= 0:
                raise ValueError('The sampler selected an unsupported opponent action')
            result.sampled_actions[token] = choice
        action = result.sampled_actions[token]
        value = walk(node.child(action))
        result.averages[token] = result.averages.get(token, np.zeros(5)) + policy
        result.average_visits[token] = result.average_visits.get(token, 0) + 1
        record({**label, 'sampled_action': action, 'sample_reused': reused,
                'average_delta': policy.tolist()})
        return value

    result.value_bb = walk(root)
    result.trace_sha256 = trace.hexdigest()
    return result


class PokerKitNode:
    """A copied PokerKit hand with explicit observable button/nonbutton roles."""
    def __init__(self, game, config):
        self.game = game
        self.config = config

    def is_terminal(self):
        return self.game.done

    def terminal_returns_bb(self):
        payoffs = self.game.view()['payoffs']
        return np.asarray([payoffs[1 - self.game.button], payoffs[self.game.button]], dtype=float) / 2

    def information_set(self):
        observation = self.game.observation()
        key, legal = abstraction(observation, self.config)
        return observation['position'], key, legal

    def child(self, action):
        game = fork_hand(self.game)
        game.act(action)
        return PokerKitNode(game, self.config)


class _RoleTable(RegretTable):
    def __init__(self, config):
        super().__init__(config)
        self.average_visits = np.zeros(0, dtype=np.uint64)

    def _grow(self, needed):
        super()._grow(needed)
        if len(self.average_visits) != self.capacity:
            prior = self.average_visits
            self.average_visits = np.zeros(self.capacity, dtype=np.uint64)
            self.average_visits[:min(len(prior), self.capacity)] = prior[:self.capacity]

    def state(self):
        return {**super().state(), 'average_visits': self.average_visits[:len(self.keys)].copy()}


class SelfPlayRegretTable:
    """Transactional two-role table; iteration boundaries include both passes.

    Config is the existing complete abstraction config. max_information_sets
    caps distinct keys across both roles. Role 0 is nonbutton/BB; role 1 is
    button/SB. Current strategy reads never insert an unvisited key.
    """
    def __init__(self, config):
        self.config = dict(config)
        self.tables = (_RoleTable(config), _RoleTable(config))
        self.iterations_completed = 0

    def strategy_for(self, role, key, legal):
        _, legal = _decision(role, key, legal)
        table = self.tables[role]
        index = table.index.get(key)
        if index is None:
            return legal.astype(float) / legal.sum()
        if not np.array_equal(table.legal[index], legal):
            raise ValueError('Stored self-play information set changed legality')
        return regret_matching(table.regrets[index], legal)

    def probabilities(self, observation):
        key, legal = abstraction(observation, self.config)
        role = observation['position']
        _decision(role, key, legal)
        return self.tables[role].probabilities(observation)

    def _validate_key(self, role, key, legal):
        try:
            value = json.loads(key)
            valid = (canonical(value) == key and value[0] == VERSION
                     and value[1] == 4 * self.config['stack_bb']
                     and value[3][0] == role and value[-1] == legal.tolist())
        except (ValueError, TypeError, IndexError, KeyError):
            valid = False
        if not valid:
            raise ValueError('A self-play key must match its public role, stack, version and legal actions')

    def _commit(self, passes):
        """Validate sparse final rows before mutation; roll back failed commits."""
        masks = {}
        for result in passes:
            for token, legal in result.legal.items():
                if token in masks and not np.array_equal(masks[token], legal):
                    raise ValueError('Two passes disagreed on information-set legality')
                masks[token] = legal
        new_keys = [[], []]
        plans = []
        uint_max = np.iinfo(np.uint64).max
        for (role, key), legal in masks.items():
            self._validate_key(role, key, legal)
            table = self.tables[role]
            index = table.index.get(key)
            if index is None:
                index = len(table.keys) + len(new_keys[role])
                new_keys[role].append((key, legal))
                regret, average, visits, average_visits = np.zeros(5), np.zeros(5), 0, 0
            else:
                if not np.array_equal(table.legal[index], legal):
                    raise ValueError('Stored self-play information set changed legality')
                regret, average = table.regrets[index].copy(), table.averages[index].copy()
                visits, average_visits = int(table.visits[index]), int(table.average_visits[index])
            token = role, key
            with np.errstate(over='raise', invalid='raise'):
                for result in passes:
                    if token in result.regrets:
                        regret += result.regrets[token]
                    if token in result.averages:
                        average += result.averages[token]
                    visits += result.regret_visits.get(token, 0)
                    average_visits += result.average_visits.get(token, 0)
            if (not np.isfinite(regret).all() or not np.isfinite(average).all()
                    or np.any(average < 0) or np.any(regret[~legal]) or np.any(average[~legal])
                    or max(visits, average_visits) > uint_max):
                raise FloatingPointError('Invalid committed self-play numeric state')
            plans.append((role, index, regret, average, visits, average_visits))
        total = sum(len(table.keys) + len(new) for table, new in zip(self.tables, new_keys))
        if total > self.config['max_information_sets']:
            raise MemoryError('Combined self-play information-set capacity exceeded')
        if self.iterations_completed >= np.iinfo(np.int64).max:
            raise OverflowError('Self-play iteration counter exhausted')

        # Allocation and row writes are inside the transaction too. Preserve
        # spare existing rows, otherwise a rolled-back insertion could leave
        # nonzero values for a later newly inserted key.
        snapshots = []
        for role, table in enumerate(self.tables):
            rows = [(index, table.regrets[index].copy(), table.averages[index].copy(),
                     table.legal[index].copy(), table.visits[index].copy(), table.average_visits[index].copy())
                    for r, index, *_ in plans if r == role and index < table.capacity]
            snapshots.append((len(table.keys), table.capacity,
                              {name: getattr(table, name) for name in _ARRAY_FIELDS}, rows))
        previous_iteration = self.iterations_completed
        try:
            for table, new in zip(self.tables, new_keys):
                table._grow(len(table.keys) + len(new))
                for key, legal in new:
                    table.lookup(key, legal)
            for role, index, regret, average, visits, average_visits in plans:
                table = self.tables[role]
                table.regrets[index] = regret
                table.averages[index] = average
                table.visits[index] = visits
                table.average_visits[index] = average_visits
            self.iterations_completed += 1
        except BaseException:
            for table, (size, capacity, arrays, rows) in zip(self.tables, snapshots):
                for key in table.keys[size:]:
                    table.index.pop(key, None)
                del table.keys[size:]
                table.capacity = capacity
                for name, array in arrays.items():
                    setattr(table, name, array)
                for index, regret, average, legal, visits, average_visits in rows:
                    table.regrets[index] = regret
                    table.averages[index] = average
                    table.legal[index] = legal
                    table.visits[index] = visits
                    table.average_visits[index] = average_visits
            self.iterations_completed = previous_iteration
            raise

    def step(self, first_hand, second_hand, rng):
        """One synchronous pair, using caller-supplied actual registered deals.

        Both inputs must be fresh ordinary equal-stack heads-up Hand objects.
        The caller owns the explicit two-deal/RNG schedule. Any incomplete
        pair restores the RNG and leaves both tables and input hands unchanged.
        """
        for hand in (first_hand, second_hand):
            if (not isinstance(hand, Hand) or hand.player_count != 2 or hand.done or hand.history
                    or hand.starting_stacks != [2 * self.config['stack_bb']] * 2):
                raise ValueError('Fresh ordinary equal-stack heads-up PokerKit hands are required')
        try:
            saved_rng = copy.deepcopy(rng.bit_generator.state)
        except AttributeError as error:
            raise ValueError('A restorable sampler state is required') from error
        roots = [copy.deepcopy(hand.serialize()) for hand in (first_hand, second_hand)]
        previous_iteration = self.iterations_completed
        try:
            passes = []
            for role, hand in enumerate((first_hand, second_hand)):
                with counterfactual_root(hand) as tree:
                    passes.append(external_sampling_traversal(
                        PokerKitNode(tree, self.config), role, self.strategy_for, rng,
                        max_nodes=self.config['max_nodes_per_traversal']))
            if roots != [first_hand.serialize(), second_hand.serialize()]:
                raise AssertionError('Self-play traversal mutated an actual input hand')
            counts = [len(table.keys) + len({key for result in passes for r, key in result.legal
                                           if r == role and key not in table.index})
                      for role, table in enumerate(self.tables)]
            row = {'schema': 'self-play-regret-iteration-v1', 'iteration': previous_iteration + 1,
                   'traversals': [{'updating_role': role, 'public_role': ROLE_NAMES[role],
                                   'root_private_checkpoint': roots[role],
                                   'sampled_policy_value_bb': result.value_bb,
                                   'nodes': result.nodes, 'terminal_branches': result.terminal_branches,
                                   'regret_information_sets': len(result.regrets),
                                   'average_information_sets': len(result.averages),
                                   'average_encounters': sum(result.average_visits.values()),
                                   'sampled_action_draws': result.sampled_action_draws,
                                   'complete_traversal_sha256': result.trace_sha256}
                                  for role, result in enumerate(passes)],
                   'information_sets_by_role': counts, 'information_sets': sum(counts),
                   'nodes': sum(result.nodes for result in passes),
                   'terminal_branches': sum(result.terminal_branches for result in passes),
                   'sampled_action_draws': sum(result.sampled_action_draws for result in passes),
                   'scope': 'Conventional sampled self-play updates; no held-out, equilibrium or fly-strength result'}
            self._commit(passes)
            return row
        except BaseException:
            if self.iterations_completed == previous_iteration:
                rng.bit_generator.state = saved_rng
            raise

    def state(self):
        return {'iterations_completed': np.asarray([self.iterations_completed], dtype=np.int64),
                **{f'role{role}_{name}': value for role, table in enumerate(self.tables)
                   for name, value in table.state().items()}}

    def restore(self, arrays):
        expected = {'iterations_completed'} | {f'role{role}_{name}' for role in (0, 1) for name in _TABLE_FIELDS}
        if set(arrays) != expected:
            raise ValueError('Complete two-role numeric self-play state is required')
        iteration = arrays['iterations_completed']
        if (iteration.dtype != np.int64 or iteration.shape != (1,) or iteration[0] < 0):
            raise ValueError('Invalid self-play iteration boundary')
        restored = []
        for role in (0, 1):
            table = _RoleTable(self.config)
            values = {name: arrays[f'role{role}_{name}'] for name in _TABLE_FIELDS}
            counts = values.pop('average_visits')
            table.restore(values)
            if counts.dtype != np.uint64 or counts.shape != (len(table.keys),):
                raise ValueError('Invalid self-play average-visit counts')
            mass = table.averages[:len(table.keys)].sum(axis=1)
            if not np.allclose(mass, counts, rtol=1e-9, atol=1e-9):
                raise ValueError('Self-play average mass differs from its visit evidence')
            table.average_visits[:len(table.keys)] = counts
            for key, legal in zip(table.keys, table.legal):
                self._validate_key(role, key, legal)
            restored.append(table)
        if sum(len(table.keys) for table in restored) > self.config['max_information_sets']:
            raise ValueError('Combined restored self-play table exceeds its capacity')
        self.tables = tuple(restored)
        self.iterations_completed = int(iteration[0])
