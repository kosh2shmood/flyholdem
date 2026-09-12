"""Exact finite-game expectations; no Monte Carlo or poker-strength thresholds."""
from dataclasses import dataclass, replace
from fractions import Fraction as F
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest


TWO_ACTIONS = np.array([True, True, False, False, False])
CHANCE = (F(1, 3), F(2, 3))
STOP = (-2, 3)
# Player 0 payoff indexed by chance, player 1's hidden move, P0's later move.
PAYOFF = (((-3, 5), (4, -2)), ((1, -4), (-6, 7)))
P0_ROOT = (0, b'player-zero-root')
P0_LATE = (0, b'player-zero-remembers-continue')
P1_PRIVATE = ((1, b'player-one-private-zero'), (1, b'player-one-private-one'))


def vector(probabilities):
    return np.array([float(value) for value in probabilities] + [0.] * (5 - len(probabilities)))


@dataclass(frozen=True)
class HiddenActionGame:
    """Chance is private to P1; its later move is hidden from P0.

    P0 remembers its own continue action, so the merged late information set
    retains perfect recall. P1 knows chance and sees that P0 continued.
    """
    chance: int
    history: tuple = ()

    def is_terminal(self):
        return self.history == (0,) or len(self.history) == 3

    def terminal_returns_bb(self):
        assert self.is_terminal()
        value = STOP[self.chance] if self.history == (0,) else PAYOFF[self.chance][self.history[1]][self.history[2]]
        return (float(value), float(-value))

    def information_set(self):
        assert not self.is_terminal()
        key = P0_ROOT if not self.history else P1_PRIVATE[self.chance] if len(self.history) == 1 else P0_LATE
        return (*key, TWO_ACTIONS.copy())

    def child(self, action):
        assert action in (0, 1) and not self.is_terminal()
        return replace(self, history=self.history + (action,))


class FixedDraws:
    """Enumerated pure-strategy draw, also verifying the distribution sampled."""
    def __init__(self, draws):
        self.draws = list(draws)
        self.bit_generator = SimpleNamespace(state={'next_draw': 0})
        self.actual = []

    def choice(self, count, *, p):
        index = self.bit_generator.state['next_draw']
        assert count == 5 and index < len(self.draws), 'Opponent infoset was independently sampled again'
        action, expected = self.draws[index]
        np.testing.assert_allclose(p, expected, atol=1e-15, rtol=0)
        assert p[action] > 0
        self.actual.append((action, np.array(p, copy=True)))
        self.bit_generator.state['next_draw'] += 1
        return action


def profile(continue_reach):
    return {
        P0_ROOT: (1 - continue_reach, continue_reach),
        P0_LATE: (F(2, 5), F(3, 5)),
        P1_PRIVATE[0]: (F(1, 3), F(2, 3)),
        P1_PRIVATE[1]: (F(3, 4), F(1, 4)),
    }


def strategy_callback(strategies):
    def lookup(role, key, legal):
        np.testing.assert_array_equal(legal, TWO_ACTIONS)
        return vector(strategies[role, key])
    return lookup


def full_cfr_by_enumeration(strategies):
    """Closed-form full CFR from every terminal payoff, using exact fractions.

    This does not invoke the traversal, its values, or its regret matching.
    Counterfactual reach includes chance and the other player's prefix only.
    """
    result = {key: [F(0)] * 5 for key in strategies}
    root_sigma, late_sigma = strategies[P0_ROOT], strategies[P0_LATE]
    for chance, chance_reach in enumerate(CHANCE):
        other_sigma = strategies[P1_PRIVATE[chance]]
        late_values = [sum(late_sigma[a] * PAYOFF[chance][b][a] for a in (0, 1)) for b in (0, 1)]
        continue_value = sum(other_sigma[b] * late_values[b] for b in (0, 1))
        root_value = root_sigma[0] * STOP[chance] + root_sigma[1] * continue_value
        for action, action_value in enumerate((STOP[chance], continue_value)):
            result[P0_ROOT][action] += chance_reach * (action_value - root_value)
        for b in (0, 1):
            for a in (0, 1):
                result[P0_LATE][a] += chance_reach * other_sigma[b] * (PAYOFF[chance][b][a] - late_values[b])
            # P1's utility has the opposite sign. P0's root reach belongs in
            # P1's counterfactual weighting; P0's own regret omits that reach.
            result[P1_PRIVATE[chance]][b] += chance_reach * root_sigma[1] * (-late_values[b] + continue_value)
    return result


def pure_action_maps(strategies, role):
    keys = [key for key in strategies if key[0] == role]
    for actions in product((0, 1), repeat=len(keys)):
        mapping = dict(zip(keys, actions))
        probability = F(1)
        for key, action in mapping.items():
            probability *= strategies[key][action]
        if probability:
            yield mapping, probability


@pytest.mark.parametrize('continue_reach', [F(0), F(1, 4)], ids=['zero-own-reach', 'quarter-own-reach'])
def test_external_sampling_regret_expectation_equals_full_cfr_for_both_players(continue_reach):
    from flyholdem.teacher.self_play_regret import external_sampling_traversal
    strategies = profile(continue_reach)
    exact = full_cfr_by_enumeration(strategies)
    measured = {key: np.zeros(5) for key in strategies}
    for updating in (0, 1):
        total_probability = F(0)
        for chance, chance_probability in enumerate(CHANCE):
            for pure, pure_probability in pure_action_maps(strategies, 1 - updating):
                if updating == 0:
                    key = P1_PRIVATE[chance]
                    draws = [(pure[key], vector(strategies[key]))]
                else:
                    draws = [(pure[key], vector(strategies[key])) for key in (P0_ROOT, P0_LATE)]
                rng = FixedDraws(draws)
                root = HiddenActionGame(chance)
                result = external_sampling_traversal(root, updating, strategy_callback(strategies), rng)
                assert root == HiddenActionGame(chance)
                weight = chance_probability * pure_probability
                total_probability += weight
                assert all(key[0] == updating for key in result.regrets)
                assert all(key[0] != updating for key in result.averages)
                for key, change in result.regrets.items():
                    measured[key] += float(weight) * change
        assert total_probability == 1
    for key, expected in exact.items():
        np.testing.assert_allclose(measured[key], [float(value) for value in expected], atol=1e-13, rtol=0)
    # The later P0 information set gets a nonzero regret even when its own
    # strategy never enters it. A prefix-reach factor would erase that update.
    assert np.any(measured[P0_LATE] != 0)
    counterpart = full_cfr_by_enumeration(profile(F(1, 4) if continue_reach == 0 else F(0)))
    assert exact[P0_LATE] == counterpart[P0_LATE]


@dataclass(frozen=True)
class AverageReachGame:
    """P0 has two decisions; one forced P1 move creates a unit history factor."""
    history: tuple = ()

    def is_terminal(self):
        return self.history == (0,) or len(self.history) == 3

    def terminal_returns_bb(self):
        assert self.is_terminal()
        value = 0 if self.history == (0,) else 2 * self.history[-1] - 1
        return (float(value), float(-value))

    def information_set(self):
        assert not self.is_terminal()
        if not self.history:
            return (*P0_ROOT, TWO_ACTIONS.copy())
        if len(self.history) == 1:
            return (1, b'forced-middle', np.array([True, False, False, False, False]))
        return (*P0_LATE, TWO_ACTIONS.copy())

    def child(self, action):
        assert self.information_set()[2][action]
        return replace(self, history=self.history + (action,))


def test_two_iteration_average_mass_has_single_sampled_own_reach_factor():
    from flyholdem.teacher.self_play_regret import external_sampling_traversal
    accumulated = np.zeros(5)
    iteration_masses = []
    for reach, later_action in ((F(1, 4), 0), (F(3, 4), 1)):
        strategies = {P0_ROOT: (1 - reach, reach), P0_LATE: (int(later_action == 0), int(later_action == 1))}
        expected = np.zeros(5)

        def lookup(role, key, legal):
            if role == 1:
                assert key == b'forced-middle'
                np.testing.assert_array_equal(legal, [True, False, False, False, False])
                return np.array([1., 0., 0., 0., 0.])
            np.testing.assert_array_equal(legal, TWO_ACTIONS)
            return vector(strategies[role, key])

        for pure, probability in pure_action_maps(strategies, 0):
            rng = FixedDraws([(pure[key], vector(strategies[key])) for key in (P0_ROOT, P0_LATE)])
            result = external_sampling_traversal(AverageReachGame(), 1, lookup, rng)
            expected += float(probability) * result.averages.get(P0_LATE, np.zeros(5))
        iteration_masses.append(expected)
        accumulated += expected
    np.testing.assert_array_equal(iteration_masses[0], [.25, 0, 0, 0, 0])
    np.testing.assert_array_equal(iteration_masses[1], [0, .75, 0, 0, 0])
    np.testing.assert_array_equal(accumulated / accumulated.sum(), [.25, .75, 0, 0, 0])
    assert not np.allclose(accumulated[:2] / accumulated.sum(), [.1, .9])  # Squared reach would yield these weights.


def test_hidden_opponent_action_reuses_one_sample_but_counts_each_average_history():
    from flyholdem.teacher.self_play_regret import external_sampling_traversal
    strategies = profile(F(1))
    strategies[P0_LATE] = (F(1, 4), F(3, 4))
    rng = FixedDraws([(1, vector(strategies[P0_ROOT])), (0, vector(strategies[P0_LATE]))])
    result = external_sampling_traversal(HiddenActionGame(0), 1, strategy_callback(strategies), rng)
    assert len(rng.actual) == 2
    # P1's two hidden action branches encounter the same P0 information set.
    # Both use the sampled P0 action zero; one independent draw per branch would
    # violate the sampled pure strategy. Both visits still add average mass.
    assert result.average_visits[P0_LATE] == 2
    np.testing.assert_array_equal(result.averages[P0_LATE], [.5, 1.5, 0, 0, 0])
    assert result.average_visits[P0_ROOT] == 1
    np.testing.assert_array_equal(result.averages[P0_ROOT], [0, 1, 0, 0, 0])
    p1_action_values = np.array([3., -4.])  # Negatives of PAYOFF[0][b][sampled P0 action 0].
    expected = p1_action_values - np.dot(vector(strategies[P1_PRIVATE[0]])[:2], p1_action_values)
    np.testing.assert_allclose(result.regrets[P1_PRIVATE[0]][:2], expected, atol=1e-15, rtol=0)


@dataclass(frozen=True)
class BoundaryProfileGame:
    keys_and_masks: dict
    history: tuple = ()

    def is_terminal(self):
        return len(self.history) == 2

    def terminal_returns_bb(self):
        assert self.is_terminal()
        value = (0, 1, -2, 4, -6)[self.history[0]] * (1 + self.history[1])
        return (float(value), float(-value))

    def information_set(self):
        role = len(self.history)
        key, legal = self.keys_and_masks[role]
        return (role, key, legal.copy())

    def child(self, action):
        assert self.information_set()[2][action]
        return replace(self, history=self.history + (action,))


def test_both_table_passes_use_the_boundary_profile_before_either_update_commits(monkeypatch):
    from flyholdem.poker.engine import Hand
    from flyholdem.teacher.regret import abstraction
    from flyholdem.teacher import self_play_regret as core
    from test_external_regret import config
    settings = config()['abstraction']
    actual = Hand(991300000)
    first_observation = actual.observation()
    actual.act(1)
    second_observation = actual.observation()
    observations = {observation['position']: observation for observation in (first_observation, second_observation)}
    assert set(observations) == {0, 1}
    keys = {role: abstraction(observation, settings) for role, observation in observations.items()}
    table = core.SelfPlayRegretTable(settings)
    initial = {}
    indices = {}
    for role in (0, 1):
        key, legal = keys[role]
        index = table.tables[role].lookup(key, legal)
        values = np.array([0, 1, 2, 3, 4] if role == 0 else [5, 4, 3, 2, 1], dtype=float)
        values[~legal] = 0
        table.tables[role].regrets[index] = values
        indices[role] = index
        initial[role] = values / values.sum()
    before_regrets = table.tables[0].regrets[indices[0]].copy()
    # Only substitute the game adapter. The two real traversal passes, pending
    # table arithmetic, commit and frozen strategy lookups remain production.
    monkeypatch.setattr(core, 'PokerKitNode', lambda game, configuration: BoundaryProfileGame(keys))
    rng = FixedDraws([(0, initial[1]), (2, initial[0])])
    first, second = Hand(991300001), Hand(991300002)
    before_hands = (first.serialize(), second.serialize())
    table.step(first, second, rng)
    assert (first.serialize(), second.serialize()) == before_hands
    assert len(rng.actual) == 2
    after_regrets = table.tables[0].regrets[indices[0]]
    assert not np.array_equal(after_regrets, before_regrets)
    positive = np.maximum(after_regrets, 0)
    assert not np.allclose(positive / positive.sum(), initial[0])
    # The second draw already asserted initial[0], so it cannot have used the
    # materially changed matching probabilities available only after commit.
    np.testing.assert_array_equal(rng.actual[1][1], initial[0])
