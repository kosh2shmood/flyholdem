"""Actual PokerKit arithmetic and atomic two-pass numeric state boundaries."""
import copy
import json

import numpy as np
import pytest
from pokerkit import Deck, StandardHighHand

from flyholdem.poker.engine import Hand
from flyholdem.provenance import canonical
from flyholdem.teacher import self_play_regret as core
from flyholdem.teacher.regret import VERSION, abstraction


def settings(stack_bb=1, **changes):
    return {'feature_version': VERSION, 'stack_bb': stack_bb, 'equity_buckets': 8,
            'equity_samples': 2, 'max_information_sets': 10000,
            'max_nodes_per_traversal': 10000, **changes}


def rigged_hand(holes, board, button=0):
    used = [card for pair in holes for card in pair] + board
    rest = [repr(card) for card in Deck.STANDARD if repr(card) not in used]
    prefix = [pair[index] for index in (0, 1) for pair in holes]
    prefix += [rest.pop(0)] + board[:3] + [rest.pop(0), board[3], rest.pop(0), board[4]]
    return Hand(991700000, button=button, stacks=(2, 2), deck=''.join(prefix + rest))


def assert_state_equal(first, second):
    assert first.keys() == second.keys()
    for name in first:
        np.testing.assert_array_equal(first[name], second[name], err_msg=name)


@pytest.mark.parametrize('button', [0, 1])
@pytest.mark.parametrize('holes,board,showdown_roles', [
    ([['2c', '3c'], ['2d', '3d']], ['Ts', 'Js', 'Qs', 'Ks', 'As'], [0., 0.]),
    ([['Ac', 'Ad'], ['Kc', 'Kd']], ['2c', '3h', '7s', '9d', 'Jc'], [1., -1.]),
])
def test_actual_blind_all_in_arithmetic_folds_ties_and_public_roles(button, holes, board, showdown_roles):
    # With one BB initial stacks, the nonbutton is already all-in. The button
    # has exactly fold/call, so both genuine terminal values are inspectable.
    first = rigged_hand(holes, board, button)
    second = rigged_hand(holes, board, button)
    original = canonical(first.serialize())
    node = core.PokerKitNode(first, settings())
    role, key, legal = node.information_set()
    assert role == 1
    np.testing.assert_array_equal(legal, [True, True, False, False, False])
    fold, call = node.child(0), node.child(1)
    assert fold.is_terminal() and call.is_terminal()
    np.testing.assert_array_equal(fold.terminal_returns_bb(), [.5, -.5])
    np.testing.assert_array_equal(call.terminal_returns_bb(), showdown_roles)
    assert canonical(first.serialize()) == original
    with pytest.raises(ValueError):
        node.child(4)

    table = core.SelfPlayRegretTable(settings())
    row = table.step(first, second, np.random.default_rng(912300))
    expected_values = np.array([-.5, showdown_roles[1]])
    stored = table.tables[1]
    index = stored.index[key]
    np.testing.assert_array_equal(stored.regrets[index, legal], expected_values - expected_values.mean())
    np.testing.assert_array_equal(stored.averages[index], [.5, .5, 0, 0, 0])
    assert stored.visits[index] == stored.average_visits[index] == 1
    assert len(table.tables[0].keys) == 0  # The BB never has a decision.
    assert row['information_sets_by_role'] == [0, 1]
    assert row['traversals'][1]['sampled_policy_value_bb'] == expected_values.mean()
    assert [item['sampled_action_draws'] for item in row['traversals']] == [1, 0]
    assert table.iterations_completed == row['iteration'] == 1
    assert canonical(first.serialize()) == canonical(second.serialize()) == original
    assert first.state.hand_types == second.state.hand_types == (StandardHighHand,)
    # The displayed average is distinct from the new regret-matching iterate.
    np.testing.assert_array_equal(table.probabilities(first.observation()), [.5, .5, 0, 0, 0])
    assert not np.array_equal(table.strategy_for(role, key, legal), table.probabilities(first.observation()))


def test_actual_numpy_draw_ledger_matches_pcg64_advancement_and_numeric_round_trip():
    config = settings(2)
    table = core.SelfPlayRegretTable(config)
    rng = np.random.default_rng(912310)
    comparison = np.random.default_rng()
    comparison.bit_generator.state = copy.deepcopy(rng.bit_generator.state)
    originals = [Hand(991700010 + i, stacks=(4, 4)) for i in range(4)]
    rows = [table.step(originals[i], originals[i + 1], rng) for i in (0, 2)]
    draws = sum(part['sampled_action_draws'] for row in rows for part in row['traversals'])
    assert draws > 0
    assert draws == sum(row['sampled_action_draws'] for row in rows)
    comparison.random(draws)
    assert rng.bit_generator.state == comparison.bit_generator.state
    assert all(not hand.history and not hand.done for hand in originals)
    assert all(len(role.keys) > 0 for role in table.tables)
    state = table.state()
    restored = core.SelfPlayRegretTable(config)
    restored.restore(state)
    assert_state_equal(state, restored.state())
    assert restored.iterations_completed == 2
    for role, stored in enumerate(restored.tables):
        for key, index in stored.index.items():
            assert json.loads(key)[3][0] == role
            np.testing.assert_allclose(stored.averages[index].sum(), stored.average_visits[index], rtol=1e-13, atol=1e-13)
    next_state = copy.deepcopy(rng.bit_generator.state)
    next_rng = np.random.default_rng()
    next_rng.bit_generator.state = next_state
    pair = [Hand(991700020 + i, stacks=(4, 4)) for i in (0, 1)]
    assert table.step(*pair, rng) == restored.step(*pair, next_rng)
    assert_state_equal(table.state(), restored.state())
    assert rng.bit_generator.state == next_rng.bit_generator.state
    # All externally returned arrays are copies.
    state['role0_regrets'][:] = 123
    assert not np.array_equal(state['role0_regrets'], restored.state()['role0_regrets'])


def test_second_pass_failure_rolls_back_existing_tables_rng_and_all_roots(monkeypatch):
    config = settings(2)
    table = core.SelfPlayRegretTable(config)
    rng = np.random.default_rng(912320)
    table.step(Hand(991700030, stacks=(4, 4)), Hand(991700031, stacks=(4, 4)), rng)
    prior = table.state()
    prior_rng = copy.deepcopy(rng.bit_generator.state)
    pair = [Hand(991700032 + i, stacks=(4, 4)) for i in (0, 1)]
    private = [canonical(hand.serialize()) for hand in pair]
    original = core.external_sampling_traversal

    def fail_second(root, updating_role, strategy_for, sampler, **kwargs):
        if updating_role == 1:
            sampler.random()  # Failure after a changed RNG, not before work.
            raise RuntimeError('injected incomplete second pass')
        return original(root, updating_role, strategy_for, sampler, **kwargs)

    monkeypatch.setattr(core, 'external_sampling_traversal', fail_second)
    with pytest.raises(RuntimeError, match='incomplete second pass'):
        table.step(*pair, rng)
    assert_state_equal(prior, table.state())
    assert rng.bit_generator.state == prior_rng
    assert [canonical(hand.serialize()) for hand in pair] == private
    monkeypatch.setattr(core, 'external_sampling_traversal', original)
    replay = core.SelfPlayRegretTable(config)
    replay.restore(prior)
    replay_rng = np.random.default_rng()
    replay_rng.bit_generator.state = prior_rng
    assert table.step(*pair, rng) == replay.step(*pair, replay_rng)
    assert_state_equal(table.state(), replay.state())


def test_failed_sparse_insertion_restores_spare_rows_and_key_order(monkeypatch):
    table = core.SelfPlayRegretTable(settings())
    game = rigged_hand([['2c', '3c'], ['2d', '3d']], ['Ts', 'Js', 'Qs', 'Ks', 'As'])
    table.step(game, game, np.random.default_rng(912330))
    prior = table.state()
    stored = table.tables[1]
    key, legal = abstraction(game.observation(), table.config)
    value = json.loads(key)
    value[2][0] = (value[2][0] + 1) % table.config['equity_buckets']
    new_key = canonical(value)
    token = 1, new_key
    assert new_key not in stored.index and stored.capacity > len(stored.keys)
    staged = core.TraversalResult(legal={token: legal}, regrets={token: np.zeros(5)},
                                  averages={token: legal / legal.sum()},
                                  regret_visits={token: 1}, average_visits={token: 1})
    lookup = stored.lookup
    spare_index = len(stored.keys)

    def fail_after_insertion(new, mask):
        index = lookup(new, mask)
        stored.regrets[index, mask] = 999.
        raise MemoryError('injected interrupted insertion')

    monkeypatch.setattr(stored, 'lookup', fail_after_insertion)
    with pytest.raises(MemoryError, match='interrupted insertion'):
        table._commit([staged, core.TraversalResult()])
    assert_state_equal(table.state(), prior)
    assert new_key not in stored.index
    np.testing.assert_array_equal(stored.regrets[spare_index], np.zeros(5))
    np.testing.assert_array_equal(stored.legal[spare_index], np.zeros(5, dtype=bool))
    monkeypatch.setattr(stored, 'lookup', lookup)
    table._commit([staged, core.TraversalResult()])
    assert stored.index[new_key] == spare_index
    np.testing.assert_array_equal(stored.regrets[spare_index], np.zeros(5))
    assert stored.visits[spare_index] == stored.average_visits[spare_index] == 1


@pytest.mark.parametrize('constraint,message', [
    ({'max_nodes_per_traversal': 1}, 'node ceiling'),
    ({'max_information_sets': 1}, 'Combined self-play'),
])
def test_node_and_combined_capacity_limits_abort_without_partial_learning(constraint, message):
    table = core.SelfPlayRegretTable(settings(2, **constraint))
    rng = np.random.default_rng(912340)
    prior, prior_rng = table.state(), copy.deepcopy(rng.bit_generator.state)
    with pytest.raises((MemoryError, RuntimeError), match=message):
        table.step(Hand(991700040, stacks=(4, 4)), Hand(991700041, stacks=(4, 4)), rng)
    assert_state_equal(table.state(), prior)
    assert rng.bit_generator.state == prior_rng


def test_restore_rejects_role_relabeling_and_inconsistent_visit_evidence_atomically():
    table = core.SelfPlayRegretTable(settings())
    game = rigged_hand([['2c', '3c'], ['2d', '3d']], ['Ts', 'Js', 'Qs', 'Ks', 'As'])
    table.step(game, game, np.random.default_rng(912350))
    good = table.state()
    bad = {name: value.copy() for name, value in good.items()}
    for name in core._TABLE_FIELDS:
        bad['role0_' + name], bad['role1_' + name] = bad['role1_' + name], bad['role0_' + name]
    with pytest.raises(ValueError, match='public role'):
        table.restore(bad)
    assert_state_equal(good, table.state())
    bad = {name: value.copy() for name, value in good.items()}
    bad['role1_average_visits'][0] += 1
    with pytest.raises(ValueError, match='average mass'):
        table.restore(bad)
    assert_state_equal(good, table.state())
    with pytest.raises(ValueError, match='Complete two-role'):
        table.restore({})
    assert_state_equal(good, table.state())


@pytest.mark.parametrize('kind', ['unequal-stacks', 'three-players', 'started', 'wrong-stack'])
def test_step_enforces_the_ordinary_fresh_equal_stack_heads_up_recall_domain(kind):
    first = Hand(991700050, stacks=(4, 4))
    if kind == 'unequal-stacks':
        first = Hand(991700050, stacks=(3, 5))
    elif kind == 'three-players':
        first = Hand(991700050, stacks=(4, 4, 4))
    elif kind == 'started':
        first.act(1)
    elif kind == 'wrong-stack':
        first = Hand(991700050)
    table = core.SelfPlayRegretTable(settings(2))
    prior = table.state()
    with pytest.raises(ValueError, match='Fresh ordinary equal-stack heads-up'):
        table.step(first, Hand(991700051, stacks=(4, 4)), np.random.default_rng(912360))
    assert_state_equal(prior, table.state())
