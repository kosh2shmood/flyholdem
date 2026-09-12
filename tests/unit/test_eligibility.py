import numpy as np
import pytest
from flyholdem.learning.eligibility import Eligibility, existing_edges
from flyholdem.learning.dopamine import PastBaseline
from flyholdem.neural.sparse import SparseBrain

CONFIG = dict(pre_tau_ms=20, eligibility_tau_ms=5000, coincidence_scale=.001,
              bin_ms=5, learning_rate=.1, weight_bounds=[.1, 2])


def brain():
    return SparseBrain([0, 2, 4, 4], [1, 2, 0, 2], [.2, -.3, .4, .5])


def test_updates_only_existing_eligible_edges_and_preserves_signs():
    b = brain()
    e = Eligibility(b, [0, 1], [2], CONFIG)
    assert e.edges.tolist() == [1, 3]
    assert existing_edges(b.ptr, b.post, [1, 0], [2])[0].tolist() == [3, 1]
    e.observe([2, 0, 3], 5)
    assert e.trace[0] > 0 and e.trace[1] == 0
    old = b.weights.copy()
    e.reinforce(1)
    assert b.weights[1] < old[1] and np.array_equal(b.weights[[0, 2, 3]], old[[0, 2, 3]])
    for dose in (1e9, -1e9):
        e.reinforce(dose)
        ratios = b.weights[e.edges] / b.initial[e.edges]
        assert np.all(ratios >= .1) and np.all(ratios <= 2)
        assert np.array_equal(np.sign(b.weights), np.sign(b.initial))
    e.reinforce(1, learning=False)
    frozen = b.weights.copy()
    e.reinforce(-1, learning=False)
    assert np.array_equal(b.weights, frozen)


def test_trace_decay_and_checkpoint_continue_exactly():
    b = brain(); e = Eligibility(b, [0, 1], [2], CONFIG)
    e.observe([2, 1, 3], 5)
    prior = e.state(); e.observe([0, 0, 0], 10)
    np.testing.assert_array_equal(e.trace, prior['eligibility_trace'] * np.exp(-10 / 5000))
    b2 = brain(); e2 = Eligibility(b2, [0, 1], [2], CONFIG)
    e2.restore_state(e.state())
    for obj in (e, e2):
        obj.observe([1, 2, 2], 5); obj.reinforce(-.3)
    assert np.array_equal(b.weights, b2.weights)
    assert np.array_equal(e.trace, e2.trace)
    with pytest.raises(ValueError): e2.restore_state({})
    with pytest.raises(ValueError): e.reinforce(float('nan'))


def test_native_binning_preserves_dynamics_and_reset_preserves_weights():
    a = brain(); b = brain(); e = Eligibility(b, [0, 1], [2], CONFIG)
    drive = np.array([12, 10, 0], dtype=np.float32)
    assert np.array_equal(a.advance(drive, 100), e.advance(drive, 100))
    # Native lazy-state rounding depends on measurement boundaries. Counts stay
    # exact here; the registered 5 ms schedule is part of the learning identity.
    np.testing.assert_allclose(a.v, b.v, rtol=0, atol=1e-4)
    np.testing.assert_allclose(a.g, b.g, rtol=0, atol=1e-4)
    e.reinforce(1)
    current = b.weights.copy(); b.reset_dynamics()
    fresh = brain(); fresh.weights[:] = current
    for key in b.state_names:
        assert np.array_equal(getattr(b, key), getattr(fresh, key)), key
    assert np.array_equal(b.advance(drive, 100), fresh.advance(drive, 100))


def test_reward_baseline_is_past_only_context_specific_and_restorable():
    a = PastBaseline()
    first = a.event(10, '20bb/button', scale=10)
    assert first['baseline'] == 0 and first['dopamine'] == np.tanh(1)
    second = a.event(-10, '20bb/button', scale=10)
    assert second['baseline'] == first['normalized_reward']
    assert a.event(3, '20bb/blind')['past_count'] == 0
    b = PastBaseline(); b.restore_state(a.state())
    assert a.event(2, '20bb/button') == b.event(2, '20bb/button')
