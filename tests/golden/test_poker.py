import random
import pytest
from flyholdem.poker.engine import Hand
from flyholdem.poker.actions import translations


def test_heads_up_blinds_button_and_postflop_order():
    for button in (0, 1):
        h = Hand(seed=42, button=button)
        assert h.actor == button
        assert h.view()['bets'][button] == 1
        assert h.view()['bets'][1-button] == 2
        h.act(1)
        assert h.actor == 1-button
        assert h.observation()['to_call'] == 0
        assert not h.legal_mask()[0]
        h.act(1)
        assert h.state.street_index == 1
        assert h.actor == 1-button
        assert len(h.observation()['board']) == 3


def test_golden_pot_after_call_and_reraise():
    h = Hand()
    assert translations(h.state) == [('fold', 0), ('call', 1), ('raise', 4), ('raise', 6), ('raise', 40)]
    h.act(3)
    assert h.observation()['pot'] == 8
    assert h.observation()['to_call'] == 4
    assert translations(h.state)[2:4] == [('raise', 12), ('raise', 18)]
    assert h.history[0]['paid'] == 5
    assert h.history[0]['wager_bb'] == 2.5


def test_duplicate_all_in_mask_and_short_call():
    h = Hand(stacks=(4, 40))
    assert h.legal_mask() == [True, True, True, False, False]
    h.act(2)
    # Raise to 4 is all-in for the button. The other player can call; excess is returned.
    h.act(1)
    assert h.done
    assert sum(h.view()['stacks']) == 44
    assert abs(h.view()['payoffs'][0]) <= 4
    assert sum(h.view()['payoffs']) == 0


def test_fold_and_illegal_action_leave_state_unchanged():
    h = Hand()
    h.act(0)
    assert h.done
    assert h.view()['payoffs'] == [-1, 1]
    before = h.view()
    with pytest.raises(ValueError):
        h.act(4)
    assert h.view() == before


def test_many_random_hands_conserve_chips_and_replay():
    for seed in range(100):
        h = Hand(seed, seed % 2)
        replay = Hand(seed, seed % 2)
        rng = random.Random(seed)
        decisions = 0
        while not h.done:
            assert h.observation() == replay.observation()
            choices = translations(h.state)
            available = [a for a in choices if a is not None]
            assert len(set(available)) == len(available)
            action = rng.choice([i for i, a in enumerate(choices) if a is not None])
            h.act(action)
            replay.act(action)
            decisions += 1
            assert decisions < 100
        assert h.view() == replay.view()
        assert sum(h.view()['stacks']) == 80
        assert sum(h.view()['payoffs']) == 0
