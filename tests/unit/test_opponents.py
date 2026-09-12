from flyholdem.poker.engine import Hand
from flyholdem.poker.opponents import Opponent, visible_equity
from flyholdem.experiments.software_gate import match


def test_fixed_opponents_legal_and_deterministic():
    for kind in ('random','calling-station','tight-aggressive','equity-bucket'):
        for seed in range(3):
            h=Hand(seed);observation=h.observation()
            a=Opponent(seed,kind).act(observation)
            assert observation['legal_mask'][a]
            assert a==Opponent(seed,kind).act(observation)


def test_visible_monte_carlo_ignores_engine_hidden_state():
    h=Hand();observation=h.observation()
    a=visible_equity(observation,samples=8)
    h.state.hole_cards[0],h.state.hole_cards[1]=h.state.hole_cards[1],h.state.hole_cards[0]
    assert visible_equity(observation,samples=8)==a


def test_exact_paired_seat_swap():
    for seed in range(20):
        assert match(seed,0,[seed+100,seed+200])==-match(seed,1,[seed+200,seed+100])
