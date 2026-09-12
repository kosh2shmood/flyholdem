from collections import deque
import random
import numpy as np
from flyholdem.poker.engine import Hand
from flyholdem.poker.observation import canonical_bytes
from flyholdem.interface.encoder import CHANNELS, encode, projection


def test_hidden_cards_future_deck_and_teacher_are_byte_invariant():
    for seed in range(12):
        h = Hand(seed)
        before = canonical_bytes(h.observation())
        encoded = encode(h.observation()).tobytes()
        # Swap hidden opponent holes with undealt cards, maintaining deck uniqueness.
        actor = h.state.actor_index
        hidden = 1-actor
        deck = list(h.state.deck_cards)
        prior = list(h.state.hole_cards[hidden])
        h.state.hole_cards[hidden] = deck[:2]
        deck = deck[2:] + prior
        random.Random(seed+90).shuffle(deck)
        h.state.deck_cards = deque(deck)
        h.state.teacher_logits = [1000, -999, 2, 3, 4]
        h.state.opponent_policy_internal = {'secret': 'not an input'}
        assert canonical_bytes(h.observation()) == before
        assert encode(h.observation()).tobytes() == encoded


def test_input_projection_nonempty_balanced_and_frozen():
    mapping = projection()
    assert mapping.shape == (len(CHANNELS), 2)
    assert np.array_equal(mapping, projection())
    counts = np.bincount(mapping.ravel(), minlength=96)
    assert counts.min() > 0 and counts.max()-counts.min() <= 1
    assert len(CHANNELS) == len(set(CHANNELS))


def test_logging_counters_never_enter_encoder():
    h = Hand()
    obs = h.observation()
    before = encode(obs).tobytes()
    obs.update(hand_number=12345, decision_count=555, source_seed=987)
    assert encode(obs).tobytes() == before
