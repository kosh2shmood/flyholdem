import copy
import random
import numpy as np
import pytest
from flyholdem.poker.engine import Hand
from flyholdem.poker.infoset import canonical_state, canonical_information_id
from flyholdem.interface.encoder import encode


def river_hand(seed=14):
    hand = Hand(seed)
    while hand.observation()['street'] < 3:
        hand.act(1)
    return hand


def test_canonical_visible_hand_orders_and_complete_history():
    obs = river_hand().observation()
    permutation = copy.deepcopy(obs)
    permutation['hole'].reverse(); permutation['board'][:3] = reversed(permutation['board'][:3])
    assert canonical_information_id(obs) == canonical_information_id(permutation)
    assert encode(obs).tobytes() == encode(canonical_state(permutation)).tobytes()
    changed = copy.deepcopy(obs); changed['board'][3:] = reversed(changed['board'][3:])
    assert canonical_information_id(changed) != canonical_information_id(obs)
    changed = copy.deepcopy(obs); changed['history'][0]['wager_bb'] += 1
    assert canonical_information_id(changed) != canonical_information_id(obs)


def test_canonical_allowlist_rejects_hidden_fields_and_malformed_targets():
    obs = Hand().observation()
    for key in ('teacher', 'opponent_hole', 'deck', 'future_cards', 'equity'):
        with pytest.raises(ValueError): canonical_state(dict(obs, **{key: []}))
    for mutate in (lambda o: o['legal_mask'].clear(), lambda o: o['hole'].append(o['hole'][0]),
                   lambda o: o.update(spr=float('nan')), lambda o: o.update(own_stack=True)):
        changed = copy.deepcopy(obs); mutate(changed)
        with pytest.raises(ValueError): canonical_state(changed)


def test_real_pokerkit_states_preserve_encoded_information():
    rng = random.Random(912)
    for seed in range(40):
        hand = Hand(seed)
        while not hand.done:
            obs = hand.observation(); normalized = canonical_state(obs)
            assert encode(obs).tobytes() == encode(normalized).tobytes()
            assert canonical_information_id(obs) == canonical_information_id(normalized)
            hand.act(rng.choice([i for i, legal in enumerate(obs['legal_mask']) if legal]))


def test_native_first_action_bit_uses_only_public_relative_history():
    from flyholdem.interface.encoder import encode_player_state
    hand = Hand(99)
    assert encode_player_state(hand.observation())[-1] == 1
    hand.act(1)
    assert hand.observation()['history'] and all(h['actor'] == 0 for h in hand.observation()['history'])
    assert encode_player_state(hand.observation())[-1] == 1
    hand.act(1)
    assert any(h['actor'] == 1 for h in hand.observation()['history'])
    assert encode_player_state(hand.observation())[-1] == 0
