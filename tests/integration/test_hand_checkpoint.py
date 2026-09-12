import json
import random
import pytest
from flyholdem.poker.engine import Hand
from flyholdem.poker.observation import canonical_bytes


def test_private_checkpoint_continuation_at_every_decision():
    for seed in range(30):
        h=Hand(seed,seed%2)
        rng=random.Random(seed+100)
        while not h.done:
            restored=Hand.restore(json.loads(json.dumps(h.serialize())))
            assert canonical_bytes(restored.observation())==canonical_bytes(h.observation())
            assert list(restored.state.deck_cards)==list(h.state.deck_cards)
            action=rng.choice([i for i,legal in enumerate(h.legal_mask()) if legal])
            h.act(action);restored.act(action)
            assert restored.view()==h.view()
        assert Hand.restore(h.serialize()).view()==h.view()


def test_checkpoint_corruption_and_observation_payload_rejected():
    h=Hand()
    with pytest.raises(ValueError):Hand.restore(h.observation())
    data=h.serialize();data['view_sha256']='wrong'
    with pytest.raises(ValueError,match='mismatch'):Hand.restore(data)
