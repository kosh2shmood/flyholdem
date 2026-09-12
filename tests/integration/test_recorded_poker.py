"""Recorded training must bind actual information, spikes, RNG and poker actions."""
import copy
import numpy as np
import pytest
from flyholdem.experiments.recorded_poker import audit_recorded_hand
from flyholdem.learning.curriculum import CURRICULA
from flyholdem.learning.poker_rollout import poker_hand
from flyholdem.interface.encoder import encode_player_state, encoded_hash
from test_poker_rollout import make, VisibleTeacher


@pytest.mark.parametrize('curriculum',list(CURRICULA))
@pytest.mark.parametrize('mode,method',[
    ('bio-plastic','terminal-local-eligibility'),
    ('distilled-connectome','teacher-advantage-local-eligibility'),
    ('distilled-connectome','direct-readout-rate-surrogate-v1')])
def test_both_seats_and_methods_replay_without_a_native_or_teacher_rerun(curriculum,mode,method,monkeypatch):
    _,player=make(mode,method);rng=np.random.default_rng(8)
    teacher=VisibleTeacher() if player.requires_teacher else None
    records=[poker_hand(player,curriculum,'calling-station',21101,seat,learning=True,teacher=teacher,temperature=.35)
        for seat in (0,1)]
    def forbidden(*args,**kwargs):raise AssertionError('Audit must not rerun native activity or teacher inference')
    monkeypatch.setattr(type(player.brain),'advance',forbidden)
    monkeypatch.setattr(VisibleTeacher,'probabilities',forbidden)
    for row in records:
        result=audit_recorded_hand(row,player.controller.registration,player.brain.n,rng,.35)
        assert result['native_decisions']==len(row['neural_decisions'])
        assert result['pokerkit_trajectory_verified'] and not result['historical_neural_execution_repeated']
    assert rng.bit_generator.state==player.controller.rng.bit_generator.state


def test_changed_observation_even_with_reencoded_hash_is_rejected():
    _,player=make();row=poker_hand(player,'hu-20bb-v1','calling-station',21103,0,learning=True,temperature=.35)
    bad=copy.deepcopy(row);d=bad['neural_decisions'][0]
    d['observation']['own_stack']+=1
    d['encoded_hash']=encoded_hash(encode_player_state(d['observation']))
    with pytest.raises(ValueError,match='actual visible input'):
        audit_recorded_hand(bad,player.controller.registration,player.brain.n,np.random.default_rng(8),.35)
    bad=copy.deepcopy(row);bad['neural_decisions'][0]['observation']['teacher_target']=[1,0,0,0,0]
    with pytest.raises(ValueError,match='actual visible input'):
        audit_recorded_hand(bad,player.controller.registration,player.brain.n,np.random.default_rng(8),.35)


def test_changed_sampling_commitment_or_missing_records_are_rejected():
    _,player=make();row=poker_hand(player,'hu-20bb-v1','calling-station',21103,0,learning=True,temperature=.35)
    assert row['neural_decisions']
    bad=copy.deepcopy(row);d=bad['neural_decisions'][0]
    d['selected']=next(i for i,legal in enumerate(d['legal_mask']) if legal and i!=d['selected'])
    with pytest.raises(ValueError,match='action RNG'):
        audit_recorded_hand(bad,player.controller.registration,player.brain.n,np.random.default_rng(8),.35)
    bad=copy.deepcopy(row);bad['neural_decisions'][0]['committed_action']['paid']+=1
    with pytest.raises(ValueError,match='PokerKit commitment'):
        audit_recorded_hand(bad,player.controller.registration,player.brain.n,np.random.default_rng(8),.35)
    bad=copy.deepcopy(row);bad['neural_decisions'].pop(0)
    with pytest.raises(ValueError):audit_recorded_hand(bad,player.controller.registration,player.brain.n,np.random.default_rng(8),.35)
    with pytest.raises(ValueError,match='actual visible input'):
        audit_recorded_hand(row,player.controller.registration,player.brain.n,np.random.default_rng(8),0)
    bad=copy.deepcopy(row);bad['private_hand_checkpoint']['hand']['initial_deck']=''.join(reversed(bad['private_hand_checkpoint']['hand']['initial_deck']))
    with pytest.raises(ValueError,match='PokerKit settlement'):
        audit_recorded_hand(bad,player.controller.registration,player.brain.n,np.random.default_rng(8),.35)


def test_frozen_snapshot_opponent_has_its_own_recorded_visible_argmax():
    from flyholdem.learning.frozen_opponent import FrozenNeuralOpponent
    from flyholdem.provenance import identity
    opponent_controller,_=make();opponent=FrozenNeuralOpponent(opponent_controller,'a'*64)
    _,player=make();row=poker_hand(player,'hu-20bb-v1',opponent,21107,1,learning=True,temperature=.35)
    audit_recorded_hand(row,player.controller.registration,player.brain.n,np.random.default_rng(8),.35)
    assert row['frozen_opponent']['decisions']
    bad=copy.deepcopy(row);snapshot=bad['frozen_opponent'];snapshot['decisions'][0]['observation_sha256']='b'*64
    snapshot['decisions_sha256']=identity(snapshot['decisions'])
    with pytest.raises(ValueError,match='visible legal argmax'):
        audit_recorded_hand(bad,player.controller.registration,player.brain.n,np.random.default_rng(8),.35)
