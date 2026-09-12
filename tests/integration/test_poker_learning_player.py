import copy
import numpy as np
import pytest
from flyholdem.interface.population import NeuralController
from flyholdem.neural.sparse import SparseBrain
from flyholdem.learning.eligibility import Eligibility
from flyholdem.learning.poker_player import PokerLearningPlayer
from flyholdem.poker.engine import Hand


def make(mode='bio-plastic',method='terminal-local-eligibility'):
    b=SparseBrain([0,5,10,10,10,10,10,10],np.tile(np.arange(2,7),2),np.linspace(3,8,10))
    p={'mode':'fixture-native','graph_hash':b.graph_hash,'input_indices':[0,1],
        'projection_indices':[[i%2] for i in range(237)],'selected_gain':12,'baseline_hz':[0]*5,
        'ensembles':[{'indices':[i]} for i in range(2,7)],
        'config':{'decision_ms':{'baseline':50,'stimulus':300,'readout':100,'inter_decision':50},'score_scale_hz':100}}
    controller=NeuralController(b,p)
    eligible=Eligibility(b,[0,1],[2,3,4,5,6],dict(pre_tau_ms=20,eligibility_tau_ms=5000,coincidence_scale=.001,bin_ms=5,learning_rate=.3,weight_bounds=[.1,2]))
    config={'learning_mode':mode,'optimization':method,'reward_scale_bb':20,'dopamine':{'pulse_ms':200,'gain':12},
        'surrogate':{'learning_rate':.1,'temperature':1.,'gradient_norm_cap':1.,'synaptic_tau_ms':5.,'threshold_distance_mv':7.}}
    return controller,PokerLearningPlayer(controller,eligible,(np.array([0]),np.array([1])),config,seed=8)


def equal_decision(a,b):
    assert set(a)==set(b)
    for key in a:
        if isinstance(a[key],np.ndarray):assert a[key].dtype==b[key].dtype and a[key].tobytes()==b[key].tobytes()
        else:assert a[key]==b[key]


def test_observer_preserves_every_actual_frozen_neural_decision_byte():
    plain,_=make();_,observed=make();hand=Hand(752)
    observed.begin_hand()
    while not hand.done:
        obs=hand.observation();a=plain.decide(obs);b=observed.decide(obs)
        equal_decision(a,b);plain.commit_interval();observed.commit_action()
        hand.act(1)
    assert np.array_equal(plain.brain.weights,observed.brain.weights)


@pytest.mark.parametrize('mode,method',[('bio-plastic','terminal-local-eligibility'),('distilled-connectome','teacher-advantage-local-eligibility'),('distilled-connectome','direct-readout-rate-surrogate-v1')])
def test_learning_state_recovery_is_exact_and_all_updates_stay_bounded(mode,method):
    _,player=make(mode,method);player.begin_hand();observation=Hand(765).observation()
    player.decide(observation)
    with pytest.raises(ValueError):player.state()
    target=[0,1,0,0,0] if mode=='distilled-connectome' else None
    player.commit_action(True,target)
    player.finish_hand(4,1,True)
    arrays,extra=player.state();_,restored=make(mode,method);restored.restore_state(arrays,extra)
    a=player.decide(observation,.2);b=restored.decide(observation,.2);equal_decision(a,b)
    assert player.commit_action(True,target)==restored.commit_action(True,target)
    assert player.finish_hand(-2,0,True)==restored.finish_hand(-2,0,True)
    pa,pe=player.state();ra,re=restored.state();assert pe==re
    assert all(np.array_equal(pa[k],ra[k]) for k in pa)
    ratios=player.brain.weights/player.brain.initial
    assert np.all(ratios>=.1-1e-6) and np.all(ratios<=2+1e-6)
    assert np.any(player.brain.weights!=player.brain.initial)


def test_frozen_or_bioplastic_player_rejects_teaching_inputs():
    for mode,method in [('bio-plastic','terminal-local-eligibility'),('distilled-connectome','direct-readout-rate-surrogate-v1')]:
        _,player=make(mode,method);player.begin_hand();before=player.brain.weights.copy()
        player.decide(Hand(71).observation())
        with pytest.raises(ValueError):player.commit_action(False,[0,1,0,0,0])
        assert np.array_equal(before,player.brain.weights)
