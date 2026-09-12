import numpy as np
import pytest
from flyholdem.interface.encoder import CHANNELS,encode
from flyholdem.interface.population import balanced_projection,population_drive,neural_scores,select_action
from flyholdem.interface.preregister import connectivity_ensembles,control_patterns
from flyholdem.poker.engine import Hand


def test_balanced_population_and_hidden_information_bytes():
    inputs=np.arange(4064);mapping=balanced_projection(inputs,32,1729)
    assert mapping.shape==(len(CHANNELS),32)
    assert all(len(set(row))==32 for row in mapping)
    counts=np.bincount(mapping.ravel());assert counts.max()-counts.min()<=1 and counts.min()>0
    hand=Hand(129);obs=hand.observation();changed=dict(obs,opponent_hole=['As','Ah'],future_cards=['Ac'],teacher=[1,0,0,0,0])
    assert population_drive(encode(obs),mapping,5000,12).tobytes()==population_drive(encode(changed),mapping,5000,12).tobytes()


def test_connectivity_selection_and_paths_ignore_column_order_metadata():
    # Five independent input groups; three reachable existing readouts each.
    weights=np.zeros((50,15))
    for action in range(5):weights[action*10:(action+1)*10,action*3:(action+1)*3]=np.arange(1,11)[:,None]
    ensembles,groups,criteria=connectivity_ensembles(weights,np.arange(100,115),3,8,1729)
    assert len(set(np.concatenate(ensembles)))==15
    patterns=control_patterns(weights,groups,8)
    for group,pattern in zip(groups,patterns):assert (weights[np.ix_(pattern,group)]>0).all()
    assert len(criteria['anchors'])==5


def test_decoder_is_neural_only_masked_and_silent_without_strategy_fallback():
    groups=[np.array([i]) for i in range(5)];rng=np.random.default_rng(8)
    rates,scores=neural_scores(np.array([0,1,2,3,4]),groups,100)
    assert rates.tolist()==[0,10,20,30,40]
    assert select_action(scores,[True,True,True,True,False],rng)==3
    assert select_action(np.zeros(5),[False,False,True,True,True],rng)==2
    with pytest.raises(ValueError):select_action([np.nan]*5,[True]*5,rng)
    with pytest.raises(ValueError):neural_scores(np.ones(5),[np.array([0])]*5,100)


def test_global_gain_sensitivity_preserves_every_edge_and_sign():
    from flyholdem.interface.population import scaled_weights
    initial=np.array([.275,-.55,9.625,0],dtype=np.float32)
    before=initial.tobytes();result=scaled_weights(initial,.5)
    assert initial.tobytes()==before
    assert np.array_equal(result,np.array([.1375,-.275,4.8125,0],dtype=np.float32))
    assert np.array_equal(np.sign(result),np.sign(initial))
    for scale in [0,-1,np.nan,np.inf]:
        with pytest.raises(ValueError):scaled_weights(initial,scale)
