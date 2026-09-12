from types import SimpleNamespace
import copy
import numpy as np
import pytest
from flyholdem.teacher.exact_cue import ExactCueTeacher
from flyholdem.interface.cue_player import CuePlayer
from flyholdem.learning.eligibility import Eligibility
from flyholdem.neural.sparse import SparseBrain


def test_exact_teacher_exhaustive_advantage_is_a_visible_token_function():
    teacher = ExactCueTeacher([0, 1])
    assert teacher.registration['exhaustively_verified_states'] == 2
    for cue in (0, 1):
        assert teacher.advantage(cue, cue) == 1
        assert teacher.advantage(cue, 1-cue) == -1
    p = teacher.probabilities(0); p[:] = 0
    assert teacher.probabilities(0).sum() == 1
    with pytest.raises(ValueError):
        teacher.probabilities({'cue': 0, 'hidden_cards': ['As', 'Ah']})
    with pytest.raises(ValueError):
        teacher.advantage(0, 4)


def test_cue_inference_matches_learning_observer_without_labels():
    brain = SparseBrain([0,5,10,10,10,10,10,10], np.tile(np.arange(2,7),2), np.linspace(.3,1,10))
    registration = {'input_indices':[0,1], 'selected_gain':12,
        'config':{'decision_ms':{'baseline':50,'stimulus':300,'readout':100},'score_scale_hz':100},
        'baseline_hz':[0]*5}
    controller = SimpleNamespace(brain=brain, registration=registration, blank=np.zeros(7,dtype=np.float32),
                                 ensembles=[np.array([i]) for i in range(2,7)])
    sensory = {'schema':'native-cue-input-v1','cells':[[0],[1]],'actions':[0,1],
               'amplitude_jitter':.15,'bin_ms':5}
    player = CuePlayer(controller, sensory)
    eligibility = Eligibility(brain,[0,1],list(range(2,7)),{'pre_tau_ms':20,'eligibility_tau_ms':5000,
        'coincidence_scale':.001,'bin_ms':5,'learning_rate':.3,'weight_bounds':[.1,2]})
    for cue in (0,1):
        observed = player.decide(cue, 74101, eligibility)
        frozen = player.decide(cue, 74101)
        assert observed == frozen
        assert frozen['selected'] == int(np.argmax(np.where([True,True,False,False,False],frozen['scores'],-np.inf)))
    bad = copy.deepcopy(sensory); bad['cells']=[[0],[0]]
    with pytest.raises(ValueError,match='disjoint'):
        CuePlayer(controller,bad)
    with pytest.raises(ValueError,match='visible cue'):
        player.decide({'cue':0,'target':1},74101)
