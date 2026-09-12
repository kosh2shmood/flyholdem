import copy
import numpy as np
import pytest
from flyholdem.interface.population import NeuralController
from flyholdem.neural.sparse import SparseBrain
from flyholdem.learning.eligibility import Eligibility
from flyholdem.learning.poker_player import PokerLearningPlayer
from flyholdem.learning.curriculum import CURRICULA,CurriculumHand
from flyholdem.learning.poker_rollout import poker_hand
from flyholdem.experiments.evaluate import frozen_hand
from flyholdem.poker.infoset import FIELDS


def make(mode='bio-plastic',method='terminal-local-eligibility'):
    brain=SparseBrain([0,5,10,10,10,10,10,10],np.tile(np.arange(2,7),2),np.linspace(3,8,10))
    registration={'mode':'fixture-native','graph_hash':brain.graph_hash,'input_indices':[0,1],
        'projection_indices':[[i%2] for i in range(237)],'selected_gain':12,'baseline_hz':[0]*5,
        'ensembles':[{'indices':[i]} for i in range(2,7)],
        'config':{'decision_ms':{'baseline':50,'stimulus':300,'readout':100,'inter_decision':50},'score_scale_hz':100}}
    controller=NeuralController(brain,registration)
    eligibility=Eligibility(brain,[0,1],[2,3,4,5,6],dict(pre_tau_ms=20,eligibility_tau_ms=5000,
        coincidence_scale=.001,bin_ms=5,learning_rate=.3,weight_bounds=[.1,2]))
    config={'learning_mode':mode,'optimization':method,'reward_scale_bb':20,'dopamine':{'pulse_ms':200,'gain':12},
        'surrogate':{'learning_rate':.1,'temperature':1.,'gradient_norm_cap':1.,'synaptic_tau_ms':5.,'threshold_distance_mv':7.}}
    return controller,PokerLearningPlayer(controller,eligibility,(np.array([0]),np.array([1])),config,seed=8)


class VisibleTeacher:
    def probabilities(self,observation):
        assert set(observation)==FIELDS
        values=np.arange(1,6)*np.array(observation['legal_mask']);return values/values.sum()


@pytest.mark.parametrize('curriculum',list(CURRICULA))
def test_complete_frozen_hands_match_the_existing_native_evaluator(curriculum):
    for seat in (0,1):
        controller,_=make();_,player=make()
        old=frozen_hand(controller,curriculum,'calling-station',845,seat)
        new=poker_hand(player,curriculum,'calling-station',845,seat)
        for key in ('neural_return_bb','action_counts','public_terminal','private_hand_checkpoint'):
            assert old[key]==new[key]
        for a,b in zip(old['neural_decisions'],new['neural_decisions']):
            assert all(a[key]==b[key] for key in a)
            assert b['teaching_after_commit'] is None and b['teacher_input_id'] is None
        assert new['weights_before_sha256']==new['weights_after_sha256']
        assert not new['teacher_connected'] and not new['terminal_reinforcement']['reward_delivered']
        assert CurriculumHand.restore(new['private_hand_checkpoint']).view()==new['public_terminal']


@pytest.mark.parametrize('mode,method',[('bio-plastic','terminal-local-eligibility'),
    ('distilled-connectome','teacher-advantage-local-eligibility'),('distilled-connectome','direct-readout-rate-surrogate-v1')])
def test_learning_hand_checkpoint_reproduces_continuation_and_records_only_its_method(mode,method):
    _,player=make(mode,method);teacher=VisibleTeacher() if mode=='distilled-connectome' else None
    first=poker_hand(player,'shove-fold-10bb-v1','random',801,0,learning=True,teacher=teacher,temperature=.2)
    arrays,extra=player.state();_,restored=make(mode,method);restored.restore_state(arrays,extra)
    kwargs={'learning':True,'teacher':teacher,'temperature':.2}
    a=poker_hand(player,'hu-20bb-v1','random',802,1,**kwargs)
    b=poker_hand(restored,'hu-20bb-v1','random',802,1,**kwargs)
    assert a==b
    assert all(np.array_equal(player.state()[0][key],restored.state()[0][key]) for key in arrays)
    assert first['weights_before_sha256']!=a['weights_after_sha256']
    if mode=='bio-plastic':
        reward=first['terminal_reinforcement']['reinforcement']
        assert reward['context']=='position-1' and reward['past_count']==0
        assert reward['raw_reward']==first['neural_return_bb']
        assert not first['teacher_connected']
    else:
        assert first['teacher_connected'] and not first['terminal_reinforcement']['reward_delivered']
        assert first['neural_decisions'][0]['teacher_input_id']


def test_teacher_is_queried_after_the_selected_action_is_committed(monkeypatch):
    committed=[];selected=[];queries=[]
    original=CurriculumHand.act
    def act(game,action):
        result=original(game,action);committed.append(result);return result
    monkeypatch.setattr(CurriculumHand,'act',act)
    _,player=make('distilled-connectome','direct-readout-rate-surrogate-v1')
    decide=player.decide
    def selecting(observation,temperature):
        decision=decide(observation,temperature);selected.append((len(committed),decision['selected']));return decision
    monkeypatch.setattr(player,'decide',selecting)
    class OrderedTeacher(VisibleTeacher):
        def probabilities(self,observation):
            count,action=selected[-1]
            assert len(committed)==count+1 and committed[-1]['action']==action
            queries.append(count);return super().probabilities(observation)
    row=poker_hand(player,'hu-20bb-v1','random',812,0,learning=True,teacher=OrderedTeacher())
    assert len(queries)==len(row['neural_decisions'])>0


def test_matched_control_targets_preserve_legal_mass_and_raw_rewards():
    _,player=make('distilled-connectome','direct-readout-rate-surrogate-v1')
    row=poker_hand(player,'hu-20bb-v1','random',901,0,learning=True,teacher=VisibleTeacher(),target_permutation_seed=55)
    for decision in row['neural_decisions']:
        original=decision['original_teacher_target'];target=decision['teaching_after_commit']['teacher_target']
        assert sorted(original)==sorted(target)
        assert all(value==0 for value,legal in zip(target,decision['legal_mask']) if not legal)
    _,player=make()
    row=poker_hand(player,'shove-fold-10bb-v1','random',903,0,learning=True,terminal_reward_override=7)
    event=row['terminal_reinforcement']
    assert event['reward_shuffled'] and event['raw_net_bb']==row['neural_return_bb']
    assert event['reinforcement']['raw_reward']==7


def test_frozen_and_biological_rollouts_reject_teacher_or_shaping_misuse():
    for kwargs in ({'teacher':VisibleTeacher()},{'temperature':.1},{'terminal_reward_override':1},{'target_permutation_seed':4}):
        _,player=make()
        with pytest.raises(ValueError):poker_hand(player,'hu-20bb-v1','random',90,0,**kwargs)
    _,player=make()
    with pytest.raises(ValueError):poker_hand(player,'hu-20bb-v1','random',90,0,learning=True,teacher=VisibleTeacher())
    _,player=make('distilled-connectome','direct-readout-rate-surrogate-v1')
    with pytest.raises(ValueError):poker_hand(player,'hu-20bb-v1','random',90,0,learning=True)


def test_frozen_training_control_can_match_exploration_without_learning():
    _,player=make()
    row=poker_hand(player,'hu-20bb-v1','random',17,0,phase='training',temperature=.2)
    assert row['phase']=='training' and not row['learning']
    assert row['weights_before_sha256']==row['weights_after_sha256']
    assert all(value['temperature']==.2 for value in row['neural_decisions'])
    assert not row['terminal_reinforcement']['reward_delivered']
    with pytest.raises(ValueError):poker_hand(player,'hu-20bb-v1','random',17,0,phase='evaluation',learning=True)


def test_rewired_control_changes_only_endpoint_order_and_preserves_checkpoint_weights():
    from flyholdem.learning.poker_controls import shuffled_connectome,shuffled_rewards
    controller,_=make();controller.brain.weights*=1.25
    original=controller.brain.state();a=shuffled_connectome(controller,42);b=shuffled_connectome(controller,42)
    assert a.brain.graph_hash==b.brain.graph_hash!=controller.brain.graph_hash
    assert np.array_equal(a.brain.ptr,controller.brain.ptr)
    assert np.array_equal(np.sort(a.brain.post),np.sort(controller.brain.post))
    assert np.array_equal(a.brain.initial,controller.brain.initial)
    assert np.array_equal(a.brain.weights,controller.brain.weights)
    assert all(np.array_equal(value,controller.brain.state()[key]) for key,value in original.items())
    assert a.registration['control']['original_graph_hash']==controller.brain.graph_hash
    with pytest.raises(ValueError,match='graph mismatch'):NeuralController(a.brain,controller.registration)
    rewards=np.array([1,-3,5,7,-9,11]);positions=np.array([0,1,0,1,0,1])
    one=shuffled_rewards(rewards,positions,71);two=shuffled_rewards(rewards,positions,71)
    assert np.array_equal(one,two) and not np.array_equal(one,rewards)
    for position in (0,1):assert np.array_equal(np.sort(one[positions==position]),np.sort(rewards[positions==position]))



def test_terminal_finetuning_preserves_distilled_label_but_uses_only_actual_chip_reward():
    _,bio=make();_,distilled=make('distilled-connectome','terminal-local-eligibility')
    a=poker_hand(bio,'hu-20bb-v1','random',904,0,learning=True,temperature=.2)
    b=poker_hand(distilled,'hu-20bb-v1','random',904,0,learning=True,temperature=.2)
    assert b['learning_mode']=='distilled-connectome' and b['optimization']=='terminal-local-eligibility'
    assert {k:v for k,v in a.items() if k!='learning_mode'}=={k:v for k,v in b.items() if k!='learning_mode'}
    assert b['terminal_reinforcement']['reward_delivered'] and not b['teacher_connected']
    assert all(d['teacher_input_id'] is None for d in b['neural_decisions'])
    arrays,extra=distilled.state();_,restored=make('distilled-connectome','terminal-local-eligibility');restored.restore_state(arrays,extra)
    assert poker_hand(distilled,'hu-20bb-v1','random',905,1,learning=True)==poker_hand(restored,'hu-20bb-v1','random',905,1,learning=True)
    with pytest.raises(ValueError,match='chip reward only'):
        poker_hand(restored,'hu-20bb-v1','random',906,0,learning=True,teacher=VisibleTeacher())



def test_frozen_native_opponent_plays_actual_scores_with_only_one_learning_seat():
    from flyholdem.learning.frozen_opponent import FrozenNeuralOpponent
    opponent_controller,_=make();opponent=FrozenNeuralOpponent(opponent_controller,'a'*64)
    _,player=make();initial=opponent.brain.weights.copy()
    a=poker_hand(player,'hu-20bb-v1',opponent,1702,0,learning=True,temperature=.2)
    assert a['frozen_opponent']['weights_unchanged'] and not a['frozen_opponent']['learning']
    assert a['frozen_opponent']['decisions'] and np.array_equal(opponent.brain.weights,initial)
    for row in a['frozen_opponent']['decisions']:
        assert row['selected']==int(np.argmax(np.where(row['legal_mask'],row['scores'],-np.inf)))
    arrays,extra=player.state();_,restored=make();restored.restore_state(arrays,extra)
    b=poker_hand(player,'hu-20bb-v1',opponent,1703,1,learning=True)
    fresh,_=make();other=FrozenNeuralOpponent(fresh,'a'*64)
    assert b==poker_hand(restored,'hu-20bb-v1',other,1703,1,learning=True)
    same=FrozenNeuralOpponent(player.controller,'b'*64)
    with pytest.raises(ValueError,match='independent mutable'):
        poker_hand(player,'hu-20bb-v1',same,1704,0,learning=True)
    opponent.brain.weights[0]*=1.1
    with pytest.raises(ValueError,match='weights changed'):
        poker_hand(player,'hu-20bb-v1',opponent,1705,0,learning=True)



def test_online_distillation_never_queries_targets_for_validation_or_test_information_ids():
    from flyholdem.teacher.corpus import split_for_id
    from flyholdem.poker.infoset import canonical_information_id
    queries=[]
    class TrainOnly(VisibleTeacher):
        def probabilities(self,observation):
            info=canonical_information_id(observation);assert split_for_id(info)=='train';queries.append(info)
            return super().probabilities(observation)
    _,player=make('distilled-connectome','direct-readout-rate-surrogate-v1')
    skipped=[];delivered=0
    for seed in range(17600,17620):
        row=poker_hand(player,'hu-20bb-v1','random',seed,seed%2,learning=True,teacher=TrainOnly(),teacher_split='train')
        delivered+=row['teacher_targets_delivered']
        for decision in row['neural_decisions']:
            info=decision['teacher_input_id']
            if split_for_id(info)!='train':
                assert decision['teacher_target_held_out'] and decision['teaching_after_commit'] is None
                assert decision['original_teacher_target'] is None;skipped.append(info)
            else:assert not decision['teacher_target_held_out'] and decision['teaching_after_commit'] is not None
    assert skipped and queries and delivered==len(queries) and not set(skipped)&set(queries)
    with pytest.raises(ValueError,match='Only teaching arms'):
        poker_hand(player,'hu-20bb-v1','random',17699,0,teacher_split='train')
