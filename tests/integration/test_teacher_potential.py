import copy
import numpy as np
import pytest
import yaml
torch=pytest.importorskip('torch',reason='Optional conventional teacher')
from flyholdem.connectome.registry import ROOT
from flyholdem.teacher.potential import transition_reward,VERSION
from flyholdem.teacher.nfsp import NFSPAgent
from flyholdem.teacher.training import self_play_hand,train


def config():
    value=yaml.safe_load((ROOT/'configs/teacher_nfsp_population_v5.yaml').read_text())
    value.update(hands=32,progress_hands=32,reward_parameterization=VERSION)
    value['agent'].update(hidden=8,replay_capacity=128,reservoir_capacity=128,batch_size=4,warmup_transitions=4,target_update_steps=4)
    return value


def test_public_stack_potential_preserves_returns_up_to_the_fixed_initial_constant():
    for final_stack in (0,23,40,80):
        stacks=[39,38,21]
        events=[transition_reward(0,stacks[i],stacks[i+1],40) for i in range(2)]
        events.append(transition_reward((final_stack-40)/40,stacks[-1],None,40))
        assert sum(e['shaped_reward'] for e in events)==pytest.approx((final_stack-39)/40)
    for stack in (1,19,39):
        assert transition_reward((stack-40)/40,stack,None,40)['shaped_reward']==0
    with pytest.raises(ValueError):transition_reward(float('nan'),20,None,40)


def test_actual_same_player_transitions_expose_only_incremental_chip_flow_without_changing_hands():
    class Recording(NFSPAgent):
        def begin_hand(self):super().begin_hand();self.previous_stack=None;self.recorded=[]
        def act(self,observation,*args,**kwargs):
            self.previous_stack=observation['own_stack'];return super().act(observation,*args,**kwargs)
        def transition(self,previous,reward,next_observation=None):
            self.recorded.append((self.previous_stack,reward,None if next_observation is None else next_observation['own_stack']))
            return super().transition(previous,reward,next_observation)
    settings=config()['agent'];torch.manual_seed(712)
    shaped=[Recording(settings,10+i) for i in range(2)]
    torch.manual_seed(712);raw=[Recording(settings,10+i) for i in range(2)]
    for seed in range(60,80):
        a=self_play_hand(raw,seed,seed%2,20,.3)
        b=self_play_hand(shaped,seed,seed%2,20,.3,reward_parameterization=VERSION)
        assert a['decisions']==b['decisions'] and a['net_bb']==b['net_bb']
        for seat,agent in enumerate(shaped):
            for before,reward,after in agent.recorded:
                end=after if after is not None else 40+2*b['net_bb'][seat]
                assert reward==pytest.approx((end-before)/40)
            assert agent.recorded==[] or agent.recorded[-1][2] is None
        assert len(b['reward_events'])==sum(b['transitions'])
    bad=copy.deepcopy(settings);bad['discount']=.99
    with pytest.raises(ValueError,match='undiscounted'):
        self_play_hand([NFSPAgent(bad,1),NFSPAgent(bad,2)],1,0,20,.1,reward_parameterization=VERSION)


def test_potential_training_reproduces_complete_optimizer_and_memory_recovery(tmp_path):
    settings=config();full=tmp_path/'full';restored=tmp_path/'restored'
    train(settings,full);train(settings,restored,stop_after=13);train(settings,restored,resume=True)
    assert (full/'hands.jsonl').read_bytes()==(restored/'hands.jsonl').read_bytes()
    settings['reward_parameterization']='unknown'
    with pytest.raises(ValueError,match='registered'):train(settings,tmp_path/'invalid')
    assert not (tmp_path/'invalid').exists()
