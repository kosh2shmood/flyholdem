import copy
import json
import numpy as np
import pytest
torch=pytest.importorskip('torch')
from flyholdem.poker.engine import Hand
from flyholdem.poker.opponents import Opponent,VERSIONS,visible_equity
from flyholdem.teacher.fast_opponents import TrainingOpponent,exact_visible_equity
from flyholdem.teacher.nfsp import NFSPAgent
from flyholdem.teacher.training import self_play_hand,train
from flyholdem.teacher.export import export_training
from flyholdem.teacher.policy import load_policy

CONFIG={'algorithm':'NFSP-fixed-policy-prior-v1','stack_bb':20,'torch_threads':1,'seed':87100,'hands':32,
    'deal_seed_start':87200,'epsilon_decay_hands':20,'epsilon_start':.9,'epsilon_end':.1,'updates_per_hand':1,'progress_hands':100,
    'agent':{'hidden':16,'q_lr':.001,'average_lr':.001,'replay_capacity':200,'reservoir_capacity':200,
        'anticipatory':.5,'batch_size':8,'warmup_transitions':8,'discount':1,'gradient_clip':5,'target_update_steps':10,
        'value_learning':'double-dqn','feature_version':'canonical-visible-uniform-equity-v3'},
    'population':{'cycle':['self-play']*4+list(VERSIONS),'fixed_hand_update':'active-agent-only','fixed_opponent_backend':'exact-v1-batched-ranks'}}


def test_accelerated_equity_and_all_opponent_actions_match_original_exactly():
    checked=0
    for seed in range(15):
        game=Hand(87500+seed)
        while not game.done:
            observation=game.observation()
            assert exact_visible_equity(observation)==visible_equity(observation)
            game.act(1);checked+=1
    assert checked==120
    for kind in VERSIONS:
        for seed in range(10):
            original=Opponent(seed,kind);fast=TrainingOpponent(seed,kind);game=Hand(87800+seed)
            while not game.done:
                observation=game.observation();action=original.act(observation)
                assert fast.act(observation)==action
                game.act(action)


def test_fixed_population_opponent_never_generates_learning_transitions():
    for fixed_seat in (0,1):
        agents=[NFSPAgent(CONFIG['agent'],89+seat) for seat in (0,1)]
        row=self_play_hand(agents,856,0,20,.3,fixed_opponent=(fixed_seat,'calling-station'))
        assert row['transitions'][fixed_seat]==0 and agents[fixed_seat].replay.size==0 and agents[fixed_seat].reservoir.size==0
        assert agents[1-fixed_seat].replay.size>=1
        assert row['learning_seats']==[1-fixed_seat]
        assert sum(row['net_bb'])==0


def test_population_recovery_and_export_pin_the_actual_training_algorithm(tmp_path):
    full=tmp_path/'full';resumed=tmp_path/'resumed'
    train(CONFIG,full);train(CONFIG,resumed,stop_after=13);result=train(CONFIG,resumed,resume=True)
    assert (full/'hands.jsonl').read_bytes()==(resumed/'hands.jsonl').read_bytes()
    assert result['algorithm']=='NFSP-fixed-policy-prior-v1' and not result['allowed_as_teacher']
    rows=[json.loads(line)['value'] for line in (full/'hands.jsonl').read_text().splitlines()]
    assert sum('fixed_opponent' in row for row in rows)==16
    for row in rows:
        if 'fixed_seat' in row:assert row['optimization'][row['fixed_seat']]==[]
    export_training(full,tmp_path/'policy');policy,record=load_policy(tmp_path/'policy')
    assert record['provenance']['algorithm']=='NFSP-fixed-policy-prior-v1'
    probabilities=policy.probabilities(Hand(1).observation());assert probabilities.sum()==pytest.approx(1)
    bad=copy.deepcopy(CONFIG);bad['population']['cycle']=['self-play','calling-station']
    with pytest.raises(ValueError,match='complete registered'):train(bad,tmp_path/'bad')
