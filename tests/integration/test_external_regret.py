"""Conventional-only exact arithmetic, PokerKit trees and numeric recovery."""
import copy
import json
import numpy as np
import pytest
from flyholdem.poker.engine import Hand
from flyholdem.learning.curriculum import CurriculumHand
from flyholdem.teacher.regret import VERSION,RegretTable,abstraction
from flyholdem.teacher.regret_training import train,completed_table,SCHEMA,AGGREGATION
from flyholdem.teacher.regret_policy import export_policy,load_policy
from flyholdem.teacher.fast_opponents import TrainingOpponent
from flyholdem.teacher.evaluation import verify_information_boundary,evaluate
from flyholdem.teacher.validation import verify_evaluation
from flyholdem.connectome.registry import digest
from flyholdem.poker.opponents import VERSIONS


def config():
    return {'schema':SCHEMA,'stack_bb':20,'iterations':12,'sampling_seed':91100,'deal_seed_start':991000000,
        'opponents':list(VERSIONS),'opponent_probabilities':[.25]*4,'aggregation':AGGREGATION,'progress_iterations':100,
        'abstraction':{'feature_version':VERSION,'stack_bb':20,'equity_buckets':8,'equity_samples':16,
            'max_information_sets':20000,'max_nodes_per_traversal':100000}}


def test_actual_shove_fold_counterfactual_regrets_and_average_have_exact_values():
    table=RegretTable({**config()['abstraction'],'stack_bb':10})
    game=CurriculumHand('shove-fold-10bb-v1',991100001);key,legal=abstraction(game.observation(),table.config)
    fold=copy.deepcopy(game);fold.act(0);shove=copy.deepcopy(game);shove.act(4);shove.act(1)
    values=np.array([fold.view()['payoffs'][0]/2,shove.view()['payoffs'][0]/2])
    result=table.step(game,0,TrainingOpponent(42,'calling-station'));i=table.index[key]
    assert result['sampled_policy_value_bb']==values.mean()
    assert np.array_equal(table.regrets[i][legal],values-values.mean())
    assert np.array_equal(table.averages[i][legal],[.5,.5]) and table.visits[i]==1
    # Counterfactual traversal leaves the real initial deck and hand untouched.
    assert game.serialize()==result['root_private_checkpoint']


def test_full_fixed_population_training_recovers_exact_table_rng_and_traversals(tmp_path):
    full=train(config(),tmp_path/'whole');first=train(config(),tmp_path/'resumed',stop_after=5)
    assert first['status']=='interrupted' and first['iterations_completed']==5
    done=train(config(),tmp_path/'resumed',resume=True)
    assert full['status']==done['status']=='trained-unvalidated'
    before=(tmp_path/'resumed/result.json').read_bytes()
    assert train(config(),tmp_path/'resumed',resume=True)==done
    assert (tmp_path/'resumed/result.json').read_bytes()==before
    assert (tmp_path/'whole/traversals.jsonl').read_bytes()==(tmp_path/'resumed/traversals.jsonl').read_bytes()
    a,_=completed_table(tmp_path/'whole');b,_=completed_table(tmp_path/'resumed')
    assert all(np.array_equal(a.state()[key],b.state()[key]) for key in a.state())
    assert not full['allowed_as_teacher'] and full['information_sets']>0
    hash_value=export_policy(tmp_path/'whole',tmp_path/'policy');policy,record=load_policy(tmp_path/'policy')
    assert hash_value==digest(tmp_path/'policy/manifest.json') and record['mode']=='conventional-teacher-control'
    for seed in range(991000000,991000004):
        game=Hand(seed)
        while not game.done:
            observation=game.observation();assert np.array_equal(a.probabilities(observation),policy.probabilities(observation));game.act(1)
    assert verify_information_boundary(policy)['decisions_checked']==32
    # A new stack domain never aliases the learned 20 BB table; its declared
    # unseen-information prior is legal uniform, with no 100 BB strength claim.
    other=Hand(1,stacks=(200,200)).observation()
    assert np.array_equal(policy.probabilities(other),np.array(other['legal_mask'],float)/sum(other['legal_mask']))
    with pytest.raises(ValueError,match='registered heads-up'):
        a.step(Hand(1,stacks=(200,200)),0,TrainingOpponent(1,'random'))
    file=tmp_path/'policy/probabilities.npy';values=np.load(file,allow_pickle=False);values[0,0]=np.nan;np.save(file,values,allow_pickle=False)
    with pytest.raises(ValueError,match='numeric regret files changed'):load_policy(tmp_path/'policy')


def test_regret_policy_uses_the_original_full_teacher_qualification_path(tmp_path,monkeypatch):
    import builtins
    original_import=builtins.__import__
    def without_torch(name,*args,**kwargs):
        if name=='torch' or name.startswith('torch.'):
            raise ModuleNotFoundError('Tabular evaluation must work without the optional neural-teacher dependency')
        return original_import(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',without_torch)
    cfg=config();cfg['iterations']=2
    train(cfg,tmp_path/'training');export_policy(tmp_path/'training',tmp_path/'policy')
    suite={'stack_bb':20,'opponents':list(VERSIONS),'bootstrap_seed':91900,'bootstrap_repeats':100,
        'profiles':{'development':{'paired_deals_per_opponent':2,'seed_start':992000000}}}
    result=evaluate(tmp_path/'policy',suite,tmp_path/'evaluation')
    assert result['sampling']=='frozen-reach-weighted-regret-average-fixed-population'
    evidence=verify_evaluation(tmp_path/'evaluation',tmp_path/'policy',suite,require_confirmatory=False)
    assert evidence['summary_recomputed'] and not evidence['allowed_as_teacher']


def test_numeric_table_rejects_illegal_averages_and_remembers_past_private_buckets():
    table=RegretTable(config()['abstraction']);game=Hand(991200001)
    game.act(1);game.act(1)
    after=json.loads(abstraction(game.observation(),table.config)[0])
    # Acting player changes at the flop; compare a fresh per-seat prefix with
    # the same visible private cards, rather than assuming identical players.
    own=game.observation();preflop=Hand(991200001);preflop.act(1)
    assert own['hole']==preflop.observation()['hole']
    assert after[2][:1]==json.loads(abstraction(preflop.observation(),table.config)[0])[2]
    table.step(Hand(991200001),0,TrainingOpponent(9,'calling-station'))
    state=table.state();bad={k:v.copy() for k,v in state.items()}
    row,action=np.argwhere(~bad['legal'])[0];bad['averages'][row,action]=1
    with pytest.raises(ValueError,match='stored legal regrets'):RegretTable(table.config).restore(bad)


def test_conventional_training_refuses_reserved_confirmation_deals_before_output(tmp_path):
    cfg=config();cfg['deal_seed_start']=3000000
    with pytest.raises(ValueError,match='exclude the registered'):
        train(cfg,tmp_path/'forbidden')
    assert not (tmp_path/'forbidden').exists()
