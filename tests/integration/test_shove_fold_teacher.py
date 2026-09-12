import copy
from collections import Counter,deque
from itertools import combinations
import json
import numpy as np
import pytest
from flyholdem.learning.curriculum import CurriculumHand
from flyholdem.teacher.shove_fold import (card_class,information_key,sample_payoffs,ShoveFoldCFR,
    ShoveFoldPolicy,export_policy,load_policy,CURRICULUM,ALGORITHM)
from flyholdem.teacher.shove_fold_training import train,export_training
from flyholdem.poker.observation import canonical_bytes


def test_all_1326_hands_map_to_the_exact_169_preflop_classes():
    cards=[r+s for r in '23456789TJQKA' for s in 'cdhs']
    classes=Counter(card_class(pair) for pair in combinations(cards,2))
    assert len(classes)==169 and Counter(classes.values())=={6:13,4:78,12:78}
    assert card_class(['As','Kd'])==card_class(['Kc','Ah'])
    assert card_class(['As','Ks'])==card_class(['Kh','Ah'])
    assert card_class(['As','Ks'])!=card_class(['As','Kh'])
    with pytest.raises(ValueError):card_class(['As','As'])


def test_counterfactual_samples_use_three_actual_pokerkit_settlements():
    for seed in range(12):
        row=sample_payoffs(seed)
        assert row['sb_terminal_bb'][:2]==[-.5,1]
        assert row['sb_terminal_bb'][2] in (-10,0,10)
        game=CurriculumHand.restore(row['private_showdown_hand'])
        assert game.done and game.view()['payoffs'][0]/2==row['sb_terminal_bb'][2]
        for position,path in enumerate(([0],[4,0],[4,1])):
            game=CurriculumHand(CURRICULUM,seed)
            for action in path:game.act(action)
            assert game.view()['payoffs'][0]/2==row['sb_terminal_bb'][position]


def test_cfr_updates_weight_bb_regret_by_sb_reach_but_not_its_average():
    solver=ShoveFoldCFR();sample={'classes':[0,1],'sb_terminal_bb':[-.5,1,10]}
    row=solver.step(sample)
    assert row['counterfactual_regret_update']==[[-3,3],[2.25,-2.25]]
    assert np.array_equal(solver.strategy_sum[:,[0,1]][[0,1],[0,1]],[[.5,.5],[.5,.5]])
    solver.regrets[0,0]=[1,-1];prior=solver.regrets[1,1].copy();average=solver.strategy_sum[1,1].copy()
    solver.step(sample)
    assert np.array_equal(solver.regrets[1,1],prior)
    assert solver.strategy_sum[1,1].sum()==average.sum()+1


@pytest.mark.parametrize('showdown,expected_sb,expected_bb',[(10,1,0),(-10,0,1)])
def test_cfr_converges_on_two_analytically_solved_one_class_games(showdown,expected_sb,expected_bb):
    solver=ShoveFoldCFR()
    for _ in range(2000):solver.step({'classes':[0,0],'sb_terminal_bb':[-.5,1,showdown]})
    assert abs(solver.average()[0,0,1]-expected_sb)<.002
    assert abs(solver.average()[1,0,1]-expected_bb)<.002


def test_tabular_policy_is_private_invariant_and_rejects_other_games(tmp_path):
    solver=ShoveFoldCFR()
    for seed in range(10):solver.step(sample_payoffs(seed))
    export_policy(solver,tmp_path/'policy',{'scope':'tiny engineering fixture; unvalidated'})
    policy,record=load_policy(tmp_path/'policy');assert not record['allowed_as_teacher']
    for seed in range(5):
        for node in (0,1):
            game=CurriculumHand(CURRICULUM,seed)
            if node:game.act(4)
            observation=game.observation();before=policy.probabilities(observation).tobytes()
            raw=game.hand;other=1-raw.state.actor_index
            raw.state.hole_cards[other][:]=list(raw.state.deck_cards)[:2]
            raw.state.deck_cards=deque(reversed(raw.state.deck_cards));raw.teacher_labels=[999]*5
            assert canonical_bytes(game.observation())==canonical_bytes(observation)
            assert before==policy.probabilities(game.observation()).tobytes()
            assert np.array_equal(policy.probabilities(observation),ShoveFoldPolicy(solver.average()).probabilities(observation))
    game=CurriculumHand('hu-20bb-v1',0)
    with pytest.raises(ValueError,match='registered'):policy.probabilities(game.observation())
    changed=dict(CurriculumHand(CURRICULUM,0).observation(),opponent_hole=['As','Ah'])
    with pytest.raises(ValueError,match='allowlist'):policy.probabilities(changed)
    path=tmp_path/'policy/probabilities.npy';path.write_bytes(path.read_bytes()+b'x')
    with pytest.raises(ValueError,match='mismatch'):load_policy(tmp_path/'policy')


def test_tabular_complete_checkpoint_and_old_checkpoint_tail_match_uninterrupted(tmp_path):
    config={'schema':'tabular-shove-fold-training-v1','algorithm':ALGORITHM,'curriculum':CURRICULUM,
        'status':'development','iterations':24,'deal_seed_start':6210000,'progress_iterations':24}
    full=tmp_path/'full';resumed=tmp_path/'resumed'
    train(config,full);train(config,resumed,stop_after=11)
    # Replay all eleven already logged traversals from the original zero state.
    checkpoints=resumed/'checkpoints';old=next(checkpoints.glob('step000000000000-*'))
    from flyholdem.connectome.registry import digest
    (checkpoints/'latest.json').write_text(json.dumps({'generation':old.name,'manifest_sha256':digest(old/'manifest.json')}))
    train(config,resumed,resume=True)
    assert (full/'hands.jsonl').read_bytes()==(resumed/'hands.jsonl').read_bytes()
    export_training(full,tmp_path/'a');export_training(resumed,tmp_path/'b')
    assert (tmp_path/'a/probabilities.npy').read_bytes()==(tmp_path/'b/probabilities.npy').read_bytes()
    assert json.loads((full/'result.json').read_text())['allowed_as_teacher'] is False
    altered=copy.deepcopy(config);altered['iterations']+=1
    with pytest.raises(ValueError,match='identity mismatch'):train(altered,resumed,resume=True)


def test_tabular_cli_runs_and_reports_actual_small_chance_traversals(tmp_path,capsys):
    import yaml
    from flyholdem.cli import main
    config={'schema':'tabular-shove-fold-training-v1','algorithm':ALGORITHM,'curriculum':CURRICULUM,
        'status':'development','iterations':3,'deal_seed_start':6210010,'progress_iterations':3}
    path=tmp_path/'config.yaml';path.write_text(yaml.safe_dump(config));run=tmp_path/'run'
    main(['teacher','train-shove-fold','--config',str(path),'--output',str(run)])
    report=json.loads((run/'report/report.json').read_text())
    assert report['status']=='trained-unvalidated' and report['journal_verification']['hands.jsonl']['rows']==3
    assert 'three terminal branches' in report['hand_unit'] and not report['allowed_as_teacher']
    main(['teacher','export-shove-fold','--run',str(run),'--output',str(tmp_path/'policy')])
    assert load_policy(tmp_path/'policy')[1]['allowed_as_teacher'] is False


def evaluation_fixture(root):
    config={'schema':'tabular-shove-fold-training-v1','algorithm':ALGORITHM,'curriculum':CURRICULUM,
        'status':'development','iterations':8,'deal_seed_start':6211000,'progress_iterations':8}
    run=root/'training';train(config,run);policy=root/'policy';export_training(run,policy)
    from flyholdem.poker.opponents import VERSIONS
    suite={'schema':'tabular-shove-fold-suite-v1','curriculum':CURRICULUM,
        'training_deal_range':[6211000,6211008],'opponents':list(VERSIONS),'bootstrap_seed':6290000,'bootstrap_repeats':100,
        'profiles':{'development':{'seed_start':6401000,'paired_deals_per_opponent':2},
                    'confirmatory':{'seed_start':6501000,'paired_deals_per_opponent':2}}}
    return run,policy,suite


def test_small_evaluation_recovery_recomputes_all_pairs_without_qualifying_full_teacher(tmp_path):
    from flyholdem.teacher.shove_fold_evaluation import evaluate,verify_run,boundary_check
    from flyholdem.connectome.registry import digest
    training,policy,config=evaluation_fixture(tmp_path)
    original=tmp_path/'evaluation';restored=tmp_path/'restored'
    result=evaluate(policy,config,original,training_run=training)
    assert not result['allowed_as_teacher'] and not result['allowed_as_small_game_teacher']
    assert result['information_boundary']['decisions_checked']==32
    with pytest.raises(KeyboardInterrupt):evaluate(policy,config,restored,training_run=training,stop_after=3)
    evaluate(policy,config,restored,resume=True,training_run=training)
    assert (original/'paired-deals.jsonl').read_bytes()==(restored/'paired-deals.jsonl').read_bytes()
    verify_run(original,config,digest(policy/'manifest.json'),'development')
    result['opponents']['random']['bb_per_hand']+=1
    (original/'result.json').write_text(json.dumps(result))
    with pytest.raises(ValueError,match='recorded paired'):verify_run(original,config,digest(policy/'manifest.json'),'development')
    with pytest.raises(ValueError,match='requires passing development'):
        evaluate(policy,config,tmp_path/'confirm',profile='confirmatory',training_run=training)
    assert not (tmp_path/'confirm').exists()


def test_small_evaluation_rejects_training_overlap_and_unrelated_training_before_running(tmp_path):
    from flyholdem.teacher.shove_fold_evaluation import evaluate
    training,policy,config=evaluation_fixture(tmp_path)
    config['profiles']['development']['seed_start']=6211001
    with pytest.raises(ValueError,match='disjoint'):evaluate(policy,config,tmp_path/'overlap',training_run=training)
    assert not (tmp_path/'overlap').exists()
    config['profiles']['development']['seed_start']=6401000
    manifest=training/'manifest.json';value=json.loads(manifest.read_text());value['config']['iterations']+=1;manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='completed registered'):evaluate(policy,config,tmp_path/'different',training_run=training)
