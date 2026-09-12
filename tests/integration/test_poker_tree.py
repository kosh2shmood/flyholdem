from contextlib import contextmanager
import copy
import numpy as np
import pytest
from pokerkit import StandardHighHand
from flyholdem.poker.engine import Hand
from flyholdem.poker.opponents import VERSIONS
from flyholdem.teacher.fast_opponents import TrainingOpponent
from flyholdem.teacher.poker_tree import counterfactual_root,fork_hand
from flyholdem.teacher.regret import RegretTable,VERSION


def config():
    return {'feature_version':VERSION,'stack_bb':20,'equity_buckets':8,'equity_samples':16,
        'max_information_sets':20000,'max_nodes_per_traversal':100000}


@pytest.mark.parametrize('restored',[False,True])
def test_cached_counterfactuals_match_original_full_records_and_numeric_state(monkeypatch,restored):
    import flyholdem.teacher.regret as regret
    @contextmanager
    def original_root(game):yield copy.deepcopy(game)
    def execute(table):
        rows=[]
        for i in range(8):
            game=Hand(991400000+i)
            if restored:game=Hand.restore(game.serialize())
            before=game.serialize();kind=list(VERSIONS)[i//2]
            rows.append(table.step(game,i%2,TrainingOpponent(97100+i,kind)))
            assert game.serialize()==before and game.state.hand_types==(StandardHighHand,)
        return rows
    original=RegretTable(config());cached=RegretTable(config())
    with monkeypatch.context() as m:
        m.setattr(regret,'counterfactual_root',original_root);m.setattr(regret,'fork_hand',copy.deepcopy)
        expected=execute(original)
    assert execute(cached)==expected
    assert all(np.array_equal(original.state()[key],cached.state()[key]) for key in original.state())


def test_all_legal_branches_keep_pokerkit_settlement_and_mutable_state_isolated():
    # Compare every available action at every street, then finish by check/call.
    for seed in range(991500000,991500004):
        parent=Hand(seed)
        while not parent.done:
            snapshot=parent.serialize()
            with counterfactual_root(parent) as tree:
                for action in np.flatnonzero(parent.legal_mask()):
                    expected=copy.deepcopy(parent);actual=fork_hand(tree)
                    expected.act(int(action));actual.act(int(action))
                    while not expected.done:
                        assert actual.observation()==expected.observation()
                        assert actual.act(1)==expected.act(1)
                    assert actual.serialize()==expected.serialize() and actual.view()==expected.view()
                    assert tree.serialize()==snapshot and parent.serialize()==snapshot
            parent.act(1)


def test_cached_hand_objects_keep_original_type_and_are_cleared_per_traversal(monkeypatch):
    original=StandardHighHand.from_game;calls=[]
    def observed(cls,hole,board=()):
        calls.append((hole,board));return original(hole,board)
    monkeypatch.setattr(StandardHighHand,'from_game',classmethod(observed))
    hole='AcAd';board='2c3d4h5s6c'
    with counterfactual_root(Hand(991600001)) as tree:
        cached=tree.state.hand_types[0]
        first=cached.from_game(hole,board);second=cached.from_game(hole,board)
        assert first is second and type(first) is StandardHighHand and len(calls)==1
    assert cached.from_game(hole,board)==first and len(calls)==2
    assert Hand(991600001).state.hand_types==(StandardHighHand,)


@pytest.mark.parametrize('holes,board,payoffs',[
    ([['2c','3c'],['2d','3d']],['Ts','Js','Qs','Ks','As'],[0,0]),
    ([['Ac','Ad'],['Kc','Kd']],['2c','3h','7s','9d','Jc'],[-40,40]),
])
def test_cached_ties_and_all_in_showdown_match_pokerkit(holes,board,payoffs):
    from pokerkit import Deck
    used=[c for pair in holes for c in pair]+board
    rest=[repr(c) for c in Deck.STANDARD if repr(c) not in used]
    prefix=[pair[r] for r in (0,1) for pair in holes]
    prefix += [rest.pop(0)]+board[:3]+[rest.pop(0),board[3],rest.pop(0),board[4]]
    game=Hand(deck=''.join(prefix+rest));before=game.serialize()
    with counterfactual_root(game) as actual:
        expected=copy.deepcopy(game)
        for action in (4,1):assert actual.act(action)==expected.act(action)
        assert actual.view()==expected.view() and actual.view()['payoffs']==payoffs
    assert game.serialize()==before
