from pokerkit import Deck
import pytest
from flyholdem.poker.engine import Hand
from flyholdem.poker.actions import translations


def deck_for(holes,board):
    """Hole order is PokerKit seat order; one card per player per deal round."""
    used=[c for pair in holes for c in pair]+board
    assert len(set(used))==len(used)
    remainder=[repr(c) for c in Deck.STANDARD if repr(c) not in used]
    prefix=[pair[round_] for round_ in (0,1) for pair in holes]
    prefix += [remainder.pop(0)]+board[:3]+[remainder.pop(0),board[3],remainder.pop(0),board[4]]
    return ''.join(prefix+remainder)


def test_showdown_aces_beat_kings_all_in():
    deck=deck_for([['Ac','Ad'],['Kc','Kd']],['2c','3h','7s','9d','Jc'])
    h=Hand(button=0,deck=deck)
    assert h.state.hole_cards[0][0].rank.value=='A'
    h.act(4);h.act(1)
    assert h.done and h.view()['payoffs']==[-40,40]
    assert h.view()['stacks']==[0,80]
    assert h.view()['board']==['2c','3h','7s','9d','Jc']


def test_royal_board_splits_even_all_in_pot():
    h=Hand(deck=deck_for([['2c','3c'],['2d','3d']],['Ts','Js','Qs','Ks','As']))
    h.act(4);h.act(1)
    assert h.done and h.view()['stacks']==[40,40]
    assert h.view()['payoffs']==[0,0]


def test_three_way_main_side_pot_and_uncalled_excess():
    deck=deck_for([['Ac','Ad'],['Kc','Kd'],['Qc','Qd']],['2c','3h','7s','9d','Jc'])
    h=Hand(button=2,stacks=(10,20,40),deck=deck)
    assert h.actor==2
    h.act(4)
    assert h.actor==0
    h.act(1);h.act(1)
    assert h.done
    # Main pot 30 to aces, side pot 20 to kings, uncalled 20 back to queens.
    assert h.view()['stacks']==[30,20,20]
    assert h.view()['payoffs']==[20,0,-20]


def test_short_all_in_does_not_reopen_action():
    h=Hand(button=2,stacks=(7,100,100))
    assert h.actor==2
    assert translations(h.state)[2]==('raise',5)
    h.act(2)
    assert h.actor==0
    assert translations(h.state)[2]==('raise',7)
    h.act(2)  # Short all-in: increment 2, less than the full raise increment 3.
    assert h.actor==1
    h.act(1)
    assert h.actor==2
    assert h.legal_mask()==[True,True,False,False,False]
    h.act(1)
    assert h.state.street_index==1


def test_multiway_neural_policy_is_explicitly_outside_registered_scope():
    h=Hand(button=2,stacks=(40,40,40))
    with pytest.raises(ValueError,match='heads-up only'):
        h.observation()


def test_noninteger_actions_rejected():
    h=Hand()
    for value in (True,False,1.0,'1',None):
        with pytest.raises(ValueError):h.act(value)
