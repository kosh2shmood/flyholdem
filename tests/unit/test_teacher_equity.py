import numpy as np
import pytest
from pokerkit import StandardHighHand
from flyholdem.teacher.equity import CARDS, INDEX, rank_batch, visible_equity


@pytest.mark.parametrize('width',[5,6,7])
def test_teacher_rank_order_and_ties_match_pokerkit(width):
    rng=np.random.default_rng(76001+width)
    cards=np.array([rng.choice(52,width,replace=False) for _ in range(600)],dtype=np.int32)
    ranks=rank_batch(cards)
    oracle=np.array([StandardHighHand.from_game(''.join(CARDS[i] for i in row[:2]),
        ''.join(CARDS[i] for i in row[2:])).entry.index for row in cards])
    order=np.argsort(oracle)
    assert np.all(np.diff(ranks[order].astype(np.int64))>=0)
    assert np.array_equal(np.diff(oracle[order])==0,np.diff(ranks[order])==0)


def test_teacher_rank_special_cases_and_rejections():
    hands=['As 2d 3c 4h 5s 8d 9c','9s Td Jc Qh Ks 2d 3c','As Ah Ad Ac Ks Qd Jc',
           'As Ah Ad Ks Kh Kd 2c','As Ks Qs Js Ts 2d 3c','As Ah Ks Kh Qs Qh Jd',
           '2s 4s 6s 8s Ts Qs As']
    cards=np.array([[INDEX[c] for c in hand.split()] for hand in hands],dtype=np.int32)
    ranks=rank_batch(cards)
    assert (ranks//(15**5)).tolist()==[4,4,7,6,8,2,5]
    assert ranks[0]<ranks[1]<ranks[3]<ranks[2]<ranks[4]
    with pytest.raises(ValueError):rank_batch(np.zeros((1,7),dtype=np.int32))
    with pytest.raises(ValueError):rank_batch(np.array([[0.,1,2,3,4]]))


def test_visible_equity_uses_hypothetical_unknown_cards_and_is_deterministic():
    assert visible_equity(('As','Kh'),('Ac','Kc','Qc','Jc','Tc'),256)==.5
    assert visible_equity(('Ac','Kc'),('Qc','Jc','Tc','2d','3h'),256)==1
    aces=visible_equity(('Ah','As'),(),10000)
    low=visible_equity(('2h','7s'),(),10000)
    assert .81<aces<.89 and .3<low<.39
    visible_equity.cache_clear()
    assert aces==visible_equity(('Ah','As'),(),10000)
    with pytest.raises(ValueError):visible_equity(('As','As'),(),256)
