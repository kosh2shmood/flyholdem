import copy
from collections import deque
import pytest
from flyholdem.learning.curriculum import CurriculumHand,CURRICULA
from flyholdem.poker.actions import translations
from flyholdem.poker.observation import canonical_bytes
from flyholdem.interface.encoder import encode_player_state


def test_shove_fold_uses_exact_pokerkit_full_stack_representatives_and_settlement():
    for seed in range(30):
        game=CurriculumHand('shove-fold-10bb-v1',seed,button=seed%2)
        assert game.legal_mask()==[True,False,False,False,True]
        before=game.serialize()
        with pytest.raises(ValueError):game.act(1)
        assert game.serialize()==before
        game.act(4)
        assert game.legal_mask()==[True,True,False,False,False]
        game.act(1)
        assert game.done and len(game.view()['board'])==5
        assert sum(game.view()['payoffs'])==0 and sum(game.view()['stacks'])==40
        assert len(game.hand.history)==2


def test_river_setup_is_declared_uniform_dealing_and_all_five_canonical_actions():
    game=CurriculumHand('river-20bb-v1',981,button=1)
    assert game.hand.state.street_index==3 and len(game.view()['board'])==5
    assert len(game.setup_actions)==6 and all(a['action']==1 for a in game.setup_actions)
    assert game.view()['pot']==4 and game.view()['stacks']==[38,38]
    assert game.legal_mask()==game.hand.legal_mask()
    assert game.observation()['history'] and game.registration['ranges']=='uniform-independent-standard-deck-v1'
    while not game.done:game.act(1)
    assert sum(game.view()['payoffs'])==0


@pytest.mark.parametrize('name',list(CURRICULA))
def test_curriculum_checkpoint_and_information_boundary(name):
    game=CurriculumHand(name,643,button=0)
    obs=canonical_bytes(game.observation());encoded=encode_player_state(game.observation()).tobytes()
    state=game.hand.state;other=1-state.actor_index
    holes=state.hole_cards[other][:];deck=state.deck_cards.copy()
    try:
        state.hole_cards[other][:]=list(deck)[:2];state.deck_cards=deque(reversed(deck))
        game.teacher_labels=[999]*5
        assert canonical_bytes(game.observation())==obs
        assert encode_player_state(game.observation()).tobytes()==encoded
    finally:state.hole_cards[other][:]=holes;state.deck_cards=deck
    choice=4 if name.startswith('shove') else 1
    game.act(choice);saved=game.serialize();restored=CurriculumHand.restore(saved)
    assert restored.serialize()==saved and restored.view()==game.view()
    while not game.done:
        action=1 if game.legal_mask()[1] else next(i for i,v in enumerate(game.legal_mask()) if v)
        game.act(action);restored.act(action)
        assert restored.view()==game.view()
    bad=copy.deepcopy(saved);bad['registration']['ranges']='private-informed'
    with pytest.raises(ValueError):CurriculumHand.restore(bad)
