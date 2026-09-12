"""Exact PokerKit reuse inside one conventional counterfactual traversal.

Only immutable cards and PokerKit's own evaluated hand objects are shared.
Mutable betting state and histories remain independent in every branch. This
module neither chooses actions nor supplies observations or neural scores.
"""
from contextlib import contextmanager
import copy
from functools import lru_cache
from pokerkit import Card, Deck, StandardHighHand

_IMMUTABLE_CARDS={id(card):card for card in (*Deck.STANDARD,Card.UNKNOWN)}


def fork_hand(game):
    """Copy all mutable state; the locked PokerKit Card values are immutable."""
    return copy.deepcopy(game,_IMMUTABLE_CARDS.copy())


@contextmanager
def counterfactual_root(game):
    """Bound repeated ordered-card evaluations to this single traversal."""
    root=fork_hand(game);hand=root.hand if hasattr(root,'hand') else root
    if hand.state.hand_types!=(StandardHighHand,):
        raise ValueError('The conventional traversal cache requires standard high Holdem')
    @lru_cache(maxsize=256)
    def evaluated(hole,board):
        # Preserve PokerKit's exact best combination and StandardHighHand type.
        return StandardHighHand.from_game(hole,board)
    class CachedStandardHighHand(StandardHighHand):
        @classmethod
        def from_game(cls,hole_cards,board_cards=()):
            return evaluated(Card.clean(hole_cards),Card.clean(board_cards))
    hand.state.hand_types=(CachedStandardHighHand,)
    try:yield root
    finally:evaluated.cache_clear()
