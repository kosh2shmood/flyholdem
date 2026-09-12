"""Conventional-teacher features derived only from canonical visible cards.

The fly continues to receive its frozen 237-channel code. These additional
features never enter corpus observation fields or the native neural interface.
PokerKit supplies visible made-hand ranks; there is no future-card sampling.
"""
from functools import lru_cache
import numpy as np
from pokerkit import StandardHighHand, Label
from flyholdem.interface.encoder import CHANNELS, encode_player_state
from flyholdem.poker.infoset import canonical_state

V1 = 'canonical-visible-237-with-public-first-action-v1'
V2 = 'canonical-visible-card-structure-v2'
RANKS = '23456789TJQKA'
SUITS = 'cdhs'
LABELS = list(Label)
CARD_NAMES = ([f'own_rank:{r}' for r in RANKS] + [f'own_pair:{r}' for r in RANKS]
    + ['suited', 'own_high_rank', 'own_low_rank', 'own_rank_gap']
    + [f'board_rank:{r}' for r in RANKS] + [f'board_suit:{s}' for s in SUITS]
    + [f'all_rank:{r}' for r in RANKS] + [f'all_suit:{s}' for s in SUITS]
    + [f'own_board_rank_match:{r}' for r in RANKS]
    + [f'straight_coverage:{i}' for i in range(10)] + [f'own_straight_coverage:{i}' for i in range(10)]
    + [f'visible_made_category:{label.name}' for label in LABELS] + ['visible_made_rank', 'own_in_best_five']
    + ['max_suit_fraction', 'four_flush', 'made_flush', 'board_paired', 'board_trips'])
PUBLIC_NAMES = ['call_price', 'own_stack_fraction', 'effective_spr', 'facing_bet']


def feature_names(version=V1):
    if version == V1:
        return list(CHANNELS)
    if version == V2:
        return list(CHANNELS) + CARD_NAMES + PUBLIC_NAMES
    raise ValueError('Unknown registered conventional-teacher representation')


@lru_cache(maxsize=20000)
def card_structure(hole, board):
    own_rank = np.bincount([RANKS.index(c[0]) for c in hole], minlength=13).astype(float)
    board_rank = np.bincount([RANKS.index(c[0]) for c in board], minlength=13).astype(float)
    board_suit = np.bincount([SUITS.index(c[1]) for c in board], minlength=4).astype(float)
    all_suit = np.bincount([SUITS.index(c[1]) for c in hole + board], minlength=4).astype(float)
    own_present = set(np.flatnonzero(own_rank).tolist())
    present = set(np.flatnonzero(own_rank + board_rank).tolist())
    runs = [{12, 0, 1, 2, 3}] + [set(range(i, i + 5)) for i in range(9)]
    ranks = [RANKS.index(c[0]) for c in hole]
    category = np.zeros(len(LABELS), dtype=float)
    made_rank = own_in_best = 0.0
    if len(board) >= 3:
        made = StandardHighHand.from_game(''.join(hole), ''.join(board))
        category[LABELS.index(made.entry.label)] = 1
        if not 0 <= made.entry.index <= 7461:
            raise ValueError('Unexpected PokerKit standard high-hand rank range')
        made_rank = made.entry.index / 7461
        own_in_best = len(set(map(repr, made.cards)) & set(hole)) / 2
    value = np.concatenate((own_rank / 2, (own_rank == 2).astype(float),
        [float(hole[0][1] == hole[1][1]), max(ranks) / 12, min(ranks) / 12, abs(ranks[0] - ranks[1]) / 12],
        board_rank / 4, board_suit / 5, (own_rank + board_rank) / 4, all_suit / 7,
        (own_rank > 0) * board_rank / 4,
        [len(present & run) / 5 for run in runs], [len(own_present & run) / 2 for run in runs],
        category, [made_rank, own_in_best, all_suit.max() / 7, float(all_suit.max() == 4),
                   float(all_suit.max() >= 5), float(board_rank.max() >= 2), float(board_rank.max() >= 3)]))
    if len(value) != len(CARD_NAMES) or not np.isfinite(value).all():
        raise AssertionError('Invalid visible card representation')
    value = value.astype(np.float32); value.flags.writeable = False
    return value


def features(observation, version=V1):
    state = canonical_state(observation)
    base = encode_player_state(state).astype(np.float32)
    if version == V1:
        return base
    if version != V2:
        raise ValueError('Unknown conventional-teacher feature version')
    cards = card_structure(tuple(state['hole']), tuple(state['board']))
    public = np.array([state['to_call'] / max(1, state['pot'] + state['to_call']),
        state['own_stack'] / max(1, state['own_stack'] + state['pot']),
        min(20, min(state['own_stack'], state['opponent_stack']) / max(1, state['pot'] + state['to_call'])) / 20,
        float(state['to_call'] > 0)], dtype=np.float32)
    return np.concatenate((base, cards, public))
