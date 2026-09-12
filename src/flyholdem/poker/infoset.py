"""Strict canonical acting-player states shared by teacher/corpus validation.

Only unordered hole-card order and the flop's simultaneous deal order are
normalized. Turn/river order and the complete public action history remain.
Suit isomorphisms are not merged: the frozen sensory code is suit specific.
"""
import hashlib
import math
from flyholdem.poker.observation import canonical_bytes
from flyholdem.interface.encoder import CARDS

FIELDS = {'schema', 'hole', 'board', 'street', 'position', 'pot', 'own_stack',
          'opponent_stack', 'contribution', 'to_call', 'min_raise', 'max_raise',
          'spr', 'legal_mask', 'history'}
HISTORY = {'street', 'actor', 'action', 'wager_bb', 'pot_fraction'}
INT_FIELDS = ('pot', 'own_stack', 'opponent_stack', 'contribution', 'to_call', 'min_raise', 'max_raise')


def canonical_state(observation):
    if set(observation) != FIELDS or observation['schema'] != 'infoset-v1':
        raise ValueError('Only the canonical acting-player observation allowlist is accepted')
    street, position = observation['street'], observation['position']
    if type(street) is not int or street not in range(4) or type(position) is not int or position not in (0, 1):
        raise ValueError('Invalid public street or relative position')
    hole, board = list(observation['hole']), list(observation['board'])
    if (len(hole) != 2 or len(board) != (0, 3, 4, 5)[street]
            or any(card not in CARDS for card in hole + board) or len(set(hole + board)) != len(hole + board)):
        raise ValueError('Invalid visible cards')
    for name in INT_FIELDS:
        if type(observation[name]) is not int or observation[name] < 0:
            raise ValueError('Nonnegative public chip counts required')
    if observation['pot'] <= 0:
        raise ValueError('A live heads-up information set must contain a pot')
    spr = observation['spr']
    if type(spr) not in (int, float) or not math.isfinite(spr) or spr < 0:
        raise ValueError('Invalid visible stack-to-pot ratio')
    if not math.isclose(spr, observation['own_stack'] / max(1, observation['pot']), rel_tol=0, abs_tol=1e-12):
        raise ValueError('Inconsistent visible stack-to-pot ratio')
    legal = list(observation['legal_mask'])
    if len(legal) != 5 or any(type(x) is not bool for x in legal) or not any(legal):
        raise ValueError('Five boolean actions with a nonempty legal set required')
    history = []
    previous_street = 0
    for item in observation['history']:
        if set(item) != HISTORY:
            raise ValueError('Unexpected hidden or extra action-history field')
        if (type(item['street']) is not int or not previous_street <= item['street'] <= street
                or type(item['actor']) is not int or item['actor'] not in (0, 1)
                or type(item['action']) is not int or item['action'] not in range(5)):
            raise ValueError('Invalid public action history')
        previous_street = item['street']
        for key in ('wager_bb', 'pot_fraction'):
            if type(item[key]) not in (int, float) or not math.isfinite(item[key]) or item[key] < 0:
                raise ValueError('Invalid visible action amount')
        history.append({**item, 'wager_bb': float(item['wager_bb']), 'pot_fraction': float(item['pot_fraction'])})
    return {**observation, 'hole': sorted(hole), 'board': sorted(board[:3]) + board[3:],
            'spr': float(spr), 'legal_mask': legal, 'history': history}


def canonical_information_id(observation):
    state = canonical_state(observation)
    return hashlib.sha256(b'canonical-visible-infoset-v1\0' + canonical_bytes(state)).hexdigest()
