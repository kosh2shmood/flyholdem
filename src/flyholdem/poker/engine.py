"""Canonical PokerKit heads-up cash hands, with an isolated deterministic deck."""
from collections import deque
import random
from pokerkit import Automation, Card, Deck, Mode, NoLimitTexasHoldem
from .actions import NAMES, translations

AUTOMATIONS = tuple(a for a in Automation if a not in (
    Automation.HOLE_DEALING, Automation.BOARD_DEALING, Automation.CARD_BURNING))


class Hand:
    def __init__(self, seed=1, button=0, stacks=(40, 40), deck=None):
        self.player_count = len(stacks)
        if not 2 <= self.player_count <= 10 or not 0 <= button < self.player_count:
            raise ValueError('Require 2–10 players and a valid button')
        if any(type(value) is not int or value < 1 for value in stacks):
            raise ValueError('Starting stacks must be positive integer chips')
        self.seed, self.button = seed, button
        self.starting_stacks = list(stacks)
        # PokerKit HU index 0 = BB, index 1 = button/SB.
        self.seats = [1 - button, button] if self.player_count == 2 else [(button + 1 + i) % self.player_count for i in range(self.player_count)]
        self.state = NoLimitTexasHoldem.create_state(
            AUTOMATIONS, True, 0, (1, 2), 2,
            tuple(stacks[i] for i in self.seats), self.player_count, mode=Mode.CASH_GAME)
        cards = list(Deck.STANDARD) if deck is None else list(Card.parse(deck))
        if deck is None:
            random.Random(seed).shuffle(cards)
        if len(cards) != 52 or len(set(cards)) != 52:
            raise ValueError('Deck must contain all 52 distinct standard cards')
        self.initial_deck = ''.join(map(repr, cards))
        self.state.deck_cards = deque(cards)
        self.history = []
        self._advance_deal()
        self.dealt_holes = [list(cards) for cards in self.state.hole_cards]

    @property
    def actor(self):
        i = self.state.actor_index
        return None if i is None else self.seats[i]

    @property
    def done(self):
        return not self.state.status

    def _advance_deal(self):
        s = self.state
        while s.status and s.actor_index is None:
            if s.can_burn_card():
                s.burn_card()
            elif s.can_deal_hole():
                s.deal_hole()
            elif s.can_deal_board():
                s.deal_board()
            else:
                raise RuntimeError('Unexpected unresolved PokerKit phase')

    def legal_mask(self):
        return [a is not None for a in translations(self.state)]

    def observation(self):
        if self.player_count != 2:
            raise ValueError('The registered neural information contract is heads-up only')
        if self.actor is None:
            raise ValueError('No acting information set')
        s, actor = self.state, self.state.actor_index
        other = 1 - actor
        return {
            'schema': 'infoset-v1',
            'hole': sorted(map(repr, s.hole_cards[actor])),
            'board': [repr(c) for group in s.board_cards for c in group],
            'street': s.street_index, 'position': int(self.actor == self.button),
            'pot': s.total_pot_amount, 'own_stack': s.stacks[actor],
            'opponent_stack': s.stacks[other], 'contribution': s.bets[actor],
            'to_call': s.checking_or_calling_amount or 0,
            'min_raise': s.min_completion_betting_or_raising_to_amount or 0,
            'max_raise': s.max_completion_betting_or_raising_to_amount or 0,
            'spr': s.stacks[actor] / max(1, s.total_pot_amount),
            'legal_mask': self.legal_mask(),
            'history': [dict(street=h['street'], actor=int(h['actor'] == self.actor),
                             action=h['action'], wager_bb=h['wager_bb'],
                             pot_fraction=h['pot_fraction']) for h in self.history],
        }

    def act(self, action):
        choices = translations(self.state)
        if type(action) is not int or not 0 <= action < 5 or choices[action] is None:
            raise ValueError('Illegal or duplicate abstract action')
        s, actor = self.state, self.actor
        kind, amount = choices[action]
        pot, street = s.total_pot_amount, s.street_index
        prior_bet = s.bets[s.actor_index]
        if kind == 'fold':
            s.fold()
        elif kind == 'call':
            s.check_or_call()
        else:
            s.complete_bet_or_raise_to(amount)
        paid = 0 if kind == 'fold' else amount if kind == 'call' else amount - prior_bet
        self.history.append({'actor': actor, 'street': street, 'action': action,
            'name': NAMES[action], 'target': amount, 'paid': paid,
            'wager_bb': paid / 2, 'pot_fraction': paid / max(1, pot)})
        self._advance_deal()
        return self.history[-1]

    def view(self):
        s = self.state
        stacks = [0] * self.player_count
        bets = [0] * self.player_count
        payoffs = [0] * self.player_count
        for i, seat in enumerate(self.seats):
            stacks[seat], bets[seat], payoffs[seat] = s.stacks[i], s.bets[i], s.payoffs[i]
        fly = self.seats.index(0)
        other = self.seats.index(1)
        showdown = self.done and not s.folded_status
        return {'button': self.button, 'actor': self.actor, 'pot': s.total_pot_amount,
                'stacks': stacks, 'bets': bets, 'hole': list(map(repr, self.dealt_holes[fly])),
                'opponent_hole': list(map(repr, s.hole_cards[other])) if showdown else ['??', '??'],
                'board': [repr(c) for group in s.board_cards for c in group],
                'street': 'SETTLED' if self.done else ['PREFLOP', 'FLOP', 'TURN', 'RIVER'][s.street_index],
                'history': list(self.history), 'legal_mask': self.legal_mask(),
                'done': self.done, 'payoffs': payoffs if self.done else None}

    def serialize(self):
        """Private checkpoint only. Never use this payload as an observation/corpus row."""
        from .observation import canonical_bytes
        import hashlib
        public = self.view()
        return {'schema': 'private-hand-checkpoint-v1', 'seed': self.seed, 'button': self.button,
                'stacks': self.starting_stacks, 'initial_deck': self.initial_deck,
                'actions': [h['action'] for h in self.history],
                'view_sha256': hashlib.sha256(canonical_bytes(public)).hexdigest()}

    @classmethod
    def restore(cls, checkpoint):
        from .observation import canonical_bytes
        import hashlib
        expected = {'schema', 'seed', 'button', 'stacks', 'initial_deck', 'actions', 'view_sha256'}
        if set(checkpoint) != expected or checkpoint['schema'] != 'private-hand-checkpoint-v1':
            raise ValueError('Invalid private hand checkpoint')
        hand = cls(checkpoint['seed'], checkpoint['button'], tuple(checkpoint['stacks']), checkpoint['initial_deck'])
        for action in checkpoint['actions']:
            hand.act(action)
        if hashlib.sha256(canonical_bytes(hand.view())).hexdigest() != checkpoint['view_sha256']:
            raise ValueError('Hand replay checkpoint mismatch')
        return hand
