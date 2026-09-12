"""Canonical, deterministic viewer events; the UI never computes policy."""
import hashlib
import json
from flyholdem.poker.engine import Hand
from flyholdem.poker.opponents import Opponent
from flyholdem.neural.simulator import FixtureBrain

LABEL = 'FIXTURE / VISUAL PROTOTYPE / NOT A FULL-CONNECTOME RESULT'


def dumps(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


class Demo:
    def __init__(self, seed=20260912):
        self.seed, self.hand_number, self.sequence = seed, 0, 0
        self.brain = FixtureBrain(seed)
        self.opponent = Opponent(seed + 1)
        self.hand = None
        self.completed = False
        self.previous_hash = '0' * 64
        self.return_bb = 0

    def next_event(self):
        if self.hand is None or self.completed:
            self.hand = Hand(self.seed + self.hand_number, button=self.hand_number % 2)
            self.hand_number += 1
            self.completed = False
            event = {'kind': 'hand_start', 'table': self.hand.view()}
        elif self.hand.done:
            net_bb = self.hand.view()['payoffs'][0] / 2
            plasticity = self.brain.reinforce(net_bb, int(self.hand.button == 0))
            self.return_bb += net_bb
            event = {'kind': 'reinforcement', 'plasticity': plasticity, 'table': self.hand.view()}
            self.completed = True
        else:
            actor = self.hand.actor
            observation = self.hand.observation()
            before = self.hand.view()
            if actor == 0:
                decision = self.brain.decide(observation)
                action = decision['selected']
            else:
                decision = None
                action = self.opponent.act(observation)
            record = self.hand.act(action)
            event = {'kind': 'decision' if actor == 0 else 'opponent_action',
                'actor': actor, 'decision': decision, 'action': record,
                'table_before': before, 'table': self.hand.view()}
        event.update(schema='flyholdem-event-v1', sequence=self.sequence, hand=self.hand_number,
            mode='fixture', learning_mode='bio-plastic', status='development',
            teacher='disconnected', encoder='engineered-kc-v1', label=LABEL,
            return_bb=self.return_bb, previous_hash=self.previous_hash)
        event['hash'] = hashlib.sha256(dumps(event).encode()).hexdigest()
        self.previous_hash = event['hash']
        self.sequence += 1
        return event


def verify_stream(events):
    previous = '0' * 64
    for sequence, event in enumerate(events):
        body = {k:v for k,v in event.items() if k != 'hash'}
        if event['sequence'] != sequence or event['previous_hash'] != previous:
            raise ValueError('Broken event ordering')
        if hashlib.sha256(dumps(body).encode()).hexdigest() != event['hash']:
            raise ValueError('Event content hash mismatch')
        previous = event['hash']
    return previous
