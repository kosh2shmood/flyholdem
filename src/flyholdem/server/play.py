"""Local play-chip heads-up sessions with private neural state and public logs.

Human seat 1 receives only its own cards and public poker information. Fly
observations, encodings, scores, activity, private seeds and private hashes never
enter the human event stream, including on folded hands.
"""
import hashlib
import os
from pathlib import Path
import secrets
import threading
import time
import numpy as np
from flyholdem.poker.engine import Hand
from flyholdem.poker.actions import translations, NAMES
from flyholdem.neural.simulator import FixtureBrain
from flyholdem.interface.population import NeuralController
from flyholdem.neural.sparse import SparseBrain
from .events import dumps


class PlayError(ValueError):
    def __init__(self, message, status=409):
        super().__init__(message);self.status=status


class FrozenPlayer:
    def __init__(self, demo):
        self.native=hasattr(demo,'controller')
        source=demo.brain
        self.weights=source.weights.copy()
        if self.native:
            brain=SparseBrain(source.ptr,source.post,source.initial)
            brain.weights[:]=self.weights
            self.controller=NeuralController(brain,demo.controller.registration,seed=0)
            self.brain=brain
        else:
            self.brain=FixtureBrain(seed=0);self.brain.weights[:]=self.weights
        self.mode=getattr(demo,'mode','fixture')
        self.learning_mode=getattr(demo,'learning_mode','bio-plastic')
        self.identity={'mode':self.mode,'learning_mode':self.learning_mode,'teacher':'disconnected',
            'weights_sha256':hashlib.sha256(self.weights.tobytes()).hexdigest(),
            'graph_sha256':source.graph_hash,'weights':'frozen at match start','policy_status':'unvalidated for poker'}

    def reset(self):
        if self.native:self.brain.reset_dynamics()
        else:
            self.brain=FixtureBrain(seed=0);self.brain.weights[:]=self.weights

    def decide(self, observation):
        if self.native:
            result=self.controller.decide(observation,temperature=0)
            self.controller.commit_interval()
            return result
        return self.brain.decide(observation,temperature=0)


class HumanSession:
    def __init__(self, player, private_root, token=None, seed_factory=None):
        self.player=player;self.token=token or secrets.token_hex(16)
        self.seed_factory=seed_factory or (lambda:secrets.randbits(63))
        self.lock=threading.RLock();self.hand=None;self.hand_number=0
        self.events=[];self.previous_hash='0'*64;self.return_bb=0
        self.halted=False;self.closed=False;self.last_access=time.monotonic()
        self.root=Path(private_root)/self.token;self.root.mkdir(parents=True,mode=0o700,exist_ok=False)
        self.stream=(self.root/'private.jsonl').open('x',buffering=1)
        os.chmod(self.root/'private.jsonl',0o600)
        changed=np.flatnonzero(player.brain.weights!=player.brain.initial)
        np.savez(self.root/'frozen-weights.npz',indices=changed,weights=player.brain.weights[changed])
        self._private({'kind':'session','identity':player.identity,'neural_start':'fresh rest at each hand'})

    def _private(self,value):
        self.stream.write(dumps(value)+'\n')

    def _table(self, view=None):
        view=dict(self.hand.view() if view is None else view)
        view['history']=[dict(item) for item in view['history']]
        showdown=view['done'] and not self.hand.state.folded_status
        # Reveal only the cards PokerKit actually showed, including muck rules.
        shown=self.hand.state.hole_cards[self.hand.seats.index(0)] if showdown else []
        view['hole']=list(map(repr,shown)) if len(shown)==2 else ['??','??']
        human_index=self.hand.seats.index(1)
        view['opponent_hole']=[repr(card) for card in self.hand.dealt_holes[human_index]]
        return view

    def _emit(self,kind,actor=None,action=None,before=None):
        event={'schema':'flyholdem-event-v1','kind':kind,'sequence':len(self.events),'hand':self.hand_number,
            'mode':self.player.mode,'learning_mode':self.player.learning_mode,'status':'development',
            'teacher':'disconnected','encoder':'engineered-kc-v1','evaluation':True,'plasticity_enabled':False,
            'viewer':'human','opponent_name':'You','label':f'{self.player.mode.upper()} / HEADS-UP PLAY / POKER POLICY UNVALIDATED',
            'table':self._table(),'return_bb':self.return_bb,'previous_hash':self.previous_hash,
            'private_diagnostics_withheld':True}
        if actor is not None:event.update(actor=actor,action=dict(action),decision=None,table_before=self._table(before))
        if kind=='settlement':event.update(net_bb=self.hand.view()['payoffs'][0]/2,reward_delivered=False)
        event['hash']=hashlib.sha256(dumps(event).encode()).hexdigest()
        self.events.append(event);self.previous_hash=event['hash']
        return event

    @property
    def revision(self):return len(self.events)-1

    def _validate(self,revision):
        if self.closed:raise PlayError('This match has ended.',410)
        if self.halted:raise PlayError('The neural controller stopped. Start a new match.',503)
        if type(revision) is not int or revision!=self.revision:raise PlayError('The hand changed. Refresh before acting.')
        self.last_access=time.monotonic()

    def _settle_or_advance(self):
        while not self.hand.done and self.hand.actor==0:
            before=self.hand.view();observation=self.hand.observation()
            try:
                decision=self.player.decide(observation)
                action=self.hand.act(decision['selected'])
            except Exception:
                self.halted=True
                self._private({'kind':'controller-error','private_hand':self.hand.serialize()})
                raise PlayError('The neural controller stopped. No fallback action was used.',503)
            self._private({'kind':'fly-decision','decision':{k:v for k,v in decision.items() if k not in ('counts','activity')},
                           'private_hand':self.hand.serialize()})
            self._emit('decision',0,action,before)
        if self.hand.done:
            self.return_bb+=self.hand.view()['payoffs'][0]/2
            self._emit('settlement')
            self.stream.flush();os.fsync(self.stream.fileno())

    def new_hand(self,revision=-1):
        with self.lock:
            self._validate(revision)
            if self.hand is not None and not self.hand.done:raise PlayError('Finish this hand before dealing again.')
            start=len(self.events);self.player.reset()
            self.hand=Hand(self.seed_factory(),button=self.hand_number%2)
            self.hand_number+=1
            self._private({'kind':'hand_start','private_hand':self.hand.serialize()})
            self._emit('hand_start');self._settle_or_advance()
            return self.response(start)

    def act(self,revision,action):
        with self.lock:
            self._validate(revision)
            if self.hand is None or self.hand.done or self.hand.actor!=1:raise PlayError('It is not your turn.')
            if type(action) is not int or action not in range(5) or not self.hand.legal_mask()[action]:
                raise PlayError('That action is illegal or duplicates another wager.',422)
            start=len(self.events);before=self.hand.view();record=self.hand.act(action)
            self._private({'kind':'human-action','action':action,'private_hand':self.hand.serialize()})
            self._emit('opponent_action',1,record,before);self._settle_or_advance()
            return self.response(start)

    def response(self,start=None):
        with self.lock:
            self.last_access=time.monotonic()
            human_turn=bool(self.hand and not self.hand.done and self.hand.actor==1 and not self.halted)
            options=[]
            if human_turn:
                choices=translations(self.hand.state)
                for index,choice in enumerate(choices):
                    name=NAMES[index]
                    if choice is not None:
                        kind,amount=choice
                        if kind=='call':name='Check' if not amount else f'Call {amount} chips'
                        elif kind=='raise':name=f'{NAMES[index]} to {amount}'
                    options.append({'index':index,'name':name,'legal':choice is not None})
            return {'session':self.token,'revision':self.revision,'human_turn':human_turn,
                    'done':bool(self.hand and self.hand.done),'halted':self.halted,
                    'actions':options,'hero_return_bb':-self.return_bb,
                    'events':self.events[start:] if start is not None else self.events[-1:]}

    def close(self):
        with self.lock:
            if not self.closed:
                self.closed=True;self.stream.flush();os.fsync(self.stream.fileno());self.stream.close()
