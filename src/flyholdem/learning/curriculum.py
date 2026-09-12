"""Registered poker subgames using PokerKit alone for dealing and settlement.

Curriculum setup is separate from policy decisions. Only the legal action mask
is restricted; no teacher target, equity or hidden state enters observation().
"""
from flyholdem.poker.engine import Hand
from flyholdem.poker.actions import translations
from flyholdem.provenance import identity

CURRICULA={
    'shove-fold-10bb-v1':{'stack_bb':10,'start_street':0,'actions':'fold-or-full-stack'},
    'river-20bb-v1':{'stack_bb':20,'start_street':3,'actions':'five-canonical'},
    'hu-20bb-v1':{'stack_bb':20,'start_street':0,'actions':'five-canonical'},
    'hu-100bb-v1':{'stack_bb':100,'start_street':0,'actions':'five-canonical'},
}
RANGES='uniform-independent-standard-deck-v1'


class CurriculumHand:
    def __init__(self,name,seed,button=0):
        if name not in CURRICULA:raise ValueError('Unknown registered poker curriculum')
        self.name=name;self.config=dict(CURRICULA[name])
        self.hand=Hand(seed,button=button,stacks=(2*self.config['stack_bb'],)*2)
        self.setup_actions=[]
        # River subgame starts after a declared check/call prefix. These actions
        # define the initial subgame, never substitute for a neural decision.
        while not self.hand.done and self.hand.state.street_index<self.config['start_street']:
            self.setup_actions.append(dict(self.hand.act(1)))
        if self.hand.done:raise ValueError('Curriculum setup unexpectedly settled before its start street')
        self.start_view=self.hand.view()

    @property
    def actor(self):return self.hand.actor
    @property
    def done(self):return self.hand.done
    @property
    def registration(self):
        return {'schema':'poker-curriculum-v1','name':self.name,**self.config,'ranges':RANGES,
            'blinds':[1,2],'engine':'PokerKit','river_prefix':'check/call every decision until river',
            'full_stack_call':'canonical check/call index 1 when it commits the complete remaining stack; no duplicate action is added'}

    def legal_mask(self):
        mask=self.hand.legal_mask()
        if self.done or self.config['actions']=='five-canonical':return mask
        s=self.hand.state;actor=s.actor_index;stack=s.stacks[actor]
        choices=translations(s);result=[False]*5
        for index,choice in enumerate(choices):
            if choice is None:continue
            kind,amount=choice
            paid=0 if kind=='fold' else amount if kind=='call' else amount-s.bets[actor]
            result[index]=(kind=='fold' or paid==stack)
        if not any(result):raise RuntimeError('No canonical fold/full-stack representative is available')
        return result

    def observation(self):
        observation=self.hand.observation();observation['legal_mask']=self.legal_mask()
        return observation

    def act(self,action):
        if type(action) is not int or action not in range(5) or not self.legal_mask()[action]:
            raise ValueError('Action is illegal in this registered curriculum')
        return self.hand.act(action)

    def view(self):
        view=self.hand.view();view['legal_mask']=self.legal_mask()
        return view

    def serialize(self):
        return {'schema':'private-curriculum-hand-v1','registration':self.registration,
            'registration_sha256':identity(self.registration),'hand':self.hand.serialize(),
            'setup_actions':self.setup_actions}

    @classmethod
    def restore(cls,value):
        if value.get('schema')!='private-curriculum-hand-v1':raise ValueError('Unknown curriculum checkpoint schema')
        game=cls(value['registration']['name'],value['hand']['seed'],value['hand']['button'])
        if value['registration']!=game.registration or value['registration_sha256']!=identity(game.registration):
            raise ValueError('Curriculum registration mismatch')
        if value['setup_actions']!=game.setup_actions:raise ValueError('Curriculum setup actions mismatch')
        game.hand=Hand.restore(value['hand'])
        if game.hand.starting_stacks!=[2*game.config['stack_bb']]*2:raise ValueError('Curriculum stack mismatch')
        if game.hand.history[:len(game.setup_actions)]!=game.setup_actions:raise ValueError('Restored river setup differs')
        # Revalidate every policy action under the same curriculum mask.
        verifier=cls(value['registration']['name'],value['hand']['seed'],value['hand']['button'])
        for action in value['hand']['actions'][len(game.setup_actions):]:verifier.act(action)
        if verifier.hand.serialize()!=game.hand.serialize():raise ValueError('Curriculum replay mismatch')
        return game
