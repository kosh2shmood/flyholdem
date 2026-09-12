"""Preregisterable opponent controls. Never imported by the neural policy."""
import hashlib
import random
from pokerkit import StandardHighHand
from .observation import canonical_bytes

VERSIONS={name:'v1' for name in ('random','calling-station','tight-aggressive','equity-bucket')}
CARDS=[rank+suit for rank in '23456789TJQKA' for suit in 'cdhs']


def visible_strength(observation):
    """Deliberately simple public-information rule score; not an equity label."""
    hole=observation['hole'];board=observation['board']
    ranks=['23456789TJQKA'.index(c[0])+2 for c in hole]
    value=sum(ranks)/28*.55
    if ranks[0]==ranks[1]:value+=.35
    if hole[0][1]==hole[1][1]:value+=.05
    matches=sum(c[0] in [h[0] for h in hole] for c in board)
    value+=min(2,matches)*.18
    return min(.98,value)


def visible_equity(observation,samples=64):
    """Monte Carlo using only this opponent's hole cards and visible board."""
    hole,board=observation['hole'],observation['board']
    remaining=[c for c in CARDS if c not in hole+board]
    seed=int.from_bytes(hashlib.sha256(canonical_bytes(observation)).digest()[:8],'big')
    rng=random.Random(seed);wins=0
    for _ in range(samples):
        draw=rng.sample(remaining,7-len(board));future=board+draw[2:]
        own=StandardHighHand.from_game(''.join(hole),''.join(future))
        other=StandardHighHand.from_game(''.join(draw[:2]),''.join(future))
        wins+=int(own>other)+.5*int(own==other)
    return wins/samples


class Opponent:
    def __init__(self, seed=1, kind='calling-station'):
        if kind not in VERSIONS:
            raise ValueError(f'Unknown opponent {kind}')
        self.rng,self.kind=random.Random(seed),kind

    def act(self,observation):
        legal=[i for i,ok in enumerate(observation['legal_mask']) if ok]
        if not legal:raise ValueError('Opponent requires legal actions')
        if self.kind=='random':return self.rng.choice(legal)
        if self.kind=='calling-station':return 1 if 1 in legal else self.rng.choice(legal)
        strength=visible_equity(observation) if self.kind=='equity-bucket' else visible_strength(observation)
        price=observation['to_call']/max(1,observation['pot']+observation['to_call'])
        if 0 in legal and strength<max(.34,price+.14):return 0
        if strength>=.67:
            for action in (3,2,4):
                if action in legal:return action
        return 1 if 1 in legal else legal[0]
