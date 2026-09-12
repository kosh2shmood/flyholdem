"""Exact acceleration of the existing fixed opponents for teacher training.

Only the rank-comparison backend changes. Card order, visible-information hash,
Python RNG samples, thresholds and action priority remain exactly v1. The
independent evaluation suite continues using its original PokerKit implementation.
"""
import hashlib
import random
import numpy as np
from flyholdem.poker.opponents import Opponent,CARDS
from flyholdem.poker.observation import canonical_bytes
from .equity import rank_batch,INDEX


def exact_visible_equity(observation,samples=64):
    hole,board=observation['hole'],observation['board']
    remaining=[c for c in CARDS if c not in hole+board]
    seed=int.from_bytes(hashlib.sha256(canonical_bytes(observation)).digest()[:8],'big')
    rng=random.Random(seed);hands=[]
    for _ in range(samples):
        draw=rng.sample(remaining,7-len(board));future=board+draw[2:]
        hands.extend([[INDEX[c] for c in hole+future],[INDEX[c] for c in draw[:2]+future]])
    values=rank_batch(hands).reshape(samples,2)
    wins=np.count_nonzero(values[:,0]>values[:,1])+.5*np.count_nonzero(values[:,0]==values[:,1])
    return float(wins/samples)


class TrainingOpponent(Opponent):
    def act(self,observation):
        if self.kind!='equity-bucket':return super().act(observation)
        legal=[i for i,ok in enumerate(observation['legal_mask']) if ok]
        if not legal:raise ValueError('Opponent requires legal actions')
        strength=exact_visible_equity(observation)
        price=observation['to_call']/max(1,observation['pot']+observation['to_call'])
        if 0 in legal and strength<max(.34,price+.14):return 0
        if strength>=.67:
            for action in (3,2,4):
                if action in legal:return action
        return 1 if 1 in legal else legal[0]
