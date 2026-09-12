"""Chance-sampled tabular CFR for the declared 10 BB shove/fold subgame.

All terminal samples come from PokerKit. Current and averaged policies depend
only on the acting player's own preflop card class and public decision node.
This small-game reference does not qualify the separate full-hand teacher.
"""
from functools import lru_cache
from pathlib import Path
import json
import numpy as np
from flyholdem.learning.curriculum import CurriculumHand
from flyholdem.poker.infoset import canonical_state
from flyholdem.provenance import identity
from flyholdem.connectome.registry import digest
from flyholdem.neural.checkpoint import atomic_json

CURRICULUM='shove-fold-10bb-v1'
ALGORITHM='chance-sampled-shove-fold-CFR-v1'
RANKS='23456789TJQKA'


def card_class(hole):
    if len(hole)!=2 or len(set(hole))!=2 or any(len(c)!=2 or c[0] not in RANKS or c[1] not in 'cdhs' for c in hole):
        raise ValueError('Two distinct standard hole cards required')
    high,low=sorted((RANKS.index(c[0]) for c in hole),reverse=True)
    return high*13+low if hole[0][1]==hole[1][1] or high==low else low*13+high


@lru_cache(maxsize=1)
def public_nodes():
    hand=CurriculumHand(CURRICULUM,0)
    sb=canonical_state(hand.observation());hand.act(4);bb=canonical_state(hand.observation())
    return tuple({k:v for k,v in state.items() if k!='hole'} for state in (sb,bb))


def information_key(observation):
    state=canonical_state(observation);node=1-state['position']
    if {k:v for k,v in state.items() if k!='hole'}!=public_nodes()[node]:
        raise ValueError('Tabular policy accepts only the registered 10 BB shove/fold nodes')
    return node,card_class(state['hole'])


def sample_payoffs(seed):
    """Traverse all three terminal branches on one common sampled deal."""
    folded=CurriculumHand(CURRICULUM,seed);sb=information_key(folded.observation());folded.act(0)
    declined=CurriculumHand(CURRICULUM,seed);declined.act(4)
    bb=information_key(declined.observation());declined.act(0)
    called=CurriculumHand(CURRICULUM,seed);called.act(4)
    if information_key(called.observation())!=bb:raise AssertionError('Counterfactual branches changed their sampled cards')
    called.act(1)
    terminal=[game.view() for game in (folded,declined,called)]
    if any(not row['done'] or sum(row['payoffs'])!=0 for row in terminal):raise AssertionError('Unsettled or nonzero-sum sampled PokerKit branch')
    return {'deal_seed':seed,'classes':[sb[1],bb[1]],
        'sb_terminal_bb':[row['payoffs'][0]/2 for row in terminal],
        'private_showdown_hand':called.serialize()}


def regret_matching(regret):
    positive=np.maximum(regret,0);total=positive.sum()
    return positive/total if total>0 else np.full(2,.5)


class ShoveFoldCFR:
    def __init__(self):
        self.regrets=np.zeros((2,169,2),dtype=np.float64)
        self.strategy_sum=np.zeros_like(self.regrets)
        self.visits=np.zeros((2,169),dtype=np.int64)

    def step(self,sample):
        a,b=sample['classes'];fold,decline,showdown=map(float,sample['sb_terminal_bb'])
        if (type(a) is not int or type(b) is not int or a not in range(169) or b not in range(169)
                or not np.isfinite([fold,decline,showdown]).all()):raise ValueError('Invalid chance sample')
        # Both players use the pre-update strategy for this complete traversal.
        sb=regret_matching(self.regrets[0,a]);bb=regret_matching(self.regrets[1,b])
        sb_values=np.array([fold,bb[0]*decline+bb[1]*showdown])
        bb_values=-np.array([decline,showdown])
        sb_delta=sb_values-np.dot(sb,sb_values)
        bb_delta=sb[1]*(bb_values-np.dot(bb,bb_values))
        self.regrets[0,a]+=sb_delta;self.regrets[1,b]+=bb_delta
        # Neither player has an earlier own decision at its information set.
        # Opponent reach weights BB regret, but must NOT weight its average.
        self.strategy_sum[0,a]+=sb;self.strategy_sum[1,b]+=bb
        self.visits[0,a]+=1;self.visits[1,b]+=1
        return {'current_strategy':[sb.tolist(),bb.tolist()],
            'counterfactual_regret_update':[sb_delta.tolist(),bb_delta.tolist()],
            'sampled_value_sb_bb':float(np.dot(sb,sb_values))}

    def average(self):
        totals=self.strategy_sum.sum(axis=2,keepdims=True)
        return np.divide(self.strategy_sum,totals,out=np.full_like(self.strategy_sum,.5),where=totals>0)

    def state(self):return {key:getattr(self,key).copy() for key in ('regrets','strategy_sum','visits')}

    def restore(self,arrays):
        expected=self.state()
        if set(arrays)!=set(expected):raise ValueError('Incomplete CFR state')
        for key,value in arrays.items():
            if value.shape!=expected[key].shape or value.dtype!=expected[key].dtype or not np.isfinite(value).all():
                raise ValueError('Invalid CFR state shape, type or values')
        if (np.any(arrays['strategy_sum']<0) or np.any(arrays['visits']<0)
                or not np.allclose(arrays['strategy_sum'].sum(axis=2),arrays['visits'],rtol=0,atol=1e-7)
                or arrays['visits'][0].sum()!=arrays['visits'][1].sum()):raise ValueError('Inconsistent CFR visitation or average mass')
        for key,value in arrays.items():getattr(self,key)[:]=value


class ShoveFoldPolicy:
    def __init__(self,probabilities):
        values=np.asarray(probabilities,dtype=np.float64)
        if (values.shape!=(2,169,2) or not np.isfinite(values).all() or np.any(values<0)
                or not np.allclose(values.sum(axis=2),1,rtol=0,atol=1e-12)):
            raise ValueError('Finite, normalized tabular action probabilities required')
        self.table=values.copy();self.table.flags.writeable=False

    def probabilities(self,observation):
        node,card=information_key(observation);values=np.zeros(5,dtype=np.float64)
        values[[0,4] if node==0 else [0,1]]=self.table[node,card]
        return values


def policy_runtime():
    import importlib
    from importlib.metadata import version
    import sys
    names=('flyholdem.learning.curriculum','flyholdem.poker.engine','flyholdem.poker.actions',
           'flyholdem.poker.infoset','flyholdem.poker.observation')
    return {'python':sys.version,'numpy':np.__version__,'pokerkit':version('pokerkit'),
        'source_files':{name:digest(importlib.import_module(name).__file__) for name in names}}


def export_policy(solver,output,provenance):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    path=output/'probabilities.npy';np.save(path,solver.average(),allow_pickle=False)
    record={'schema':'tabular-shove-fold-policy-v1','algorithm':ALGORITHM,'curriculum':CURRICULUM,
        'files':{'probabilities.npy':digest(path)},'provenance':provenance,
        'class_mapping':'13x13 ranks 2..A; diagonal pairs, lower suited, upper offsuit',
        'public_nodes_sha256':identity(public_nodes()),'source_sha256':digest(__file__),'implementation':policy_runtime(),
        'allowed_as_teacher':False,'status':'frozen-small-game-reference-unvalidated',
        'scope':'10 BB shove/fold only; not the registered full 20 BB teacher or a fly controller'}
    atomic_json(output/'manifest.json',record);return record


def load_policy(output):
    output=Path(output);record=json.loads((output/'manifest.json').read_text())
    if (record.get('schema')!='tabular-shove-fold-policy-v1' or record.get('algorithm')!=ALGORITHM
            or record.get('curriculum')!=CURRICULUM or record.get('source_sha256')!=digest(__file__)
            or record.get('implementation')!=policy_runtime() or record.get('public_nodes_sha256')!=identity(public_nodes())
            or record.get('files')!={'probabilities.npy':digest(output/'probabilities.npy')}):
        raise ValueError('Tabular policy artifact, source or public-node mismatch')
    return ShoveFoldPolicy(np.load(output/'probabilities.npy',allow_pickle=False)),record
