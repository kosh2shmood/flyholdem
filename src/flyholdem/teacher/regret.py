"""External-sampling tabular regret against a fixed conventional population.

Own actions are enumerated; chance and fixed-opponent actions are sampled.
This is a population-response experiment, not a Nash-equilibrium claim. The
card abstraction remembers every earlier street bucket and own/public action.
No biological controller imports this module.
"""
import copy
import hashlib
import numpy as np
from flyholdem.poker.infoset import canonical_state
from flyholdem.provenance import canonical,identity
from flyholdem.teacher.equity import visible_equity
from .poker_tree import fork_hand,counterfactual_root

VERSION='visible-equity-recall-public-history-v1'


def abstraction(observation,config):
    state=canonical_state(observation)
    total_chips=state['own_stack']+state['opponent_stack']+state['pot']
    hole=tuple(state['hole']);board=tuple(state['board']);buckets=[]
    for length in (0,3,4,5)[:state['street']+1]:
        equity=visible_equity(hole,board[:length],config['equity_samples'])
        buckets.append(min(config['equity_buckets']-1,int(equity*config['equity_buckets'])))
    # Full public action/paid-chip sequence retains the earlier price, pot and
    # stack states. Every earlier private bucket is recomputed from the visible
    # board prefix, never the opponent's cards or the actual unrevealed deck.
    history=[[h['street'],h['actor'],h['action'],int(h['wager_bb']*2)] for h in state['history']]
    public=[state[k] for k in ('position','street','pot','own_stack','opponent_stack','contribution','to_call','min_raise','max_raise')]
    legal=np.asarray(state['legal_mask'],dtype=bool)
    key=canonical([VERSION,total_chips,buckets,public,history,legal.tolist()])
    return key,legal


def regret_matching(values,legal):
    values=np.asarray(values,dtype=np.float64);legal=np.asarray(legal,dtype=bool)
    if values.shape!=(5,) or legal.shape!=(5,) or not legal.any() or not np.isfinite(values).all():
        raise ValueError('Finite regrets and canonical legal actions required')
    positive=np.where(legal,np.maximum(values,0),0)
    return positive/positive.sum() if positive.sum()>0 else legal.astype(float)/legal.sum()


class RegretTable:
    def __init__(self,config):
        if (config.get('feature_version')!=VERSION or type(config['stack_bb']) is not int or config['stack_bb']<=0
                or type(config['equity_buckets']) is not int or not 2<=config['equity_buckets']<=100
                or type(config['equity_samples']) is not int or not 1<=config['equity_samples']<=4096
                or type(config['max_information_sets']) is not int or config['max_information_sets']<1
                or type(config['max_nodes_per_traversal']) is not int or config['max_nodes_per_traversal']<1):
            raise ValueError('Complete registered conventional abstraction required')
        self.config=dict(config);self.keys=[];self.index={};self.capacity=0
        self.regrets=np.zeros((0,5),dtype=np.float64);self.averages=self.regrets.copy()
        self.legal=np.zeros((0,5),dtype=bool);self.visits=np.zeros(0,dtype=np.uint64)

    def _grow(self,needed):
        if needed>self.config['max_information_sets']:raise MemoryError('Registered information-set capacity exceeded')
        if needed<=self.capacity:return
        capacity=min(self.config['max_information_sets'],max(1024,2*self.capacity,needed))
        for name in ('regrets','averages','legal','visits'):
            prior=getattr(self,name);value=np.zeros((capacity,*prior.shape[1:]),dtype=prior.dtype)
            value[:len(self.keys)]=prior[:len(self.keys)];setattr(self,name,value)
        self.capacity=capacity

    def lookup(self,key,legal):
        if key in self.index:
            i=self.index[key]
            if not np.array_equal(self.legal[i],legal):raise ValueError('Abstraction merged different legal action sets')
            return i
        self._grow(len(self.keys)+1);i=len(self.keys);self.keys.append(key);self.index[key]=i;self.legal[i]=legal
        return i

    def probabilities(self,observation):
        key,legal=abstraction(observation,self.config);i=self.index.get(key)
        if i is None:return legal.astype(float)/legal.sum()
        if not np.array_equal(self.legal[i],legal):raise ValueError('Stored conventional information set changed legality')
        values=self.averages[i]
        if np.any(values<0) or not np.isfinite(values).all():raise ValueError('Nonfinite/negative average strategy')
        return values/values.sum() if values.sum()>0 else legal.astype(float)/legal.sum()

    def step(self,game,seat,opponent):
        """One actual PokerKit chance deal and a sampled fixed-opponent tree.

        Against a stationary sampled population, sampled opponent/chance reach
        is constant in expectation. Its factor cancels when normalizing the
        own-reach-weighted average at an information set. Regret updates are
        counterfactual: they omit the updating player's prefix reach.
        """
        if game.done or seat not in (0,1):raise ValueError('A fresh actual heads-up traversal is required')
        raw=game.hand if hasattr(game,'hand') else game
        if raw.starting_stacks!=[2*self.config['stack_bb']]*2 or raw.history:
            raise ValueError('Start a fresh registered heads-up training hand')
        trace=hashlib.sha256();nodes=leaves=0;created_before=len(self.keys);seen=set()
        def record(value):trace.update(canonical(value)+b'\n')
        def walk(hand,other,own_reach):
            nonlocal nodes,leaves
            nodes+=1
            if nodes>self.config['max_nodes_per_traversal']:raise RuntimeError('Registered complete-traversal node ceiling exceeded')
            if hand.done:
                leaves+=1;table=hand.view()
                if sum(table['payoffs'])!=0:raise AssertionError('Counterfactual PokerKit settlement is not zero sum')
                value=table['payoffs'][seat]/2
                record({'terminal':hand.serialize(),'return_bb':value});return float(value)
            observation=hand.observation()
            if hand.actor!=seat:
                action=other.act(observation);committed=hand.act(action)
                record({'fixed_opponent_action':committed,'visible_observation':observation})
                return walk(hand,other,own_reach)
            key,legal=abstraction(observation,self.config)
            if key in seen:raise AssertionError('A traversal revisited an abstract information set despite complete public recall')
            seen.add(key);i=self.lookup(key,legal)
            strategy=regret_matching(self.regrets[i],legal).copy();values=np.zeros(5,dtype=float)
            for action in np.flatnonzero(legal):
                child=fork_hand(hand);child.act(int(action))
                values[action]=walk(child,copy.deepcopy(other),own_reach*strategy[action])
            value=float(np.dot(strategy,values));delta=np.where(legal,values-value,0)
            # Reacquire array rows after recursion: lookup may have grown them.
            self.regrets[i]+=delta;self.averages[i]+=own_reach*strategy;self.visits[i]+=1
            if not np.isfinite(self.regrets[i]).all() or not np.isfinite(self.averages[i]).all():
                raise FloatingPointError('Nonfinite conventional regret update')
            record({'information_key_sha256':hashlib.sha256(key).hexdigest(),'strategy':strategy.tolist(),
                'action_values_bb':values.tolist(),'own_reach':float(own_reach),'regret_delta':delta.tolist(),
                'regrets_after':self.regrets[i].tolist(),'average_after':self.averages[i].tolist()})
            return value
        root=game.serialize()
        with counterfactual_root(game) as tree:value=walk(tree,copy.deepcopy(opponent),1.)
        if root!=game.serialize():raise AssertionError('Counterfactual traversal mutated its initial PokerKit hand')
        return {'root_private_checkpoint':root,'learning_seat':seat,'sampled_policy_value_bb':value,
            'nodes':nodes,'terminal_branches':leaves,'information_sets':len(self.keys),
            'new_information_sets':len(self.keys)-created_before,'complete_traversal_sha256':trace.hexdigest(),
            'scope':'conventional counterfactual training values; not held-out returns or a fly result'}

    def state(self):
        lengths=np.fromiter((len(key) for key in self.keys),dtype=np.int64,count=len(self.keys))
        offsets=np.concatenate((np.zeros(1,dtype=np.int64),np.cumsum(lengths)))
        return {'key_offsets':offsets,'key_bytes':np.frombuffer(b''.join(self.keys),dtype=np.uint8).copy(),
            **{name:getattr(self,name)[:len(self.keys)].copy() for name in ('regrets','averages','legal','visits')}}

    def restore(self,arrays):
        expected={'key_offsets','key_bytes','regrets','averages','legal','visits'}
        if set(arrays)!=expected:raise ValueError('Incomplete numeric regret table')
        offsets=arrays['key_offsets'];data=arrays['key_bytes'];n=len(offsets)-1
        if (offsets.dtype!=np.int64 or offsets.ndim!=1 or n<0 or offsets[0]!=0 or np.any(np.diff(offsets)<=0)
                or data.dtype!=np.uint8 or data.ndim!=1 or offsets[-1]!=len(data)
                or arrays['regrets'].dtype!=np.float64 or arrays['regrets'].shape!=(n,5)
                or arrays['averages'].dtype!=np.float64 or arrays['averages'].shape!=(n,5)
                or arrays['legal'].dtype!=np.bool_ or arrays['legal'].shape!=(n,5)
                or arrays['visits'].dtype!=np.uint64 or arrays['visits'].shape!=(n,)):
            raise ValueError('Invalid regret table numeric shapes')
        if (not np.isfinite(arrays['regrets']).all() or not np.isfinite(arrays['averages']).all()
                or np.any(arrays['averages']<0) or np.any(~arrays['legal'].any(axis=1))
                or np.any(arrays['regrets'][~arrays['legal']]) or np.any(arrays['averages'][~arrays['legal']])):
            raise ValueError('Invalid stored legal regrets/average strategy')
        keys=[data[a:b].tobytes() for a,b in zip(offsets[:-1],offsets[1:])]
        if len(set(keys))!=n:raise ValueError('Duplicate regret information key')
        import json
        for key,legal in zip(keys,arrays['legal']):
            decoded=json.loads(key)
            if (canonical(decoded)!=key or decoded[0]!=VERSION or decoded[1]!=4*self.config['stack_bb']
                    or decoded[-1]!=legal.tolist()):raise ValueError('Invalid canonical stored information key')
        self.keys=[];self.index={};self.capacity=0
        for name in ('regrets','averages','legal','visits'):setattr(self,name,arrays[name][:0].copy())
        self._grow(n);self.keys=keys;self.index={key:i for i,key in enumerate(keys)}
        for name in ('regrets','averages','legal','visits'):getattr(self,name)[:n]=arrays[name]
