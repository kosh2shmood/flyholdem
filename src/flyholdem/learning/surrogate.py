"""Explicitly nonbiological direct-readout surrogate gradient on existing edges.

Forward scores are the actual native LIF scores. Backward derivatives use a
declared rate approximation of direct KC-to-readout paths, ignoring recurrent
and threshold derivatives. No approximate score is used to select an action.
"""
import numpy as np
from flyholdem.provenance import identity


class ReadoutSurrogate:
    def __init__(self, eligibility, ensembles, config):
        self.eligible = eligibility
        self.config = dict(config)
        if set(config) != {'learning_rate', 'temperature', 'gradient_norm_cap', 'synaptic_tau_ms', 'threshold_distance_mv'}:
            raise ValueError('Explicit registered surrogate parameters required')
        if any(not np.isfinite(v) or v <= 0 for v in config.values()):
            raise ValueError('Positive finite surrogate parameters required')
        if len(ensembles) != 5 or any(len(e) == 0 for e in ensembles):
            raise ValueError('Five fixed nonempty readout ensembles required')
        neurons = np.concatenate(ensembles)
        if len(neurons) != len(np.unique(neurons)):
            raise ValueError('Surrogate readouts must be disjoint')
        lookup = np.full(eligibility.brain.n, -1, dtype=np.int32)
        for action, group in enumerate(ensembles): lookup[group] = action
        self.action = lookup[eligibility.brain.post[eligibility.edges]]
        self.direct = np.flatnonzero(self.action >= 0)
        if not len(self.direct): raise ValueError('No preregistered direct readout edges')
        self.sizes = np.asarray([len(e) for e in ensembles], dtype=np.float64)

    @property
    def registration(self):
        return {'method':'direct-readout-rate-surrogate-v1','biological_learning':False,
            'score_source':'actual-native-LIF-forward-pass','trainable':'existing-registered-KC-MBON-multipliers',
            'direct_edge_count':len(self.direct),
            'direct_edge_indices_hash':identity(self.eligible.edges[self.direct].tolist()),
            'jacobian':'original_weight * filtered_pre_count * tau_syn / tau_pre / threshold_distance / ensemble_size',
            'ignored_derivatives':'recurrent paths, presynaptic activity and hard spike thresholds',
            'config':self.config}

    def jacobian(self):
        e = self.eligible
        values = np.zeros(len(e.edges), dtype=np.float64)
        d = self.direct
        values[d] = (e.initial[d].astype(np.float64) * e.pretrace[e.pre[d]]
            * self.config['synaptic_tau_ms'] / e.config['pre_tau_ms']
            / self.config['threshold_distance_mv'] / self.sizes[self.action[d]])
        return values

    def loss_gradient(self, native_scores, legal, target):
        scores=np.asarray(native_scores,dtype=np.float64);legal=np.asarray(legal,dtype=bool)
        target=np.asarray(target,dtype=np.float64)
        if (scores.shape!=(5,) or legal.shape!=(5,) or target.shape!=(5,) or not legal.any()
                or not np.isfinite(scores).all() or not np.isfinite(target).all()
                or np.any(target<0) or np.any(target[~legal]!=0) or not np.isclose(target.sum(),1,atol=1e-10,rtol=0)):
            raise ValueError('Finite native scores and a legal teacher distribution required')
        temperature=self.config['temperature']
        logits=scores[legal]/temperature;logits-=logits.max()
        log_probability=logits-np.log(np.exp(logits).sum())
        probability=np.zeros(5);probability[legal]=np.exp(log_probability)
        loss=-float(np.sum(target[legal]*log_probability))
        error=(probability-target)/temperature
        gradient=np.zeros(len(self.eligible.edges),dtype=np.float64)
        gradient[self.direct]=error[self.action[self.direct]]*self.jacobian()[self.direct]
        return loss,gradient

    def step(self, native_scores, legal, target, learning=True):
        e=self.eligible
        loss,gradient=self.loss_gradient(native_scores,legal,target)
        norm=float(np.linalg.norm(gradient))
        gradient*=min(1.0,self.config['gradient_norm_cap']/max(norm,1e-30))
        old=e.brain.weights[e.edges].copy()
        ratios=old.astype(np.float64)/e.initial
        proposed=ratios-self.config['learning_rate']*gradient
        bounded=np.clip(proposed,e.lower,e.upper)
        if learning:
            e.brain.weights[e.edges]=(e.initial*bounded).astype(np.float32)
        delta=e.brain.weights[e.edges]-old
        actual=e.brain.weights[e.edges].astype(np.float64)/e.initial
        return {'optimization':'nonbiological-surrogate-gradient','learning':bool(learning),
            'native_cross_entropy':loss,'surrogate_gradient_norm':norm,
            'gradient_clipped':norm>self.config['gradient_norm_cap'],
            'changed_synapses':int(np.count_nonzero(delta)),'absolute_update':float(np.abs(delta.astype(float)).sum()),
            'clipped_updates':int(np.count_nonzero(proposed!=bounded)) if learning else 0,
            'weight_ratio_range':[float(actual.min()),float(actual.max())],
            'dopamine_delivered':False,'native_scores_overridden':False}
