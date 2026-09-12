"""Fixed population code and native neural-only action interface."""
import numpy as np
from .encoder import CHANNELS,encode_player_state,encoded_hash


def balanced_projection(inputs,fanout,seed):
    inputs=np.asarray(inputs,dtype=np.int32)
    if len(np.unique(inputs))!=len(inputs) or not 1<=fanout<=len(inputs):raise ValueError('Unique inputs and a valid fanout required')
    order=np.random.default_rng(seed).permutation(inputs)
    return order[np.arange(len(CHANNELS)*fanout).reshape(len(CHANNELS),fanout)%len(inputs)]


def population_drive(x,mapping,n,gain):
    x=np.asarray(x,dtype=np.float64);mapping=np.asarray(mapping,dtype=np.int32)
    if x.shape!=(len(CHANNELS),) or mapping.ndim!=2 or mapping.shape[0]!=len(x):raise ValueError('Population shape mismatch')
    if not np.isfinite(x).all() or np.any(x<0) or np.any(mapping<0) or np.any(mapping>=n) or not np.isfinite(gain) or gain<=0:raise ValueError('Invalid population code')
    drive=np.zeros(n,dtype=np.float32)
    np.add.at(drive,mapping.ravel(),np.repeat(x,mapping.shape[1]).astype(np.float32))
    np.minimum(drive*gain,gain*2,out=drive)
    return drive


def neural_scores(counts,ensembles,duration_ms,baseline_hz=None,scale_hz=100):
    if duration_ms<=0 or scale_hz<=0:raise ValueError('Positive neural measurement interval required')
    counts=np.asarray(counts)
    if len(ensembles)!=5 or any(not len(e) for e in ensembles):raise ValueError('Five nonempty ensembles required')
    flattened=np.concatenate(ensembles)
    if len(np.unique(flattened))!=len(flattened) or np.any(flattened<0) or np.any(flattened>=len(counts)):raise ValueError('Invalid/disjoint readout membership')
    rates=np.array([np.mean(counts[e])*1000/duration_ms for e in ensembles])
    baseline=np.zeros(5) if baseline_hz is None else np.asarray(baseline_hz)
    scores=(rates-baseline)/scale_hz
    if not np.isfinite(scores).all():raise ValueError('Nonfinite neural scores')
    return rates,scores


def select_action(scores,legal,rng,temperature=0):
    scores=np.asarray(scores);legal=np.asarray(legal,dtype=bool)
    if scores.shape!=(5,) or legal.shape!=(5,) or not legal.any() or not np.isfinite(scores).all():raise ValueError('Finite scores and legal actions required')
    if not np.isfinite(temperature) or temperature<0:raise ValueError('Nonnegative temperature required')
    masked=np.where(legal,scores,-np.inf)
    if temperature==0:return int(np.argmax(masked))
    probabilities=np.exp((masked-np.max(masked))/temperature);probabilities/=probabilities.sum()
    return int(rng.choice(5,p=probabilities))


class NeuralController:
    """Consumes an information-set dictionary. No game/teacher/policy imports."""
    def __init__(self,brain,preregistration,seed=0):
        if brain.graph_hash!=preregistration['graph_hash']:raise ValueError('Controller graph mismatch')
        self.brain=brain;self.registration=preregistration;self.rng=np.random.default_rng(seed)
        self.mapping=np.array(preregistration['projection_indices'],dtype=np.int32)
        self.ensembles=[np.array(e['indices'],dtype=np.int32) for e in preregistration['ensembles']]
        self.blank=np.zeros(brain.n,dtype=np.float32)

    def decide(self,observation,temperature=0,first_in_hand=None):
        p=self.registration;t=p['config']['decision_ms']
        x=encode_player_state(observation)
        if first_in_hand is not None and (type(first_in_hand) is not bool or first_in_hand != bool(x[-1])):
            raise ValueError('First-action flag disagrees with visible action history')
        drive=population_drive(x,self.mapping,self.brain.n,p['selected_gain'])
        self.brain.advance(self.blank,t['baseline']);self.brain.advance(drive,t['stimulus'])
        counts=self.brain.advance(drive,t['readout'])
        rates,scores=neural_scores(counts,self.ensembles,t['readout'],p['baseline_hz'],p['config']['score_scale_hz'])
        chosen=select_action(scores,observation['legal_mask'],self.rng,temperature)
        return {'raw_rates_hz':rates.tolist(),'scores':scores.tolist(),'selected':chosen,'legal_mask':observation['legal_mask'],
            'temperature':temperature,'silent':bool(np.max(rates[np.asarray(observation['legal_mask'],dtype=bool)])==0),
            'decoder_fallback':False,'tie_break':'lowest-legal-index','score_source':'MaleCNS-native-LIF-spikes',
            'encoded_hash':encoded_hash(x),'encoded':[(CHANNELS[i],float(x[i])) for i in np.flatnonzero(x)],
            'observation':observation,'virtual_time_ms':self.brain.time_ms,'counts':counts}

    def commit_interval(self):self.brain.advance(self.blank,self.registration['config']['decision_ms']['inter_decision'])


def scaled_weights(weights,scale):
    """Uniform declared dynamics sensitivity; topology and transmitter signs stay fixed."""
    weights=np.asarray(weights,dtype=np.float32)
    if not np.isfinite(scale) or scale<=0:raise ValueError('Positive finite synaptic scale required')
    result=weights*np.float32(scale)
    if not np.isfinite(result).all() or np.any((weights!=0)&(result==0)):raise ValueError('Invalid scaled synapses')
    return result


def load_controller(graph_path,preregistration_path,seed=0):
    import json
    from pathlib import Path
    from flyholdem.connectome.prepare import load_graph
    from flyholdem.neural.sparse import SparseBrain
    graph,prepared=load_graph(graph_path)
    registration=json.loads(Path(preregistration_path).read_text())
    if prepared['graph_hash']!=registration.get('base_graph_hash',registration['graph_hash']):raise ValueError('Preregistration base graph mismatch')
    brain=SparseBrain(graph['ptr'],graph['post'],scaled_weights(graph['weight'],registration['config'].get('global_weight_scale',1)))
    return NeuralController(brain,registration,seed)
