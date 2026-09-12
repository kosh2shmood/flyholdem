"""Training observer around the unchanged native poker information interface.

The native controller alone emits scores/actions. Teaching targets are accepted
only after action selection and never passed into its observation or readouts.
"""
import numpy as np
from flyholdem.interface.population import NeuralController
from .dopamine import PastBaseline,pulse
from .surrogate import ReadoutSurrogate


class _ObservedBrain:
    def __init__(self,eligible):self.eligible=eligible;self.raw=eligible.brain
    def __getattr__(self,key):return getattr(self.raw,key)
    def advance(self,drive,duration_ms):
        counts=self.eligible.advance(drive,duration_ms)
        if np.any(counts>np.iinfo(np.int32).max):raise OverflowError('Observed native counts exceed the native counter range')
        return counts.astype(np.int32)


class PokerLearningPlayer:
    def __init__(self,controller,eligible,dopamine_populations,config,seed=0):
        if eligible.brain is not controller.brain:raise ValueError('Observer and controller must share the same native brain')
        self.brain=controller.brain;self.eligible=eligible
        self.controller=NeuralController(_ObservedBrain(eligible),controller.registration,seed)
        self.config=dict(config);self.populations=dopamine_populations;self.baseline=PastBaseline()
        self.mode=config['learning_mode'];self.optimization=config['optimization']
        if (self.mode,self.optimization) not in (
            ('bio-plastic','terminal-local-eligibility'),
            ('distilled-connectome','terminal-local-eligibility'),
            ('distilled-connectome','teacher-advantage-local-eligibility'),
            ('distilled-connectome','direct-readout-rate-surrogate-v1')):
            raise ValueError('Unsupported registered poker learning method')
        self.requires_teacher=self.optimization!='terminal-local-eligibility'
        self.surrogate=ReadoutSurrogate(eligible,controller.ensembles,config['surrogate']) if self.optimization=='direct-readout-rate-surrogate-v1' else None
        if config['reward_scale_bb']<=0:raise ValueError('Positive registered terminal reward scale required')
        self.last_decision=None

    def begin_hand(self):
        self.brain.reset_dynamics();self.eligible.clear_traces();self.last_decision=None

    def decide(self,observation,temperature=0):
        if self.last_decision is not None:raise ValueError('Commit the preceding action before another neural decision')
        value=self.controller.decide(observation,temperature)
        self.last_decision=value
        return value

    def commit_action(self,learning=False,teacher_target=None,override_advantage=None):
        if self.last_decision is None:raise ValueError('A selected neural action is required')
        decision=self.last_decision;event=None
        if not self.requires_teacher and (teacher_target is not None or override_advantage is not None):
            raise ValueError('Terminal poker learning uses chip reward only')
        if self.requires_teacher and learning:
            target=np.asarray(teacher_target,dtype=float);legal=np.asarray(decision['legal_mask'],dtype=bool)
            if target.shape!=(5,) or not np.isfinite(target).all() or np.any(target<0) or np.any(target[~legal]) or not np.isclose(target.sum(),1,atol=1e-10,rtol=0):
                raise ValueError('A legal teacher distribution is required after action selection')
            if self.surrogate:
                if override_advantage is not None:raise ValueError('Surrogate shuffled controls shuffle distributions, not scalar advantages')
                event={'teacher_connected':True,'teacher_target':target.tolist(),
                    'plasticity':self.surrogate.step(decision['scores'],legal,target,True),'dopamine_pulse':{'delivered':False}}
            else:
                advantage=2*(target[decision['selected']]-1/legal.sum())
                supplied=advantage if override_advantage is None else float(override_advantage)
                reward=self.baseline.event(supplied,'teacher-advantage',transform='identity')
                event={'teacher_connected':True,'teacher_target':target.tolist(),'teacher_advantage':float(advantage),
                    'advantage_shuffled':override_advantage is not None,'reinforcement':reward,
                    'plasticity':self.eligible.reinforce(reward['dopamine'],True),
                    'dopamine_pulse':pulse(self.brain,reward['dopamine'],self.populations,self.config['dopamine']['pulse_ms'],self.config['dopamine']['gain'])}
        elif teacher_target is not None or override_advantage is not None:
            raise ValueError('Frozen decisions must not query or receive teacher targets')
        self.controller.commit_interval();self.last_decision=None
        return event

    def finish_hand(self,net_bb,position,learning=False,override_reward=None):
        if self.last_decision is not None:raise ValueError('Commit the last action before terminal reinforcement')
        if not learning:return {'teacher_connected':False,'reward_delivered':False,'raw_net_bb':float(net_bb)}
        if self.requires_teacher:
            if override_reward is not None:raise ValueError('Distillation does not use terminal reward shaping')
            return {'teacher_connected':False,'reward_delivered':False,'raw_net_bb':float(net_bb)}
        raw=net_bb if override_reward is None else override_reward
        reward=self.baseline.event(raw,'position-'+str(position),self.config['reward_scale_bb'])
        update=self.eligible.reinforce(reward['dopamine'],True)
        activity=pulse(self.brain,reward['dopamine'],self.populations,self.config['dopamine']['pulse_ms'],self.config['dopamine']['gain'])
        return {'teacher_connected':False,'reward_delivered':True,'raw_net_bb':float(net_bb),
            'reward_shuffled':override_reward is not None,'reinforcement':reward,'plasticity':update,'dopamine_pulse':activity}

    def state(self):
        if self.last_decision is not None:raise ValueError('Checkpoint only after a complete committed action')
        arrays={'brain_'+k:v for k,v in self.brain.state().items()};arrays.update(self.eligible.state())
        return arrays,{'baseline':self.baseline.state(),'controller_rng':self.controller.rng.bit_generator.state,
                       'config':self.config,'pending_decision':None}

    def restore_state(self,arrays,extra):
        if extra['config']!=self.config or extra['pending_decision'] is not None:raise ValueError('Learning-player checkpoint configuration mismatch')
        self.brain.restore_state({key:arrays['brain_'+key] for key in self.brain.state_names})
        self.eligible.restore_state({key:arrays[key] for key in self.eligible.state()})
        self.baseline.restore_state(extra['baseline']);self.controller.rng.bit_generator.state=extra['controller_rng'];self.last_decision=None
