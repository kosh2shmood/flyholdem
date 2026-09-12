"""Frozen native-connectome spectator loop and annotated neuron point cloud.

The viewer consumes committed native decisions. This demonstration makes no
poker-learning claim, never loads a conventional teacher, and never updates
weights. Neurons without annotated soma coordinates occupy a labeled grid.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from flyholdem.connectome.registry import ROOT, digest
from flyholdem.interface.frozen import load_frozen
from flyholdem.interface.population import load_controller
from flyholdem.poker.engine import Hand
from flyholdem.poker.opponents import Opponent
from .events import dumps


class NativeGraph:
    def __init__(self, rows, controller):
        self.rows = rows
        n = controller.brain.n
        if len(rows) != n:
            raise ValueError('Annotation rows must match the retained native graph')
        raw = np.full((n,3),np.nan,dtype=np.float64)
        for i,row in enumerate(rows):
            xyz = row.get('somaLocation')
            if xyz is not None and len(xyz) == 3 and np.isfinite(xyz).all():
                raw[i] = xyz
        known = np.isfinite(raw).all(axis=1)
        positions = np.zeros((n,3),dtype='<f4')
        if known.any():
            low,high = raw[known].min(axis=0),raw[known].max(axis=0)
            positions[known] = (raw[known]-(low+high)/2) / max(float((high-low).max())/2,1)
            positions[known,1] *= -1
        missing = np.flatnonzero(~known)
        width = max(1,int(np.ceil(np.sqrt(len(missing)))))
        if len(missing):
            positions[missing,0] = 1.5 + np.arange(len(missing)) % width / width * 1.1
            positions[missing,1] = .65 - np.arange(len(missing)) // width / width * 1.3
        roles = np.zeros(n,dtype=np.uint8)
        for i,row in enumerate(rows):
            if row.get('class') == 'DAN': roles[i]=7
            elif row.get('class') == 'MBON': roles[i]=8
        roles[controller.registration['input_indices']]=1
        for action,ensemble in enumerate(controller.ensembles): roles[ensemble]=2+action
        self.positions,self.roles = positions,roles
        self.known = known
        self.classes = sorted({r.get('class') or r.get('superclass') or 'unclassified' for r in rows})
        lookup = {name:i for i,name in enumerate(self.classes)}
        self.class_index = np.array([lookup[r.get('class') or r.get('superclass') or 'unclassified'] for r in rows])
        self.class_sizes = np.bincount(self.class_index,minlength=len(self.classes))
        registration_hash=hashlib.sha256(dumps(controller.registration).encode()).hexdigest()
        self.meta = {'registration_sha256':registration_hash,'mode':controller.registration['mode'],'native_cloud':True,'neuron_count':n,
            'edge_count':len(controller.brain.weights),'hash':controller.brain.graph_hash,
            'located_neurons':int(known.sum()),'unlocated_neurons':int((~known).sum()),
            'layout':'Annotated soma coordinates; missing coordinates in a separate schematic grid',
            'edges_rendered':0,'positions_sha256':hashlib.sha256(positions.tobytes()).hexdigest(),
            'roles_sha256':hashlib.sha256(roles.tobytes()).hexdigest(),
            'role_names':['Other','KC input','Fold readout','Check/call readout','Half-pot readout',
                          'Pot readout','All-in readout','Dopamine','Other MBON']}

    def node(self,index):
        if not 0 <= index < len(self.rows): raise IndexError('Neuron index out of range')
        row = self.rows[index]
        return {'index':index,'body_id':str(row['bodyId']),'class':row.get('class'),
                'superclass':row.get('superclass'),'type':row.get('type'),
                'role':self.meta['role_names'][self.roles[index]],
                'has_soma_coordinate':bool(self.known[index]),'soma_coordinate':row.get('somaLocation')}

    def activity(self,counts):
        counts=np.asarray(counts)
        if counts.shape != (len(self.rows),): raise ValueError('Native activity shape mismatch')
        indices=np.flatnonzero(counts)
        total=np.bincount(self.class_index,weights=counts,minlength=len(self.classes))
        return {'activity_indices':indices.tolist(),'activity_counts':counts[indices].tolist(),
                'activity_total':int(counts.sum()),'population_activity':[
                    {'class':name,'neurons':int(self.class_sizes[i]),'spikes':int(total[i])}
                    for i,name in enumerate(self.classes) if total[i]]}


class NativeDemo:
    def __init__(self,mode,seed=20260912,preregistration=None,model=None):
        if mode not in ('circuit','full'): raise ValueError('Explicit circuit or full mode required')
        graph = ROOT/'connectome_data/malecns_v1'/('prepared-'+mode)
        if model:
            self.controller=load_frozen(model,graph,seed=seed)
            self.learning_mode=self.controller.frozen_model['learning_mode']
            model_hash=self.controller.frozen_model['sha256']
        else:
            default='controllability-circuit-v1' if mode=='circuit' else 'controllability-full-quarter-v2'
            preregistration=Path(preregistration) if preregistration else ROOT/'runs'/default/'preregistration.json'
            self.controller=load_controller(graph,preregistration,seed)
            self.learning_mode='bio-plastic';model_hash=None
        if self.controller.registration['mode']!=mode: raise ValueError('Requested mode and native model differ')
        self.brain=self.controller.brain
        from pyarrow import feather
        rows=feather.read_table(graph/'neurons.feather',columns=['bodyId','class','superclass','type','somaLocation']).to_pylist()
        self.graph=NativeGraph(rows,self.controller)
        self.seed,self.mode,self.model_hash=seed,mode,model_hash
        self.hand_number=self.sequence=0;self.hand=None;self.completed=False
        self.opponent=Opponent(seed+1);self.previous_hash='0'*64;self.return_bb=0
        self.initial_weights_hash=hashlib.sha256(self.brain.weights.tobytes()).hexdigest()
        self.identity={'graph_sha256':self.brain.graph_hash,'model_sha256':model_hash,
            'registration_sha256':self.graph.meta['registration_sha256'],
            'binary_sha256':self.brain.build['binary_sha256'],
            'policy_status':'unvalidated for poker','weights':'frozen','neural_reset':'fresh rest at each hand'}

    def next_event(self):
        if self.hand is None or self.completed:
            self.brain.reset_dynamics()
            self.hand=Hand(self.seed+self.hand_number,button=self.hand_number%2)
            self.hand_number+=1;self.completed=False
            event={'kind':'hand_start','table':self.hand.view()}
        elif self.hand.done:
            net_bb=self.hand.view()['payoffs'][0]/2;self.return_bb+=net_bb
            # No reinforcement is delivered during this frozen spectator evaluation.
            event={'kind':'settlement','net_bb':net_bb,'table':self.hand.view(),
                   'plasticity_enabled':False,'reward_delivered':False}
            self.completed=True
        else:
            actor=self.hand.actor;observation=self.hand.observation();before=self.hand.view()
            if actor==0:
                decision=self.controller.decide(observation)
                decision.update(self.graph.activity(decision.pop('counts')))
                action=decision['selected'];self.controller.commit_interval()
            else:
                decision=None;action=self.opponent.act(observation)
            record=self.hand.act(action)
            event={'kind':'decision' if actor==0 else 'opponent_action','actor':actor,'decision':decision,
                   'action':record,'table_before':before,'table':self.hand.view()}
        event.update(schema='flyholdem-event-v1',sequence=self.sequence,hand=self.hand_number,
            mode=self.mode,learning_mode=self.learning_mode,status='development',teacher='disconnected',
            encoder='engineered-kc-v1',evaluation=True,plasticity_enabled=False,
            label=f'{self.mode.upper()} CONNECTOME / FROZEN EVALUATION / POKER POLICY UNVALIDATED',
            return_bb=self.return_bb,previous_hash=self.previous_hash,run_identity=self.identity)
        event['hash']=hashlib.sha256(dumps(event).encode()).hexdigest()
        self.previous_hash=event['hash'];self.sequence+=1
        return event
