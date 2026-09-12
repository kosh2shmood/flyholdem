"""Conventional learned Q mixture with the exact stack-potential fold boundary.

For V7's undiscounted reward representation, a legal fold terminates with zero
shaped return. Other action values remain the fixed learned Q outputs. This is
never a fly score source and does not claim an equilibrium strategy.
"""
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from flyholdem.connectome.registry import digest
from flyholdem.neural.checkpoint import atomic_json
from .q_policy import BestResponsePolicy,load_policy as load_q,implementation as q_implementation
from .features import features,feature_names
from .nfsp import network
from .potential import VERSION

SCHEMA='teacher-potential-boundary-policy-v1'
ALGORITHM='population-Double-DQN-exact-fold-boundary-v1'
AGGREGATION='equal-legal-greedy-Q-mixture-with-exact-zero-fold-value'


def implementation(version):
    return {'potential_policy_sha256':digest(__file__),'potential_reward_sha256':digest(Path(__file__).with_name('potential.py')),
        'shared_q_runtime':q_implementation(version)}


class PotentialBoundaryPolicy(BestResponsePolicy):
    @torch.no_grad()
    def probabilities(self,observation):
        x=torch.from_numpy(features(observation,self.feature_version));legal=torch.tensor(observation['legal_mask'],dtype=torch.bool)
        probabilities=np.zeros(5,dtype=np.float64)
        for model in self.models:
            values=model(x)
            if values.shape!=(5,) or not torch.isfinite(values).all():raise ValueError('Finite learned Q outputs required before the exact terminal boundary')
            values=values.clone();values[0]=0
            probabilities[int(values.masked_fill(~legal,-torch.inf).argmax())]+=.5
        return probabilities


def export_boundary(config,output):
    if (config.get('schema')!='teacher-potential-boundary-extraction-v1' or config.get('algorithm')!=ALGORITHM
            or config.get('aggregation')!=AGGREGATION or config.get('reward_parameterization')!=VERSION
            or config.get('status')!='development'):
        raise ValueError('Registered stack-potential boundary extraction required')
    source=Path(config['source_policy']);training=Path(config['training_run'])
    if digest(source/'manifest.json')!=config['source_policy_sha256'] or digest(training/'manifest.json')!=config['training_manifest_sha256']:
        raise ValueError('Exact source Q policy and training identities required')
    policy,base=load_q(source);saved=json.loads((training/'manifest.json').read_text());result=json.loads((training/'result.json').read_text())
    parameters=saved['config'];origin=base['provenance']
    if (parameters.get('reward_parameterization')!=VERSION or parameters['agent']['discount']!=1
            or origin['training_manifest_sha256']!=config['training_manifest_sha256']
            or origin['training_source_hash']!=saved['source_hash'] or origin['checkpoint_step']!=parameters['hands']
            or result['hands_completed']!=parameters['hands'] or result['status']!='trained-unvalidated'):
        raise ValueError('Zero fold value requires the matching completed undiscounted stack-potential training')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    for name in base['files']:shutil.copy2(source/name,output/name)
    record={**base,'schema':SCHEMA,'algorithm':ALGORITHM,'aggregation':AGGREGATION,
        'implementation':implementation(base['feature_version']),
        'boundary':{'reward_parameterization':VERSION,'discount':1.0,'action':0,'value':0.0},
        'provenance':{**origin,'source_policy_sha256':config['source_policy_sha256'],'boundary_extraction_config':config},
        'scope':'Conventional final learned Q mixture with an exact terminal payoff boundary; not a fly policy, NFSP average or equilibrium claim'}
    atomic_json(output/'manifest.json',record)
    return {'policy_sha256':digest(output/'manifest.json'),'allowed_as_teacher':False}


def load_policy(output):
    output=Path(output);record=json.loads((output/'manifest.json').read_text())
    if (record.get('schema')!=SCHEMA or record.get('algorithm')!=ALGORITHM or record.get('aggregation')!=AGGREGATION
            or record.get('agents')!=2 or record.get('implementation')!=implementation(record['feature_version'])
            or record.get('boundary')!={'reward_parameterization':VERSION,'discount':1.0,'action':0,'value':0.0}):
        raise ValueError('Potential-boundary policy source/runtime or exact payoff mismatch')
    models=[network(record['hidden'],len(feature_names(record['feature_version']))) for _ in range(2)]
    expected={f'agent{i}-{name}.npy' for i,model in enumerate(models) for name in model.state_dict()}
    if set(record['files'])!=expected:raise ValueError('Unexpected potential-boundary tensor files')
    for i,model in enumerate(models):
        state={}
        for name,original in model.state_dict().items():
            path=output/f'agent{i}-{name}.npy'
            if digest(path)!=record['files'][path.name]:raise ValueError('Potential-boundary tensor checksum mismatch')
            value=np.load(path,allow_pickle=False)
            if value.dtype!=np.float32 or value.shape!=tuple(original.shape) or not np.isfinite(value).all():
                raise ValueError('Invalid potential-boundary numeric tensor')
            state[name]=torch.from_numpy(value)
        model.load_state_dict(state)
    return PotentialBoundaryPolicy(models,record['feature_version']),record
