"""Frozen legal-greedy best-response components from a completed NFSP run.

This is a conventional population-trained reference, not the historical NFSP
average strategy, an equilibrium certificate, or a fly action-score source.
"""
import importlib
import json
from pathlib import Path
import numpy as np
import torch
from flyholdem.connectome.registry import digest
from flyholdem.provenance import source_identity,environment,identity
from flyholdem.neural.checkpoint import Checkpoints,atomic_json
from .features import features,feature_names,feature_runtime_identity
from .nfsp import network
from .policy import policy_implementation_identity
from .serialization import unpack

AGGREGATION='equal-legal-greedy-best-response-mixture'
ALGORITHM='population-Double-DQN-best-response-mixture-v1'


def implementation(feature_version):
    return {'q_policy_sha256':digest(__file__),'shared_policy_runtime':policy_implementation_identity(feature_version)}


class BestResponsePolicy:
    def __init__(self,models,feature_version):
        self.models=list(models);self.feature_version=feature_version
        if len(self.models)!=2:raise ValueError('Two registered best-response components required')
        for model in self.models:model.eval();model.requires_grad_(False)

    @torch.no_grad()
    def probabilities(self,observation):
        x=torch.from_numpy(features(observation,self.feature_version))
        legal=torch.tensor(observation['legal_mask'],dtype=torch.bool)
        probabilities=np.zeros(5,dtype=np.float64)
        for model in self.models:
            values=model(x)
            if values.shape!=(5,) or not torch.isfinite(values).all():raise ValueError('Finite conventional Q values required')
            action=int(values.masked_fill(~legal,-torch.inf).argmax())
            probabilities[action]+=.5
        return probabilities


def export_components(config,output):
    if (config.get('schema')!='teacher-best-response-extraction-v1' or config.get('algorithm')!=ALGORITHM
            or config.get('aggregation')!=AGGREGATION or config.get('status')!='development'):
        raise ValueError('Registered best-response extraction configuration required')
    run=Path(config['training_run']);source=Path(config['training_source'])
    saved=json.loads((run/'manifest.json').read_text());trained=json.loads((run/'result.json').read_text())
    source_hash=source_identity(source);training=saved['config'];version=training['agent']['feature_version']
    if (digest(run/'manifest.json')!=config['training_manifest_sha256']
            or source_hash!=config['training_source_sha256'] or source_hash!=saved['source_hash']
            or saved['environment']!=environment()
            or saved['binary_hash']!=identity({'pytorch':'cpu','features':feature_runtime_identity(version)})
            or trained.get('status')!='trained-unvalidated' or trained['hands_completed']!=config['completed_hands']
            or training['hands']!=config['completed_hands'] or training['agent'].get('value_learning')!='double-dqn'
            or training.get('algorithm')!='NFSP-fixed-policy-prior-v1'):
        raise ValueError('Best-response extraction requires the matching completed population training source and environment')
    shared=policy_implementation_identity(version)
    # The new wrapper may be added later, but every original tensor/feature
    # implementation must match the actual immutable training source exactly.
    for name,checksum in shared['source_files'].items():
        relative=Path(*name.split('.')).with_suffix('.py')
        if name.endswith('.cpp'):relative=Path(*name[:-4].split('.')).with_suffix('.cpp')
        if digest(source/'src'/relative)!=checksum:raise ValueError('Original Q/feature implementation differs from training: '+name)
    arrays,extra,metadata=Checkpoints(run/'checkpoints').load(saved)
    if extra['completed_hands']!=config['completed_hands'] or extra['journal_head']!=trained['journal_head']:
        raise ValueError('Best-response checkpoint is not the completed registered run')
    from flyholdem.experiments.reports import audit_journal
    audited=audit_journal(run/'hands.jsonl')
    if audited['rows']!=config['completed_hands'] or audited['head']!=trained['journal_head']:
        raise ValueError('Completed population training journal mismatch')
    output=Path(output);output.mkdir(parents=True,exist_ok=False);entries={}
    for i in range(2):
        state=unpack(extra['agents'][i]['tree'],arrays)['q']
        model=network(training['agent']['hidden'],len(feature_names(version)));model.load_state_dict(state)
        for name,tensor in model.state_dict().items():
            path=output/f'agent{i}-{name}.npy';value=tensor.detach().cpu().numpy()
            if value.dtype!=np.float32 or not np.isfinite(value).all():raise ValueError('Invalid source Q tensor')
            np.save(path,value,allow_pickle=False);entries[path.name]=digest(path)
    record={'schema':'teacher-best-response-policy-v1','mode':'conventional-teacher-control','algorithm':ALGORITHM,
        'aggregation':AGGREGATION,'hidden':training['agent']['hidden'],'agents':2,'feature_version':version,
        'files':entries,'implementation':implementation(version),'allowed_as_teacher':False,
        'provenance':{'stack_bb':training['stack_bb'],'training_manifest_sha256':digest(run/'manifest.json'),
            'training_source_hash':source_hash,'checkpoint_step':metadata['step'],
            'checkpoint_pointer_sha256':digest(run/'checkpoints/latest.json'),'extraction_config':config},
        'scope':'Final learned best-response components only; not the historical NFSP average or an equilibrium claim'}
    atomic_json(output/'manifest.json',record);return {'policy_sha256':digest(output/'manifest.json'),'allowed_as_teacher':False}


def load_policy(output):
    output=Path(output);record=json.loads((output/'manifest.json').read_text())
    if (record.get('schema')!='teacher-best-response-policy-v1' or record.get('algorithm')!=ALGORITHM
            or record.get('aggregation')!=AGGREGATION or record.get('agents')!=2
            or record.get('implementation')!=implementation(record['feature_version'])):
        raise ValueError('Best-response policy source/runtime or aggregation mismatch')
    models=[network(record['hidden'],len(feature_names(record['feature_version']))) for _ in range(2)]
    expected={f'agent{i}-{name}.npy' for i,model in enumerate(models) for name in model.state_dict()}
    if set(record['files'])!=expected:raise ValueError('Unexpected best-response tensor files')
    for i,model in enumerate(models):
        state={}
        for name,original in model.state_dict().items():
            path=output/f'agent{i}-{name}.npy'
            if digest(path)!=record['files'][path.name]:raise ValueError('Best-response tensor checksum mismatch')
            value=np.load(path,allow_pickle=False)
            if value.dtype!=np.float32 or value.shape!=tuple(original.shape) or not np.isfinite(value).all():
                raise ValueError('Invalid best-response numeric tensor')
            state[name]=torch.from_numpy(value)
        model.load_state_dict(state)
    return BestResponsePolicy(models,record['feature_version']),record
