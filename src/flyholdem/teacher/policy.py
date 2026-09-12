"""Small frozen conventional average-policy exports, never a fly controller."""
import json
from pathlib import Path
import numpy as np
import torch
from flyholdem.connectome.registry import digest
from flyholdem.provenance import identity
from flyholdem.neural.checkpoint import atomic_json
from .nfsp import network
from .features import features, feature_names, feature_runtime_identity, V1, V3


def policy_implementation_identity(feature_version=V1):
    import importlib
    import platform
    import sys
    names = ('flyholdem.teacher.nfsp', 'flyholdem.teacher.policy', 'flyholdem.teacher.features', 'flyholdem.teacher.memory',
             'flyholdem.teacher.serialization', 'flyholdem.poker.infoset',
             'flyholdem.poker.observation', 'flyholdem.interface.encoder')
    files = {name: digest(importlib.import_module(name).__file__) for name in names}
    if feature_version == V3:
        module=importlib.import_module('flyholdem.teacher.equity')
        files['flyholdem.teacher.equity']=digest(module.__file__)
        files['flyholdem.teacher.equity.cpp']=digest(Path(module.__file__).with_suffix('.cpp'))
    return {'source_files': files, 'source_hash': identity(files), 'python': sys.version,
            'platform': platform.platform(), 'numpy': np.__version__, 'torch': torch.__version__,
            'pokerkit': __import__('importlib.metadata', fromlist=['version']).version('pokerkit'),
            'feature_runtime':feature_runtime_identity(feature_version)}


class TeacherPolicy:
    def __init__(self, models, feature_version=V1):
        self.feature_version = feature_version
        self.models = list(models)
        for model in self.models:
            model.eval(); model.requires_grad_(False)

    @torch.no_grad()
    def probabilities(self, observation):
        x = torch.from_numpy(features(observation, self.feature_version))
        legal = torch.tensor(observation['legal_mask'], dtype=torch.bool)
        values = [torch.softmax(model(x).masked_fill(~legal, -torch.inf), dim=-1).numpy().astype(np.float64)
                  for model in self.models]
        probabilities = np.mean(values, axis=0)
        probabilities /= probabilities.sum()
        if (not np.isfinite(probabilities).all() or np.any(probabilities < 0)
                or np.any(probabilities[~legal.numpy()] != 0)):
            raise ValueError('Invalid frozen conventional policy probabilities')
        return probabilities


def export_policy(agents, output, provenance):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    entries = {}
    for i, agent in enumerate(agents):
        for name, tensor in agent.average.state_dict().items():
            path = output / f'agent{i}-{name}.npy'
            np.save(path, tensor.detach().cpu().numpy(), allow_pickle=False)
            entries[path.name] = digest(path)
    record = {'schema': 'teacher-average-policy-v1', 'mode': 'conventional-teacher-control',
              'hidden': agents[0].config['hidden'], 'agents': len(agents),
              'aggregation': 'equal-probability-mixture', 'files': entries,
              'feature_version': agents[0].feature_version,
              'provenance': provenance, 'implementation': policy_implementation_identity(agents[0].feature_version), 'allowed_as_teacher': False}
    atomic_json(output / 'manifest.json', record)
    return digest(output / 'manifest.json')


def load_policy(output):
    output = Path(output)
    record = json.loads((output / 'manifest.json').read_text())
    if (record['schema'] != 'teacher-average-policy-v1' or record['aggregation'] != 'equal-probability-mixture'
            or record['agents'] != 2):
        raise ValueError('Unsupported conventional teacher policy')
    if record['implementation'] != policy_implementation_identity(record['feature_version']):
        raise ValueError('Teacher inference source/runtime mismatch')
    dimension = len(feature_names(record['feature_version']))
    models = [network(record['hidden'], dimension) for _ in range(record['agents'])]
    expected = {f'agent{i}-{name}.npy' for i, model in enumerate(models) for name in model.state_dict()}
    if set(record['files']) != expected:
        raise ValueError('Unexpected teacher tensor files')
    for i, model in enumerate(models):
        weights = {}
        for name, original in model.state_dict().items():
            path = output / f'agent{i}-{name}.npy'
            if digest(path) != record['files'][path.name]:
                raise ValueError('Teacher tensor checksum mismatch')
            value = np.load(path, allow_pickle=False)
            if value.shape != tuple(original.shape) or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError('Invalid teacher tensor')
            weights[name] = torch.from_numpy(value)
        model.load_state_dict(weights)
    return TeacherPolicy(models, record['feature_version']), record
