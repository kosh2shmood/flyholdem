"""Export a checked numeric NFSP training generation as a small frozen policy."""
import argparse
import json
from pathlib import Path
import torch
from flyholdem.neural.checkpoint import Checkpoints
from flyholdem.provenance import manifest, identity
from flyholdem.connectome.registry import digest
from .features import feature_names, feature_runtime_identity, V1
from .nfsp import NFSPAgent
from .policy import export_policy


def export_training(run_path, output, which='latest'):
    root = Path(run_path)
    saved = json.loads((root / 'manifest.json').read_text())
    config = saved['config']
    torch.set_num_threads(config['torch_threads']); torch.use_deterministic_algorithms(True)
    runtime = manifest(config, 'conventional-teacher-no-connectome', identity(feature_names(config['agent'].get('feature_version', V1))),
                       identity({'algorithm': config.get('algorithm','NFSP')+'-average-policy', 'actions': 5}),
                       identity({'pytorch':'cpu','features':feature_runtime_identity(config['agent'].get('feature_version',V1))}))
    arrays, extra, metadata = Checkpoints(root / 'checkpoints').load(runtime, which)
    agents = [NFSPAgent(config['agent'], config['seed'] + 100 * seat) for seat in range(2)]
    for seat, agent in enumerate(agents):
        agent.restore_state(arrays, extra['agents'][seat], f'agent{seat}')
    policy_hash = export_policy(agents, output, {'stack_bb': config['stack_bb'],
        'training_manifest_sha256': digest(root / 'manifest.json'), 'checkpoint_step': metadata['step'],
        'checkpoint_pointer_sha256': digest(root / 'checkpoints' / (which + '.json')),
        'training_source_hash': saved['source_hash'], 'training_commit': saved['commit'],
        'algorithm': config.get('algorithm','NFSP'), 'hands_completed': extra['completed_hands'], 'seed': config['seed']})
    return {'policy_sha256': policy_hash, 'status': 'frozen-conventional-policy-unvalidated', 'allowed_as_teacher': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True); p.add_argument('--output', required=True)
    p.add_argument('--which', choices=['latest','best'], default='latest')
    args = p.parse_args()
    print(json.dumps(export_training(args.run, args.output, args.which), indent=2))


if __name__ == '__main__':
    main()
