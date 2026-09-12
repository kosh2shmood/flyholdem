"""Export actual locally distilled circuit weights with frozen cue inference."""
import argparse
import json
from pathlib import Path
import numpy as np
from flyholdem.connectome.registry import ROOT, digest
from flyholdem.interface.population import load_controller
from flyholdem.interface.frozen import export_frozen
from flyholdem.interface import cue_player
from flyholdem.learning.eligibility import Eligibility
from flyholdem.neural.checkpoint import Checkpoints
from flyholdem.provenance import manifest, assert_compatible, identity


def export(run, output):
    from pyarrow import feather
    run = Path(run)
    saved = json.loads((run / 'manifest.json').read_text())
    result = json.loads((run / 'result.json').read_text())
    if (result['schema'] != 'exact-transfer-result-v1'
            or not (result['development_criteria_met'] or result['status'] == 'pass')
            or result['manifest_sha256'] != digest(run / 'manifest.json')):
        raise ValueError('A matching qualifying exact-transfer result is required')
    config = saved['config']
    base = ROOT / config['preregistration']
    if digest(base) != config['preregistration_sha256']:
        raise ValueError('Frozen registration mismatch')
    graph = ROOT / 'connectome_data/malecns_v1' / ('prepared-' + config['mode'])
    controller = load_controller(graph, base)
    current = manifest(config, controller.brain.graph_hash, config['cue_hash'],
                       identity(controller.registration['ensembles']), controller.brain.build['binary_sha256'])
    assert_compatible(saved, current)
    arrays, extra, checkpoint = Checkpoints(run / 'checkpoints').load(current, 'best')
    controller.brain.restore_state({key: arrays['brain_' + key] for key in controller.brain.state_names})
    nodes = feather.read_table(graph / 'neurons.feather').to_pandas()
    eligible = Eligibility(controller.brain, np.flatnonzero(nodes['class'].eq('Kenyon_Cell').to_numpy()),
                           np.flatnonzero(nodes['class'].eq('MBON').to_numpy()),
                           {**config['plasticity'], 'learning_rate': config['selected_learning_rate']})
    if eligible.registration != config['plasticity_registration']:
        raise ValueError('Actual trained edge subset differs from preregistration')
    reference = {'manifest_sha256': digest(run / 'manifest.json'),
                 'best_checkpoint': json.loads((run / 'checkpoints/best.json').read_text()),
                 'best_validation': extra['best_validation'],
                 'optimization': config.get('optimization', 'teacher-advantage-local-eligibility'),
                 'teacher_table_sha256': config['teacher_registration']['table_sha256'],
                 'cue_player': {'implementation_sha256': digest(Path(cue_player.__file__)),
                                'sensory_protocol': config['sensory_protocol']}}
    model_hash = export_frozen(controller, eligible.edges, config['plasticity']['weight_bounds'],
        'distilled-connectome', output, reference,
        {'result_sha256': digest(run / 'result.json'), 'status': result['status'],
         'scope': 'Small exact-cue circuit transfer only; no poker claim or complete Gate 2A pass'})
    return {'model_sha256': model_hash, 'selected_seed': extra['best_validation']['seed'],
            'learning_mode': 'distilled-connectome', 'teacher_connected': False, 'mode': config['mode']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True); p.add_argument('--output', required=True)
    args = p.parse_args()
    print(json.dumps(export(args.run, args.output), indent=2))


if __name__ == '__main__':
    main()
