"""Canonical, stratified teacher targets with disjoint information-set splits."""
import json
from pathlib import Path
import numpy as np
from flyholdem.connectome.registry import digest
from flyholdem.neural.checkpoint import atomic_json
from flyholdem.poker.engine import Hand
from flyholdem.poker.infoset import canonical_state, canonical_information_id
from flyholdem.poker.observation import canonical_bytes
from flyholdem.poker.opponents import Opponent, visible_strength
from flyholdem.provenance import identity
from .loaders import load_policy


def split_for_id(information_id):
    value = int(information_id[:16], 16) % 100
    return 'train' if value < 80 else 'validation' if value < 90 else 'test'


def stratum(observation):
    # Strength is a named coarse visible-information proxy, not hidden equity.
    strength = min(4, int(visible_strength(observation) * 5))
    spr = int(np.searchsorted([.5, 1, 3, 10], observation['spr']))
    price = observation['to_call'] / max(1, observation['pot'] + observation['to_call'])
    price_bucket = int(np.searchsorted([0, .1, .25, .5], price))
    legal = ''.join('1' if x else '0' for x in observation['legal_mask'])
    history = observation['history']
    shape = str(min(len(history), 8)) + ':' + ''.join(str(item['action']) for item in history[-3:])
    return [observation['street'], observation['position'], strength, spr, price_bucket, legal, shape]


def validate_targets(observation, probabilities):
    state = canonical_state(observation)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    legal = np.asarray(state['legal_mask'], dtype=bool)
    if (probabilities.shape != (5,) or not np.isfinite(probabilities).all() or np.any(probabilities < 0)
            or np.any(probabilities[~legal] != 0) or not np.isclose(probabilities.sum(), 1, rtol=0, atol=1e-12)):
        raise ValueError('Invalid canonical teacher targets')
    return state, probabilities


def export_corpus(policy_path, validation_path, config, output, resume=False):
    validation = json.loads(Path(validation_path).read_text())
    policy_hash = digest(Path(policy_path) / 'manifest.json')
    if (validation.get('schema') != 'teacher-evaluation-v1' or validation.get('profile') != 'confirmatory'
            or not validation.get('passes_fixed_suite') or not validation.get('information_boundary_verified')
            or validation.get('allowed_as_teacher') is not True
            or validation.get('policy_sha256') != policy_hash or validation.get('stack_bb') != config['stack_bb']):
        raise ValueError('Corpus export requires a matching validated frozen teacher')
    if Path(validation_path).name != "result.json":
        raise ValueError("Teacher validation must reference result.json in its complete evaluated run")
    from .validation import verify_evaluation
    verify_evaluation(Path(validation_path).parent, policy_path)
    policy, policy_record = load_policy(policy_path)
    from .evaluation import verify_information_boundary
    verify_information_boundary(policy)
    from flyholdem.experiments.journal import Journal
    from flyholdem.provenance import manifest, assert_compatible
    output = Path(output); output.mkdir(parents=True, exist_ok=resume)
    runtime = manifest({**config, 'teacher_sha256': policy_hash, 'validation_sha256': digest(validation_path)},
                       'conventional-teacher-corpus', identity('canonical-visible-infoset-v1'), policy_hash, 'pytorch-cpu')
    if resume:
        assert_compatible(json.loads((output / 'run-manifest.json').read_text()), runtime)
        if (output / 'manifest.json').exists():
            verify_corpus(output)
            return json.loads((output / 'manifest.json').read_text())
    else:
        atomic_json(output / 'run-manifest.json', runtime)
    journal = Journal(output / 'collection.jsonl', resume)
    import signal
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    rows = {}; occurrences = {}; strata = {}; streets = [0] * 4
    try:
        for hand_index in range(config['maximum_hands']):
            if hand_index < len(journal.rows):
                cached = journal.rows[hand_index]
                if cached['label'] != ['collection-hand', hand_index]:
                    raise ValueError('Corpus collection journal plan changed')
                states = cached['value']['visible_states']
            else:
                seed = config['seed_start'] + hand_index
                hand = Hand(seed, button=hand_index % 2, stacks=(config['stack_bb'] * 2,) * 2)
                rng = np.random.default_rng(seed + 1000000)
                random_opponent = Opponent(seed + 2000000, 'random')
                station = Opponent(seed + 3000000, 'calling-station')
                states = []
                while not hand.done:
                    observation = canonical_state(hand.observation())
                    probabilities = policy.probabilities(observation)
                    validate_targets(observation, probabilities)
                    states.append({'observation': observation, 'probabilities': probabilities.tolist()})
                    mixture = rng.random()
                    if mixture < config['calling_station_collection_probability']:
                        action = station.act(observation)
                    elif mixture < config['calling_station_collection_probability'] + config['random_collection_probability']:
                        action = random_opponent.act(observation)
                    else:
                        action = int(rng.choice(5, p=probabilities))
                    hand.act(action)
                journal.record(hand_index, ['collection-hand', hand_index], {'visible_states': states})
            for item in states:
                observation, probabilities = validate_targets(item['observation'], item['probabilities'])
                info = canonical_information_id(observation)
                encoded = canonical_bytes(observation); targets = probabilities.tobytes()
                if info in occurrences and occurrences[info] != (encoded, targets):
                    raise AssertionError('Repeated canonical state changed observation bytes or teacher targets')
                occurrences[info] = (encoded, targets)
                bucket = stratum(observation); key = identity(bucket)
                if info not in rows and strata.get(key, 0) < config['maximum_per_stratum']:
                    rows[info] = {'schema': 'distillation-row-v1', 'information_id': info,
                                  'observation': observation, 'teacher_probabilities': probabilities.tolist(),
                                  'split': split_for_id(info), 'stratum': bucket}
                    strata[key] = strata.get(key, 0) + 1; streets[observation['street']] += 1
            if stopped:
                raise KeyboardInterrupt('Corpus stopped after a complete hand; use --resume')
            if len(rows) >= config['minimum_rows'] and min(streets) >= config['minimum_per_street']:
                break
    finally:
        journal.close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    if len(rows) < config['minimum_rows'] or min(streets) < config['minimum_per_street']:
        raise ValueError('Registered corpus coverage was not reached')
    counts = {}
    for split in ('train', 'validation', 'test'):
        selected = sorted((r for r in rows.values() if r['split'] == split), key=lambda r: r['information_id'])
        if not selected:
            raise ValueError('Empty canonical corpus split')
        with (output / (split + '.jsonl')).open('wb') as stream:
            for row in selected:
                stream.write(canonical_bytes(row) + b'\n')
            stream.flush(); __import__('os').fsync(stream.fileno())
        counts[split] = len(selected)
    record = {'schema': 'distillation-corpus-v1', 'config': config, 'config_hash': identity(config),
              'teacher_policy_sha256': policy_hash, 'teacher_validation_sha256': digest(validation_path),
              'collection_sha256': digest(output / 'collection.jsonl'), 'run_manifest_sha256': digest(output / 'run-manifest.json'),
              'files': {split + '.jsonl': digest(output / (split + '.jsonl')) for split in counts},
              'split_counts': counts, 'street_counts': streets, 'strata': len(strata),
              'unique_information_sets': len(rows), 'collection_hands': hand_index + 1,
              'strength_bucket': 'visible-strength-rule-v1; coarse proxy', 'hidden_information': False}
    atomic_json(output / 'manifest.json', record)
    verify_corpus(output)
    return record


def verify_corpus(output):
    output = Path(output); manifest = json.loads((output / 'manifest.json').read_text())
    if manifest['schema'] != 'distillation-corpus-v1' or set(manifest['files']) != {'train.jsonl', 'validation.jsonl', 'test.jsonl'}:
        raise ValueError('Unsupported corpus')
    for name, key in [('collection.jsonl','collection_sha256'), ('run-manifest.json','run_manifest_sha256')]:
        if digest(output / name) != manifest[key]:
            raise ValueError('Corpus collection/provenance checksum mismatch')
    seen = set(); counts = {}
    for filename, checksum in manifest['files'].items():
        if digest(output / filename) != checksum:
            raise ValueError('Corpus checksum mismatch')
        split = filename.split('.')[0]; count = 0
        for line in (output / filename).read_text().splitlines():
            row = json.loads(line)
            if set(row) != {'schema', 'information_id', 'observation', 'teacher_probabilities', 'split', 'stratum'} or row['schema'] != 'distillation-row-v1':
                raise ValueError('Unexpected corpus field')
            state, probabilities = validate_targets(row['observation'], row['teacher_probabilities'])
            if canonical_bytes(state) != canonical_bytes(row['observation']):
                raise ValueError('Stored observation is not normalized')
            info = canonical_information_id(state)
            if (info != row['information_id'] or info in seen or split_for_id(info) != split
                    or row['split'] != split or row['stratum'] != stratum(state)):
                raise ValueError('Canonical duplicate, split overlap or row mismatch')
            seen.add(info); count += 1
        counts[split] = count
    if counts != manifest['split_counts'] or len(seen) != manifest['unique_information_sets']:
        raise ValueError('Corpus counts mismatch')
    return {'canonical_ids_disjoint': True, 'unique_information_sets': len(seen), 'split_counts': counts}


def main():
    import argparse,yaml
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy', required=True); p.add_argument('--validation', required=True)
    p.add_argument('--config', required=True); p.add_argument('--output', required=True)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    result = export_corpus(args.policy, args.validation, yaml.safe_load(Path(args.config).read_text()), args.output, args.resume)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
