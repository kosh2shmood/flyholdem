"""Paired frozen conventional-teacher evaluation with disjoint locked deals."""
import numpy as np
from flyholdem.poker.engine import Hand
from flyholdem.poker.opponents import Opponent, VERSIONS


def paired_match(policy, kind, deal_seed, stack_bb):
    returns = []
    frequencies = np.zeros(5, dtype=np.int64)
    for seat in (0, 1):
        hand = Hand(deal_seed, button=0, stacks=(2 * stack_bb,) * 2)
        opponent = Opponent(deal_seed + 2000000, kind)
        rng = np.random.default_rng(deal_seed + 1000000)
        while not hand.done:
            observation = hand.observation()
            if hand.actor == seat:
                action = int(rng.choice(5, p=policy.probabilities(observation)))
                frequencies[action] += 1
            else:
                action = opponent.act(observation)
            hand.act(action)
        returns.append(hand.view()['payoffs'][seat] / 2)
    return {'deal_seed': deal_seed, 'seat_returns_bb': returns, 'paired_bb_per_hand': float(np.mean(returns)),
            'action_counts': frequencies.tolist()}


def evaluation_summary(rows_by_opponent, config):
    summaries = {}
    for i, (kind, rows) in enumerate(rows_by_opponent.items()):
        values = np.asarray([row['paired_bb_per_hand'] for row in rows], dtype=np.float64)
        rng = np.random.default_rng(config['bootstrap_seed'] + i)
        means = rng.choice(values, size=(config['bootstrap_repeats'], len(values)), replace=True).mean(axis=1)
        # Bonferroni over the fixed suite, while keeping each paired deal intact.
        alpha = .05 / len(config['opponents'])
        interval = np.quantile(means, [alpha / 2, 1 - alpha / 2])
        frequencies = np.sum([row['action_counts'] for row in rows], axis=0)
        summaries[kind] = {'opponent_version': VERSIONS[kind], 'paired_deals': len(values),
                           'bb_per_hand': float(values.mean()), 'suite_adjusted_bootstrap_ci': interval.tolist(),
                           'paired_standard_error': float(values.std(ddof=1) / np.sqrt(len(values))),
                           'action_counts': frequencies.tolist(), 'positive_lower_bound': bool(interval[0] > 0)}
    return {'mode': 'conventional-teacher-control', 'sampling': 'frozen-average-policy-probabilities',
            'opponents': summaries, 'passes_fixed_suite': all(s['positive_lower_bound'] for s in summaries.values()),
            'learning_claim_about_fly': False}


def verify_information_boundary(policy, seeds=(821001, 821002, 821003, 821004)):
    from collections import deque
    from flyholdem.poker.observation import canonical_bytes
    from flyholdem.poker.infoset import canonical_information_id
    checked = 0
    for seed in seeds:
        hand = Hand(seed)
        while not hand.done:
            observation = hand.observation()
            before = policy.probabilities(observation).tobytes()
            other = 1 - hand.state.actor_index
            holes = list(hand.state.hole_cards[other])
            deck = deque(hand.state.deck_cards)
            hand.state.hole_cards[other][:] = list(deck)[:2]
            hand.state.deck_cards = deque(reversed(deck))
            hand.teacher_labels = [999, -999, 999, -999, 999]
            try:
                changed = hand.observation()
                if canonical_bytes(observation) != canonical_bytes(changed):
                    raise AssertionError('Hidden data changed the acting-player information set')
                if before != policy.probabilities(changed).tobytes():
                    raise AssertionError('Hidden data changed conventional teacher targets')
                if canonical_information_id(observation) != canonical_information_id(changed):
                    raise AssertionError('Hidden data changed canonical information identity')
            finally:
                hand.state.hole_cards[other][:] = holes
                hand.state.deck_cards = deck
            hand.act(1)
            checked += 1
    return {'hidden_hole_future_deck_and_teacher_label_invariance': True, 'decisions_checked': checked}


def evaluate(policy_path, config, output, profile='development', resume=False):
    import json
    from pathlib import Path
    import signal
    from flyholdem.connectome.registry import digest
    from flyholdem.neural.checkpoint import atomic_json
    from flyholdem.experiments.journal import Journal
    from flyholdem.provenance import manifest, identity, assert_compatible
    from .loaders import load_policy, sampling
    if profile not in config['profiles'] or set(config['opponents']) != set(VERSIONS):
        raise ValueError('Use the complete registered opponent suite and profile')
    policy, policy_record = load_policy(policy_path)
    if policy_record['schema'] != 'teacher-external-regret-policy-v1':
        import torch
        torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    if policy_record['provenance'].get('stack_bb') != config['stack_bb']:
        raise ValueError('Teacher evaluation stack differs from the trained stack')
    selected = config['profiles'][profile]
    if selected['paired_deals_per_opponent'] < 2:
        raise ValueError('At least two independent paired deals required')
    output = Path(output); output.mkdir(parents=True, exist_ok=resume)
    policy_hash = digest(Path(policy_path) / 'manifest.json')
    runtime_config = {**config, 'profile': profile, 'policy_sha256': policy_hash}
    runtime = manifest(runtime_config, 'conventional-teacher-no-connectome',
                       identity(policy_record['feature_version']), identity(policy_record['aggregation']), policy_record.get('inference_backend','pytorch-cpu'))
    if resume:
        assert_compatible(json.loads((output / 'manifest.json').read_text()), runtime)
    else:
        atomic_json(output / 'manifest.json', runtime)
    boundary = verify_information_boundary(policy)
    journal = Journal(output / 'paired-deals.jsonl', resume)
    rows_by_opponent = {}
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    index = 0
    try:
        for kind in config['opponents']:
            rows = []
            for repeat in range(selected['paired_deals_per_opponent']):
                label = [kind, repeat]
                if index < len(journal.rows):
                    saved = journal.rows[index]
                    if saved['label'] != label:
                        raise ValueError('Teacher evaluation journal plan mismatch')
                    row = saved['value']
                else:
                    row = paired_match(policy, kind, selected['seed_start'] + repeat, config['stack_bb'])
                    journal.record(index, label, row)
                rows.append(row); index += 1
                if stopped:
                    raise KeyboardInterrupt('Frozen evaluation stopped at complete paired deal; use --resume')
            rows_by_opponent[kind] = rows
            print(json.dumps({'opponent': kind, 'paired_deals': len(rows),
                              'mean_bb_per_hand': float(np.mean([r['paired_bb_per_hand'] for r in rows]))}), flush=True)
        if index != len(journal.rows):
            raise ValueError('Unexpected trailing evaluation rows')
        summary = evaluation_summary(rows_by_opponent, config)
        result = {'schema': 'teacher-evaluation-v1', **summary, 'sampling': sampling(policy_record), 'profile': profile,
                  'policy_sha256': policy_hash, 'stack_bb': config['stack_bb'],
                  'information_boundary_verified': True, 'information_boundary': boundary,
                  'allowed_as_teacher': bool(profile == 'confirmatory' and summary['passes_fixed_suite']),
                  'journal_head': journal.rows[-1]['hash'], 'manifest_sha256': digest(output / 'manifest.json')}
        atomic_json(output / 'result.json', result)
        return result
    finally:
        journal.close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def main():
    import argparse,json,yaml
    from pathlib import Path
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy', required=True); p.add_argument('--config', required=True)
    p.add_argument('--output', required=True); p.add_argument('--profile', choices=['development','confirmatory'], default='development')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    print(json.dumps(evaluate(args.policy, yaml.safe_load(Path(args.config).read_text()), args.output,
                              args.profile, args.resume), indent=2))


if __name__ == '__main__':
    main()
