"""Prescribed abstract-recall witness; no learning, evaluation or selection.

Run with the repository Python environment. The exact engineering deals,
action patterns, abstraction and expected trace are independent of candidate
training configs. Verification remains active under Python's -O option.
"""
import hashlib
import json
from pathlib import Path

from flyholdem.connectome.registry import ROOT
from flyholdem.poker.engine import Hand
from flyholdem.poker.infoset import canonical_state
from flyholdem.provenance import canonical
from flyholdem.teacher.regret import VERSION, abstraction


ABSTRACTION_CONFIG = {
    'feature_version': 'visible-equity-recall-public-history-v1',
    'stack_bb': 20,
    'equity_buckets': 8,
    'equity_samples': 64,
    'max_information_sets': 2000000,
    'max_nodes_per_traversal': 100000,
}
DEAL_SEEDS = range(991920000, 991920008)
RECONSTRUCTION_SEED = 991990000
PATTERNS = ('call-through', 'half-open', 'preflop-reraise', 'flop-pot',
            'turn-reraise', 'river-shove', 'raise-fold')
PUBLIC_NAMES = ('position', 'street', 'pot', 'own_stack', 'opponent_stack',
                'contribution', 'to_call', 'min_raise', 'max_raise')
EXPECTED_COUNTS = {'hands': 112, 'current_keys_reconstructed': 832,
                   'earlier_own_key_action_pairs': 1280, 'button_remappings': 56}
EXPECTED_STREETS = [240, 192, 208, 192]
EXPECTED_ACTIONS = [16, 688, 64, 48, 16]
EXPECTED_MINIMA = [0, 2, 4, 6, 8, 18, 20]
EXPECTED_DISTINCT_KEYS = 234
EXPECTED_TRACE = 'f365bb31ffc4773692d598cb90445c5fa285fa156e48afa0768a4da853068d99'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reconstruct(key):
    """Recover earlier own abstract keys/actions from one current key alone."""
    try:
        version, total, buckets, public, history, mask = json.loads(key)
    except (ValueError, TypeError) as error:
        raise ValueError('Invalid abstract-recall key structure') from error
    require(version == VERSION == ABSTRACTION_CONFIG['feature_version'],
            'Abstract-recall feature version differs from the prescribed witness')
    require(total == 4 * ABSTRACTION_CONFIG['stack_bb'],
            'Abstract-recall key differs from the fixed equal-stack 20 BB domain')
    require(isinstance(public, list) and len(public) == len(PUBLIC_NAMES)
            and type(public[0]) is int and public[0] in (0, 1),
            'Abstract-recall key has no valid public button role')
    # Seat zero denotes the player whose key is reconstructed. PokerKit uses
    # an arbitrary independent deck only to reconstruct public betting state;
    # its cards must never be used to recompute any abstract bucket.
    game = Hand(RECONSTRUCTION_SEED, button=0 if public[0] else 1,
                stacks=(total // 2,) * 2)
    sequence = []
    for index, (street, own_actor, action, paid) in enumerate(history):
        require(not game.done, 'Recalled history continued after PokerKit settlement')
        observation = canonical_state(game.observation())
        require(observation['street'] == street and (game.actor == 0) == bool(own_actor),
                f'Recalled street or relative actor differs at history entry {index}')
        if own_actor:
            earlier = canonical([VERSION, total, buckets[:street + 1],
                                 [observation[name] for name in PUBLIC_NAMES],
                                 history[:index], observation['legal_mask']])
            sequence.append((earlier, action))
        committed = game.act(action)
        require(committed['paid'] == paid,
                f'Recalled paid-chip amount differs at history entry {index}')
    require(not game.done and game.actor == 0,
            'Recalled key does not end at the original player live decision')
    observation = canonical_state(game.observation())
    current = canonical([VERSION, total, buckets,
                         [observation[name] for name in PUBLIC_NAMES], history,
                         observation['legal_mask']])
    require(current == key, 'Reconstructed current abstract key differs from the actual key')
    return sequence


def choose(game, pattern):
    """Execute the prescribed action pattern, without values or payoff input."""
    require(pattern in PATTERNS, 'Unknown prescribed recall action pattern')
    observation = game.observation()
    on_street = sum(row['street'] == observation['street'] for row in game.history)
    action = 1
    if pattern == 'half-open' and not game.history:
        action = 2
    elif pattern == 'preflop-reraise' and observation['street'] == 0 and on_street < 2:
        action = (2, 3)[on_street]
    elif pattern == 'flop-pot' and observation['street'] == 1 and on_street == 0:
        action = 3
    elif pattern == 'turn-reraise' and observation['street'] == 2 and on_street < 2:
        action = (2, 3)[on_street]
    elif pattern == 'river-shove' and observation['street'] == 3 and on_street == 0:
        action = 4
    elif pattern == 'raise-fold' and observation['street'] == 0 and on_street < 2:
        action = (2, 0)[on_street]
    require(observation['legal_mask'][action],
            f'Prescribed {pattern} action {action} is illegal on street {observation["street"]}')
    return action


def audit_recall():
    require(VERSION == ABSTRACTION_CONFIG['feature_version'],
            'The current abstraction differs from the prescribed recall witness')
    counts = dict.fromkeys(EXPECTED_COUNTS, 0)
    streets, action_counts = [0] * 4, [0] * 5
    minima, seen = set(), {}
    witness = None
    trace = hashlib.sha256()
    for seed in DEAL_SEEDS:
        for pattern in PATTERNS:
            role_traces = []
            for button in (0, 1):
                game = Hand(seed, button=button,
                            stacks=(2 * ABSTRACTION_CONFIG['stack_bb'],) * 2)
                own_history, role_trace = {0: [], 1: []}, []
                while not game.done:
                    observation, seat = game.observation(), game.actor
                    key, _ = abstraction(observation, ABSTRACTION_CONFIG)
                    recovered = reconstruct(key)
                    require(recovered == own_history[seat],
                            f'Recalled earlier own decisions differ: seed {seed}, pattern {pattern}, button {button}')
                    counts['current_keys_reconstructed'] += 1
                    counts['earlier_own_key_action_pairs'] += len(recovered)
                    streets[observation['street']] += 1
                    minima.add(observation['min_raise'])
                    if key in seen:
                        previous = seen[key]
                        require(previous['sequence'] == recovered,
                                'Equal abstract keys recalled different earlier own decisions')
                        if witness is None and previous['hole'] != observation['hole']:
                            witness = {'key_sha256': hashlib.sha256(key).hexdigest(),
                                       'first_hole': previous['hole'], 'second_hole': observation['hole'],
                                       'street': observation['street'], 'previous_own_decisions': len(recovered)}
                    else:
                        seen[key] = {'sequence': recovered, 'hole': observation['hole']}
                    action = choose(game, pattern)
                    action_counts[action] += 1
                    own_history[seat].append((key, action))
                    game.act(action)
                    role_trace.append([json.loads(key), action])
                trace.update(canonical(role_trace) + b'\n')
                role_traces.append(role_trace)
                counts['hands'] += 1
            require(role_traces[0] == role_traces[1],
                    f'Public-role trace changed under button remapping: seed {seed}, pattern {pattern}')
            counts['button_remappings'] += 1
    checksum = trace.hexdigest()
    require(counts == EXPECTED_COUNTS, 'Prescribed recall coverage counts changed')
    require(streets == EXPECTED_STREETS and action_counts == EXPECTED_ACTIONS,
            'Prescribed recall street or action coverage changed')
    require(sorted(minima) == EXPECTED_MINIMA and len(seen) == EXPECTED_DISTINCT_KEYS,
            'Prescribed recall raise-minimum or distinct-key coverage changed')
    require(witness is not None, 'Prescribed witness did not demonstrate deliberate card coarsening')
    require(checksum == EXPECTED_TRACE, 'Prescribed abstract-recall witness trace changed')
    return {'schema': 'self-play-abstract-recall-witness-v1',
            'scope': 'prescribed engineering-deal recall witness, no learning/evaluation/parameter search',
            'diagnostic_source': str(Path(__file__).resolve().relative_to(ROOT)),
            'abstraction_config': dict(ABSTRACTION_CONFIG),
            'deal_range': [DEAL_SEEDS.start, DEAL_SEEDS.stop - 1],
            'reconstruction_deal': RECONSTRUCTION_SEED, 'patterns': PATTERNS,
            **counts, 'decisions_by_street': streets, 'abstract_action_counts': action_counts,
            'observed_min_raise_to_chips': sorted(minima), 'distinct_keys': len(seen),
            'distinct_exact_cards_same_abstract_key': witness, 'trace_sha256': checksum}


def main():
    print(json.dumps(audit_recall(), indent=2))


if __name__ == '__main__':
    main()
