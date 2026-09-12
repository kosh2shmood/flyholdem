"""Complete-hand NFSP self-play and numeric, resumable training checkpoints."""
import argparse
import json
from pathlib import Path
import signal
import time
import numpy as np
import torch
import yaml
from .features import feature_names, feature_runtime_identity, V1
from flyholdem.poker.engine import Hand
from flyholdem.poker.infoset import canonical_information_id
from flyholdem.provenance import manifest, identity, assert_compatible
from flyholdem.neural.checkpoint import Checkpoints, atomic_json
from flyholdem.experiments.journal import Journal
from .nfsp import NFSPAgent


def self_play_hand(agents, seed, button, stack_bb, epsilon, training=True, fixed_opponent=None, reward_parameterization=None):
    if reward_parameterization not in (None,'own-stack-potential-v1'):
        raise ValueError('Unknown conventional reward parameterization')
    if reward_parameterization and any(agent.config['discount']!=1 for agent in agents):
        raise ValueError('The registered stack potential requires undiscounted finite hands')
    reward_events=[]
    hand = Hand(seed, button=button, stacks=(2 * stack_bb,) * 2)
    fixed_seat=None
    if fixed_opponent is not None:
        from .fast_opponents import TrainingOpponent
        fixed_seat,kind=fixed_opponent
        if fixed_seat not in (0,1):raise ValueError('Fixed opponent requires one heads-up seat')
        fixed=TrainingOpponent(seed+2000000,kind)
    pending = [None, None]
    transitions = [0, 0]
    decisions = []
    def transition(seat,raw_reward,next_observation=None):
        reward=raw_reward
        if reward_parameterization:
            from .potential import transition_reward
            event=transition_reward(raw_reward,pending[seat][2],None if next_observation is None else next_observation['own_stack'],2*stack_bb)
            reward=event['shaped_reward'];reward_events.append({'seat':seat,**event})
        agents[seat].transition(pending[seat][:2],reward,next_observation)

    for agent in agents:
        agent.begin_hand()
    while not hand.done:
        seat = hand.actor
        observation = hand.observation()
        if seat==fixed_seat:
            action=fixed.act(observation)
            decisions.append({'seat':seat,'information_id':canonical_information_id(observation),
                              'action':action,'best_response_episode':False,'fixed_opponent':kind})
            hand.act(action);continue
        agent = agents[seat]
        # Bootstrap only at this SAME player's next decision, never on the other
        # player's private information set. Raw intermediate reward remains zero.
        if training and pending[seat] is not None:
            transition(seat,0,observation)
            transitions[seat] += 1
        action, x = agent.act(observation, epsilon, record=training)
        decisions.append({'seat': seat, 'information_id': canonical_information_id(observation),
                          'action': action, 'best_response_episode': agent.episode_br})
        pending[seat] = (x, action, observation['own_stack']) if reward_parameterization else (x, action)
        hand.act(action)
    payoffs = hand.view()['payoffs']
    if sum(payoffs) != 0:
        raise AssertionError('Teacher hand violates chip conservation')
    if training:
        for seat, agent in enumerate(agents):
            if pending[seat] is not None:
                transition(seat,payoffs[seat] / (2 * stack_bb))
                transitions[seat] += 1
    result={'deal_seed': seed, 'button': button, 'net_bb': [value / 2 for value in payoffs],
            'decisions': decisions, 'transitions': transitions, 'epsilon': epsilon}
    if reward_parameterization:
        result.update(reward_parameterization=reward_parameterization,reward_events=reward_events)
        for seat in (0,1):
            events=[value for value in reward_events if value['seat']==seat]
            if events and not np.isclose(sum(value['shaped_reward'] for value in events),
                    payoffs[seat]/(2*stack_bb)-events[0]['previous_potential'],rtol=0,atol=1e-12):
                raise AssertionError('Stack-potential rewards failed their telescoping identity')
    if fixed_opponent is not None:result.update(fixed_opponent=kind,fixed_seat=fixed_seat,learning_seats=[1-fixed_seat])
    return result


def train(config, output, resume=False, stop_after=None):
    if config['stack_bb'] not in (10, 20):
        raise ValueError('Independent teacher prototype is registered at 10 or 20 BB only')
    reward_parameterization=config.get('reward_parameterization')
    if reward_parameterization not in (None,'own-stack-potential-v1') or reward_parameterization and config['agent']['discount']!=1:
        raise ValueError('Use the registered undiscounted stack potential or unchanged terminal reward')
    population=config.get('population')
    if population is not None:
        from flyholdem.poker.opponents import VERSIONS
        expected=['self-play']*4+list(VERSIONS)
        if (not isinstance(population,dict) or config.get('algorithm')!='NFSP-fixed-policy-prior-v1' or population.get('cycle')!=expected
                or population.get('fixed_hand_update')!='active-agent-only'
                or population.get('fixed_opponent_backend')!='exact-v1-batched-ranks'):
            raise ValueError('Use the complete registered fixed-policy-prior cycle and update contract')
    elif config.get('algorithm','NFSP')!='NFSP':raise ValueError('Unknown conventional self-play algorithm')
    torch.set_num_threads(config['torch_threads'])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config['seed'])
    agents = [NFSPAgent(config['agent'], config['seed'] + 100 * seat) for seat in range(2)]
    runtime = manifest(config, 'conventional-teacher-no-connectome', identity(feature_names(config['agent'].get('feature_version', V1))),
                       identity({'algorithm': config.get('algorithm','NFSP')+'-average-policy', 'actions': 5}),
                       identity({'pytorch':'cpu','features':feature_runtime_identity(config['agent'].get('feature_version',V1))}))
    output = Path(output); output.mkdir(parents=True, exist_ok=resume)
    if resume:
        assert_compatible(json.loads((output / 'manifest.json').read_text()), runtime)
    else:
        atomic_json(output / 'manifest.json', runtime)
    journal = Journal(output / 'hands.jsonl', resume)
    checkpoints = Checkpoints(output / 'checkpoints')
    start = 0
    if resume:
        arrays, extra, metadata = checkpoints.load(runtime)
        start = extra['completed_hands']
        if start > len(journal.rows) or extra['journal_head'] != (journal.rows[start - 1]['hash'] if start else '0' * 64):
            raise ValueError('Teacher checkpoint/journal mismatch')
        for seat, agent in enumerate(agents):
            agent.restore_state(arrays, extra['agents'][seat], f'agent{seat}')
        torch.set_rng_state(torch.from_numpy(arrays['torch_rng']))
    completed = start
    last_checkpoint = time.monotonic()
    started = time.perf_counter()
    stopped = False
    in_hand = False

    def checkpoint():
        nonlocal last_checkpoint
        if in_hand:
            return  # An incomplete hand cannot replace a valid saved boundary.
        arrays = {'torch_rng': torch.get_rng_state().numpy().copy()}
        metadata = []
        for seat, agent in enumerate(agents):
            values, meta = agent.state(f'agent{seat}')
            arrays.update(values); metadata.append(meta)
        checkpoints.save(arrays, runtime, {'completed_hands': completed, 'agents': metadata,
            'journal_head': journal.rows[completed - 1]['hash'] if completed else '0' * 64,
            'private_hand': None, 'boundary': 'complete settled hand; no pending transitions'}, completed)
        last_checkpoint = time.monotonic()

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    if not resume:
        checkpoint()
    try:
        for hand_index in range(start, config['hands']):
            in_hand = True
            progress = min(1, hand_index / config['epsilon_decay_hands'])
            epsilon = config['epsilon_start'] + progress * (config['epsilon_end'] - config['epsilon_start'])
            fixed_opponent=None
            if population is not None:
                kind=population['cycle'][hand_index%len(population['cycle'])]
                if kind!='self-play':fixed_opponent=((hand_index//len(population['cycle']))%2,kind)
            row = self_play_hand(agents, config['deal_seed_start'] + hand_index, hand_index % 2,
                                 config['stack_bb'], epsilon,fixed_opponent=fixed_opponent,reward_parameterization=reward_parameterization)
            row['optimization'] = []
            for seat,agent in enumerate(agents):
                updates = [agent.train_step() for _ in range(config['updates_per_hand'])] if fixed_opponent is None or seat!=fixed_opponent[0] else []
                row['optimization'].append(updates)
            journal.record(hand_index, ['self-play-hand', hand_index], row)
            completed = hand_index + 1
            in_hand = False
            if time.monotonic() - last_checkpoint >= 300:
                checkpoint()
            if completed % config['progress_hands'] == 0:
                print(json.dumps({'hands': completed, 'replay': [a.replay.size for a in agents],
                    'reservoir': [a.reservoir.size for a in agents], 'last_losses': row['optimization']}), flush=True)
            if stopped or stop_after is not None and completed >= stop_after:
                break
        result = {'schema': 'nfsp-training-v1', 'algorithm': config.get('algorithm','NFSP'), 'mode': 'conventional-teacher-control',
                  'hands_completed': completed, 'planned_hands': config['hands'],
                  'status': 'trained-unvalidated' if completed == config['hands'] else 'interrupted',
                  'allowed_as_teacher': False, 'learning_claim_about_fly': False,
                  'elapsed_seconds_this_invocation': time.perf_counter() - started,
                  'journal_head': journal.rows[completed - 1]['hash'] if completed else '0' * 64}
        atomic_json(output / 'result.json', result)
        return result
    finally:
        checkpoint(); journal.close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--output', required=True)
    p.add_argument('--resume', action='store_true'); p.add_argument('--stop-after', type=int)
    args = p.parse_args()
    result = train(yaml.safe_load(Path(args.config).read_text()), args.output, args.resume, args.stop_after)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
