import copy
import json
import numpy as np
import pytest
torch = pytest.importorskip('torch', reason='Optional teacher extra')
from flyholdem.poker.engine import Hand
from flyholdem.teacher.memory import Memory
from flyholdem.teacher.nfsp import NFSPAgent, features
from flyholdem.teacher.training import train, self_play_hand
from flyholdem.teacher.policy import TeacherPolicy, export_policy, load_policy

CONFIG = dict(stack_bb=20, torch_threads=1, seed=71100, hands=30, deal_seed_start=71200,
              epsilon_decay_hands=20, epsilon_start=.9, epsilon_end=.1, updates_per_hand=1, progress_hands=30,
              agent=dict(hidden=16, q_lr=.001, average_lr=.001, replay_capacity=200, reservoir_capacity=200,
                         anticipatory=.5, batch_size=8, warmup_transitions=8, discount=1,
                         gradient_clip=5, target_update_steps=10))


@pytest.mark.parametrize('feature_version', [None, 'canonical-visible-card-structure-v2', 'canonical-visible-uniform-equity-v3'])
def test_teacher_complete_checkpoint_matches_uninterrupted_optimizer_and_replay(tmp_path, feature_version):
    full, resumed = tmp_path / 'full', tmp_path / 'resumed'
    config = copy.deepcopy(CONFIG)
    if feature_version:
        config['agent']['feature_version'] = feature_version
    train(config, full)
    train(config, resumed, stop_after=13)
    train(config, resumed, resume=True)
    assert (full / 'hands.jsonl').read_bytes() == (resumed / 'hands.jsonl').read_bytes()
    assert json.loads((resumed / 'result.json').read_text())['allowed_as_teacher'] is False
    assert (resumed / 'checkpoints/latest.json').exists()


def test_teacher_terminal_targets_and_episode_mixture_are_information_safe():
    torch.set_num_threads(1)
    agents = [NFSPAgent(CONFIG['agent'], 91 + seat) for seat in range(2)]
    row = self_play_hand(agents, 852, 0, 20, .3)
    assert sum(row['net_bb']) == 0
    for seat, agent in enumerate(agents):
        memory = agent.replay
        count = memory.size
        assert row['transitions'][seat] == count
        if count:
            assert memory.arrays['terminal'][:count].sum() == 1
            assert memory.arrays['reward'][count - 1] == pytest.approx(row['net_bb'][seat] / 20)
            assert np.all(memory.arrays['reward'][:count - 1] == 0)
            assert np.all(memory.arrays['next_legal'][:count - 1].any(axis=1))
            assert not memory.arrays['next_legal'][count - 1].any()
        if not agent.episode_br:
            assert agent.reservoir.size == 0
        assert all(item['best_response_episode'] == agent.episode_br for item in row['decisions'] if item['seat'] == seat)


def test_frozen_teacher_targets_are_canonical_and_export_exact(tmp_path):
    torch.set_num_threads(1)
    agents = [NFSPAgent(CONFIG['agent'], 51 + seat) for seat in range(2)]
    original = TeacherPolicy([agent.average for agent in agents])
    export_policy(agents, tmp_path / 'policy', {'stack_bb': 20, 'status': 'untrained-engineering-fixture'})
    restored, _ = load_policy(tmp_path / 'policy')
    hand = Hand(813)
    for _ in range(4):
        obs = hand.observation()
        target = original.probabilities(obs)
        assert np.array_equal(target, restored.probabilities(obs))
        altered = copy.deepcopy(obs); altered['hole'].reverse(); altered['board'][:3] = reversed(altered['board'][:3])
        assert np.array_equal(target, restored.probabilities(altered))
        assert np.all(target[~np.array(obs['legal_mask'])] == 0)
        assert target.sum() == pytest.approx(1)
        hand.act(1)
    changed = dict(hand.observation(), opponent_hole=['As','Ah'])
    with pytest.raises(ValueError): restored.probabilities(changed)
    path = next((tmp_path / 'policy').glob('*.npy'))
    data = bytearray(path.read_bytes()); data[-1] ^= 1; path.write_bytes(data)
    with pytest.raises(ValueError, match='checksum'): load_policy(tmp_path / 'policy')


def test_uniform_reservoir_and_ring_restore_own_rng_and_count():
    fields = {'value': ((), np.int64)}
    for reservoir in (False, True):
        a = Memory(5, fields, 19, reservoir)
        for i in range(25): a.add(value=i)
        arrays, state = a.state()
        b = Memory(5, fields, 19, reservoir); b.restore_state(arrays, state)
        for i in range(25, 50):
            a.add(value=i); b.add(value=i)
            assert np.array_equal(a.sample(3)['value'], b.sample(3)['value'])
        assert np.array_equal(a.arrays['value'], b.arrays['value'])


def test_teacher_checks_actual_private_holes_and_future_deck_invariance():
    from flyholdem.teacher.evaluation import verify_information_boundary
    agents = [NFSPAgent(CONFIG['agent'], 51 + seat) for seat in range(2)]
    report = verify_information_boundary(TeacherPolicy([a.average for a in agents]), seeds=(713, 714))
    assert report['hidden_hole_future_deck_and_teacher_label_invariance']
    assert report['decisions_checked'] >= 8


def test_visible_card_structure_stays_inside_teacher_and_has_no_hidden_inputs(tmp_path):
    from flyholdem.teacher.features import V2, feature_names
    from flyholdem.interface.encoder import encode_player_state, CHANNELS
    from flyholdem.teacher.evaluation import verify_information_boundary
    hand = Hand(19); hand.act(1); hand.act(1)
    obs = dict(hand.observation(), hole=['As','Ah'], board=['Ac','2d','3h'])
    x = features(obs, V2); names = feature_names(V2)
    assert x.shape == (354,) and len(CHANNELS) == 237
    assert np.array_equal(x[:237], encode_player_state(obs).astype(np.float32))
    assert x[names.index('own_pair:A')] == 1
    assert x[names.index('visible_made_category:THREE_OF_A_KIND')] == 1
    config = dict(CONFIG['agent'], feature_version=V2)
    agents = [NFSPAgent(config, 201 + seat) for seat in range(2)]
    policy = TeacherPolicy([a.average for a in agents], V2)
    report = verify_information_boundary(policy, seeds=(713, 714))
    assert report['hidden_hole_future_deck_and_teacher_label_invariance']
    export_policy(agents, tmp_path/'v2-policy', {'stack_bb':20,'status':'engineering-fixture'})
    restored,_ = load_policy(tmp_path/'v2-policy')
    assert np.array_equal(policy.probabilities(obs), restored.probabilities(obs))


def test_teacher_equity_representation_is_private_invariant_and_export_pins_binary(tmp_path):
    from flyholdem.teacher.features import V3, features, feature_names
    from flyholdem.teacher.equity import backend_identity
    from flyholdem.interface.encoder import encode_player_state
    config=dict(CONFIG['agent'],feature_version=V3)
    agents=[NFSPAgent(config,301+seat) for seat in range(2)]
    policy=TeacherPolicy([a.average for a in agents],V3)
    obs=Hand(1967).observation();x=features(obs,V3)
    assert x.shape==(357,) and np.array_equal(x[:237],encode_player_state(obs).astype(np.float32))
    assert 0<=x[feature_names(V3).index('teacher_visible_uniform_equity')]<=1
    from flyholdem.teacher.evaluation import verify_information_boundary
    assert verify_information_boundary(policy,seeds=(713,714))['hidden_hole_future_deck_and_teacher_label_invariance']
    export_policy(agents,tmp_path/'v3',{'stack_bb':20,'status':'engineering-fixture'})
    restored,record=load_policy(tmp_path/'v3')
    assert np.array_equal(policy.probabilities(obs),restored.probabilities(obs))
    assert record['implementation']['feature_runtime']['visible_equity_native']['binary_sha256']==backend_identity()['binary_sha256']
