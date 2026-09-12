"""Actual tiny teacher play, observed at the original PokerKit action boundary."""
from collections import Counter
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from flyholdem.connectome.registry import ROOT, digest
from flyholdem.experiments.journal import Journal
from flyholdem.poker.engine import Hand
from flyholdem.poker.opponents import VERSIONS
from flyholdem.teacher import evaluation
from flyholdem.teacher.regret_training import train
from flyholdem.teacher.regret_policy import export_policy
from flyholdem.teacher.validation import verify_evaluation
from test_external_regret import config as training_config


@pytest.fixture(scope='module')
def audit_module():
    spec = importlib.util.spec_from_file_location(
        'actual_teacher_play_audit', ROOT / 'scripts/audit_teacher_play.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def numeric_training(tmp_path_factory):
    root = tmp_path_factory.mktemp('actual-teacher-play')
    settings = training_config()
    settings['iterations'] = 4
    train(settings, root / 'training')
    return root, settings


@pytest.fixture(scope='module', params=['average', 'final-current', 'self-play'])
def measured_play(request, numeric_training):
    root, settings = numeric_training
    policy_path = root / (request.param + '-policy')
    if request.param == 'self-play':
        from flyholdem.teacher.self_play_training import train as train_self_play
        from flyholdem.teacher.self_play_policy import export_policy as export_self_play
        from flyholdem.teacher.self_play_regret import SCHEMA, AGGREGATION
        current = {key: settings[key] for key in ('iterations', 'stack_bb', 'sampling_seed', 'deal_seed_start', 'abstraction')}
        current.update(schema=SCHEMA, aggregation=AGGREGATION)
        train_self_play(current, root / 'self-play-training')
        export_self_play(root / 'self-play-training', policy_path)
    elif request.param == 'average':
        export_policy(root / 'training', policy_path)
    else:
        from flyholdem.teacher.regret_current_policy import EXTRACTION, AGGREGATION, export_current
        manifest = json.loads((root / 'training/manifest.json').read_text())
        export_current({
            'schema': EXTRACTION, 'status': 'development', 'aggregation': AGGREGATION,
            'training_run': str(root / 'training'), 'iterations': settings['iterations'],
            'training_manifest_sha256': digest(root / 'training/manifest.json'),
            'training_source_sha256': manifest['source_hash'],
            'training_config_sha256': manifest['config_hash'],
        }, policy_path)
    suite = {
        'schema': 'conventional-teacher-suite-v1', 'stack_bb': 20,
        'opponents': list(VERSIONS), 'bootstrap_seed': 91900, 'bootstrap_repeats': 100,
        'profiles': {'development': {'paired_deals_per_opponent': 8, 'seed_start': 992100000}},
    }
    decisions = []
    original_match = evaluation.paired_match

    def observed_match(policy, kind, deal_seed, stack_bb):
        # Preserve the real evaluator and its RNG. Observe the committed moves
        # during its two PokerKit hands rather than implement another simulator.
        hands = []

        class ObservedHand(Hand):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.teacher_seat = len(hands)
                hands.append(self)

            def act(self, action):
                own_turn = self.actor == self.teacher_seat
                observation = self.observation()
                committed = super().act(action)
                if own_turn:
                    decisions.append({
                        'opponent': kind, 'seat': self.teacher_seat,
                        'street': observation['street'], 'to_call': observation['to_call'],
                        'action': action, 'paid': committed['paid'],
                    })
                return committed

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(evaluation, 'Hand', ObservedHand)
            result = original_match(policy, kind, deal_seed, stack_bb)
        assert len(hands) == 2 and all(hand.done for hand in hands)
        return result

    run = root / (request.param + '-evaluation')
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(evaluation, 'paired_match', observed_match)
        evaluation.evaluate(policy_path, suite, run)
    return root, run, policy_path, suite, decisions


def artifact_hashes(root):
    return {str(path.relative_to(root)): digest(path)
            for path in root.rglob('*') if path.is_file()}


def test_actual_play_audit_matches_original_observed_actions_without_mutation(
        measured_play, audit_module, monkeypatch):
    import builtins
    root, run, policy, suite, observed = measured_play
    original_import = builtins.__import__

    def without_torch(name, *args, **kwargs):
        if name == 'torch' or name.startswith('torch.'):
            raise AssertionError('Numeric teacher play audit must not import Torch')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', without_torch)
    before = artifact_hashes(root)
    source_before = {str(path): digest(path) for path in (ROOT / 'src').rglob('*.py')}
    source_before[str(Path(audit_module.__file__))] = digest(audit_module.__file__)
    first = audit_module.audit_play(run, policy, suite)
    assert first == audit_module.audit_play(run, policy, suite)
    assert before == artifact_hashes(root)
    assert source_before == {path: digest(path) for path in source_before}
    assert first['actual_paired_records_reproduced'] == 8 * len(VERSIONS)
    assert first['decisions'] == len(observed)
    assert first['grants_teacher_qualification'] is False
    assert not first['verified_evaluation']['allowed_as_teacher']
    assert len(first['decision_trace_sha256']) == 64
    assert {row['seat'] for row in observed} == {0, 1}

    names = ('fold', 'check', 'call', 'half_pot_raise', 'pot_raise', 'all_in')
    streets = ('preflop', 'flop', 'turn', 'river')
    # The engine's committed payment separates check from call independently
    # of the diagnostic's to_call classification. Verify both against the real
    # acting observation, so a reversed check/call label cannot pass.
    def name(row):
        if row['action'] == 1:
            assert (row['paid'] == 0) == (row['to_call'] == 0)
            return 'check' if row['paid'] == 0 else 'call'
        return {0: 'fold', 2: 'half_pot_raise', 3: 'pot_raise', 4: 'all_in'}[row['action']]

    expected = Counter(name(row) for row in observed)
    assert all(expected[action] > 0 for action in names)
    assert first['actions'] == {action: expected[action] for action in names}
    abstract = Counter(row['action'] for row in observed)
    assert first['abstract_action_counts'] == [abstract[action] for action in range(5)]
    assert expected['check'] + expected['call'] == abstract[1]
    for index, street in enumerate(streets):
        street_rows = [row for row in observed if row['street'] == index]
        assert street_rows
        counts = Counter(name(row) for row in street_rows)
        assert first['by_street'][street] == {action: counts[action] for action in names}
        for kind in suite['opponents']:
            counts = Counter(name(row) for row in street_rows if row['opponent'] == kind)
            assert first['by_opponent'][kind][street] == {action: counts[action] for action in names}
    entries = [json.loads(line)['value'] for line in (run / 'paired-deals.jsonl').read_text().splitlines()]
    assert np.sum([entry['action_counts'] for entry in entries], axis=0).tolist() == first['abstract_action_counts']


@pytest.mark.parametrize('false_field', ['payoff', 'action-count'])
def test_actual_replay_rejects_coherently_rehashed_false_evaluation(
        measured_play, audit_module, tmp_path, false_field):
    _, original, policy, suite, _ = measured_play
    run = tmp_path / 'changed-evaluation'
    shutil.copytree(original, run)
    path = run / 'paired-deals.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    false = rows[0]['value']
    if false_field == 'payoff':
        old = false['seat_returns_bb'][0]
        false['seat_returns_bb'][0] += -.5 if old == suite['stack_bb'] else .5
        false['paired_bb_per_hand'] = float(np.mean(false['seat_returns_bb']))
    else:
        counts = false['action_counts']
        source = next(index for index, count in enumerate(counts) if count > 0)
        counts[source] -= 1
        counts[(source + 1) % 5] += 1
    path.unlink()
    writer = Journal(path)
    grouped = {kind: [] for kind in suite['opponents']}
    try:
        for row in rows:
            writer.record(row['index'], row['label'], row['value'])
            grouped[row['label'][0]].append(row['value'])
        head = writer.rows[-1]['hash']
    finally:
        writer.close()
    result_path = run / 'result.json'
    result = json.loads(result_path.read_text())
    sampling = result['sampling']
    result.update(evaluation.evaluation_summary(grouped, suite))
    result.update(journal_head=head, sampling=sampling)
    result_path.write_text(json.dumps(result))
    # Consistency alone accepts this fully rehashed fiction. Executing the
    # frozen policy on the original deals must independently reject it.
    checked = verify_evaluation(run, policy, suite, require_confirmatory=False)
    assert checked['summary_recomputed'] and checked['registered_schedule_verified']
    with pytest.raises(ValueError, match='Actual teacher play differs'):
        audit_module.audit_play(run, policy, suite)


def test_public_audit_cli_does_not_allow_a_replacement_registration(audit_module, monkeypatch, capsys, tmp_path):
    import sys
    monkeypatch.setattr(sys, 'argv', [
        'audit_teacher_play.py', '--run', 'unused-run', '--policy', 'unused-policy',
        '--output', str(tmp_path / 'audit.json'), '--config', 'replacement.yaml',
    ])
    with pytest.raises(SystemExit) as stopped:
        audit_module.main()
    assert stopped.value.code == 2
    assert 'unrecognized arguments' in capsys.readouterr().err
    assert not (tmp_path / 'audit.json').exists()
