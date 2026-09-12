import json
import numpy as np
import pytest
from flyholdem.experiments.conditioning import paired_evidence, cue_registration
from flyholdem.experiments.journal import Journal


def test_journal_replays_checkpoint_tail_and_detects_divergence(tmp_path):
    path = tmp_path / 'trials.jsonl'
    j = Journal(path)
    j.record(0, ['train', 0], {'reward': -1})
    j.record(1, ['train', 1], {'reward': 1})
    j.close(); original = path.read_bytes()
    j = Journal(path, resume=True)
    j.record(1, ['train', 1], {'reward': 1})
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match='differs'):
        j.record(1, ['train', 1], {'reward': -1})
    j.close()
    row = json.loads(path.read_text().splitlines()[1]); row['value']['reward'] = -1
    path.write_text(path.read_text().splitlines()[0] + '\n' + json.dumps(row) + '\n')
    with pytest.raises(ValueError, match='checksum'):
        Journal(path, resume=True)


def test_sign_flip_test_requires_independent_positive_seed_evidence():
    good = paired_evidence([.4] * 5, 1, 100)
    assert good['one_sided_sign_flip_p'] == 1 / 32
    assert good['paired_seed_bootstrap_95'] == [.4, .4]
    bad = paired_evidence([.4, -.4, .4, -.4, 0], 1, 100)
    assert bad['one_sided_sign_flip_p'] >= .5


def test_conditioning_cues_are_disjoint_existing_shared_paths():
    class Brain:
        ptr = np.array([0, 2, 4, 6, 8, 8, 8])
        post = np.tile([4, 5], 4)
        initial = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
    registration = {'input_indices': [0, 1, 2, 3], 'ensembles': [{'indices': [4]}, {'indices': [5]}]}
    config = {'actions': [0, 1], 'cells_per_cue': 2, 'seed': 31, 'selection': 'minimum-original-strength'}
    first = cue_registration(Brain(), registration, config)
    assert first == cue_registration(Brain(), registration, config)
    assert sorted(np.array(first['indices']).ravel().tolist()) == [0, 1, 2, 3]
    assert first['shared_candidates'] == 4
    assert np.all(np.array(first['selected_strengths']) > 0)
