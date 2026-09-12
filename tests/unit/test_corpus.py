import json
from pathlib import Path
import numpy as np
import pytest
torch = pytest.importorskip('torch', reason='Optional teacher extra')
from flyholdem.connectome.registry import digest
from flyholdem.poker.engine import Hand
from flyholdem.poker.infoset import canonical_state, canonical_information_id
from flyholdem.teacher.corpus import split_for_id, stratum, verify_corpus, export_corpus


def fixture_corpus(root):
    rows = {split: [] for split in ('train','validation','test')}
    seen = set()
    for seed in range(50):
        obs = canonical_state(Hand(seed).observation())
        info = canonical_information_id(obs); split = split_for_id(info)
        if info in seen:
            continue
        seen.add(info)
        target = np.array(obs['legal_mask'], dtype=float); target /= target.sum()
        rows[split].append({'schema':'distillation-row-v1', 'information_id':info, 'observation':obs,
                            'teacher_probabilities':target.tolist(), 'split':split, 'stratum':stratum(obs)})
    for split, values in rows.items():
        assert values
        (root / (split + '.jsonl')).write_text(''.join(json.dumps(value) + '\n' for value in values))
    (root / 'collection.jsonl').write_text('engineering-fixture\n')
    (root / 'run-manifest.json').write_text('{"scope":"engineering fixture"}')
    manifest = {'schema':'distillation-corpus-v1', 'files':{split+'.jsonl':digest(root/(split+'.jsonl')) for split in rows},
                'split_counts':{split:len(values) for split,values in rows.items()}, 'unique_information_sets':len(seen),
                'collection_sha256':digest(root/'collection.jsonl'), 'run_manifest_sha256':digest(root/'run-manifest.json')}
    (root / 'manifest.json').write_text(json.dumps(manifest))
    return rows, manifest


def rewrite(root, rows, manifest):
    for split, values in rows.items():
        path = root / (split + '.jsonl')
        path.write_text(''.join(json.dumps(value) + '\n' for value in values))
        manifest['files'][path.name] = digest(path)
    (root / 'manifest.json').write_text(json.dumps(manifest))


def test_corpus_canonical_splits_have_no_overlapping_ids(tmp_path):
    rows, manifest = fixture_corpus(tmp_path)
    assert verify_corpus(tmp_path)['canonical_ids_disjoint']
    rows['validation'].append(rows['train'][0])
    rewrite(tmp_path, rows, manifest)
    with pytest.raises(ValueError, match='split overlap'):
        verify_corpus(tmp_path)


def test_corpus_rejects_hidden_fields_even_with_recomputed_file_checksums(tmp_path):
    rows, manifest = fixture_corpus(tmp_path)
    rows['train'][0]['observation']['opponent_hole'] = ['As','Ah']
    rewrite(tmp_path, rows, manifest)
    with pytest.raises(ValueError, match='allowlist'):
        verify_corpus(tmp_path)


def test_unvalidated_teacher_cannot_generate_a_training_corpus(tmp_path):
    policy = tmp_path / 'policy'; policy.mkdir(); (policy/'manifest.json').write_text('{}')
    report = tmp_path / 'evaluation.json'; report.write_text('{"schema":"teacher-evaluation-v1","profile":"confirmatory","passes_fixed_suite":false}')
    with pytest.raises(ValueError, match='validated'):
        export_corpus(policy, report, {'stack_bb':20}, tmp_path / 'corpus')
    assert not (tmp_path / 'corpus').exists()
