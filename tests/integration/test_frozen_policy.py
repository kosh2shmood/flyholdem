import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import pytest
from flyholdem.connectome.registry import ROOT
from flyholdem.inference_identity import RUNTIME_FILES
from flyholdem.interface.frozen import verify_changed_edges
from flyholdem.neural.kernel.build import LIBRARY
from flyholdem.neural.sparse import SparseBrain
from flyholdem.poker.engine import Hand


def test_frozen_weights_reject_new_edges_outside_changes_and_bad_bounds():
    brain = SparseBrain([0, 2, 2, 2], [1, 2], [.2, -.3])
    brain.weights[1] *= 1.2
    assert verify_changed_edges(brain, np.array([1]), [.1, 2]).tolist() == [1]
    with pytest.raises(ValueError, match='outside'):
        verify_changed_edges(brain, np.array([0]), [.1, 2])
    with pytest.raises(ValueError, match='Unique'):
        verify_changed_edges(brain, np.array([1, 1]), [.1, 2])
    brain.weights[1] = brain.initial[1] * .05
    with pytest.raises(ValueError, match='bounds'):
        verify_changed_edges(brain, np.array([1]), [.1, 2])


def test_deleting_teacher_and_corpus_preserves_native_decision_bytes(tmp_path):
    source = tmp_path / 'src/flyholdem'
    for name in RUNTIME_FILES:
        target = source / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / 'src/flyholdem' / name, target)
    build = tmp_path / 'runs/build'; build.mkdir(parents=True)
    for path in (LIBRARY, LIBRARY.with_suffix(LIBRARY.suffix + '.json')):
        shutil.copy2(path, build / path.name)
    # Include the actual optional teacher implementation before deleting it.
    shutil.copytree(ROOT / 'src/flyholdem/teacher', source / 'teacher', ignore=shutil.ignore_patterns('__pycache__'))
    corpus = tmp_path / 'corpus'; corpus.mkdir()
    (corpus / 'labels.json').write_text('{"fixture_teacher_targets":[1,0,0,0,0]}')
    graph = tmp_path / 'graph'; graph.mkdir()
    arrays = {'ptr': np.array([0,5,5,5,5,5,5], dtype=np.int64), 'post': np.arange(1,6,dtype=np.int32),
              'weight': np.array([.275,.55,.825,1.1,1.375],dtype=np.float32),
              'contacts': np.arange(1,6,dtype=np.uint32), 'ids': np.arange(6,dtype=np.uint64),
              'signs': np.ones(6,dtype=np.int8), 'uncertain': np.zeros(6,dtype=bool)}
    entries = {}
    for name, array in arrays.items():
        path = graph / (name + '.npy'); np.save(path, array, allow_pickle=False)
        entries[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    graph_hash = hashlib.sha256(b''.join(arrays[k].tobytes() for k in ('ptr','post','weight'))).hexdigest()
    (graph / 'manifest.json').write_text(json.dumps({'files': entries, 'graph_hash': graph_hash, 'mode': 'fixture-native'}))
    registration = {'stage':'frozen','mode':'fixture-native','graph_hash':graph_hash,'base_graph_hash':graph_hash,
                    'config':{'decision_ms':{'baseline':50,'stimulus':300,'readout':100,'inter_decision':50},'score_scale_hz':100},
                    'projection_indices':[[0]]*237,'ensembles':[{'indices':[i]} for i in range(1,6)],
                    'baseline_hz':[0]*5,'selected_gain':12}
    (tmp_path / 'registration.json').write_text(json.dumps(registration))
    (tmp_path / 'observations.json').write_text(json.dumps([Hand(i).observation() for i in range(4)]))
    env = dict(os.environ, PYTHONPATH=str(tmp_path / 'src'))
    create = '''import numpy as np
from flyholdem.interface.population import load_controller
from flyholdem.interface.frozen import export_frozen
c=load_controller('graph','registration.json');c.brain.weights[1]*=1.2
export_frozen(c,np.arange(5),[.1,2],'distilled-connectome','model',{'teacher':'teacher','corpus':'corpus'},{'status':'engineering-test-only'})
'''
    subprocess.run([sys.executable, '-c', create], cwd=tmp_path, env=env, check=True)
    probe = '''import json,sys
from flyholdem.interface.frozen import load_frozen
c=load_frozen('model','graph');out=[]
for obs in json.load(open('observations.json')):
 result=c.decide(obs,first_in_hand=True);result['counts']=result['counts'].tolist();out.append(result)
assert not any(name.startswith('flyholdem.teacher') for name in sys.modules)
print(json.dumps(out,sort_keys=True,separators=(',',':')))
'''
    before = subprocess.check_output([sys.executable, '-c', probe], cwd=tmp_path, env=env)
    shutil.rmtree(source / 'teacher'); shutil.rmtree(corpus)
    after = subprocess.check_output([sys.executable, '-c', probe], cwd=tmp_path, env=env)
    assert before == after
    path = source / 'interface/encoder.py'; path.write_text(path.read_text() + '\n# changed runtime\n')
    changed = subprocess.run([sys.executable, '-c', probe], cwd=tmp_path, env=env, capture_output=True)
    assert changed.returncode != 0 and b'runtime/source mismatch' in changed.stderr
