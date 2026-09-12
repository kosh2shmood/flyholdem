"""Verify an actual exact-transfer model after deleting teacher/corpus files."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from flyholdem.connectome.registry import ROOT, digest
from flyholdem.inference_identity import RUNTIME_FILES
from flyholdem.neural.kernel.build import LIBRARY
from flyholdem.provenance import identity


def verify(run, model, output):
    run = Path(run).resolve(); model = Path(model).resolve()
    config = json.loads((run / 'manifest.json').read_text())['config']
    record = json.loads((model / 'manifest.json').read_text())
    if record['validation_reference']['result_sha256'] != digest(run / 'result.json'):
        raise ValueError('Model and transfer result differ')
    seed = record['training_reference']['best_validation']['seed']
    count = config['profiles'][config['profile']]['evaluation_trials_per_cue'] * 2
    requests = [{'cue':i%2,'stimulus_seed':int(identity([seed,'evaluation',i])[:16],16)} for i in range(count)]
    rows = [json.loads(line) for line in (run / 'trials.jsonl').read_text().splitlines()]
    expected = [r['value'] for r in rows if r['label'][:3] == [seed,'plastic','post']]
    if len(expected) != count:
        raise ValueError('Missing actual trained-model evaluation decisions')
    with tempfile.TemporaryDirectory(prefix='flyholdem-exact-inference-') as temporary:
        temporary = Path(temporary); source = temporary / 'src/flyholdem'
        for name in (*RUNTIME_FILES, 'interface/cue_player.py'):
            target = source / name; target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT / 'src/flyholdem' / name,target)
        build = temporary / 'runs/build'; build.mkdir(parents=True)
        for path in (LIBRARY,LIBRARY.with_suffix(LIBRARY.suffix+'.json')):
            shutil.copy2(path,build/path.name)
        shutil.copytree(ROOT / 'src/flyholdem/teacher', source / 'teacher', ignore=shutil.ignore_patterns('__pycache__'))
        corpus = temporary / 'corpus'; corpus.mkdir()
        shutil.copy2(run / 'teacher.json',corpus/'targets.json')
        (temporary/'requests.json').write_text(json.dumps(requests))
        graph = ROOT / 'connectome_data/malecns_v1' / ('prepared-'+config['mode'])
        probe = '''import json,sys
from flyholdem.interface.cue_player import load_cue_player
player=load_cue_player(sys.argv[1],sys.argv[2],sys.argv[3])
out=[player.decide(**request) for request in json.load(open('requests.json'))]
assert not any(name.startswith(('flyholdem.teacher','flyholdem.learning','flyholdem.experiments')) for name in sys.modules)
print(json.dumps(out,sort_keys=True,separators=(',',':')))
'''
        env = dict(os.environ,PYTHONPATH=str(temporary/'src'))
        command = [sys.executable,'-c',probe,str(model),str(graph),digest(model/'manifest.json')]
        before = subprocess.check_output(command,cwd=temporary,env=env)
        actual = json.loads(before)
        for found, historical in zip(actual,expected):
            if any(historical[key] != value for key,value in found.items()):
                raise AssertionError('Frozen inference differs from actual trained evaluation')
        shutil.rmtree(source/'teacher'); shutil.rmtree(corpus)
        after = subprocess.check_output(command,cwd=temporary,env=env)
        if before != after:
            raise AssertionError('Deleting actual teacher/corpus changed inference bytes')
        # The separate cue driver is pinned in addition to the common native runtime.
        driver = source/'interface/cue_player.py'; driver.write_text(driver.read_text()+'\n# changed cue driver\n')
        changed = subprocess.run(command,cwd=temporary,env=env,capture_output=True)
        if changed.returncode == 0 or b'Cue inference implementation checksum mismatch' not in changed.stderr:
            raise AssertionError('Changed cue inference source was not rejected')
        result = {'schema':'actual-exact-transfer-inference-audit-v1','model_sha256':digest(model/'manifest.json'),
            'mode':config['mode'],'learning_mode':'distilled-connectome','decisions_checked':count,
            'actual_trained_evaluation_reproduced':True,'teacher_and_corpus_deleted':True,
            'no_teacher_learning_or_experiment_imports':True,'decision_bytes_identical':True,
            'cue_source_mutation_rejected':True,'decision_bytes_sha256':hashlib.sha256(before).hexdigest(),
            'scope':'Actual exact-cue circuit model; not poker strength or complete Gate 2A'}
    Path(output).write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True);p.add_argument('--model',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();print(json.dumps(verify(args.run,args.model,args.output),indent=2))
