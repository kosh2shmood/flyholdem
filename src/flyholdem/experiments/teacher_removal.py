"""Execute frozen cue inference before and after deleting its teacher artifacts.

Only inference modules, a verified native binary and a numeric model are needed.
The temporary package cannot fall through to the project's training package.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from flyholdem.connectome.registry import ROOT,digest
from flyholdem.provenance import canonical,identity
from flyholdem.inference_identity import RUNTIME_FILES
from flyholdem.neural.kernel.build import LIBRARY

PROBE='''import json,sys
from pathlib import Path
from flyholdem.interface.cue_player import load_cue_player
import flyholdem
assert Path(flyholdem.__file__).resolve().is_relative_to(Path('src').resolve())
player=load_cue_player('model',sys.argv[1]);rows=json.load(open('probes.json'));out=[]
for row in rows:out.append(player.decide(row['cue'],row['stimulus_seed']))
assert not any(name.startswith(('flyholdem.teacher','flyholdem.learning','flyholdem.experiments')) for name in sys.modules)
print(json.dumps(out,sort_keys=True,separators=(',',':'),allow_nan=False))
'''


def verify_cue_teacher_removal(model,graph,transfer_run):
    model=Path(model).resolve();graph=Path(graph).resolve();run=Path(transfer_run).resolve()
    record=json.loads((model/'manifest.json').read_text());reference=record['training_reference']
    if (record['learning_mode']!='distilled-connectome' or record['teacher_connected'] is not False
            or reference['manifest_sha256']!=digest(run/'manifest.json')
            or record['validation_reference']['result_sha256']!=digest(run/'result.json')):
        raise ValueError('Teacher removal requires the actual frozen model from this transfer result')
    seed=reference['best_validation']['seed']
    probes=[]
    for line in (run/'trials.jsonl').read_text().splitlines():
        row=json.loads(line)
        if row['label'][:3]==[seed,'plastic','post']:probes.append(row['value'])
    runtime=json.loads((run/'manifest.json').read_text())['config']
    count=2*runtime['profiles']['confirmatory']['evaluation_trials_per_cue']
    if len(probes)!=count or count<2:raise ValueError('Incomplete actual post-training probe schedule')
    with tempfile.TemporaryDirectory(prefix='flyholdem-teacher-removal-') as temporary:
        root=Path(temporary);package=root/'src/flyholdem'
        for name in (*RUNTIME_FILES,'interface/cue_player.py'):
            target=package/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/'src/flyholdem'/name,target)
        build=root/'runs/build';build.mkdir(parents=True)
        for path in (LIBRARY,LIBRARY.with_suffix(LIBRARY.suffix+'.json')):shutil.copy2(path,build/path.name)
        shutil.copytree(model,root/'model')
        shutil.copytree(ROOT/'src/flyholdem/teacher',package/'teacher',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        artifacts=root/'teaching-artifacts';artifacts.mkdir()
        shutil.copy2(run/'teacher.json',artifacts/'teacher.json');shutil.copy2(run/'cues.json',artifacts/'cue-corpus.json')
        (root/'probes.json').write_bytes(canonical([{'cue':row['cue'],'stimulus_seed':row['stimulus_seed']} for row in probes]))
        env=dict(os.environ,PYTHONPATH=str(root/'src'),PYTHONNOUSERSITE='1')
        def execute():
            return subprocess.check_output([sys.executable,'-c',PROBE,str(graph)],cwd=root,env=env,timeout=120)
        before=execute()
        actual=json.loads(before)
        keys=('cue','selected','rates_hz','scores','readout_spikes','stimulus_seed')
        if [{key:row[key] for key in keys} for row in actual]!=[{key:row[key] for key in keys} for row in probes]:
            raise ValueError('Exported native model does not reproduce actual recorded post-training decisions')
        shutil.rmtree(package/'teacher');shutil.rmtree(artifacts)
        after=execute()
        if before!=after:raise ValueError('Deleting teacher/corpus changed actual frozen decision bytes')
        if (package/'teacher').exists() or artifacts.exists():raise AssertionError('Teaching files were not removed')
    return {'schema':'actual-cue-teacher-removal-v1','model_sha256':digest(model/'manifest.json'),
        'transfer_result_sha256':digest(run/'result.json'),'decisions':count,'selected_checkpoint_seed':seed,
        'decisions_sha256':__import__('hashlib').sha256(before).hexdigest(),'decision_schedule_sha256':identity(probes),
        'matches_recorded_post_training_decisions':True,'teacher_and_corpus_deleted':True,
        'no_teacher_learning_or_experiment_imports':True,'decision_bytes_unchanged':True,'whole_gate2a_passed':False}
