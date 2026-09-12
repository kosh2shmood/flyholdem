"""Create and execute a source-pinned native evaluation runtime without a teacher."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import yaml
from flyholdem.connectome.registry import ROOT,digest
from flyholdem.inference_identity import RUNTIME_FILES
from flyholdem.neural.kernel.build import LIBRARY
from flyholdem.neural.checkpoint import atomic_json
from flyholdem.provenance import identity

EXTRA_FILES=('poker/__init__.py','poker/engine.py','poker/actions.py','poker/observation.py','poker/opponents.py','poker/infoset.py',
    'learning/__init__.py','learning/curriculum.py','experiments/__init__.py','experiments/journal.py',
    'experiments/evaluate.py','experiments/disconnected_worker.py')


def evaluate_disconnected(config,output,model,graph=None,resume=False,stop_after=None):
    if model is None:raise ValueError('Disconnected evaluation requires an exported numeric frozen model')
    root=Path(output).resolve();runtime=root/'runtime';model=Path(model).resolve()
    graph=Path(graph or ROOT/'connectome_data/malecns_v1'/('prepared-'+config['mode'])).resolve()
    names=tuple(dict.fromkeys((*RUNTIME_FILES,*EXTRA_FILES)))
    sources={name:digest(ROOT/'src/flyholdem'/name) for name in names}
    git_directory=subprocess.check_output(['git','rev-parse','--absolute-git-dir'],cwd=ROOT,text=True).strip()
    contract={'schema':'isolated-frozen-evaluation-runtime-v1','config':config,'config_hash':identity(config),
        'model_sha256':digest(model/'manifest.json'),'graph_manifest_sha256':digest(graph/'manifest.json'),
        'project':str(ROOT.resolve()),'graph':str(graph),'git_directory':git_directory,'source_files':sources,
        'source_files_hash':identity(sources),'lock_sha256':digest(ROOT/'uv.lock'),'binary_sha256':digest(LIBRARY)}
    if resume:
        if list((runtime/'src').rglob('*.pyc')):raise ValueError('Isolated runtime must compile its verified source without cached bytecode')
        if json.loads((root/'isolation.json').read_text())!=contract:raise ValueError('Isolated evaluation source/config/model/graph changed')
        for name,checksum in sources.items():
            if digest(runtime/'src/flyholdem'/name)!=checksum:raise ValueError('Isolated runtime copy changed')
        if (digest(runtime/'uv.lock')!=contract['lock_sha256'] or digest(runtime/'model/manifest.json')!=contract['model_sha256']
                or digest(runtime/'runs/build'/LIBRARY.name)!=contract['binary_sha256']
                or yaml.safe_load((runtime/'config.yaml').read_text())!=config):
            raise ValueError('Isolated runtime artifact changed')
    else:
        root.mkdir(parents=True,exist_ok=False)
        for name in names:
            target=runtime/'src/flyholdem'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/'src/flyholdem'/name,target)
        shutil.copy2(ROOT/'uv.lock',runtime/'uv.lock');shutil.copytree(model,runtime/'model')
        build=runtime/'runs/build';build.mkdir(parents=True)
        for path in (LIBRARY,LIBRARY.with_suffix(LIBRARY.suffix+'.json')):shutil.copy2(path,build/path.name)
        data=runtime/'connectome_data/malecns_v1';data.mkdir(parents=True)
        (data/('prepared-'+config['mode'])).symlink_to(graph,target_is_directory=True)
        (runtime/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=True));atomic_json(root/'isolation.json',contract)
    env=dict(os.environ,PYTHONPATH=str(runtime/'src'),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',GIT_DIR=git_directory,GIT_WORK_TREE=str(ROOT.resolve()))
    command=[sys.executable,'-m','flyholdem.experiments.disconnected_worker','--root',str(root)]
    if resume:command.append('--resume')
    if stop_after is not None:command.extend(['--stop-after',str(stop_after)])
    process=subprocess.Popen(command,cwd=runtime,env=env)
    def forward(signum,frame):
        if process.poll() is None:process.send_signal(signum)
    handlers={sig:signal.signal(sig,forward) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        code=process.wait()
        if code:raise RuntimeError(f'Isolated native evaluation exited with code {code}; completed checkpoints remain resumable')
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)
    result=json.loads((root/'evaluation/result.json').read_text());isolation=json.loads((root/'isolation-result.json').read_text())
    if not isolation['teacher_import_denied'] or not isolation['external_training_files_denied'] or isolation['teacher_modules_loaded']:
        raise ValueError('Isolated evaluation did not enforce its teacher boundary')
    return {'run':str(root/'evaluation'),'isolation_sha256':digest(root/'isolation.json'),'isolation':isolation,'result':result}
