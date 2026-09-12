"""Canonical identities shared by experiments and checkpoints."""
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from importlib.metadata import distributions
from flyholdem.connectome.registry import digest, ROOT


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def identity(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def source_identity(root=ROOT):
    root=Path(root)
    paths=sorted([*root.joinpath('src').rglob('*.py'), *root.joinpath('src').rglob('*.cpp'),root/'uv.lock'])
    return identity({str(p.relative_to(root)):digest(p) for p in paths if p.is_file()})


def environment():
    versions={d.metadata['Name']:d.version for d in distributions() if d.metadata.get('Name')}
    tools={}
    for key, command in {'node':['node','--version'],'compiler':['c++','--version']}.items():
        try:
            tools[key]=subprocess.check_output(command,text=True,stderr=subprocess.STDOUT).splitlines()[0]
        except (OSError,subprocess.CalledProcessError):
            tools[key]=None
    return {'python':sys.version,'platform':platform.platform(),'packages':dict(sorted(versions.items())),**tools}


def manifest(config, graph_hash, encoder_hash, decoder_hash, binary_hash='python-fixture', initial_checkpoint_hash=None):
    return {'schema':'flyholdem-run-manifest-v1','config_hash':identity(config),'source_hash':source_identity(),
      'graph_hash':graph_hash,'encoder_hash':encoder_hash,'decoder_hash':decoder_hash,'binary_hash':binary_hash,
      'initial_checkpoint_hash':initial_checkpoint_hash,'environment':environment(),
      'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      'dirty_tree':bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)),
      'config':config}


def assert_compatible(saved, current):
    for key in ('config_hash','source_hash','graph_hash','encoder_hash','decoder_hash','binary_hash'):
        if key not in saved or saved[key]!=current.get(key):
            raise ValueError(f'Checkpoint identity mismatch: {key}')
    if saved.get('environment') != current.get('environment'):
        raise ValueError('Checkpoint runtime environment mismatch')
