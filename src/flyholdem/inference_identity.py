"""Explicit frozen-policy runtime identity, independent of training artifacts.

Historical training checkpoints keep their existing whole-source/environment
contract. This separate identity covers every imported frozen-policy module and
runtime dependency, while retaining full training provenance as inert metadata.
"""
import platform
import sys
from importlib.metadata import version
from flyholdem.connectome.registry import ROOT, digest
from flyholdem.provenance import identity

RUNTIME_FILES = (
    '__init__.py', 'provenance.py', 'inference_identity.py',
    'connectome/__init__.py', 'connectome/registry.py', 'connectome/prepare.py',
    'neural/__init__.py', 'neural/sparse.py', 'neural/checkpoint.py',
    'neural/kernel/__init__.py', 'neural/kernel/build.py', 'neural/kernel/doomfly_lif.cpp',
    'interface/__init__.py', 'interface/encoder.py', 'interface/population.py', 'interface/frozen.py',
)


def frozen_runtime_identity():
    source = {name: digest(ROOT / 'src/flyholdem' / name) for name in RUNTIME_FILES}
    environment = {'python': sys.version, 'platform': platform.platform(),
                   'packages': {name: version(name) for name in ('numpy', 'PyYAML')}}
    return {'schema': 'frozen-inference-runtime-v1', 'source_files': source,
            'source_hash': identity(source), 'environment': environment}


def assert_frozen_runtime(saved):
    if saved != frozen_runtime_identity():
        raise ValueError('Frozen policy runtime/source mismatch')
