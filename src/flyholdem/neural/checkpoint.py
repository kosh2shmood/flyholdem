"""Atomic, hash-verified generations: latest three plus best validation checkpoint.

A generation becomes visible only after all arrays and metadata are fsynced.
Pointers are atomically replaced. No pickle, executable objects or silent
identity migration. Callers must include every mutable controller/game state.
"""
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import uuid
import numpy as np
from flyholdem.provenance import canonical, assert_compatible
from flyholdem.connectome.registry import digest

GENERATION=re.compile(r'^step[0-9]{12}-[0-9a-f]{8}$')


def atomic_json(path,value):
    path=Path(path)
    temporary=path.with_name(path.name+'.partial')
    with temporary.open('wb') as stream:
        stream.write(canonical(value));stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


def sync_dir(path):
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


class Checkpoints:
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)

    def save(self,arrays,identity,extra,step,best=False):
        if type(step) is not int or not 0<=step<10**12:raise ValueError('Invalid checkpoint step')
        name=f'step{step:012d}-{uuid.uuid4().hex[:8]}'
        temporary=Path(tempfile.mkdtemp(prefix='.writing-',dir=self.root))
        try:
            entries={}
            for key,value in sorted(arrays.items()):
                if not key.isidentifier():raise ValueError('Invalid checkpoint array name')
                value=np.ascontiguousarray(value)
                if value.dtype.kind not in 'biuf' or not np.isfinite(value).all():raise ValueError('Checkpoint arrays must be finite numeric arrays')
                path=temporary/f'{key}.npy'
                with path.open('wb') as stream:
                    np.save(stream,value,allow_pickle=False);stream.flush();os.fsync(stream.fileno())
                entries[key]={'shape':list(value.shape),'dtype':value.dtype.str,'sha256':digest(path)}
            meta={'schema':'atomic-checkpoint-v1','identity':identity,'step':step,'arrays':entries,'extra':extra}
            atomic_json(temporary/'manifest.json',meta);sync_dir(temporary)
            temporary.rename(self.root/name);sync_dir(self.root)
            pointer={'generation':name,'manifest_sha256':digest(self.root/name/'manifest.json')}
            atomic_json(self.root/'latest.json',pointer)
            if best:atomic_json(self.root/'best.json',pointer)
            sync_dir(self.root)
            self._prune()
            return name
        finally:
            if temporary.exists():shutil.rmtree(temporary)

    def _prune(self):
        generations=sorted((p for p in self.root.iterdir() if p.is_dir() and GENERATION.fullmatch(p.name)),
                           key=lambda p:p.stat().st_mtime_ns,reverse=True)
        keep={p.name for p in generations[:3]}
        for pointer in ('latest.json','best.json'):
            path=self.root/pointer
            if path.exists():keep.add(json.loads(path.read_text())['generation'])
        for path in generations:
            if path.name not in keep:shutil.rmtree(path)

    def load(self,current_identity,which='latest'):
        if which not in ('latest','best'):raise ValueError('Select latest or best checkpoint')
        pointer=json.loads((self.root/f'{which}.json').read_text())
        name=pointer['generation']
        if not GENERATION.fullmatch(name):raise ValueError('Invalid generation pointer')
        root=self.root/name
        if digest(root/'manifest.json')!=pointer['manifest_sha256']:raise ValueError('Checkpoint manifest checksum mismatch')
        meta=json.loads((root/'manifest.json').read_text())
        if meta['schema']!='atomic-checkpoint-v1':raise ValueError('Unsupported checkpoint schema')
        assert_compatible(meta['identity'],current_identity)
        arrays={}
        for key,entry in meta['arrays'].items():
            if not key.isidentifier():raise ValueError('Invalid array key')
            path=root/f'{key}.npy'
            if digest(path)!=entry['sha256']:raise ValueError(f'Checkpoint array checksum mismatch: {key}')
            array=np.load(path,allow_pickle=False)
            if list(array.shape)!=entry['shape'] or array.dtype.str!=entry['dtype'] or array.dtype.kind not in 'biuf' or not np.isfinite(array).all():raise ValueError('Invalid checkpoint array metadata')
            arrays[key]=array
        return arrays,meta['extra'],meta
