"""Deterministic teacher-only equity from own cards and the public board.

Sampled cards are hypothetical, drawn from the unknown-card set. This module
accepts no game object, actual opponent cards or actual future deck. PokerKit
remains the canonical dealing/betting/showdown engine.
"""
import argparse
import ctypes as C
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import uuid
import numpy as np
from flyholdem.connectome.registry import ROOT, digest
from flyholdem.neural.checkpoint import atomic_json
from flyholdem.provenance import canonical

SOURCE = Path(__file__).with_suffix('.cpp')
LIBRARY = ROOT/'runs/build'/('teacher_equity.dylib' if sys.platform=='darwin' else 'teacher_equity.so')
METADATA = LIBRARY.with_suffix(LIBRARY.suffix+'.json')
CARDS = tuple(r+s for r in '23456789TJQKA' for s in 'cdhs')
INDEX = {card:i for i,card in enumerate(CARDS)}
FLAGS = ['-O3','-std=c++17'] + (['-dynamiclib'] if sys.platform=='darwin' else ['-shared','-fPIC'])


def expected_build():
    compiler=subprocess.check_output(['c++','--version'],text=True).splitlines()[0]
    return {'schema':'teacher-visible-equity-native-v1','source_sha256':digest(SOURCE),
            'compiler':compiler,'flags':FLAGS,'machine':platform.machine(),'platform':sys.platform}


def build(force=False):
    expected=expected_build()
    if LIBRARY.exists() and METADATA.exists() and not force:
        record=json.loads(METADATA.read_text())
        if any(record.get(k)!=v for k,v in expected.items()) or record.get('binary_sha256')!=digest(LIBRARY):
            raise ValueError('Teacher equity binary/source mismatch; rebuild only outside active runs')
        return record
    LIBRARY.parent.mkdir(parents=True,exist_ok=True)
    temporary=LIBRARY.with_name('teacher-equity-'+uuid.uuid4().hex+LIBRARY.suffix)
    try:
        subprocess.run(['c++',*FLAGS,str(SOURCE),'-o',str(temporary)],check=True,capture_output=True)
        temporary.replace(LIBRARY)
        record={**expected,'binary_sha256':digest(LIBRARY)}
        atomic_json(METADATA,record)
        return record
    finally:
        temporary.unlink(missing_ok=True)


@lru_cache(maxsize=1)
def backend():
    record=build();library=C.CDLL(str(LIBRARY))
    library.rank_hands.argtypes=[C.c_int,C.c_int,C.c_void_p,C.c_void_p];library.rank_hands.restype=None
    library.visible_equity.argtypes=[C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_uint64]
    library.visible_equity.restype=C.c_double
    return library,record


def backend_identity():
    return dict(backend()[1])


def rank_batch(cards):
    raw=np.asarray(cards)
    if raw.dtype.kind not in 'iu' or np.any(raw<0) or np.any(raw>=52):
        raise ValueError('Standard integer card indices required')
    values=np.ascontiguousarray(raw,dtype=np.int32)
    if values.ndim!=2 or values.shape[1] not in (5,6,7) or values.shape[0]>1000000:
        raise ValueError('Batch of five to seven distinct standard cards required')
    ranks=np.zeros(len(values),dtype=np.uint32)
    backend()[0].rank_hands(len(values),values.shape[1],values.ctypes.data,ranks.ctypes.data)
    if np.any(ranks==0): raise ValueError('Invalid or duplicate card in native rank batch')
    return ranks


@lru_cache(maxsize=50000)
def visible_equity(hole,board,samples=256):
    if (len(hole)!=2 or len(board) not in (0,3,4,5) or any(c not in INDEX for c in hole+board)
            or len(set(hole+board))!=len(hole)+len(board) or type(samples) is not int or not 1<=samples<=1000000):
        raise ValueError('Own hole cards, public board and a bounded sample count required')
    h=np.array([INDEX[c] for c in hole],dtype=np.int32);b=np.array([INDEX[c] for c in board],dtype=np.int32)
    # A pure visible-card function, independent of hand counters and actual deck.
    seed=int.from_bytes(hashlib.sha256(b'visible-equity-v1'+canonical([hole,board])).digest()[:8],'little')
    value=float(backend()[0].visible_equity(h.ctypes.data,b.ctypes.data,len(b),samples,seed))
    if not np.isfinite(value) or not 0<=value<=1: raise FloatingPointError('Invalid teacher equity estimate')
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--rebuild',action='store_true')
    args=parser.parse_args();print(json.dumps(build(force=args.rebuild),indent=2))
