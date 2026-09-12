"""Validated native all-edge LIF wrapper; no game or teacher dependency."""
import ctypes as C
import hashlib
import math
import numpy as np
from .kernel.build import LIBRARY, verify_build


class SparseBrain:
    state_names=('weights','v','g','refractory','drive','previous_drive','queue','queue_count','clock',
                 'counts','active','flags','nactive','last')

    def __init__(self,ptr,post,weights):
        self.ptr=np.ascontiguousarray(ptr,dtype=np.int64)
        self.post=np.ascontiguousarray(post,dtype=np.int32)
        self.initial=np.ascontiguousarray(weights,dtype=np.float32)
        self.n=len(self.ptr)-1
        if self.n<1 or self.ptr[0]!=0 or self.ptr[-1]!=len(self.post) or np.any(np.diff(self.ptr)<0):raise ValueError('Invalid CSR pointers')
        if self.initial.shape!=self.post.shape or np.any(self.post<0) or np.any(self.post>=self.n) or not np.all(np.isfinite(self.initial)):raise ValueError('Invalid CSR edges')
        self.build=verify_build()
        self._library=C.CDLL(str(LIBRARY))
        self._advance=self._library.neural_advance
        self._advance.argtypes=[C.c_int]+[C.c_void_p]*11+[C.c_int,C.c_float]+[C.c_void_p]*5
        self._advance.restype=None
        self.weights=self.initial.copy()
        self.v=np.full(self.n,-52,dtype=np.float32);self.g=np.zeros(self.n,dtype=np.float32)
        self.refractory=np.zeros(self.n,dtype=np.int16)
        self.drive=np.zeros(self.n,dtype=np.float32);self.previous_drive=self.drive.copy()
        self.queue=np.zeros((19,self.n),dtype=np.int32);self.queue_count=np.zeros(19,dtype=np.int32)
        self.clock=np.zeros(1,dtype=np.int64);self.counts=np.zeros(self.n,dtype=np.int32)
        self.active=np.zeros(self.n,dtype=np.int32);self.flags=np.zeros(self.n,dtype=np.uint8)
        self.nactive=np.zeros(1,dtype=np.int32);self.last=np.full(self.n,-1,dtype=np.int64)

    @property
    def graph_hash(self):
        h=hashlib.sha256()
        for x in (self.ptr,self.post,self.initial):h.update(memoryview(x))
        return h.hexdigest()

    @property
    def time_ms(self):return int(self.clock[0])/10

    def advance(self,drive,duration_ms):
        ticks=round(duration_ms*10)
        if ticks<1 or ticks>2**31-1 or not math.isclose(ticks/10,duration_ms,abs_tol=1e-9):raise ValueError('Duration must be positive .1ms ticks')
        drive=np.asarray(drive,dtype=np.float32)
        if drive.shape!=(self.n,) or not np.isfinite(drive).all():raise ValueError('Finite per-node stimulus required')
        self.drive[:]=drive;self.counts.fill(0)
        args=(self.ptr,self.post,self.weights,self.v,self.g,self.refractory,self.drive,self.previous_drive,self.queue,self.queue_count,self.clock)
        tail=(self.counts,self.active,self.flags,self.nactive,self.last)
        self._advance(self.n,*[x.ctypes.data for x in args],ticks,.1,*[x.ctypes.data for x in tail])
        if not np.isfinite(self.v).all() or not np.isfinite(self.g).all():raise FloatingPointError('Nonfinite neural state')
        return self.counts.copy()

    def reset_dynamics(self):
        """Fresh neural trial, preserving every current synaptic weight."""
        self.v.fill(-52)
        self.last.fill(-1)
        for name in self.state_names:
            if name not in ('weights', 'v', 'last'):
                getattr(self, name).fill(0)

    def state(self):return {name:getattr(self,name).copy() for name in self.state_names}

    def restore_state(self,state):
        if set(state)!=set(self.state_names):raise ValueError('Incomplete neural state')
        for name in self.state_names:
            old=getattr(self,name);new=np.asarray(state[name])
            if old.shape!=new.shape or old.dtype!=new.dtype or not new.flags.c_contiguous:raise ValueError(f'State shape/dtype mismatch: {name}')
            if not np.isfinite(new).all():raise ValueError(f'Nonfinite state: {name}')
        if np.any(state['queue_count']<0) or np.any(state['queue_count']>self.n):raise ValueError('Invalid delayed spike counts')
        if not 0<=state['nactive'][0]<=self.n or np.any(state['queue']<0) or np.any(state['queue']>=self.n):raise ValueError('Invalid state node indices')
        if np.any(state['active']<0) or np.any(state['active']>=self.n) or np.any(state['refractory']<0) or np.any(state['refractory']>22):raise ValueError('Invalid active/refractory state')
        if np.any(state['flags']>1) or state['clock'][0]<0 or np.any(state['last']>=state['clock'][0]):raise ValueError('Invalid state schedule')
        frontier=state['active'][:state['nactive'][0]]
        if len(np.unique(frontier))!=len(frontier) or not np.all(state['flags'][frontier]==1) or int(state['flags'].sum())!=len(frontier):raise ValueError('Inconsistent active frontier')
        # Validation completes before mutating the live state.
        for name in self.state_names:getattr(self,name)[:]=state[name]
