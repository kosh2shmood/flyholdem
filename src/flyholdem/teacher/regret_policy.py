"""Numeric frozen conventional regret-average policy, separate from every fly."""
import importlib
import json
import platform
from pathlib import Path
import sys
import numpy as np
from flyholdem.connectome.registry import digest
from flyholdem.provenance import identity,canonical
from flyholdem.neural.checkpoint import atomic_json
from .regret import abstraction,VERSION
from .equity import backend_identity

SCHEMA='teacher-external-regret-policy-v1'
AGGREGATION='own-reach-weighted-average-stationary-population-v1'


def implementation():
    names=('flyholdem.teacher.regret','flyholdem.teacher.regret_policy','flyholdem.teacher.poker_tree','flyholdem.teacher.equity',
        'flyholdem.poker.infoset','flyholdem.poker.observation','flyholdem.interface.encoder')
    files={name:digest(importlib.import_module(name).__file__) for name in names}
    files['flyholdem.teacher.equity.cpp']=digest(Path(importlib.import_module('flyholdem.teacher.equity').__file__).with_suffix('.cpp'))
    return {'source_files':files,'source_hash':identity(files),'python':sys.version,'platform':platform.platform(),
        'numpy':np.__version__,'pokerkit':importlib.import_module('importlib.metadata').version('pokerkit'),
        'equity_backend':backend_identity()}


class FrozenRegretPolicy:
    def __init__(self,config,keys,probabilities,legal):
        self.config=dict(config);self.index={key:i for i,key in enumerate(keys)}
        self.values=probabilities;self.legal=legal

    def probabilities(self,observation):
        key,legal=abstraction(observation,self.config);i=self.index.get(key)
        if i is None:return legal.astype(float)/legal.sum()
        if not np.array_equal(legal,self.legal[i]):raise ValueError('Frozen regret information set changed legality')
        return self.values[i].copy()


def export_policy(run,output):
    from .regret_training import completed_table
    table,provenance=completed_table(run);state=table.state();n=len(table.keys)
    averages=state['averages'];legal=state['legal'];total=averages.sum(axis=1,keepdims=True)
    probabilities=np.divide(averages,total,out=legal.astype(float)/legal.sum(axis=1,keepdims=True),where=total>0)
    arrays={'key_offsets':state['key_offsets'],'key_bytes':state['key_bytes'],'probabilities':probabilities,'legal':legal}
    import tempfile
    from flyholdem.neural.checkpoint import sync_dir
    output=Path(output)
    if output.exists():raise FileExistsError('Write a new frozen conventional policy directory')
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=output.name+'-partial-',dir=output.parent) as temporary:
        root=Path(temporary)/'model';root.mkdir();files={}
        for name,value in arrays.items():
            path=root/(name+'.npy')
            with path.open('wb') as stream:np.save(stream,value,allow_pickle=False);stream.flush();__import__('os').fsync(stream.fileno())
            files[path.name]=digest(path)
        runtime=implementation()
        record={'schema':SCHEMA,'mode':'conventional-teacher-control','aggregation':AGGREGATION,'feature_version':VERSION,
            'config':table.config,'information_sets':n,'files':files,'provenance':provenance,'implementation':runtime,
            'inference_backend':identity(runtime['equity_backend']),'allowed_as_teacher':False,
            'unseen_state_prior':'uniform-over-legal-actions; includes untrained stack domains',
            'scope':'Learned tabular response to a fixed opponent population; no Nash equilibrium or fly-learning claim'}
        atomic_json(root/'manifest.json',record);sync_dir(root);root.rename(output);sync_dir(output.parent)
    return digest(output/'manifest.json')

def load_policy(root):
    root=Path(root);record=json.loads((root/'manifest.json').read_text())
    if (record['schema']!=SCHEMA or record['aggregation']!=AGGREGATION or record['feature_version']!=VERSION
            or record['unseen_state_prior']!='uniform-over-legal-actions; includes untrained stack domains'
            or record['implementation']!=implementation()):
        raise ValueError('Frozen regret source/runtime or policy definition mismatch')
    from .regret import RegretTable
    RegretTable(record['config'])
    if record['config']['stack_bb']!=record['provenance']['stack_bb'] or record['config']['feature_version']!=VERSION:
        raise ValueError('Frozen regret abstraction/training stack mismatch')
    names={'key_offsets.npy','key_bytes.npy','probabilities.npy','legal.npy'}
    if set(record['files'])!=names or any(digest(root/name)!=record['files'][name] for name in names):
        raise ValueError('Frozen numeric regret files changed')
    arrays={name:np.load(root/(name+'.npy'),allow_pickle=False) for name in ('key_offsets','key_bytes','probabilities','legal')}
    offsets,data,p,legal=[arrays[k] for k in ('key_offsets','key_bytes','probabilities','legal')];n=record['information_sets']
    if (type(n) is not int or n<1 or offsets.dtype!=np.int64 or offsets.shape!=(n+1,) or offsets[0]!=0
            or np.any(np.diff(offsets)<=0) or data.dtype!=np.uint8 or data.ndim!=1 or offsets[-1]!=len(data)
            or p.dtype!=np.float64 or p.shape!=(n,5) or legal.dtype!=np.bool_ or legal.shape!=(n,5)
            or not np.isfinite(p).all() or np.any(p<0) or np.any(p[~legal]) or not np.allclose(p.sum(axis=1),1,atol=1e-12,rtol=0)):
        raise ValueError('Invalid numeric conventional regret policy')
    keys=[data[a:b].tobytes() for a,b in zip(offsets[:-1],offsets[1:])]
    if len(set(keys))!=n:raise ValueError('Duplicate frozen information key')
    for key,mask in zip(keys,legal):
        decoded=json.loads(key)
        if (canonical(decoded)!=key or decoded[0]!=VERSION or decoded[1]!=4*record['provenance']['stack_bb']
                or decoded[-1]!=mask.tolist()):raise ValueError('Frozen information key binding changed')
    return FrozenRegretPolicy(record['config'],keys,p,legal),record
