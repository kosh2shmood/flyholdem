"""Compile every retained edge into stable CSR using bounded Arrow batches."""
import argparse
import hashlib
import json
from pathlib import Path
import os
import uuid
import numpy as np
import yaml
from .registry import ROOT, digest, registry
from flyholdem.provenance import identity


def transmitter_signs(values,ambiguous_sign=1):
    """Declared sign proxy, informed by DOOMFLY's MIT transmitters.py.

    Acetylcholine positive; GABA/glutamate/histamine negative. No dropped edges.
    Conflicting/missing/modulator-only cells use the explicit sensitivity sign.
    """
    if ambiguous_sign not in (-1,1):raise ValueError('Ambiguous sign must be +1 or -1')
    signs=[];uncertain=[]
    for value in values:
        tokens={t.strip() for t in str(value).lower().split(',')}
        fast=({1} if 'acetylcholine' in tokens else set())|({-1} if tokens & {'gaba','glutamate','histamine'} else set())
        ambiguous=len(fast)!=1
        signs.append(ambiguous_sign if ambiguous else next(iter(fast)));uncertain.append(ambiguous)
    return np.array(signs,dtype=np.int8),np.array(uncertain,dtype=bool)


def compile_csr(nodes,batches,output,gain=.275,ambiguous_sign=1):
    """Two passes; stable per-source order, no all-edge sorting in RAM.

    batches is a repeatable factory of (pre,post,contacts) index arrays.
    """
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if not np.isfinite(gain) or gain<=0:raise ValueError('Positive finite gain required')
    n=len(nodes);degree=np.zeros(n,dtype=np.int64)
    for pre,post,contacts in batches():
        if len(pre)!=len(post) or len(pre)!=len(contacts) or np.any(pre<0) or np.any(pre>=n) or np.any(post<0) or np.any(post>=n):raise ValueError('Invalid indexed edge batch')
        if np.asarray(contacts).dtype.kind not in 'iu' or np.any(contacts<1) or np.any(contacts>2**32-1):raise ValueError('Positive integer contacts required')
        degree+=np.bincount(pre,minlength=n)
    ptr=np.r_[0,np.cumsum(degree)].astype(np.int64);edges=int(ptr[-1])
    np.save(output/'ptr.npy',ptr,allow_pickle=False)
    signs,uncertain=transmitter_signs(nodes.neurotransmitter,ambiguous_sign)
    for name,arr in [('ids',nodes.source_id.to_numpy(dtype=np.uint64)),('signs',signs),('uncertain',uncertain)]:np.save(output/(name+'.npy'),arr,allow_pickle=False)
    arrays={name:np.lib.format.open_memmap(output/(name+'.npy'),mode='w+',dtype=dtype,shape=(edges,)) for name,dtype in [('post',np.int32),('weight',np.float32),('contacts',np.uint32)]}
    cursor=np.zeros(n,dtype=np.int64)
    for pre,post,contacts in batches():
        if not len(pre):continue
        order=np.argsort(pre,kind='stable');s=pre[order]
        unique,starts,counts=np.unique(s,return_index=True,return_counts=True)
        rank=np.arange(len(s))-np.repeat(starts,counts)
        dest=ptr[s]+cursor[s]+rank
        arrays['post'][dest]=post[order];arrays['contacts'][dest]=contacts[order]
        arrays['weight'][dest]=contacts[order].astype(np.float32)*signs[s]*np.float32(gain)
        cursor[unique]+=counts
    if not np.array_equal(cursor,degree):raise ValueError('CSR cursor mismatch')
    for a in arrays.values():a.flush()
    return {'nodes':n,'edges':edges,'ambiguous_sign_nodes':int(uncertain.sum())}


def load_graph(path,verify=True):
    path=Path(path);manifest=json.loads((path/'manifest.json').read_text())
    if verify:
        for name,sha in manifest['files'].items():
            if digest(path/name)!=sha:raise ValueError(f'Prepared graph hash mismatch: {name}')
    result={name:np.load(path/(name+'.npy'),mmap_mode='r',allow_pickle=False) for name in ('ptr','post','weight','contacts','ids','signs','uncertain')}
    h=hashlib.sha256()
    for name in ('ptr','post','weight'):h.update(memoryview(result[name]))
    if h.hexdigest()!=manifest['graph_hash']:raise ValueError('Prepared graph identity mismatch')
    return result,manifest


def prepare(config,mode='full',root=None,output=None):
    import pyarrow as pa
    import pyarrow.feather as feather
    import pyarrow.ipc as ipc
    root=Path(root or ROOT/'connectome_data/malecns_v1');normalized=root/'normalized'
    config=dict(config,mode=mode)
    if mode not in ('full','circuit'):raise ValueError('Unknown graph mode')
    report=json.loads((normalized/'report.json').read_text())
    expected={name:{'bytes':item['bytes'],'sha256':item['sha256']} for name,item in registry().items()}
    if report['sources']!=expected:raise ValueError('Normalized source registry mismatch')
    for name,sha in report['files'].items():
        if digest(normalized/name)!=sha:raise ValueError('Normalized artifact hash mismatch')
    output=Path(output or root/('prepared-'+mode));config_hash=identity(config)
    if output.exists():
        _,manifest=load_graph(output)
        if manifest['config_hash']!=config_hash or manifest['normalized_sha256']!=digest(normalized/'report.json') or manifest['preparer_sha256']!=digest(__file__):raise ValueError('Prepared config/source mismatch; use a new output directory')
        return manifest
    temp=output.with_name(output.name+'.partial-'+uuid.uuid4().hex[:8]);temp.mkdir(parents=True)
    allnodes=feather.read_table(normalized/'neurons.feather').to_pandas()
    selected=np.ones(len(allnodes),dtype=bool) if mode=='full' else (allnodes['class'].isin(config['circuit_classes']) | allnodes['type'].isin(config.get('circuit_types',[]))).to_numpy()
    remap=np.full(len(allnodes),-1,dtype=np.int32);remap[selected]=np.arange(int(selected.sum()),dtype=np.int32)
    nodes=allnodes.loc[selected].reset_index(drop=True).copy();nodes['node_index']=np.arange(len(nodes),dtype=np.uint32)
    if not len(nodes):raise ValueError('Empty graph selection')
    def batches():
        reader=ipc.open_file(pa.memory_map(str(normalized/'edges.arrow'),'r'))
        for k in range(reader.num_record_batches):
            b=reader.get_batch(k);pre,post,count=[b.column(i).to_numpy() for i in range(3)]
            keep=selected[pre]&selected[post]
            yield remap[pre[keep]],remap[post[keep]],count[keep]
    metrics=compile_csr(nodes,batches,temp,config['weight_gain'],config['ambiguous_sign'])
    feather.write_feather(nodes,temp/'neurons.feather')
    h=hashlib.sha256()
    for name in ('ptr','post','weight'):h.update(memoryview(np.load(temp/(name+'.npy'),mmap_mode='r',allow_pickle=False)))
    contacts=np.load(temp/'contacts.npy',mmap_mode='r',allow_pickle=False)
    metrics['synaptic_contacts']=int(contacts.sum(dtype=np.uint64))
    if mode=='full' and any(metrics[k]!=v for k,v in config['reference_counts'].items()):raise ValueError(f'Official retained-count mismatch: {metrics}')
    manifest={'schema':'malecns-csr-v1','mode':mode,'config':config,'config_hash':config_hash,'graph_hash':h.hexdigest(),
        'normalized_sha256':digest(normalized/'report.json'),'preparer_sha256':digest(__file__),'sources':report['sources'],
        'metrics':metrics,'files':{p.name:digest(p) for p in sorted(temp.iterdir()) if p.is_file()},
        'selection':'All retained released edges and neurons' if mode=='full' else 'Annotation-only mushroom-body induced subgraph: '+','.join(config['circuit_classes']+config.get('circuit_types',[]))}
    (temp/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
    for path in temp.iterdir():
        with path.open('rb') as f:os.fsync(f.fileno())
    temp.rename(output);return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='configs/runtime.yaml');parser.add_argument('--mode',choices=['full','circuit'],default='full');args=parser.parse_args()
    result=prepare(yaml.safe_load(Path(args.config).read_text()),args.mode)
    print(json.dumps({k:result[k] for k in ('mode','metrics','graph_hash')},indent=2))
