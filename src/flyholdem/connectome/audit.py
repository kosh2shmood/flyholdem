"""Independent prepared-graph accounting and resource-aware neural smoke run."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
import yaml
from .prepare import load_graph
from .registry import ROOT, digest, registry, verify
from flyholdem.provenance import manifest as run_manifest


def audit(path):
    graph,manifest=load_graph(path)
    ptr,post,contacts,weights=graph['ptr'],graph['post'],graph['contacts'],graph['weight'];n=len(graph['ids']);edges=len(post)
    if ptr.dtype!=np.int64 or post.dtype!=np.int32 or weights.dtype!=np.float32 or graph['ids'].dtype!=np.uint64:raise ValueError('Unexpected prepared dtype')
    if len(ptr)!=n+1 or ptr[0]!=0 or ptr[-1]!=edges or np.any(np.diff(ptr)<0):raise ValueError('Invalid CSR pointer accounting')
    if len(contacts)!=edges or len(weights)!=edges or np.any(graph['ids'][1:]<=graph['ids'][:-1]):raise ValueError('Invalid IDs/edge arrays')
    stats={'nodes':n,'edges':edges,'synaptic_contacts':0,'self_edges':0,'weight_one_edges':0,'positive_edges':0,'negative_edges':0}
    incoming=np.zeros(n,dtype=np.int64)
    for start in range(0,edges,500_000):
        end=min(edges,start+500_000);pre=np.searchsorted(ptr[1:],np.arange(start,end),side='right')
        j=post[start:end];c=contacts[start:end];w=weights[start:end]
        if np.any(j<0) or np.any(j>=n) or np.any(c<1):raise ValueError('Invalid retained endpoint/contact')
        expected=c.astype(np.float32)*graph['signs'][pre]*np.float32(manifest['config']['weight_gain'])
        if not np.array_equal(w,expected):raise ValueError('Contact/sign/gain mismatch')
        stats['synaptic_contacts']+=int(c.sum(dtype=np.uint64));stats['self_edges']+=int(np.count_nonzero(pre==j));stats['weight_one_edges']+=int(np.count_nonzero(c==1))
        stats['positive_edges']+=int(np.count_nonzero(w>0));stats['negative_edges']+=int(np.count_nonzero(w<0));incoming+=np.bincount(j,minlength=n)
    if any(stats[k]!=manifest['metrics'][k] for k in ('nodes','edges','synaptic_contacts')):raise ValueError('Manifest accounting mismatch')
    stats['isolated_nodes']=int(np.count_nonzero((incoming==0)&(np.diff(ptr)==0)))
    stats['array_bytes']=sum(a.nbytes for a in graph.values())
    return {'status':'pass','mode':manifest['mode'],'graph_hash':manifest['graph_hash'],'counts':stats,'source_hashes':manifest['sources']}


def benchmark(path,config,output):
    import pyarrow.feather as feather
    from flyholdem.neural.sparse import SparseBrain
    graph,prepared=load_graph(path);nodes=feather.read_table(Path(path)/'neurons.feather',columns=['class']).to_pandas()
    cfg=config['benchmark'];pool=np.flatnonzero(nodes['class'].eq(cfg['input_class']).to_numpy())
    if len(pool)<cfg['input_cells']:raise ValueError('Not enough declared benchmark inputs')
    rng=np.random.default_rng(cfg['seed']);inputs=np.sort(rng.choice(pool,cfg['input_cells'],replace=False))
    started=time.perf_counter();brain=SparseBrain(graph['ptr'],graph['post'],graph['weight']);load_seconds=time.perf_counter()-started
    drive=np.zeros(brain.n,dtype=np.float32)
    start=time.perf_counter();baseline=brain.advance(drive,cfg['baseline_ms']);baseline_seconds=time.perf_counter()-start
    before=brain.state();drive[inputs]=cfg['stimulus']
    start=time.perf_counter();first=brain.advance(drive,cfg['stimulus_ms']);stimulus_seconds=time.perf_counter()-start
    after=brain.state();brain.restore_state(before);second=brain.advance(drive,cfg['stimulus_ms'])
    if not np.array_equal(first,second) or any(not np.array_equal(after[k],v) for k,v in brain.state().items()):raise ValueError('Full-state deterministic continuation failed')
    result={'schema':'neural-resource-smoke-v1','status':'pass','mode':prepared['mode'],'learning_claim':False,
        'manifest':run_manifest(dict(config,mode=prepared['mode']),brain.graph_hash,'benchmark-direct-KC-v1','none',brain.build['binary_sha256']),
        'graph':prepared['metrics'],'inputs':[str(graph['ids'][i]) for i in inputs],
        'baseline_spikes':int(baseline.sum()),'stimulus_spikes':int(first.sum()),'stimulated_cell_spikes':int(first[inputs].sum()),
        'active_cells':int(np.count_nonzero(first)),'load_seconds':load_seconds,'baseline_seconds':baseline_seconds,
        'stimulus_seconds':stimulus_seconds,'wall_seconds_per_simulated_second':stimulus_seconds/(cfg['stimulus_ms']/1000),
        'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
        'mutable_state_bytes':sum(x.nbytes for x in after.values()),'exact_state_continuation':True}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/runtime.yaml');p.add_argument('--mode',choices=['full','circuit'],default='full');p.add_argument('--benchmark',action='store_true');args=p.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text());root=ROOT/'connectome_data/malecns_v1'
    for name,item in registry().items():verify(root/name,item)
    path=root/('prepared-'+args.mode);result=audit(path)
    output=ROOT/'runs/data-audit';output.mkdir(parents=True,exist_ok=True);(output/(args.mode+'.json')).write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    if args.benchmark:
        result=benchmark(path,cfg,output/(args.mode+'-benchmark.json'))
        print(json.dumps({k:v for k,v in result.items() if k not in ('manifest','inputs')},indent=2),flush=True)
