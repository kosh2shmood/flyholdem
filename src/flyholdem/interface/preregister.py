"""Annotation/connectivity-only candidates and registered neural interventions."""
import argparse
import itertools
import json
import os
import signal
from pathlib import Path
import time
import numpy as np
import yaml
from .encoder import CHANNELS
from .population import balanced_projection,neural_scores,select_action
from flyholdem.connectome.registry import ROOT,digest
from flyholdem.connectome.prepare import load_graph
from flyholdem.provenance import identity,manifest,assert_compatible
from flyholdem.neural.checkpoint import Checkpoints


def connectivity_ensembles(matrix,readouts,size,minimum_inputs,seed):
    """Farthest cosine anchors and their nearest unused readouts; no game data."""
    matrix=np.asarray(matrix,dtype=float);readouts=np.asarray(readouts,dtype=np.int32)
    eligible=np.flatnonzero(np.count_nonzero(matrix,axis=0)>=minimum_inputs)
    if len(eligible)<5*size:raise ValueError('Insufficient reachable annotated readouts')
    vectors=matrix[:,eligible];norm=np.linalg.norm(vectors,axis=0);unit=vectors/norm
    cosine=unit.T@unit
    priority=np.random.default_rng(seed).permutation(len(eligible));rank=np.argsort(priority)
    anchors=[int(np.lexsort((rank,-vectors.sum(axis=0)))[0])]
    while len(anchors)<5:
        distance=np.max(cosine[:,anchors],axis=1);distance[anchors]=np.inf
        anchors.append(int(np.lexsort((rank,distance))[0]))
    used=set(anchors);groups=[]
    for anchor in anchors:
        members=[anchor]
        for candidate in np.lexsort((rank,-cosine[anchor])):
            if len(members)>=size:break
            if int(candidate) not in used:members.append(int(candidate));used.add(int(candidate))
            if len(members)==size:break
        groups.append(eligible[members])
    return [readouts[g] for g in groups],groups,{'anchors':readouts[eligible[anchors]].tolist(),
        'anchor_cosines':cosine[np.ix_(anchors,anchors)].tolist(),'eligible_readouts':len(eligible)}


def control_patterns(matrix,groups,count):
    profiles=np.array([matrix[:,g].mean(axis=1) for g in groups]).T
    profiles/=np.maximum(profiles.sum(axis=0),1)
    result=[]
    for action in range(5):
        contrast=profiles[:,action]-np.max(np.delete(profiles,action,axis=1),axis=1)
        order=np.lexsort((np.arange(len(matrix)),-contrast))
        order=order[profiles[order,action]>0]
        if len(order)<count:raise ValueError('Insufficient connected control inputs')
        result.append(order[:count])
    return result


def candidates(graph_path,config):
    import pyarrow.feather as feather
    graph,prepared=load_graph(graph_path);nodes=feather.read_table(Path(graph_path)/'neurons.feather').to_pandas()
    inputs=np.flatnonzero(nodes['class'].eq(config['input_class']).to_numpy()).astype(np.int32)
    readouts=np.flatnonzero(nodes['class'].eq(config['readout_class']).to_numpy()).astype(np.int32)
    lookup=np.full(len(nodes),-1,dtype=np.int32);lookup[readouts]=np.arange(len(readouts))
    matrix=np.zeros((len(inputs),len(readouts)),dtype=np.float64)
    for row,i in enumerate(inputs):
        start,end=graph['ptr'][i:i+2];post=graph['post'][start:end];local=lookup[post];keep=local>=0
        np.add.at(matrix[row],local[keep],graph['contacts'][start:end][keep])
    ensembles,groups,criteria=connectivity_ensembles(matrix,readouts,config['ensemble_size'],config['minimum_presynaptic_inputs'],config['mapping_seed'])
    projection=balanced_projection(inputs,config['fanout'],config['mapping_seed'])
    usage=np.bincount(projection.ravel(),minlength=len(nodes))[inputs]
    def cell(i):
        row=nodes.iloc[i]
        def val(key):
            value=row.get(key);return None if value is None or isinstance(value,float) and np.isnan(value) else str(value)
        return {'index':int(i),'id':str(graph['ids'][i]),'class':val('class'),'type':val('type'),'side':val('somaSide') or val('rootSide')}
    registered=[]
    for action,(ensemble,group) in enumerate(zip(ensembles,groups)):
        paths=[]
        for j in group:
            row=int(np.argmax(matrix[:,j]));paths.append({'pre_id':str(graph['ids'][inputs[row]]),'post_id':str(graph['ids'][readouts[j]]),'contacts':int(matrix[row,j])})
        registered.append({'action':action,'indices':ensemble.tolist(),'cells':[cell(i) for i in ensemble],'direct_input_paths':paths})
    artifact={'schema':'neural-preregistration-v1','stage':'candidate','mode':prepared['mode'],'graph_hash':prepared['graph_hash'],
        'config':config,'config_hash':identity(config),'selection':criteria,'input_ids':[str(graph['ids'][i]) for i in inputs],
        'input_indices':inputs.tolist(),'channels':CHANNELS,'projection_indices':projection.tolist(),
        'input_fanin_range':[int(usage.min()),int(usage.max())],'ensembles':registered,
        'no_poker_information_used':True,'selected_gain':None,'baseline_hz':[0]*5}
    return graph,artifact,matrix,groups


def run_gate(graph_path,config,output,resume=False):
    from flyholdem.neural.sparse import SparseBrain
    graph,artifact,matrix,groups=candidates(graph_path,config);output=Path(output)
    output.mkdir(parents=True,exist_ok=resume)
    brain=SparseBrain(graph['ptr'],graph['post'],graph['weight']);inputs=np.array(artifact['input_indices']);ensembles=[np.array(e['indices']) for e in artifact['ensembles']]
    runtime=manifest(config,brain.graph_hash,identity(artifact['projection_indices']),identity(artifact['ensembles']),brain.build['binary_sha256'])
    if resume:
        assert_compatible(json.loads((output/'manifest.json').read_text()),runtime)
        if json.loads((output/'candidate.json').read_text())!=artifact:raise ValueError('Candidate mismatch on resume')
    else:
        (output/'manifest.json').write_text(json.dumps(runtime,indent=2,sort_keys=True)+'\n');(output/'candidate.json').write_text(json.dumps(artifact,indent=2,sort_keys=True)+'\n')
    initial=brain.state();blank=np.zeros(brain.n,dtype=np.float32);timing=config['decision_ms'];started=time.perf_counter()
    records={};trial_path=output/'trials.jsonl'
    def trial_key(row):return tuple(row[k] for k in ('phase','target','repeat','gain','input_count','seed'))
    if resume:
        for line in trial_path.read_text().splitlines():
            row=json.loads(line);key=trial_key(row)
            if key in records:raise ValueError('Duplicate trial checkpoint')
            records[key]=row
    journal=trial_path.open('a' if resume else 'x');calibrations=[];last_checkpoint=time.monotonic()
    checkpoints=Checkpoints(output/'checkpoints')
    def checkpoint():
        nonlocal last_checkpoint
        checkpoints.save(brain.state(),runtime,{'trial_count':len(records),'next_trial_resets_to_initial':True,
            'rng':'Independent recorded seed per trial; no mutable shared RNG'},len(records))
        last_checkpoint=time.monotonic()
    if resume:checkpoints.load(runtime)  # Verify the previous complete-state boundary.
    else:checkpoint()
    stopped=False
    def stop(signum,frame):
        nonlocal stopped
        stopped=True
    previous_handler=signal.signal(signal.SIGTERM,stop)
    def trial(pattern,gain,seed):
        brain.restore_state(initial);rng=np.random.default_rng(seed);drive=blank.copy()
        drive[pattern]=gain*rng.uniform(1-config['amplitude_jitter'],1+config['amplitude_jitter'],len(pattern))
        baseline=brain.advance(blank,timing['baseline']);brain.advance(drive,timing['stimulus']);counts=brain.advance(drive,timing['readout'])
        rates,scores=neural_scores(counts,ensembles,timing['readout'],artifact['baseline_hz'],config['score_scale_hz'])
        return {'selected':select_action(scores,[True]*5,rng),'rates_hz':rates.tolist(),'baseline_spikes':int(baseline.sum()),'spikes':int(counts.sum())}
    def block(patterns,gain,phase,trials,seed):
        wins=np.zeros(5,dtype=int);confusion=np.zeros((5,5),dtype=int)
        for action,pattern in enumerate(patterns):
            for repeat in range(trials):
                row={'phase':phase,'target':action,'repeat':repeat,'gain':gain,'input_count':len(pattern),'seed':seed+action*1000+repeat}
                key=trial_key(row)
                if key in records:row=records[key]
                else:
                    row.update(trial(pattern,gain,seed+action*1000+repeat));records[key]=row
                    journal.write(json.dumps(row,sort_keys=True)+'\n');journal.flush();os.fsync(journal.fileno())
                if time.monotonic()-last_checkpoint>=300:checkpoint()
                if stopped:raise KeyboardInterrupt('Stopped at complete neural-trial boundary; resume with --resume')
                wins[action]+=row['selected']==action;confusion[action,row['selected']]+=1
        result={'phase':phase,'gain':gain,'control_cells':len(patterns[0]),'success':(wins/trials).tolist(),'confusion':confusion.tolist()}
        print(json.dumps(result),flush=True);return result
    try:
        selected=None;selected_patterns=None
        for gain,count in itertools.product(config['calibration']['gains'],config['calibration']['control_cell_counts']):
            patterns=[inputs[p] for p in control_patterns(matrix,groups,count)]
            result=block(patterns,gain,'development',config['calibration']['trials_per_action'],config['calibration']['seed']);calibrations.append(result)
            if min(result['success'])>=config['confirmatory']['minimum_success']:
                selected=result;selected_patterns=patterns;break
        if selected is None:
            result={'schema':'controllability-result-v1','status':'fail','mode':artifact['mode'],'gate':1,'reason':'No registered calibration candidate reached the criterion','calibration':calibrations,'elapsed_seconds':time.perf_counter()-started}
        else:
            artifact['stage']='frozen';artifact['selected_gain']=selected['gain'];artifact['control_cell_count']=selected['control_cells'];artifact['control_indices']=[p.tolist() for p in selected_patterns]
            (output/'preregistration.json').write_text(json.dumps(artifact,sort_keys=True,indent=2)+'\n')
            confirmation=block(selected_patterns,selected['gain'],'confirmatory',config['confirmatory']['trials_per_action'],config['confirmatory']['seed'])
            shuffled=np.random.default_rng(config['mapping_seed']+1).permutation(inputs);lookup=dict(zip(inputs,shuffled))
            shuffled_patterns=[np.array([lookup[i] for i in p]) for p in selected_patterns]
            shuffle=block(shuffled_patterns,selected['gain'],'shuffled-input',config['confirmatory']['trials_per_action'],config['confirmatory']['seed'])
            black=trial([],selected['gain'],config['confirmatory']['seed'])
            result={'schema':'controllability-result-v1','status':'pass' if min(confirmation['success'])>=config['confirmatory']['minimum_success'] else 'fail',
                'mode':artifact['mode'],'gate':1,'preregistration_sha256':digest(output/'preregistration.json'),
                'calibration':calibrations,'confirmatory':confirmation,'shuffled_input':shuffle,'black_input':black,
                'elapsed_seconds':time.perf_counter()-started,'learning_claim':False}
    finally:
        journal.close();checkpoint();signal.signal(signal.SIGTERM,previous_handler)
    (output/'result.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',default='configs/controllability.yaml');p.add_argument('--mode',choices=['circuit','full'],default='circuit');p.add_argument('--output',required=True);p.add_argument('--resume',action='store_true');args=p.parse_args()
    result=run_gate(ROOT/'connectome_data/malecns_v1'/('prepared-'+args.mode),yaml.safe_load(Path(args.config).read_text()),args.output,args.resume)
    print(json.dumps(result,indent=2))
