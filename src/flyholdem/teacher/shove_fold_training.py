"""Resumable conventional tabular self-play for the small shove/fold curriculum."""
import argparse
import json
from pathlib import Path
import signal
import time
import yaml
from flyholdem.provenance import identity,manifest,assert_compatible
from flyholdem.connectome.registry import digest
from flyholdem.neural.checkpoint import Checkpoints,atomic_json
from flyholdem.experiments.journal import Journal
from .shove_fold import ShoveFoldCFR,sample_payoffs,export_policy,public_nodes,ALGORITHM,CURRICULUM


def train(config,output,resume=False,stop_after=None):
    if (config.get('schema')!='tabular-shove-fold-training-v1' or config.get('algorithm')!=ALGORITHM
            or config.get('curriculum')!=CURRICULUM or config.get('status')!='development'
            or type(config.get('iterations')) is not int or config['iterations']<1
            or type(config.get('deal_seed_start')) is not int or type(config.get('progress_iterations')) is not int
            or config['progress_iterations']<1):raise ValueError('Registered tabular shove/fold development configuration required')
    runtime=manifest(config,'conventional-tabular-no-connectome',identity(public_nodes()),identity(ALGORITHM),'numpy-tabular')
    output=Path(output);output.mkdir(parents=True,exist_ok=resume)
    if resume:assert_compatible(json.loads((output/'manifest.json').read_text()),runtime)
    else:atomic_json(output/'manifest.json',runtime)
    solver=ShoveFoldCFR();journal=Journal(output/'hands.jsonl',resume)
    checkpoints=Checkpoints(output/'checkpoints');completed=0
    if resume:
        arrays,extra,_=checkpoints.load(runtime);completed=extra['completed_iterations'];solver.restore(arrays)
        if (completed>len(journal.rows) or solver.visits[0].sum()!=completed
                or extra['journal_head']!=(journal.rows[completed-1]['hash'] if completed else '0'*64)):
            raise ValueError('Tabular checkpoint/journal mismatch')
    last_checkpoint=time.monotonic();started=time.perf_counter();stopped=False;in_traversal=False
    def checkpoint():
        nonlocal last_checkpoint
        if in_traversal:return
        checkpoints.save(solver.state(),runtime,{'completed_iterations':completed,
            'journal_head':journal.rows[completed-1]['hash'] if completed else '0'*64,
            'private_hand':None,'boundary':'complete three-branch chance traversal',
            'rng':'Independent deterministic PokerKit deal seed from registered start plus iteration'},completed)
        last_checkpoint=time.monotonic()
    def stop(signum,frame):
        nonlocal stopped
        stopped=True
    handlers={sig:signal.signal(sig,stop) for sig in (signal.SIGINT,signal.SIGTERM)}
    if not resume:checkpoint()
    try:
        for index in range(completed,config['iterations']):
            in_traversal=True;sample=sample_payoffs(config['deal_seed_start']+index)
            update=solver.step(sample)
            journal.record(index,['chance-traversal',index],{**sample,**update})
            completed=index+1;in_traversal=False
            if time.monotonic()-last_checkpoint>=300:checkpoint()
            if completed%config['progress_iterations']==0:
                print(json.dumps({'chance_traversals':completed,'minimum_class_visits':int(solver.visits.min())}),flush=True)
            if stopped or stop_after is not None and completed>=stop_after:break
        result={'schema':'tabular-shove-fold-training-result-v1','algorithm':ALGORITHM,
            'mode':'conventional-tabular-control','curriculum':CURRICULUM,'profile':'development',
            'status':'trained-unvalidated' if completed==config['iterations'] else 'interrupted',
            'iterations_completed':completed,'planned_iterations':config['iterations'],
            'hands_completed':completed,'planned_hands':config['iterations'],
            'hand_unit':'one sampled deal traversed through all three terminal branches',
            'minimum_class_visits':int(solver.visits.min()),'allowed_as_teacher':False,'learning_claim':False,
            'journal_head':journal.rows[completed-1]['hash'] if completed else '0'*64,
            'manifest_sha256':digest(output/'manifest.json'),'elapsed_seconds_this_invocation':time.perf_counter()-started,
            'scope':'Small 10 BB tabular reference only; no equilibrium certificate, fly learning or full teacher qualification'}
        atomic_json(output/'result.json',result);return result
    finally:
        checkpoint();journal.close()
        for sig,handler in handlers.items():signal.signal(sig,handler)


def export_training(run,output):
    run=Path(run);saved=json.loads((run/'manifest.json').read_text());config=saved['config']
    runtime=manifest(config,'conventional-tabular-no-connectome',identity(public_nodes()),identity(ALGORITHM),'numpy-tabular')
    assert_compatible(saved,runtime)
    arrays,extra,_=Checkpoints(run/'checkpoints').load(runtime);solver=ShoveFoldCFR();solver.restore(arrays)
    return export_policy(solver,output,{'training_manifest_sha256':digest(run/'manifest.json'),
        'training_source_hash':runtime['source_hash'],'iterations_completed':extra['completed_iterations'],
        'checkpoint_pointer_sha256':digest(run/'checkpoints/latest.json')})


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    training=sub.add_parser('train');training.add_argument('--config',required=True);training.add_argument('--output',required=True)
    training.add_argument('--resume',action='store_true');training.add_argument('--stop-after',type=int)
    export=sub.add_parser('export');export.add_argument('--run',required=True);export.add_argument('--output',required=True)
    args=p.parse_args()
    result=train(yaml.safe_load(Path(args.config).read_text()),args.output,args.resume,args.stop_after) if args.command=='train' else export_training(args.run,args.output)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
