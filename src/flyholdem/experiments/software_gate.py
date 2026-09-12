"""Rules/seat symmetry on fixed seeds, with a declared sampling-error bound."""
import json
from pathlib import Path
import time
import numpy as np
import yaml
from flyholdem.poker.engine import Hand
from flyholdem.poker.opponents import Opponent
from flyholdem.provenance import identity, source_identity, environment, ROOT


def match(seed,button,policy_seeds):
    hand=Hand(seed,button)
    policies=[Opponent(s,'random') for s in policy_seeds]
    while not hand.done:
        hand.act(policies[hand.actor].act(hand.observation()))
    return hand.view()['payoffs'][0]/2


def run(config_path=None,output=None):
    config=yaml.safe_load(Path(config_path or ROOT/'configs/software_gate.yaml').read_text())
    start=time.perf_counter();seed=config['seed_start']
    independent=np.array([match(seed+i,i%2,[seed+100000+2*i,seed+100001+2*i]) for i in range(config['independent_hands'])])
    pairs=[]
    for i in range(config['paired_deals']):
        s=seed+200000+i;p=[s+10000,s+20000]
        # Same positional cards and RNG streams, with physical player labels swapped.
        pairs.append(match(s,0,p)+match(s,1,p[::-1]))
    se=float(independent.std(ddof=1)/np.sqrt(len(independent)))
    mean=float(independent.mean());bound=config['symmetry_threshold_standard_errors']*se
    result={'schema':'software-gate0-v1','scope':config['claim'],'config':config,'config_hash':identity(config),
      'source_hash':source_identity(),'environment':environment(),
      'mean_bb_per_hand':mean,'standard_error':se,'sampling_interval_bb':[mean-bound,mean+bound],
      'within_registered_sampling_error':abs(mean)<=bound,'max_seat_swapped_pair_residual_bb':float(max(map(abs,pairs))),
      'seat_swap_exact':all(x==0 for x in pairs),'hands':len(independent)+2*len(pairs),'seconds':time.perf_counter()-start}
    result['status']='pass' if result['seat_swap_exact'] and result['within_registered_sampling_error'] else 'fail'
    path=Path(output or ROOT/'docs/review/software-gate0.json');path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    result=run();print(json.dumps({k:v for k,v in result.items() if k not in ('environment','config')},indent=2))
    if result['status']!='pass':raise SystemExit(1)
