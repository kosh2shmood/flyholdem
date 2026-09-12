"""Replay a completed teacher evaluation and report actual action use.

Run with the policy's exact original PYTHONPATH. This script grants no
qualification, changes no policy and never samples additional evaluation deals.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from flyholdem.connectome.registry import digest
from flyholdem.poker.engine import Hand
from flyholdem.poker.opponents import Opponent
from flyholdem.provenance import canonical

ACTION_NAMES=('fold','check','call','half_pot_raise','pot_raise','all_in')
STREETS=('preflop','flop','turn','river')


def action_counter():
    return {name:0 for name in ACTION_NAMES}


def audit_play(run,policy_path,registered_config=None):
    from flyholdem.teacher.validation import verify_evaluation,registered_suite
    from flyholdem.teacher.loaders import load_policy
    run=Path(run);policy_path=Path(policy_path)
    config=registered_suite() if registered_config is None else registered_config
    verified=verify_evaluation(run,policy_path,config,require_confirmatory=False)
    policy,record=load_policy(policy_path)
    if record['schema'] not in ('teacher-external-regret-policy-v1','teacher-final-regret-policy-v1','teacher-self-play-regret-policy-v1'):
        import torch
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    by_street={street:action_counter() for street in STREETS}
    by_opponent={kind:{street:action_counter() for street in STREETS} for kind in config['opponents']}
    actions=action_counter();reproduced=0;trace=hashlib.sha256();abstract_counts=np.zeros(5,dtype=np.int64)
    with (run/'paired-deals.jsonl').open() as stream:
        for line in stream:
            entry=json.loads(line);kind,_=entry['label'];expected=entry['value'];deal=expected['deal_seed']
            returns=[];frequencies=np.zeros(5,dtype=np.int64)
            for seat in (0,1):
                hand=Hand(deal,button=0,stacks=(2*config['stack_bb'],)*2)
                opponent=Opponent(deal+2000000,kind);rng=np.random.default_rng(deal+1000000)
                while not hand.done:
                    observation=hand.observation()
                    if hand.actor==seat:
                        probabilities=policy.probabilities(observation)
                        action=int(rng.choice(5,p=probabilities));frequencies[action]+=1
                        name='check' if action==1 and observation['to_call']==0 else 'call' if action==1 else ACTION_NAMES[{0:0,2:3,3:4,4:5}[action]]
                        street=STREETS[observation['street']]
                        actions[name]+=1;by_street[street][name]+=1;by_opponent[kind][street][name]+=1
                        # Hash the actual visible information, distribution and
                        # selected action without exporting per-hand private cards.
                        trace.update(canonical({'opponent':kind,'deal_seed':deal,'seat':seat,
                            'observation':observation,'probabilities':probabilities.tolist(),'selected':action})+b'\n')
                    else:action=opponent.act(observation)
                    hand.act(action)
                returns.append(hand.view()['payoffs'][seat]/2)
            actual={'deal_seed':deal,'seat_returns_bb':returns,'paired_bb_per_hand':float(np.mean(returns)),
                'action_counts':frequencies.tolist()}
            if actual!=expected:
                raise ValueError(f'Actual teacher play differs from recorded paired deal: {kind} / {deal}')
            reproduced+=1;abstract_counts+=frequencies
    if verify_evaluation(run,policy_path,config,require_confirmatory=False)!=verified:
        raise ValueError('Teacher evaluation or policy changed during actual-play audit')
    if reproduced!=verified['paired_deals']:
        raise ValueError('Actual-play audit did not consume the entire evaluation')
    return {'schema':'actual-teacher-play-audit-v1','scope':'Descriptive replay of the same fixed evaluated policy and deals; no new trial or qualification',
        'diagnostic_source_sha256':digest(__file__),'verified_evaluation':verified,
        'policy_schema':record['schema'],'sampling':record['aggregation'],
        'actual_paired_records_reproduced':reproduced,'decisions':sum(actions.values()),
        'actions':actions,'abstract_action_counts':abstract_counts.tolist(),
        'by_street':by_street,'by_opponent':by_opponent,'decision_trace_sha256':trace.hexdigest(),
        'grants_teacher_qualification':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True);parser.add_argument('--policy',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();output=Path(args.output)
    if output.exists():raise FileExistsError('Choose a new actual-play audit output')
    result=audit_play(args.run,args.policy)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({'output':str(output),'paired_records':result['actual_paired_records_reproduced'],
        'decisions':result['decisions'],'actions':result['actions'],'grants_teacher_qualification':False},indent=2))


if __name__=='__main__':main()
