"""Conventional reference on the native curriculum's exact held-out deals.

This policy acts only in its separately labeled reference match. It never
supplies native scores or substitutes for a fly decision.
"""
import json
from pathlib import Path
import signal
import numpy as np
from flyholdem.connectome.registry import digest
from flyholdem.provenance import manifest,identity,assert_compatible
from flyholdem.neural.checkpoint import atomic_json
from flyholdem.learning.curriculum import CurriculumHand
from flyholdem.poker.infoset import canonical_state
from flyholdem.poker.opponents import Opponent,VERSIONS
from flyholdem.experiments.journal import Journal
from flyholdem.experiments.poker_evidence import read_journal
from .evaluation import evaluation_summary


def reference_hand(policy,config,kind,seed,seat):
    game=CurriculumHand(config['curriculum'],seed,button=0);other=Opponent(seed+2000000,kind)
    rng=np.random.default_rng(seed+1000000);decisions=[];counts=np.zeros(5,dtype=np.int64)
    while not game.done:
        observation=game.observation()
        if game.actor!=seat:game.act(other.act(observation));continue
        visible=canonical_state(observation);before=identity(visible)
        probabilities=np.asarray(policy.probabilities(visible),dtype=float)
        legal=np.asarray(visible['legal_mask'],dtype=bool)
        if (identity(visible)!=before or probabilities.shape!=(5,) or not np.isfinite(probabilities).all()
                or np.any(probabilities<0) or np.any(probabilities[~legal]) or not np.isclose(probabilities.sum(),1,atol=1e-12,rtol=0)):
            raise ValueError('Reference policy must supply legal visible-information probabilities')
        action=int(rng.choice(5,p=probabilities));counts[action]+=1;committed=game.act(action)
        decisions.append({'observation':visible,'probabilities':probabilities.tolist(),'selected':action,'committed_action':committed})
    return {'scope':'conventional-reference-only','opponent':kind,'deal_seed':seed,'seat':seat,
        'return_bb':game.view()['payoffs'][seat]/2,'action_counts':counts.tolist(),'decisions':decisions,
        'private_hand_checkpoint':game.serialize()}


def _summary(rows,config):
    grouped={kind:[] for kind in config['opponents']}
    for a,b in zip(rows[::2],rows[1::2]):
        grouped[a['opponent']].append({'deal_seed':a['deal_seed'],'seat_returns_bb':[a['return_bb'],b['return_bb']],
            'paired_bb_per_hand':(a['return_bb']+b['return_bb'])/2,
            'action_counts':(np.asarray(a['action_counts'])+b['action_counts']).tolist()})
    return evaluation_summary(grouped,config)


def run_reference(policy,reference,config,output,resume=False):
    if tuple(config['opponents'])!=tuple(VERSIONS):raise ValueError('The complete original reference opponent suite is required')
    root=Path(output);runtime=manifest({'evaluation':config,'reference':reference},'not-neural','canonical-visible-information','teacher-probabilities')
    root.mkdir(parents=True,exist_ok=resume)
    if resume:assert_compatible(json.loads((root/'manifest.json').read_text()),runtime)
    else:atomic_json(root/'manifest.json',runtime)
    if resume and (root/'result.json').exists():
        result=json.loads((root/'result.json').read_text())
        if result['status']=='complete':
            verify_reference(policy,reference,config,root);return result
    journal=Journal(root/'hands.jsonl',resume);stopped=False
    def stop(signum,frame):
        nonlocal stopped
        stopped=True
    handlers={sig:signal.signal(sig,stop) for sig in (signal.SIGINT,signal.SIGTERM)}
    schedule=[(kind,config['deal_seed_start']+i,seat) for kind in config['opponents']
        for i in range(config['paired_deals_per_opponent']) for seat in (0,1)]
    try:
        for index in range(len(journal.rows),len(schedule)):
            kind,seed,seat=schedule[index];journal.record(index,[kind,seed,seat],reference_hand(policy,config,kind,seed,seat))
            if stopped:break
        complete=len(journal.rows)==len(schedule)
        result={'schema':'conventional-curriculum-reference-v1','scope':'Conventional reference, never a fly or a new teacher qualification',
            'status':'complete' if complete else 'interrupted','reference':reference,'manifest_sha256':digest(root/'manifest.json'),
            'journal_head':journal.rows[-1]['hash'] if journal.rows else '0'*64,'hands_completed':len(journal.rows),
            'learning_claim':False,'allowed_as_teacher':False,
            'summary':_summary([r['value'] for r in journal.rows],config) if complete else None}
        atomic_json(root/'result.json',result);return result
    finally:
        journal.close()
        for sig,handler in handlers.items():signal.signal(sig,handler)


def verify_reference(policy,reference,config,root):
    root=Path(root);result=json.loads((root/'result.json').read_text());runtime=json.loads((root/'manifest.json').read_text())
    if (runtime['config']!={'evaluation':config,'reference':reference} or runtime['config_hash']!=identity(runtime['config'])
            or result['manifest_sha256']!=digest(root/'manifest.json') or result['status']!='complete'
            or result['reference']!=reference or result['learning_claim'] is not False or result['allowed_as_teacher'] is not False):
        raise ValueError('Complete registered conventional reference required')
    journal=read_journal(root/'hands.jsonl');n=config['paired_deals_per_opponent']
    if len(journal)!=2*n*len(VERSIONS) or result['hands_completed']!=len(journal) or result['journal_head']!=journal[-1]['hash']:
        raise ValueError('Incomplete conventional reference paired schedule')
    for index,item in enumerate(journal):
        kind=config['opponents'][index//(2*n)];seed=config['deal_seed_start']+(index//2)%n;seat=index%2
        if item['label']!=[kind,seed,seat] or item['value']!=reference_hand(policy,config,kind,seed,seat):
            raise ValueError('Conventional reference differs from exact policy/PokerKit replay')
    if result['summary']!=_summary([r['value'] for r in journal],config):raise ValueError('Conventional reference summary mismatch')
    return {name:digest(root/name) for name in ('manifest.json','result.json','hands.jsonl')}
