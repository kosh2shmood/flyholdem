"""Complete-hand training arm used only behind the curriculum gate orchestrator.

This module implements execution/recovery, not prerequisite authorization or a
scientific learning claim. Its private runner also supports native CI fixtures.
"""
import hashlib
import json
from pathlib import Path
import signal
import time
import zlib
import numpy as np
from flyholdem.provenance import identity,manifest,assert_compatible
from flyholdem.connectome.registry import digest
from flyholdem.neural.checkpoint import Checkpoints,atomic_json
from flyholdem.poker.opponents import VERSIONS
from flyholdem.learning.curriculum import CURRICULA
from flyholdem.learning.poker_rollout import poker_hand
from .journal import Journal


def _train_arm(player,config,output,*,teacher=None,teacher_sha256=None,shuffled_returns=None,
               frozen_opponents=None,resume=False,stop_after=None):
    arm=config['arm'];hands=config['hands'];cycle=config['opponent_cycle'];snapshots=frozen_opponents or {}
    if (config.get('schema')!='poker-training-arm-v1' or config['curriculum'] not in CURRICULA
            or type(hands) is not int or hands<2 or hands%2 or not cycle
            or any(kind not in VERSIONS and kind not in snapshots for kind in cycle)
            or arm not in ('plastic','frozen','shuffled-reward','shuffled-teacher','shuffled-encoder','shuffled-connectome')):
        raise ValueError('A complete registered poker arm and paired training schedule are required')
    temperature=config['temperature']
    if (set(temperature)!={'start','end','decay_hands'} or not np.isfinite([temperature['start'],temperature['end']]).all()
            or min(temperature['start'],temperature['end'])<0 or temperature['decay_hands']<1):
        raise ValueError('Finite registered exploration schedule required')
    learning=arm!='frozen'
    if (learning and player.requires_teacher)!=(teacher is not None) or (teacher is None)!=(teacher_sha256 is None):
        raise ValueError('Teaching arms require a separately qualified target provider and its identity')
    if arm=='shuffled-teacher' and not player.requires_teacher or arm=='shuffled-reward' and player.requires_teacher:
        raise ValueError('Control labels must match the actual supervision method')
    rewards=None if shuffled_returns is None else np.asarray(shuffled_returns,dtype=float)
    if (arm=='shuffled-reward')!=(rewards is not None) or rewards is not None and (rewards.shape!=(hands,) or not np.isfinite(rewards).all()):
        raise ValueError('Shuffled terminal control requires one finite matched reward per complete hand')
    if arm.startswith('shuffled-') and arm in ('shuffled-encoder','shuffled-connectome'):
        if player.controller.registration.get('control',{}).get('kind')!=arm:
            raise ValueError('Control graph/encoder must be explicitly constructed and identified')
    # Entry uses a freshly loaded starting model; restore happens only after its
    # immutable identity has been checked against the saved run.
    if player.baseline.contexts or player.last_decision is not None or player.brain.time_ms!=0:
        raise ValueError('Construct a fresh registered starting player before training or recovery')
    initial_hash=hashlib.sha256(player.brain.weights.tobytes()).hexdigest()
    config_record={**config,'activity_recording':'lossless-sparse-readout-window-v1','activity_codec_runtime':zlib.ZLIB_RUNTIME_VERSION,'neuron_count':player.brain.n,'learning_mode':player.mode,'optimization':player.optimization,
        'player_config':player.config,'plasticity_registration':player.eligible.registration,
        'controller_registration':player.controller.registration,'initial_weights_sha256':initial_hash,
        'initial_rng':player.controller.rng.bit_generator.state,'teacher_sha256':teacher_sha256,
        'control_rewards_sha256':None if rewards is None else hashlib.sha256(rewards.tobytes()).hexdigest(),
        'opponent_versions':{name:VERSIONS[name] if name in VERSIONS else snapshots[name].version for name in cycle}}
    runtime=manifest(config_record,player.brain.graph_hash,identity(player.controller.registration['projection_indices']),
        identity(player.controller.registration['ensembles']),player.brain.build['binary_sha256'],initial_hash)
    out=Path(output);out.mkdir(parents=True,exist_ok=resume)
    if resume:assert_compatible(json.loads((out/'manifest.json').read_text()),runtime)
    else:atomic_json(out/'manifest.json',runtime)
    journal=Journal(out/'hands.jsonl',resume);checkpoints=Checkpoints(out/'checkpoints')
    completed=0;in_hand=False;stopped=False;started=time.perf_counter();last_checkpoint=time.monotonic()
    if resume:
        arrays,extra,_=checkpoints.load(runtime);completed=extra['completed_hands']
        if completed>len(journal.rows) or extra['journal_head']!=(journal.rows[completed-1]['hash'] if completed else '0'*64):
            raise ValueError('Poker arm checkpoint/journal mismatch')
        player.restore_state(arrays,extra['player'])
        if completed==hands and (out/'result.json').is_file():
            result=json.loads((out/'result.json').read_text())
            if (len(journal.rows)!=hands or result['status']!='training-complete'
                    or result['journal_head']!=journal.rows[-1]['hash']
                    or result['manifest_sha256']!=digest(out/'manifest.json')
                    or result['final_weights_sha256']!=hashlib.sha256(player.brain.weights.tobytes()).hexdigest()
                    or result['initial_weights_sha256']!=initial_hash
                    or result['hands_completed']!=hands or result['planned_hands']!=hands):
                journal.close();raise ValueError('Completed training result differs from its restored final state')
            journal.close();return result
    def checkpoint():
        nonlocal last_checkpoint
        if in_hand:return
        arrays,state=player.state()
        checkpoints.save(arrays,runtime,{'completed_hands':completed,'journal_head':journal.rows[completed-1]['hash'] if completed else '0'*64,
            'player':state,'private_hand':None,'selection':'Only the registered final training boundary; no poker-based selection'},completed,best=completed==hands)
        last_checkpoint=time.monotonic()
    def stop(signum,frame):
        nonlocal stopped
        stopped=True
    handlers={sig:signal.signal(sig,stop) for sig in (signal.SIGINT,signal.SIGTERM)}
    if not resume:checkpoint()
    try:
        for index in range(completed,hands):
            in_hand=True;kind=cycle[(index//2)%len(cycle)];deal=config['deal_seed_start']+index//2;seat=index%2
            progress=min(1,index/temperature['decay_hands']);temp=temperature['start']+progress*(temperature['end']-temperature['start'])
            target_seed=int(identity([config['seed'],'shuffled-teacher',index])[:16],16) if arm=='shuffled-teacher' else None
            row=poker_hand(player,config['curriculum'],snapshots[kind] if kind in snapshots else kind,deal,seat,
                learning=learning,temperature=temp,teacher=teacher,phase='training',
                terminal_reward_override=None if rewards is None else float(rewards[index]),target_permutation_seed=target_seed,
                teacher_split=config.get('teacher_split') if learning and player.requires_teacher else None)
            journal.record(index,[kind,deal,seat],row);completed=index+1;in_hand=False
            if time.monotonic()-last_checkpoint>=300:checkpoint()
            if completed%config.get('progress_hands',32)==0:print(json.dumps({'arm':arm,'hands':completed,'planned':hands}),flush=True)
            if stopped or stop_after is not None and completed>=stop_after:break
        weights_hash=hashlib.sha256(player.brain.weights.tobytes()).hexdigest()
        if not learning and weights_hash!=initial_hash:raise AssertionError('Frozen training arm changed its initial weights')
        if completed==hands and len(journal.rows)!=hands:raise ValueError('Unexpected trailing training hands')
        result={'schema':'poker-training-arm-result-v1','status':'training-complete' if completed==hands else 'interrupted',
            'arm':arm,'mode':player.controller.registration['mode'],'learning_mode':player.mode,'optimization':player.optimization,
            'curriculum':config['curriculum'],'hands_completed':completed,'planned_hands':hands,
            'teacher_connected':teacher is not None,'learning_claim':False,'initial_weights_sha256':initial_hash,'final_weights_sha256':weights_hash,
            'journal_head':journal.rows[completed-1]['hash'] if completed else '0'*64,'manifest_sha256':digest(out/'manifest.json'),
            'elapsed_seconds_this_invocation':time.perf_counter()-started,'scope':'Training mechanism only; no held-out result or curriculum gate claim'}
        atomic_json(out/'result.json',result);return result
    finally:
        checkpoint();journal.close()
        for sig,handler in handlers.items():signal.signal(sig,handler)
