"""Historical training viewer built from audited records, without neural execution."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from flyholdem.connectome.registry import digest
from flyholdem.connectome.prepare import load_graph
from flyholdem.interface.encoder import CHANNELS, encode_player_state
from flyholdem.interface.population import scaled_weights
from flyholdem.learning.curriculum import CurriculumHand, CURRICULA
from flyholdem.learning.spike_record import restore_spikes
from flyholdem.provenance import identity
from .events import dumps, verify_stream
from .native import NativeGraph


def _viewer_table(game,seat):
    """Orient the actual neural player at visual seat zero, retaining card privacy."""
    table=game.view();hand=game.hand
    own=hand.seats.index(seat);other=hand.seats.index(1-seat)
    table['hole']=list(map(repr,hand.dealt_holes[own]))
    showdown=game.done and not hand.state.folded_status
    table['opponent_hole']=list(map(repr,hand.state.hole_cards[other])) if showdown else ['??','??']
    for field in ('stacks','bets','payoffs'):
        if table[field] is not None:table[field]=[table[field][seat],table[field][1-seat]]
    for field in ('actor','button'):
        if table[field] is not None:table[field]=int(table[field]!=seat)
    table['history']=[{**action,'actor':int(action['actor']!=seat)} for action in table['history']]
    return table


def _recorded_graph(path,registration,neuron_count):
    path=Path(path);arrays,prepared=load_graph(path)
    if prepared['graph_hash']!=registration.get('base_graph_hash',registration['graph_hash']):
        raise ValueError('Recorded training base graph mismatch')
    weights=scaled_weights(arrays['weight'],registration['config'].get('global_weight_scale',1))
    h=hashlib.sha256()
    for array in (arrays['ptr'],arrays['post'],weights):h.update(memoryview(array))
    n=len(arrays['ptr'])-1
    if h.hexdigest()!=registration['graph_hash'] or n!=neuron_count or len(arrays['ids'])!=n:
        raise ValueError('Recorded training graph/size mismatch')
    mode=registration['mode']
    if prepared['mode']!=mode:raise ValueError('Recorded training graph mode mismatch')
    if 'neurons.feather' in prepared['files']:
        from pyarrow import feather
        rows=feather.read_table(path/'neurons.feather',columns=['bodyId','class','superclass','type','somaLocation']).to_pylist()
    elif mode.startswith('fixture-native'):
        rows=[{'bodyId':int(body),'class':'synthetic','superclass':'synthetic','type':None,'somaLocation':None} for body in arrays['ids']]
    else:raise ValueError('Recorded MaleCNS viewer requires hash-bound anatomical annotations')
    # NativeGraph only needs graph metadata and registration, never a solver.
    brain=SimpleNamespace(n=n,weights=range(len(weights)),graph_hash=h.hexdigest())
    controller=SimpleNamespace(brain=brain,registration=registration,
        ensembles=[group['indices'] for group in registration['ensembles']])
    graph=NativeGraph(rows,controller)
    graph.meta.update(activity_source='recorded-native-readout-spikes',historical_training=True)
    return brain,graph


class RecordedTraining:
    def __init__(self,run,graph_path,start_hand=0,hands=16):
        from flyholdem.experiments.poker_curriculum import _training_rows
        run=Path(run).resolve();parent=run.parents[2]
        if type(start_hand) is not int or start_hand<0 or type(hands) is not int or not 1<=hands<=128:
            raise ValueError('Choose a nonnegative starting hand and 1–128 recorded hands')
        outer=json.loads((parent/'manifest.json').read_text());protocol=outer['config']
        runtime=json.loads((run/'manifest.json').read_text());config=runtime['config']
        if (outer['config_hash']!=identity(protocol) or protocol['schema']!='native-poker-curriculum-run-v1'
                or run!=parent/str(config['seed'])/config['arm']/'training'):
            raise ValueError('Recorded training requires its matching curriculum parent')
        plan=protocol['plan'];registration=config['controller_registration'];self.mode=registration['mode']
        if plan['profile'] not in ('development','confirmatory'):
            raise ValueError('Recorded training profile is unregistered')
        if not self.mode.startswith('fixture-native') and protocol['authorization'] is None:
            raise ValueError('Recorded native training is missing its enclosing gate authorization')
        # This verifies complete hand/weight/control/target/graph-size bindings.
        rows=_training_rows(run,plan,config['seed'],config['arm'],include_records=True,record_range=(start_hand,start_hand+hands))
        if not rows:raise ValueError('Starting hand is outside the completed training arm')
        self.brain,self.graph=_recorded_graph(graph_path,registration,config['neuron_count'])
        self.learning_mode=config['learning_mode'];self.recorded_training=True
        self.identity={'graph_sha256':self.brain.graph_hash,'registration_sha256':self.graph.meta['registration_sha256'],
            'training_manifest_sha256':digest(run/'manifest.json'),'training_result_sha256':digest(run/'result.json'),
            'training_journal_sha256':digest(run/'hands.jsonl'),'curriculum_manifest_sha256':digest(parent/'manifest.json'),
            'neural_execution':'recorded only; no solver or teacher inference runs in this viewer',
            'policy_status':'poker learning unvalidated','source_start_hand':start_hand,'source_hands':len(rows)}
        self.events=[];total=0.;previous='0'*64
        for offset,row in enumerate(rows):
            game=CurriculumHand(row['curriculum'],row['deal_seed'],button=0);seat=row['neural_seat']
            learning=row['learning'];teacher=row['teacher_connected'];decisions=iter(row['neural_decisions'])
            def emit(event):
                nonlocal previous
                event.update(schema='flyholdem-event-v1',sequence=len(self.events),hand=start_hand+offset+1,
                    mode=self.mode,learning_mode=self.learning_mode,status=plan['profile'],evaluation=False,
                    optimization=row['optimization'],arm=config['arm'],recorded_training=True,
                    curriculum=row['curriculum'],setup_action_count=len(game.setup_actions),
                    teacher='connected' if teacher else 'disconnected',encoder='engineered-kc-v1',
                    plasticity_enabled=learning,opponent=row['opponent'],stack_bb=CURRICULA[row['curriculum']]['stack_bb'],
                    label=f'{self.mode.upper()} / RECORDED {plan["profile"].upper()} TRAINING / {config["arm"].upper()} / POKER LEARNING UNVALIDATED',
                    return_bb=total,run_identity=self.identity,previous_hash=previous)
                event['hash']=hashlib.sha256(dumps(event).encode()).hexdigest();previous=event['hash'];self.events.append(event)
            def plasticity(signal,terminal=False):
                value=dict(signal['plasticity'])
                value.update(raw_net_bb=row['neural_return_bb'] if terminal else None,dopamine=signal.get('reinforcement',{}).get('dopamine'),
                    eligibility_mean=value.get('eligibility_mean'),virtual_time_ms=None,
                    activity_recording='pulse-summary-only',dopamine_pulse=signal['dopamine_pulse'])
                return value
            emit({'kind':'hand_start','table':_viewer_table(game,seat)})
            actions=row['private_hand_checkpoint']['hand']['actions'][len(game.setup_actions):]
            for action in actions:
                actor=game.actor;before=_viewer_table(game,seat);decision=None;record=None
                if actor==seat:
                    record=next(decisions)
                    decision={key:record[key] for key in ('observation','encoded_hash','raw_rates_hz','scores','selected',
                        'legal_mask','temperature','silent','decoder_fallback','tie_break','score_source','virtual_time_ms')}
                    encoded=encode_player_state(record['observation'])
                    decision['encoded']=[(CHANNELS[i],float(encoded[i])) for i in np.flatnonzero(encoded)]
                    counts=restore_spikes(record['recorded_spikes'],record['counts_sha256'],self.brain.n)
                    decision.update(self.graph.activity(counts))
                committed=game.act(action);visual_action={**committed,'actor':int(actor!=seat)}
                emit({'kind':'decision' if actor==seat else 'opponent_action','actor':int(actor!=seat),
                    'decision':decision,'action':visual_action,'table_before':before,'table':_viewer_table(game,seat)})
                if record is not None and record['teaching_after_commit'] is not None:
                    emit({'kind':'teaching','table':_viewer_table(game,seat),
                        'plasticity':plasticity(record['teaching_after_commit'])})
            total+=row['neural_return_bb'];terminal=row['terminal_reinforcement']
            event={'kind':'reinforcement' if terminal['reward_delivered'] else 'settlement',
                'table':_viewer_table(game,seat),'net_bb':row['neural_return_bb'],'reward_delivered':terminal['reward_delivered']}
            if terminal['reward_delivered']:event['plasticity']=plasticity(terminal,True)
            emit(event)
        verify_stream(self.events)
