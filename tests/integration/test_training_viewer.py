"""Training replay uses actual private fixture records but emits public viewer state."""
import copy
import json
import numpy as np
import pytest
from flyholdem.experiments.poker_training import _train_arm
from flyholdem.learning.curriculum import CURRICULA,CurriculumHand
from flyholdem.learning.spike_record import restore_spikes
from flyholdem.neural.checkpoint import atomic_json
from flyholdem.neural.sparse import SparseBrain
from flyholdem.provenance import identity
from flyholdem.server.training import RecordedTraining
from flyholdem.server.events import verify_stream
from test_poker_curriculum import fixture
from test_poker_rollout import VisibleTeacher


def training_fixture(root,curriculum='hu-20bb-v1',teaching=False):
    plan,factory,graph=fixture(root,teaching=teaching)
    plan['seeds']=[51];plan['curriculum']=curriculum
    plan['training'].update(hands=8,opponent_cycle=['calling-station'],temperature={'start':.5,'end':.2,'decay_hands':8})
    parent=root/'experiment';run=parent/'51/plastic/training';run.parent.mkdir(parents=True)
    protocol={'schema':'native-poker-curriculum-run-v1','plan':plan,'authorization':None,'scope':'numerical-engineering-fixture-only'}
    atomic_json(parent/'manifest.json',{'config':protocol,'config_hash':identity(protocol)})
    player,_=factory(51,'plastic',root/'unused-null')
    config={**plan['training'],'schema':'poker-training-arm-v1','arm':'plastic','curriculum':curriculum,
        'seed':51,'deal_seed_start':plan['training_starts']['51'],'teacher_split':'train' if teaching else None}
    options={'teacher':VisibleTeacher(),'teacher_sha256':'a'*64} if teaching else {}
    _train_arm(player,config,run,**options)
    return run,graph


@pytest.mark.parametrize('curriculum',list(CURRICULA))
@pytest.mark.parametrize('teaching',[False,True])
def test_training_viewer_preserves_both_seats_actual_spikes_and_hidden_cards(tmp_path,curriculum,teaching,monkeypatch):
    run,graph=training_fixture(tmp_path,curriculum,teaching)
    rows=[json.loads(line)['value'] for line in (run/'hands.jsonl').read_text().splitlines()]
    def forbidden(*args,**kwargs):raise AssertionError('Recorded viewer must not construct/advance a solver or query a teacher')
    monkeypatch.setattr(SparseBrain,'__init__',forbidden);monkeypatch.setattr(SparseBrain,'advance',forbidden)
    monkeypatch.setattr(VisibleTeacher,'probabilities',forbidden)
    viewer=RecordedTraining(run,graph,hands=8)
    assert viewer.mode=='fixture-native' and viewer.graph.meta['neuron_count']==7
    assert viewer.graph.meta['located_neurons']==0 and viewer.recorded_training
    verify_stream(viewer.events)
    assert {event['teacher'] for event in viewer.events}=={'connected' if teaching else 'disconnected'}
    for number,row in enumerate(rows,1):
        events=[event for event in viewer.events if event['hand']==number]
        game=CurriculumHand(curriculum,row['deal_seed'],button=0);seat=row['neural_seat']
        own=list(map(repr,game.hand.dealt_holes[game.hand.seats.index(seat)]))
        assert events[0]['setup_action_count']==len(game.setup_actions)
        assert events[0]['curriculum']==curriculum
        assert events[0]['table']['hole']==own
        assert events[0]['table']['button']==int(seat!=0)
        decisions=[event for event in events if event.get('decision')]
        assert len(decisions)==len(row['neural_decisions'])
        for event,decision in zip(decisions,row['neural_decisions']):
            assert event['actor']==0 and event['action']['actor']==0
            assert event['table_before']['stacks'][0]==decision['observation']['own_stack']
            assert event['table_before']['stacks'][1]==decision['observation']['opponent_stack']
            assert event['decision']['observation']==decision['observation']
            assert event['decision']['scores']==decision['scores']
            counts=restore_spikes(decision['recorded_spikes'],decision['counts_sha256'],7)
            reconstructed=np.zeros(7,dtype=np.int32)
            reconstructed[event['decision']['activity_indices']]=event['decision']['activity_counts']
            assert np.array_equal(counts,reconstructed)
        for event in events:
            assert event['table']['hole']==own and event['recorded_training']
            if not event['table']['done']:assert event['table']['opponent_hole']==['??','??']
            assert not {'private_hand_checkpoint','initial_deck','deal_seed','teacher_target'}&set(event)
            if event['kind']=='teaching':
                assert event['plasticity']['raw_net_bb'] is None
                assert event['plasticity']['dopamine'] is None
                assert event['plasticity']['eligibility_mean'] is None
        final=CurriculumHand.restore(row['private_hand_checkpoint'])
        if final.hand.state.folded_status:assert events[-1]['table']['opponent_hole']==['??','??']
        assert events[-1]['net_bb']==row['neural_return_bb']
        assert events[-1]['table']['payoffs']==[row['public_terminal']['payoffs'][seat],row['public_terminal']['payoffs'][1-seat]]
    if teaching:assert any(event['kind']=='teaching' for event in viewer.events)
    else:assert any(event['kind']=='reinforcement' for event in viewer.events)


def test_training_viewer_ranges_and_graph_bindings_fail_closed(tmp_path):
    run,graph=training_fixture(tmp_path)
    viewer=RecordedTraining(run,graph,start_hand=3,hands=2)
    assert {event['hand'] for event in viewer.events}=={4,5}
    assert viewer.events[0]['sequence']==0 and viewer.identity['source_hands']==2
    for start,count in [(-1,2),(8,2),(0,0),(0,129)]:
        with pytest.raises(ValueError):RecordedTraining(run,graph,start,count)
    weights=graph/'weight.npy';value=np.load(weights);value[0]+=1;np.save(weights,value,allow_pickle=False)
    with pytest.raises(ValueError,match='graph hash mismatch'):RecordedTraining(run,graph)


def test_training_server_serves_verified_events_without_a_live_player(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    from flyholdem.server.app import create_app
    from flyholdem.server.native import NativeDemo
    from flyholdem.server.play import FrozenPlayer
    run,graph=training_fixture(tmp_path)
    expected=RecordedTraining(run,graph,hands=2)
    hashes={event['hash'] for event in expected.events}
    def forbidden(*args,**kwargs):raise AssertionError('No live player or native worker in training replay')
    monkeypatch.setattr(NativeDemo,'__init__',forbidden);monkeypatch.setattr(FrozenPlayer,'__init__',forbidden)
    monkeypatch.setattr(SparseBrain,'__init__',forbidden);monkeypatch.setattr(SparseBrain,'advance',forbidden)
    log=tmp_path/'no-live-log/events.jsonl'
    with TestClient(create_app(training_run=run,graph_path=graph,hands=2,interval=.01,log_path=log)) as client:
        health=client.get('/api/health').json()
        assert health['ok'] and health['recorded_training'] and health['replay']
        assert health['weights_frozen'] is None and not health['human_play_available']
        assert client.post('/api/play/start').status_code==409
        assert client.get('/api/graph').json()['neuron_count']==7
        assert len(client.get('/api/graph/positions').content)==7*3*4
        assert len(client.get('/api/graph/roles').content)==7
        assert client.get('/api/graph/neuron/0').json()['class']=='synthetic'
        assert client.get('/api/graph?mode=fixture').json()['neuron_count']==126
        with client.websocket_connect('/ws') as socket:
            for _ in range(4):
                event=socket.receive_json();assert event['hash'] in hashes
                assert event['recorded_training'] and event['mode']=='fixture-native'
    assert not log.exists()
    for kwargs in ({'model':'no-model'},{'mode':'full'},{'replay':'no-replay'},{'preregistration':'no-registration'}):
        with pytest.raises(ValueError,match='no --mode'):
            create_app(training_run=run,graph_path=graph,**kwargs)
    with pytest.raises(ValueError,match='require --training-run'):
        create_app(graph_path=graph)


@pytest.mark.parametrize('teaching',[False,True])
def test_all_matched_control_recordings_keep_their_exact_graph_and_arm(tmp_path,teaching):
    from flyholdem.experiments.poker_curriculum import _execute_curriculum
    plan,factory,graph=fixture(tmp_path,teaching=teaching)
    plan['seeds']=[51];plan['training_starts']={'51':17800}
    parent=tmp_path/'experiment'
    _execute_curriculum(plan,parent,factory,teacher=VisibleTeacher() if teaching else None,
        teacher_sha256='a'*64 if teaching else None)
    for arm in ['plastic',*plan['controls']]:
        actual=parent/'51/null-graph' if arm=='shuffled-connectome' else graph
        viewer=RecordedTraining(parent/'51'/arm/'training',actual,hands=4)
        assert {event['arm'] for event in viewer.events}=={arm}
        assert {event['plasticity_enabled'] for event in viewer.events}=={arm!='frozen'}
        assert {event['teacher'] for event in viewer.events}=={'connected' if teaching and arm!='frozen' else 'disconnected'}
        if arm=='shuffled-connectome':
            assert viewer.mode=='fixture-native-shuffled-control'
            with pytest.raises(ValueError,match='base graph mismatch'):
                RecordedTraining(parent/'51'/arm/'training',graph,hands=4)
        else:assert viewer.mode=='fixture-native'
