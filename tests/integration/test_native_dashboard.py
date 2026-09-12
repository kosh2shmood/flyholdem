import hashlib
import numpy as np
import pytest
from flyholdem.interface.population import NeuralController
from flyholdem.neural.sparse import SparseBrain
from flyholdem.poker.opponents import Opponent
from flyholdem.server.events import verify_stream
from flyholdem.server.native import NativeGraph, NativeDemo


def fixture_controller():
    brain=SparseBrain([0,5,10,10,10,10,10,10],np.tile(np.arange(2,7),2),np.linspace(.3,1,10))
    registration={'mode':'fixture-native','graph_hash':brain.graph_hash,'input_indices':[0,1],
        'projection_indices':[[i%2] for i in range(237)],'selected_gain':12,'baseline_hz':[0]*5,
        'ensembles':[{'indices':[i]} for i in range(2,7)],
        'config':{'decision_ms':{'baseline':50,'stimulus':300,'readout':100,'inter_decision':50},'score_scale_hz':100}}
    c=NeuralController(brain,registration)
    rows=[{'bodyId':2**54+i,'class':'Kenyon_Cell' if i<2 else 'MBON','superclass':'central',
           'type':'fixture','somaLocation':[i,i*2,i*3] if i<6 else None} for i in range(7)]
    return c,NativeGraph(rows,c)


def test_native_view_preserves_all_neurons_and_marks_missing_coordinates():
    c,graph=fixture_controller()
    assert graph.meta['neuron_count']==7 and graph.meta['located_neurons']==6
    assert graph.meta['unlocated_neurons']==1 and graph.meta['edges_rendered']==0
    assert graph.positions.shape==(7,3) and graph.positions.dtype==np.dtype('<f4')
    assert graph.positions[6,0]>=1.5 and np.abs(graph.positions[:6]).max()<=1
    assert graph.node(6)['has_soma_coordinate'] is False
    assert graph.node(0)['body_id']==str(2**54)
    counts=np.arange(7,dtype=np.int64);view=graph.activity(counts)
    restored=np.zeros(7,dtype=np.int64);restored[view['activity_indices']]=view['activity_counts']
    assert np.array_equal(counts,restored)
    assert sum(row['spikes'] for row in view['population_activity'])==view['activity_total']==21
    with pytest.raises(IndexError): graph.node(-1)


def test_native_spectator_commits_only_neural_scores_and_never_updates_weights():
    c,graph=fixture_controller()
    # Native numerical fixture exercises the real spectator state machine without
    # downloading the connectome or mislabeling a synthetic graph as biological.
    demo=NativeDemo.__new__(NativeDemo)
    demo.controller=c;demo.brain=c.brain;demo.graph=graph
    demo.mode='fixture-native';demo.learning_mode='bio-plastic';demo.model_hash=None
    demo.seed=123;demo.hand_number=demo.sequence=0;demo.hand=None;demo.completed=False
    demo.opponent=Opponent(124);demo.previous_hash='0'*64;demo.return_bb=0;demo.identity={'scope':'native-fixture-test'}
    initial=c.brain.weights.copy();events=[];settled=0
    while settled<4:
        event=demo.next_event();events.append(event)
        assert event['plasticity_enabled'] is False
        if event.get('decision'):
            d=event['decision'];expected=int(np.argmax(np.where(d['legal_mask'],d['scores'],-np.inf)))
            assert event['action']['action']==d['selected']==expected
            assert d['score_source']=='MaleCNS-native-LIF-spikes'
            assert 'opponent_hole' not in d['observation'] and 'counts' not in d
            assert d['activity_total']==sum(d['activity_counts'])
        if event['kind']=='settlement':
            settled+=1;assert event['reward_delivered'] is False
    assert np.array_equal(initial,c.brain.weights)
    assert verify_stream(events)==events[-1]['hash']


def test_fixture_server_reports_evidence_and_rejects_unavailable_native_buffers(tmp_path):
    from fastapi.testclient import TestClient
    from flyholdem.server.app import create_app
    with TestClient(create_app(interval=.01,log_path=tmp_path/'events.jsonl')) as client:
        graph=client.get('/api/graph').json()
        assert graph['mode']=='fixture' and graph['neuron_count']==126
        assert client.get('/api/graph/positions').status_code==404
        assert client.get('/api/graph?mode=full').status_code==404
        assert client.get('/api/evidence').json()['scope'].startswith('Separate registered experiments')


def test_native_replay_refuses_a_different_population_registration(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    import flyholdem.server.native as native
    from flyholdem.server.app import create_app
    from flyholdem.server.events import dumps
    controller,graph=fixture_controller()
    fake=SimpleNamespace(brain=controller.brain,graph=graph)
    monkeypatch.setattr(native,'NativeDemo',lambda *args,**kwargs:fake)
    event={'schema':'flyholdem-event-v1','sequence':0,'previous_hash':'0'*64,'mode':'circuit',
           'run_identity':{'graph_sha256':controller.brain.graph_hash,'registration_sha256':'different'},
           'kind':'hand_start'}
    event['hash']=hashlib.sha256(dumps(event).encode()).hexdigest()
    path=tmp_path/'wrong-registration.jsonl';path.write_text(dumps(event)+'\n')
    with pytest.raises(ValueError,match='population registration'):
        create_app(mode='circuit',replay=path)
