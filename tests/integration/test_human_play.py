import json
import numpy as np
import pytest
from fastapi.testclient import TestClient
from flyholdem.server.events import Demo, verify_stream
from flyholdem.server.play import FrozenPlayer, HumanSession, PlayError
from flyholdem.server.app import create_app


def test_human_cards_are_private_and_neural_state_is_independent(tmp_path):
    source=Demo(123);player=FrozenPlayer(source)
    assert not np.shares_memory(player.brain.weights,source.brain.weights)
    assert not np.shares_memory(player.brain.v,source.brain.v)
    seeds=iter(range(123,140));s=HumanSession(player,tmp_path,seed_factory=lambda:next(seeds))
    try:
        for hand in range(4):
            s.new_hand(s.revision)
            while not s.hand.done:
                assert s.hand.actor==1
                s.act(s.revision,1 if s.hand.legal_mask()[1] else next(i for i,v in enumerate(s.hand.legal_mask()) if v))
            own=[repr(c) for c in s.hand.dealt_holes[s.hand.seats.index(1)]]
            for event in [e for e in s.events if e['hand']==s.hand_number]:
                assert event['table']['opponent_hole']==own
                for view in [event['table']]+([event['table_before']] if 'table_before' in event else []):
                    if not view['done'] or len(view['board'])<5:assert view['hole']==['??','??']
                assert event.get('decision') is None
                assert 'encoded_hash' not in json.dumps(event)
                assert 'private_hand' not in json.dumps(event) and 'initial_deck' not in json.dumps(event)
                assert 'activity_counts' not in json.dumps(event) and 'scores' not in json.dumps(event)
            assert np.array_equal(player.brain.weights,player.weights)
        assert verify_stream(s.events)==s.events[-1]['hash']
        assert s.return_bb==sum(e['net_bb'] for e in s.events if e['kind']=='settlement')
    finally:s.close()


def test_stale_duplicate_illegal_and_out_of_turn_actions_cannot_mutate_hand(tmp_path):
    s=HumanSession(FrozenPlayer(Demo(44)),tmp_path,seed_factory=lambda:444)
    try:
        result=s.new_hand()
        if result['done']:result=s.new_hand(s.revision)
        assert result['human_turn']
        before=s.hand.serialize();revision=s.revision
        for action in (True,5,-1,'1'):
            with pytest.raises(PlayError):s.act(revision,action)
            assert s.hand.serialize()==before and s.revision==revision
        with pytest.raises(PlayError):s.new_hand(revision)
        legal=next(a['index'] for a in result['actions'] if a['legal'])
        s.act(revision,legal)
        after=s.hand.serialize()
        with pytest.raises(PlayError,match='changed'):s.act(revision,legal)
        assert s.hand.serialize()==after
    finally:s.close()


def test_play_api_enforces_origin_session_and_server_revision(tmp_path):
    with TestClient(create_app(interval=.01,log_path=tmp_path/'events.jsonl')) as client:
        assert client.post('/api/play/start',headers={'origin':'https://unrelated.example'}).status_code==403
        response=client.post('/api/play/start');assert response.status_code==200
        state=response.json();token=state['session']
        assert client.get('/api/health').json()['spectator_paused_for_play']
        if state['done']:
            state=client.post(f'/api/play/{token}/hand',json={'revision':state['revision']}).json()
        assert state['human_turn']
        assert client.post(f'/api/play/{token}/action',json={'revision':state['revision'],'action':True}).status_code==422
        assert client.post(f'/api/play/{token}/action',json={'revision':state['revision'],'action':1,'seed':3}).status_code==422
        assert client.post(f'/api/play/{token}/action',json={'revision':-50,'action':1}).status_code==409
        action=next(a['index'] for a in state['actions'] if a['legal'])
        accepted=client.post(f'/api/play/{token}/action',json={'revision':state['revision'],'action':action})
        assert accepted.status_code==200
        assert client.post(f'/api/play/{token}/action',json={'revision':state['revision'],'action':action}).status_code==409
        assert client.post(f'/api/play/{token}/end').json()['closed']
        assert client.get(f'/api/play/{token}').status_code==410
        assert not client.get('/api/health').json()['spectator_paused_for_play']


def test_native_human_player_clones_state_and_commits_only_native_scores(tmp_path):
    from types import SimpleNamespace
    from flyholdem.interface.population import NeuralController
    from flyholdem.neural.sparse import SparseBrain
    from flyholdem.poker.engine import Hand
    brain=SparseBrain([0,5,10,10,10,10,10,10],np.tile(np.arange(2,7),2),np.linspace(.3,1,10))
    registration={'mode':'fixture-native','graph_hash':brain.graph_hash,'input_indices':[0,1],
        'projection_indices':[[i%2] for i in range(237)],'selected_gain':12,'baseline_hz':[0]*5,
        'ensembles':[{'indices':[i]} for i in range(2,7)],
        'config':{'decision_ms':{'baseline':50,'stimulus':300,'readout':100,'inter_decision':50},'score_scale_hz':100}}
    controller=NeuralController(brain,registration)
    player=FrozenPlayer(SimpleNamespace(controller=controller,brain=brain,mode='fixture-native',learning_mode='bio-plastic'))
    observation=Hand(345).observation();initial_v=brain.v.copy()
    result=player.decide(observation)
    assert result['selected']==int(np.argmax(np.where(result['legal_mask'],result['scores'],-np.inf)))
    assert result['score_source']=='MaleCNS-native-LIF-spikes'
    assert np.array_equal(brain.v,initial_v) and np.array_equal(player.brain.weights,brain.weights)
    assert not np.shares_memory(player.brain.weights,brain.weights)
    player.reset();again=player.decide(observation)
    for key in result:
        if isinstance(result[key],np.ndarray):assert np.array_equal(result[key],again[key])
        else:assert result[key]==again[key]


def test_controller_failure_halts_without_fallback_and_fold_never_reveals(tmp_path):
    player=FrozenPlayer(Demo(9))
    def fail(_):raise ArithmeticError('bad neural scores')
    player.decide=fail
    s=HumanSession(player,tmp_path,seed_factory=lambda:5)
    try:
        with pytest.raises(PlayError,match='No fallback'):s.new_hand()
        assert s.halted and not s.hand.history
        assert s.response()['halted'] and not s.response()['human_turn']
        assert s.events[-1]['table']['hole']==['??','??']
        with pytest.raises(PlayError):s.act(s.revision,1)
    finally:s.close()
    player=FrozenPlayer(Demo(9));player.decide=lambda _: {'selected':0}
    s=HumanSession(player,tmp_path,seed_factory=lambda:5)
    try:
        result=s.new_hand();assert result['done']
        assert all(e['table']['hole']==['??','??'] for e in s.events)
    finally:s.close()
