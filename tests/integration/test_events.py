import copy
import json
import hashlib
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from flyholdem.server.app import create_app
from flyholdem.server.events import Demo, dumps, verify_stream

ROOT = Path(__file__).resolve().parents[2]


def compare_cross_platform(actual, recorded, path='event'):
    # IEEE floating-point transcendentals can differ in the last bit across ARM
    # and x86 NumPy/libm builds. Categories, cards, counts and actions stay exact.
    if isinstance(recorded, dict):
        assert actual.keys() == recorded.keys(), path
        for key in recorded:
            if key not in ('hash', 'previous_hash', 'encoded_hash'):
                compare_cross_platform(actual[key], recorded[key], path+'.'+key)
    elif isinstance(recorded, list):
        assert len(actual) == len(recorded), path
        for i, (a,b) in enumerate(zip(actual,recorded)):
            compare_cross_platform(a,b,f'{path}[{i}]')
    elif isinstance(recorded, float):
        assert actual == pytest.approx(recorded, rel=0, abs=1e-12), path
    else:
        assert actual == recorded, path


def test_checked_in_replay_is_exact_and_simulation_is_repeatable():
    recorded = [json.loads(line) for line in (ROOT/'examples/fixture-demo.jsonl').read_text().splitlines()]
    live = Demo()
    actual = [live.next_event() for _ in recorded]
    second = Demo()
    assert dumps(actual) == dumps([second.next_event() for _ in recorded])
    compare_cross_platform(actual, recorded)
    # Both complete hash chains and the committed example digest must validate.
    verify_stream(actual)
    manifest = json.loads((ROOT/'examples/fixture-manifest.json').read_text())
    assert hashlib.sha256((ROOT/'examples/fixture-demo.jsonl').read_bytes()).hexdigest() == manifest['example_sha256']
    assert verify_stream(recorded) == recorded[-1]['hash']
    assert any(e.get('plasticity',{}).get('changed_synapses',0)>0 for e in actual)
    corrupted = copy.deepcopy(recorded)
    corrupted[1]['table']['pot'] += 1
    with pytest.raises(ValueError, match='content hash'):
        verify_stream(corrupted)


def test_live_websocket_has_actual_fly_decision_and_reinforcement(tmp_path):
    app = create_app(interval=.001, log_path=tmp_path/'events.jsonl')
    with TestClient(app) as client:
        assert client.get('/').status_code == 200
        assert client.get('/api/graph').json()['edge_count'] == 1920
        assert client.get('/api/health').json()['ok']
        with client.websocket_connect('/ws') as ws:
            events=[]
            for _ in range(30):
                e=ws.receive_json()
                events.append(e)
                if e['kind']=='reinforcement':
                    break
        assert any(e['kind']=='decision' for e in events)
        assert events[-1]['kind']=='reinforcement'
        for e in events:
            assert e['mode']=='fixture' and e['teacher']=='disconnected'
            if e['kind']=='decision':
                d=e['decision']
                assert d['legal_mask'][d['selected']]
                assert d['observation']['hole']==sorted(e['table_before']['hole'])
    lines=(tmp_path/'events.jsonl').read_text().splitlines()
    verify_stream([json.loads(line) for line in lines])


def test_server_replay_preserves_logged_bytes(tmp_path):
    recorded = [json.loads(line) for line in (ROOT/'examples/fixture-demo.jsonl').read_text().splitlines()]
    app=create_app(replay=ROOT/'examples/fixture-demo.jsonl',interval=.05)
    with TestClient(app) as client:
        with client.websocket_connect('/ws') as ws:
            event=ws.receive_json()
            assert event==recorded[event['sequence']]
