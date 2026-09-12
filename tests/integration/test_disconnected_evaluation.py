"""Actual frozen native PokerKit evaluation in an isolated package and process."""
import hashlib
import json
import numpy as np
import pytest
from flyholdem.connectome.registry import digest
from flyholdem.interface.population import load_controller
from flyholdem.interface.frozen import export_frozen
from flyholdem.experiments.disconnected_evaluation import evaluate_disconnected
from flyholdem.experiments.evaluate import evaluate
from flyholdem.poker.opponents import VERSIONS


def native_model(root):
    graph=root/'graph';graph.mkdir()
    arrays={'ptr':np.array([0,5,10,10,10,10,10,10],dtype=np.int64),'post':np.tile(np.arange(2,7,dtype=np.int32),2),
        'weight':np.linspace(.3,1,10,dtype=np.float32),'contacts':np.ones(10,dtype=np.uint32),
        'ids':np.arange(7,dtype=np.uint64),'signs':np.ones(7,dtype=np.int8),'uncertain':np.zeros(7,dtype=bool)}
    files={}
    for name,value in arrays.items():
        np.save(graph/(name+'.npy'),value,allow_pickle=False);files[name+'.npy']=digest(graph/(name+'.npy'))
    graph_hash=hashlib.sha256(b''.join(arrays[k].tobytes() for k in ('ptr','post','weight'))).hexdigest()
    (graph/'manifest.json').write_text(json.dumps({'files':files,'graph_hash':graph_hash,'mode':'fixture-native'}))
    registration={'stage':'frozen','mode':'fixture-native','graph_hash':graph_hash,'base_graph_hash':graph_hash,
        'input_indices':[0,1],'projection_indices':[[i%2] for i in range(237)],'selected_gain':12,'baseline_hz':[0]*5,
        'ensembles':[{'indices':[i]} for i in range(2,7)],
        'config':{'decision_ms':{'baseline':50,'stimulus':300,'readout':100,'inter_decision':50},'score_scale_hz':100}}
    path=root/'registration.json';path.write_text(json.dumps(registration));controller=load_controller(graph,path)
    model=root/'model';export_frozen(controller,np.arange(10),[.1,2],'distilled-connectome',model,
        {'scope':'fixture, no teacher or learning result'},{'status':'engineering-only'})
    config={'schema':'frozen-poker-evaluation-v1','status':'development','mode':'fixture-native','curriculum':'shove-fold-10bb-v1',
        'opponents':list(VERSIONS),'paired_deals_per_opponent':2,'deal_seed_start':17101,'bootstrap_seed':17102,'bootstrap_repeats':100,'progress_hands':100}
    return config,model,graph


def test_isolated_native_poker_evaluation_denies_teacher_and_resumes_exactly(tmp_path,monkeypatch):
    config,model,graph=native_model(tmp_path)
    import flyholdem.experiments.evaluate as original
    load=original.load_frozen
    monkeypatch.setattr(original,'load_frozen',lambda path,unused:load(path,graph))
    direct=evaluate(config,tmp_path/'direct',model)
    first=evaluate_disconnected(config,tmp_path/'isolated',model,graph,stop_after=7)
    assert first['result']['status']=='interrupted' and first['isolation']['hands_completed']==7
    result=evaluate_disconnected(config,tmp_path/'isolated',model,graph,resume=True)
    assert result['isolation']['teacher_import_denied'] and result['isolation']['external_training_files_denied']
    assert result['result']['opponents']==direct['opponents'] and result['result']['hands_completed']==16
    assert (tmp_path/'direct/hands.jsonl').read_bytes()==(tmp_path/'isolated/evaluation/hands.jsonl').read_bytes()
    assert not (tmp_path/'isolated/runtime/src/flyholdem/teacher').exists()
    path=tmp_path/'isolated/runtime/src/flyholdem/poker/engine.py';path.write_text(path.read_text()+'\n# changed\n')
    with pytest.raises(ValueError,match='runtime copy changed'):
        evaluate_disconnected(config,tmp_path/'isolated',model,graph,resume=True)



def test_synthetic_topology_exports_with_its_own_graph_and_reloads_exact_native_decisions(tmp_path):
    from flyholdem.learning.poker_controls import shuffled_connectome,shuffled_encoder,export_control_graph
    from flyholdem.interface.frozen import load_frozen
    from flyholdem.poker.engine import Hand
    config,model,graph=native_model(tmp_path);original=load_frozen(model,graph)
    encoder=shuffled_encoder(original,42)
    assert encoder.brain is original.brain and encoder.registration['ensembles']==original.registration['ensembles']
    assert sorted(encoder.registration['projection_indices'])==sorted(original.registration['projection_indices'])
    assert encoder.registration['projection_indices']!=original.registration['projection_indices']
    control=shuffled_connectome(original,42);exported=export_control_graph(control,graph,tmp_path/'null-graph')
    null_record=json.loads((tmp_path/'null-graph/manifest.json').read_text())
    assert not null_record['retained_malecns_graph'] and exported.registration['mode']=='fixture-native-shuffled-control'
    export_frozen(exported,np.arange(10),[.1,2],'bio-plastic',tmp_path/'null-model',{'scope':'synthetic control only'},{'status':'engineering-only'})
    restored=load_frozen(tmp_path/'null-model',tmp_path/'null-graph')
    a=exported.decide(Hand(17400).observation());b=restored.decide(Hand(17400).observation())
    assert a['counts'].tobytes()==b['counts'].tobytes() and a['scores']==b['scores'] and a['selected']==b['selected']
    with pytest.raises(ValueError,match='base graph mismatch'):load_frozen(tmp_path/'null-model',graph)
    again=export_control_graph(control,graph,tmp_path/'null-graph');assert again.registration==exported.registration
