import hashlib
import json
import numpy as np
from flyholdem.interface.population import NeuralController
from flyholdem.neural.sparse import SparseBrain
from flyholdem.poker.opponents import VERSIONS
from flyholdem.experiments.evaluate import frozen_hand,evaluate,summarize
from flyholdem.experiments.reports import write_report


def controller():
    b=SparseBrain([0,5,10,10,10,10,10,10],np.tile(np.arange(2,7),2),np.linspace(.3,1,10))
    registration={'mode':'fixture-native','graph_hash':b.graph_hash,'input_indices':[0,1],
        'projection_indices':[[i%2] for i in range(237)],'selected_gain':12,'baseline_hz':[0]*5,
        'ensembles':[{'indices':[i]} for i in range(2,7)],
        'config':{'decision_ms':{'baseline':50,'stimulus':300,'readout':100,'inter_decision':50},'score_scale_hz':100}}
    c=NeuralController(b,registration);c.frozen_model={'learning_mode':'bio-plastic'};return c


def test_native_frozen_poker_rollout_is_deterministic_and_neural_only():
    c=controller();weights=c.brain.weights.copy()
    for seat in (0,1):
        a=frozen_hand(c,'hu-20bb-v1','calling-station',825,seat)
        b=frozen_hand(c,'hu-20bb-v1','calling-station',825,seat)
        assert a==b and np.array_equal(weights,c.brain.weights)
        assert not a['learning'] and not a['teacher_connected'] and not a['reward_delivered']
        for d in a['neural_decisions']:
            assert d['selected']==int(np.argmax(np.where(d['legal_mask'],d['scores'],-np.inf)))
            assert d['score_source']=='MaleCNS-native-LIF-spikes' and not d['decoder_fallback']
            assert not {'opponent_hole','teacher_labels','deck'}&set(d['observation'])
            assert len(d['counts_sha256'])==64


def test_complete_native_evaluation_recovery_and_report(tmp_path,monkeypatch):
    import flyholdem.experiments.evaluate as module
    monkeypatch.setattr(module,'load_frozen',lambda *a,**k:controller())
    model=tmp_path/'model';model.mkdir();(model/'manifest.json').write_text('{}')
    config={'schema':'frozen-poker-evaluation-v1','status':'development','mode':'fixture-native',
        'curriculum':'shove-fold-10bb-v1','opponents':list(VERSIONS),'paired_deals_per_opponent':2,
        'deal_seed_start':51234,'bootstrap_seed':51235,'bootstrap_repeats':100,'progress_hands':100}
    full=tmp_path/'full';resumed=tmp_path/'resumed'
    result=evaluate(config,full,model)
    partial=evaluate(config,resumed,model,stop_after=7);assert partial['status']=='interrupted'
    recovered=evaluate(config,resumed,model,resume=True)
    assert result['opponents']==recovered['opponents']
    assert (full/'hands.jsonl').read_bytes()==(resumed/'hands.jsonl').read_bytes()
    assert result['hands_completed']==16 and result['status']=='baseline-complete'
    assert all(v['paired_deals']==2 for v in result['opponents'].values())
    report=write_report(full);value=json.loads(open(report['json']).read())
    assert value['status']=='baseline-complete' and not value['recorded_learning_claim']
    assert value['journal_verification']['hands.jsonl']['rows']==16


def test_paired_statistics_keep_same_deal_seats_together():
    config={'opponents':['random'],'bootstrap_seed':1,'bootstrap_repeats':100}
    rows=[{'opponent':'random','deal_seed':seed,'neural_seat':seat,'neural_return_bb':value,'action_counts':[1,0,0,0,0]}
          for seed,seat,value in [(1,0,10),(1,1,-8),(2,0,-4),(2,1,10),(3,0,999)]]
    summary=summarize(rows,config)['random']
    assert summary['bb_per_hand']==2 and summary['paired_deals']==2
    assert summary['unpaired_completed_hands']==1 and summary['action_counts']==[4,0,0,0,0]
