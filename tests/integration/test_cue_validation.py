"""Actual native cue decisions in tiny frozen evidence; no learned gate claim."""
import copy
import json
from types import SimpleNamespace
import numpy as np
import pytest
from flyholdem.neural.sparse import SparseBrain
from flyholdem.interface.cue_player import CuePlayer
from flyholdem.provenance import identity
from flyholdem.connectome.registry import digest
from flyholdem.experiments.journal import Journal
from flyholdem.experiments.conditioning import paired_evidence
from flyholdem.experiments.cue_validation import verify_cue_evidence


def fixture(root):
    brain=SparseBrain([0,5,10,10,10,10,10,10],np.tile(np.arange(2,7),2),np.linspace(.3,1,10))
    registration={'input_indices':[0,1],'selected_gain':12,'config':{'decision_ms':{'baseline':50,'stimulus':300,'readout':100},'score_scale_hz':100},'baseline_hz':[0]*5}
    controller=SimpleNamespace(brain=brain,registration=registration,blank=np.zeros(7,dtype=np.float32),ensembles=[np.array([i]) for i in range(2,7)])
    player=CuePlayer(controller,{'schema':'native-cue-input-v1','cells':[[0],[1]],'actions':[0,1],'amplitude_jitter':.15,'bin_ms':5})
    protocol={'schema':'conditioning-local-v1','mode':'fixture-native','learning_mode':'bio-plastic','cue':{'actions':[0,1]},'learning_rates':[.3],
        'profiles':{'confirmatory':{'seeds':[1,2,3,4,5],'training_trials':2,'evaluation_trials_per_cue':1}},'retention_ms':5,
        'criterion':{'mean_accuracy':.8,'minimum_seed_accuracy':.7,'mean_improvement':.15,'maximum_retention_drop':.05,'one_sided_p':.05,'bootstrap_seed':42,'bootstrap_repeats':100}}
    cues={'targets':[0,1],'indices':[[0],[1]]};plastic={'edge_count':10};config={**protocol,'profile':'confirmatory','selected_learning_rate':.3,'cue_hash':identity(cues),'plasticity_registration':plastic}
    (root/'manifest.json').write_text(json.dumps({'config':config,'config_hash':identity(config)}));(root/'cues.json').write_text(json.dumps(cues))
    journal=Journal(root/'trials.jsonl');summaries=[]
    def put(label,value):journal.record(len(journal.rows),label,value)
    for seed in range(1,6):
        measures={}
        for arm in ('plastic','frozen','shuffled-reward'):
            put([seed,arm,'initialize'],{'weights':'original','neural_state':'rest','baseline':'empty'})
            phases=('pre','train','post','retention','erased') if arm=='plastic' else ('train','post')
            for phase in phases:
                if phase=='retention':put([seed,arm,'retention-interval'],{'interval_ms':5,'weights_unchanged':True})
                if phase=='erased':put([seed,arm,'restore-original-weights'],{'restored_existing_edges':10})
                sequence=np.random.default_rng(seed).permutation([0,1]) if phase=='train' else [0,1]
                results=[]
                for repeat,cue in enumerate(sequence):
                    row=player.decide(int(cue),int(identity([seed,'train' if phase=='train' else 'evaluation',repeat])[:16],16))
                    row.update(target=int(cue),correct=row['selected']==cue);row['correct']=bool(row['correct'])
                    put([seed,arm,phase,repeat],row);results.append(row['correct'])
                if phase!='train':measures[arm+'/'+phase]=float(np.mean(results))
        summaries.append({'seed':seed,**measures,'restored_decisions_equal_initial':True,'frozen_decisions_equal_initial':True})
    evidence={arm:paired_evidence([row['plastic/post']-row[arm] for row in summaries],42,100) for arm in ('frozen/post','shuffled-reward/post','plastic/erased')}
    result={'schema':'conditioning-result-v1','gate':2,'learning_mode':'bio-plastic','mode':'fixture-native','profile':'confirmatory',
        'status':'fail','learning_claim':False,'development_criteria_met':False,'protocol_hash':identity(protocol),'learning_rate':.3,
        'manifest_sha256':digest(root/'manifest.json'),'cue_hash':identity(cues),'plasticity':plastic,'operations':len(journal.rows),'journal_head':journal.rows[-1]['hash'],
        'seed_results':summaries,'paired_evidence':evidence,'retention_criterion_met':True}
    journal.close();(root/'result.json').write_text(json.dumps(result));return protocol


def test_native_frozen_evidence_recomputes_as_failed_and_cannot_qualify(tmp_path):
    protocol=fixture(tmp_path);found=verify_cue_evidence(tmp_path,protocol,False)
    assert not found['passed'] and found['statistics_recomputed'] and not found['whole_gate2a_passed']
    with pytest.raises(ValueError,match='failed'):verify_cue_evidence(tmp_path,protocol)


@pytest.mark.parametrize('mutation',['statistics','schedule','argmax','learning'])
def test_rehashed_cue_evidence_cannot_change_statistics_or_native_decisions(tmp_path,mutation):
    protocol=fixture(tmp_path);path=tmp_path/'result.json';result=json.loads(path.read_text())
    if mutation=='statistics':result['seed_results'][0]['plastic/post']=1
    else:
        journal_path=tmp_path/'trials.jsonl';rows=[json.loads(line) for line in journal_path.read_text().splitlines()];journal_path.unlink()
        row=rows[1]['value']
        if mutation=='schedule':row['stimulus_seed']+=1
        elif mutation=='argmax':row['selected']=4
        else:row['plasticity']={'delta':0}
        journal=Journal(journal_path)
        for value in rows:journal.record(value['index'],value['label'],value['value'])
        result['journal_head']=journal.rows[-1]['hash'];journal.close()
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError):verify_cue_evidence(tmp_path,protocol,False)
