"""Actual native training mechanics; fixture targets do not qualify a teacher."""
import copy
import numpy as np
import pytest
from flyholdem.experiments.poker_training import _train_arm
from test_poker_rollout import make,VisibleTeacher


def config(arm='plastic'):
    return {'schema':'poker-training-arm-v1','arm':arm,'curriculum':'hu-20bb-v1','seed':71,'hands':12,
        'opponent_cycle':['random','calling-station','tight-aggressive'],'deal_seed_start':17300,
        'temperature':{'start':.3,'end':.1,'decay_hands':12},'progress_hands':12}


@pytest.mark.parametrize('mode,method,arm',[
    ('bio-plastic','terminal-local-eligibility','plastic'),('bio-plastic','terminal-local-eligibility','frozen'),
    ('bio-plastic','terminal-local-eligibility','shuffled-reward'),
    ('distilled-connectome','teacher-advantage-local-eligibility','plastic'),
    ('distilled-connectome','direct-readout-rate-surrogate-v1','shuffled-teacher'),
    ('distilled-connectome','terminal-local-eligibility','plastic')])
def test_complete_native_arm_recovers_optimizer_traces_rng_and_journal(tmp_path,mode,method,arm):
    _,player=make(mode,method)
    kwargs={'teacher':VisibleTeacher(),'teacher_sha256':'a'*64} if player.requires_teacher and arm!='frozen' else {}
    if arm=='shuffled-reward':kwargs['shuffled_returns']=np.arange(12)-6
    full=_train_arm(player,config(arm),tmp_path/'full',**kwargs)
    _,interrupted=make(mode,method);part=_train_arm(interrupted,config(arm),tmp_path/'restored',stop_after=5,**kwargs)
    assert part['status']=='interrupted' and part['hands_completed']==5
    _,restored=make(mode,method);done=_train_arm(restored,config(arm),tmp_path/'restored',resume=True,**kwargs)
    assert full['final_weights_sha256']==done['final_weights_sha256']
    assert (tmp_path/'full/hands.jsonl').read_bytes()==(tmp_path/'restored/hands.jsonl').read_bytes()
    a,am=player.state();b,bm=restored.state();assert am==bm
    assert all(np.array_equal(a[k],b[k]) for k in a)
    assert done['status']=='training-complete' and not done['learning_claim']
    assert (tmp_path/'restored/checkpoints/best.json').exists()


def test_unmatched_controls_and_changed_recovery_config_are_rejected(tmp_path):
    _,player=make()
    with pytest.raises(ValueError,match='matched reward'):_train_arm(player,config('shuffled-reward'),tmp_path/'missing')
    assert not (tmp_path/'missing').exists()
    _train_arm(player,config(),tmp_path/'run',stop_after=4)
    _,fresh=make();changed=config();changed['deal_seed_start']+=1
    with pytest.raises(ValueError,match='identity mismatch'):_train_arm(fresh,changed,tmp_path/'run',resume=True)
    with pytest.raises(ValueError,match='fresh registered'):_train_arm(player,config(),tmp_path/'bad')
