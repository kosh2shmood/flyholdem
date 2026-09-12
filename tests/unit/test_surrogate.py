import numpy as np
import pytest
from flyholdem.neural.sparse import SparseBrain
from flyholdem.learning.eligibility import Eligibility
from flyholdem.learning.surrogate import ReadoutSurrogate


def optimizer(rate=.1):
    brain=SparseBrain([0,7,14,14,14,14,14,14,14,14],np.tile(np.arange(2,9),2),np.linspace(.2,1.2,14))
    e=Eligibility(brain,[0,1],list(range(2,8)),{'pre_tau_ms':20,'eligibility_tau_ms':5000,
        'coincidence_scale':.001,'bin_ms':5,'learning_rate':.3,'weight_bounds':[.1,2]})
    e.pretrace[:]=[3.5,2.25]
    s=ReadoutSurrogate(e,[np.array([i]) for i in range(2,7)],{'learning_rate':rate,'temperature':1.0,
        'gradient_norm_cap':1.0,'synaptic_tau_ms':5.0,'threshold_distance_mv':7.0})
    return brain,e,s


def test_surrogate_gradient_matches_declared_linearization_finite_difference():
    _,e,s=optimizer()
    scores=np.array([.2,-.1,.4,0,.6]);legal=np.array([True,True,False,False,False]);target=np.array([0,1,0,0,0])
    loss,gradient=s.loss_gradient(scores,legal,target)
    assert np.isfinite(loss)
    jac=s.jacobian()
    for edge in range(len(e.edges)):
        direction=np.zeros(5)
        if s.action[edge]>=0: direction[s.action[edge]]=jac[edge]
        epsilon=1e-5
        upper=s.loss_gradient(scores+epsilon*direction,legal,target)[0]
        lower=s.loss_gradient(scores-epsilon*direction,legal,target)[0]
        assert gradient[edge]==pytest.approx((upper-lower)/(2*epsilon),abs=1e-10)
    assert np.count_nonzero(gradient)==4


def test_surrogate_updates_only_existing_direct_legal_edges_and_freeze_is_exact():
    brain,e,s=optimizer(1e6)
    initial=brain.weights.copy()
    scores=[.2,-.1,.4,0,.6];legal=[True,True,False,False,False];target=[0,1,0,0,0]
    s.step(scores,legal,target,learning=False)
    assert np.array_equal(brain.weights,initial)
    result=s.step(scores,legal,target)
    assert result['native_scores_overridden'] is False and result['dopamine_delivered'] is False
    assert result['changed_synapses']==4 and result['clipped_updates']==4
    changed=np.flatnonzero(brain.weights!=initial)
    assert np.all(np.isin(changed,e.edges)) and np.all(np.isin(brain.post[changed],[2,3]))
    assert np.all(brain.weights/initial>=.1-1e-7) and np.all(brain.weights/initial<=2+1e-7)
    saved=brain.weights.copy()
    with pytest.raises(ValueError,match='legal teacher'):
        s.step(scores,legal,[0,0,1,0,0])
    assert np.array_equal(brain.weights,saved)
