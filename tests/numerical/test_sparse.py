import numpy as np
import pytest
from flyholdem.neural.reference import ReferenceBrain
from flyholdem.neural.sparse import SparseBrain
from flyholdem.neural.kernel.build import build, LIBRARY


@pytest.fixture(scope='module',autouse=True)
def compile_kernel():build()


def graph():return np.array([0,3,5,6,7]),np.array([0,1,2,2,3,1,0]),np.array([100,35,-20,45,-15,20,-10],dtype=np.float32)


@pytest.mark.parametrize('cadence',[.1,10])
def test_native_against_independent_python_reference(cadence):
    ptr,post,w=graph();a=ReferenceBrain(ptr,post,w);b=SparseBrain(ptr,post,w)
    for drive,duration in [(12,40),(0,10),(18,40)]:
        for _ in range(round(duration/cadence)):
            stimulus=[drive,0,0,drive]
            np.testing.assert_array_equal(a.advance(stimulus,cadence),b.advance(stimulus,cadence))
            np.testing.assert_allclose(a.v,b.v,atol=.002,rtol=0)
            np.testing.assert_allclose(a.g,b.g,atol=.002,rtol=0)
            np.testing.assert_array_equal(a.refractory,b.refractory)


def test_complete_state_preserves_pending_spikes_and_lazy_evolution():
    ptr,post,w=graph();a=SparseBrain(ptr,post,w)
    a.advance([20,0,0,0],17.3)
    saved=a.state();b=SparseBrain(ptr,post,w);b.restore_state(saved)
    for drive in ([0,0,0,0],[0,20,0,0],[30,0,0,0]):
        np.testing.assert_array_equal(a.advance(drive,20),b.advance(drive,20))
        for name in a.state_names:np.testing.assert_array_equal(getattr(a,name),getattr(b,name))


def test_native_interventions_and_invalid_state():
    a=SparseBrain(*graph())
    assert not a.advance([0]*4,20).any()
    counts=a.advance([20,0,0,0],40)
    assert counts[0]>0 and counts[1]>0  # only an actual existing edge drives node 1
    state=a.state();state['queue_count'][0]=a.n+1
    before=a.v.copy()
    with pytest.raises(ValueError):a.restore_state(state)
    np.testing.assert_array_equal(before,a.v)
    with pytest.raises(ValueError):a.advance([float('nan')]*4,1)
    with pytest.raises(ValueError):a.advance([0]*4,.15)
