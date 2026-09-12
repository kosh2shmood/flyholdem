"""Independent Brian2 oracle inspired by DOOMFLY's MIT numerical audit.

Original test adaptation, upstream revision recorded in THIRD_PARTY.md.
"""
import numpy as np
import pytest
from flyholdem.neural.reference import ReferenceBrain
from flyholdem.neural.sparse import SparseBrain
from flyholdem.neural.kernel.build import build


@pytest.mark.parametrize('backend',['python','native'])
def test_against_brian2_exact_lif_and_refractory_schedule(backend):
    b2=pytest.importorskip('brian2')
    if backend=='native':build()
    b2.start_scope();b2.prefs.codegen.target='numpy';b2.defaultclock.dt=.1*b2.ms
    pre=np.array([0,0,0,1,1,2,3]);post=np.array([0,1,2,2,3,1,0])
    w=np.array([100,35,-20,45,-15,20,-10],dtype=np.float32)
    ptr=np.array([0,3,5,6,7])
    brain=(ReferenceBrain if backend=='python' else SparseBrain)(ptr,post,w)
    cells=b2.NeuronGroup(4,'''dv/dt = (-52*mV-v+drive+g)/(20*ms) : volt (unless refractory)
        dg/dt = -g/(5*ms) : volt (unless refractory)
        drive : volt''',method='exact',threshold='v > -45*mV',reset='v=-52*mV; g=0*mV',refractory=2.2*b2.ms)
    cells.v=-52*b2.mV
    synapses=b2.Synapses(cells,cells,'w : volt',on_pre='g += w',delay=1.8*b2.ms)
    synapses.connect(i=pre,j=post);synapses.w=w*b2.mV
    spikes=b2.SpikeMonitor(cells);states=b2.StateMonitor(cells,['v','g'],record=True,when='end')
    network=b2.Network(cells,synapses,spikes,states)
    actual_counts=[];actual_v=[];actual_g=[]
    for drive,duration in [(12,40),(0,10),(18,40)]:
        stimulation=[drive,0,0,drive];cells.drive=np.array(stimulation)*b2.mV
        network.run(duration*b2.ms)
        for _ in range(duration*10):
            actual_counts.append(brain.advance(stimulation,.1));actual_v.append(list(brain.v));actual_g.append(list(brain.g))
    expected=np.zeros((900,4),dtype=int)
    for cell,t in zip(spikes.i,spikes.t/b2.ms):expected[round(float(t)*10),cell]+=1
    np.testing.assert_array_equal(actual_counts,expected)
    np.testing.assert_allclose(np.array(actual_v).T,states.v/b2.mV,atol=.002,rtol=0)
    np.testing.assert_allclose(np.array(actual_g).T,states.g/b2.mV,atol=.002,rtol=0)
