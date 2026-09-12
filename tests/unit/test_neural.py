import numpy as np
from flyholdem.interface.decoder import decode
from flyholdem.neural.simulator import FixtureBrain
from flyholdem.poker.engine import Hand


def test_scores_are_only_readout_spikes():
    brain = FixtureBrain()
    result = brain.decide(Hand().observation(), temperature=0)
    expected = [float(np.mean(np.array(result['activity'])[ids]))/.1/100 for ids in brain.ensembles]
    assert result['scores'] == expected
    assert not result['decoder_fallback']
    assert result['selected'] == int(np.argmax(np.where(result['legal_mask'], expected, -np.inf)))


def test_no_input_silence_uses_declared_tie_break():
    brain = FixtureBrain()
    counts = brain.advance(np.zeros(brain.n), 500)
    assert not counts.any()
    result = decode(counts, brain.ensembles, [True]*5, brain.rng)
    assert result['silent'] and not result['decoder_fallback']
    assert result['selected'] == 0


def test_fixture_plasticity_changes_only_existing_bounded_edges():
    brain = FixtureBrain()
    brain.decide(Hand().observation())
    topology = brain.graph_hash
    first = brain.weights.copy()
    event = brain.reinforce(20, 0)
    assert event['changed_synapses'] > 0
    assert np.any(brain.weights != first)
    assert np.all(brain.weights >= .1*brain.initial)
    assert np.all(brain.weights <= 2*brain.initial)
    assert brain.graph_hash == topology
    first = brain.weights.copy()
    frozen = brain.reinforce(-20, 1, learning=False)
    assert frozen['changed_synapses'] == 0
    assert np.array_equal(first, brain.weights)


def test_refractory_blocks_two_complete_ticks_and_discards_input():
    brain = FixtureBrain()
    brain.refractory[0] = 2
    brain.pending[0] = 999
    drive = np.zeros(brain.n)
    drive[0] = 10
    assert brain.advance(drive, 2)[0] == 0
    assert brain.v[0] == 0
    assert brain.current[0] == 0
    assert brain.refractory[0] == 0
    brain.advance(drive, 1)
    assert brain.v[0] == .5
