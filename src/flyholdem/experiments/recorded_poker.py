"""Audit recorded native decisions against the exact PokerKit hand and RNG.

No neural solver or teacher executes here. Reconstructed spike bytes establish
record consistency, not independent attestation of historical neural activity.
"""
import hashlib
import numpy as np
from flyholdem.interface.encoder import encode_player_state, encoded_hash
from flyholdem.interface.population import neural_scores, select_action
from flyholdem.learning.curriculum import CurriculumHand
from flyholdem.learning.spike_record import restore_spikes
from flyholdem.poker.infoset import canonical_state
from flyholdem.poker.observation import canonical_bytes
from flyholdem.poker.opponents import Opponent, VERSIONS
from flyholdem.provenance import identity


def audit_recorded_hand(row, registration, neuron_count, rng, temperature):
    """Consume the same continuous action RNG as the original training arm.

    Seed/deal schedules, source bindings, teaching signals and weight continuity
    are checked by the enclosing arm audit. This function checks the complete
    actual hand, including the canonical river prefix and both player seats.
    """
    seat=row['neural_seat'];seed=row['deal_seed'];kind=row['opponent']
    if type(seat) is not int or seat not in (0,1) or type(seed) is not int:
        raise ValueError('Recorded native seat/deal is invalid')
    if not np.isfinite(temperature) or temperature<0:
        raise ValueError('Finite registered action temperature required')
    game=CurriculumHand(row['curriculum'],seed,button=0)
    ensembles=[np.asarray(group['indices'],dtype=np.int64) for group in registration['ensembles']]
    timing=registration['config']['decision_ms']
    decisions=iter(row['neural_decisions']);action_counts=np.zeros(5,dtype=np.int64)
    snapshot=row['frozen_opponent'];snapshot_decisions=iter(())
    if kind in VERSIONS:
        if row['opponent_version']!=VERSIONS[kind] or snapshot is not None:
            raise ValueError('Recorded original opponent identity changed')
        other=Opponent(seed+2000000,kind)
    else:
        if (not isinstance(snapshot,dict) or kind!='frozen-native:'+snapshot['model_sha256']
                or row['opponent_version']!='frozen-native-opponent-v1:'+snapshot['model_sha256']
                or snapshot['weights_unchanged'] is not True or snapshot['learning'] is not False
                or snapshot['teacher_connected'] is not False
                or snapshot['decisions_sha256']!=identity(snapshot['decisions'])):
            raise ValueError('Recorded frozen opponent binding changed')
        other=None;snapshot_decisions=iter(snapshot['decisions'])
    total_spikes=0;checked=0
    while not game.done:
        observation=game.observation()
        if game.actor!=seat:
            if other is not None:action=other.act(observation)
            else:
                d=next(snapshot_decisions,None)
                if d is None:raise ValueError('Missing frozen opponent decision')
                scores=np.asarray(d['scores'],dtype=float)
                legal=np.asarray(observation['legal_mask'],dtype=bool)
                if (scores.shape!=(5,) or not np.isfinite(scores).all()
                        or d['legal_mask']!=legal.tolist()
                        or d['observation_sha256']!=hashlib.sha256(canonical_bytes(canonical_state(observation))).hexdigest()
                        or type(d['selected']) is not int
                        or d['selected']!=int(np.argmax(np.where(legal,scores,-np.inf)))):
                    raise ValueError('Recorded frozen opponent differs from its visible legal argmax')
                action=d['selected']
            game.act(action);continue
        decision=next(decisions,None)
        if decision is None:raise ValueError('Missing recorded native decision')
        counts=restore_spikes(decision['recorded_spikes'],decision['counts_sha256'],neuron_count)
        rates,scores=neural_scores(counts,ensembles,timing['readout'],registration['baseline_hz'],registration['config']['score_scale_hz'])
        legal=np.asarray(observation['legal_mask'],dtype=bool)
        if (canonical_bytes(decision['observation'])!=canonical_bytes(observation)
                or decision['legal_mask']!=legal.tolist()
                or decision['encoded_hash']!=encoded_hash(encode_player_state(observation))
                or decision['raw_rates_hz']!=rates.tolist() or decision['scores']!=scores.tolist()
                or decision['temperature']!=temperature or decision['decoder_fallback'] is not False
                or decision['silent']!=bool(np.max(rates[legal])==0)
                or decision['score_source']!='MaleCNS-native-LIF-spikes'
                or decision['tie_break']!='lowest-legal-index' or type(decision['selected']) is not int):
            raise ValueError('Recorded native decision differs from its actual visible input or spike-derived scores')
        action=select_action(scores,legal,rng,temperature)
        if action!=decision['selected']:
            raise ValueError('Recorded native action differs from registered scores and action RNG')
        if game.act(action)!=decision['committed_action']:
            raise ValueError('Recorded native action differs from actual PokerKit commitment')
        action_counts[action]+=1;checked+=1;total_spikes+=int(counts.sum())
    if (next(decisions,None) is not None or next(snapshot_decisions,None) is not None
            or row['public_terminal']!=game.view() or row['private_hand_checkpoint']!=game.serialize()
            or row['neural_return_bb']!=game.view()['payoffs'][seat]/2
            or row['action_counts']!=action_counts.tolist()):
        raise ValueError('Recorded native hand differs from its actual PokerKit settlement')
    return {'native_decisions':checked,'readout_spikes':total_spikes,
        'action_rng_verified':True,'visible_observations_verified':True,'pokerkit_trajectory_verified':True,
        'historical_neural_execution_repeated':False}
