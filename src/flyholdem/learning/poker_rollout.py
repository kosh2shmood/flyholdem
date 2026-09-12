"""Complete PokerKit hands with learning delivered after neural action commitment.

This mechanism is shared by gated experiments. It does not qualify a teacher,
choose parameters, publish a policy, or claim a scientific gate on its own.
"""
import hashlib
import numpy as np
from flyholdem.poker.infoset import canonical_state, canonical_information_id
from flyholdem.poker.observation import canonical_bytes
from flyholdem.poker.opponents import Opponent, VERSIONS
from .curriculum import CurriculumHand
from .spike_record import record_spikes


def poker_hand(player, curriculum, opponent, deal_seed, seat, *, learning=False,
               temperature=0, teacher=None, terminal_reward_override=None,
               target_permutation_seed=None, phase=None, teacher_split=None):
    if type(seat) is not int or seat not in (0,1):raise ValueError('One heads-up learning seat is required')
    phase=phase or ('training' if learning else 'evaluation')
    if phase not in ('training','evaluation') or phase=='evaluation' and learning:
        raise ValueError('Learning is allowed only during an explicit training phase')
    if phase=='evaluation' and temperature!=0:
        raise ValueError('Frozen evaluation has no exploration')
    if not learning and (teacher is not None or terminal_reward_override is not None or target_permutation_seed is not None):
        raise ValueError('Frozen weights accept no teacher or control reward inputs')
    distilled=player.requires_teacher
    if teacher_split is not None and (teacher_split!='train' or not learning or not distilled):
        raise ValueError('Only teaching arms may restrict supervision to the canonical train split')
    if learning and distilled and teacher is None:raise ValueError('Distillation requires an independently qualified teacher')
    if not distilled and (teacher is not None or target_permutation_seed is not None):
        raise ValueError('Terminal learning accepts chip reward only')
    if distilled and terminal_reward_override is not None:raise ValueError('Distillation cannot use terminal shaping')
    if terminal_reward_override is not None and not np.isfinite(terminal_reward_override):
        raise ValueError('Finite matched-control reward required')
    if target_permutation_seed is not None and type(target_permutation_seed) is not int:
        raise ValueError('An explicit deterministic target permutation seed is required')
    from .frozen_opponent import FrozenNeuralOpponent
    snapshot=isinstance(opponent,FrozenNeuralOpponent)
    if snapshot:
        if opponent.brain is player.brain or np.shares_memory(opponent.brain.weights,player.brain.weights):
            raise ValueError('Learning player and frozen opponent require independent mutable neural states')
        other=opponent;opponent_name=other.name;opponent_version=other.version;other.begin_hand()
    else:
        if opponent not in VERSIONS:raise ValueError('Use a registered fixed opponent or an explicit frozen native snapshot')
        other=Opponent(deal_seed+2000000,opponent);opponent_name=opponent;opponent_version=VERSIONS[opponent]
    game=CurriculumHand(curriculum,deal_seed,button=0)
    target_rng=np.random.default_rng(target_permutation_seed) if target_permutation_seed is not None else None
    player.begin_hand();decisions=[];counts=np.zeros(5,dtype=np.int64)
    weights_before=hashlib.sha256(player.brain.weights.tobytes()).hexdigest()
    while not game.done:
        observation=game.observation()
        if game.actor!=seat:
            game.act(other.act(observation));continue
        decision=player.decide(observation,temperature)
        scores=np.asarray(decision['scores']);legal=np.asarray(observation['legal_mask'],dtype=bool)
        action=decision['selected']
        if (decision['decoder_fallback'] or type(action) is not int or action not in range(5) or not legal[action]
                or not np.isfinite(scores).all() or scores.shape!=(5,)
                or temperature==0 and action!=int(np.argmax(np.where(legal,scores,-np.inf)))):
            raise AssertionError('Committed poker action must come from finite legal native scores')
        spikes=record_spikes(decision['counts'])
        # Canonical action is committed before any teaching target is requested.
        committed=game.act(action);counts[action]+=1
        target=None;original_target=None;target_id=None
        if learning and distilled:
            visible=canonical_state(observation);before=canonical_bytes(visible)
            target_id=canonical_information_id(visible)
            from flyholdem.teacher.corpus import split_for_id
            if teacher_split is None or split_for_id(target_id)==teacher_split:
                target=np.asarray(teacher.probabilities(visible),dtype=float).copy()
                if before!=canonical_bytes(visible):raise ValueError('Teacher modified its canonical input')
                if (target.shape!=(5,) or not np.isfinite(target).all() or np.any(target<0)
                        or np.any(target[~legal]) or not np.isclose(target.sum(),1,rtol=0,atol=1e-12)):
                    raise ValueError('Teacher supplied an invalid legal distribution')
                original_target=target.tolist()
                if target_rng is not None:
                    # Preserve this target's mass and entropy; break its action label
                    # association only within the currently legal action set.
                    target[legal]=target_rng.permutation(target[legal])
        event=player.commit_action(learning and (not distilled or target is not None),teacher_target=target)
        record={key:value for key,value in decision.items() if key not in ('counts','encoded')}
        record.update(recorded_spikes=spikes,counts_sha256=hashlib.sha256(decision['counts'].tobytes()).hexdigest(),
            committed_action=committed,teaching_after_commit=event,
            teacher_input_id=target_id,original_teacher_target=original_target,
            legal_target_permutation=target_rng is not None and target is not None,
            teacher_target_held_out=target_id is not None and target is None)
        decisions.append(record)
    table=game.view()
    if sum(table['payoffs'])!=0:raise AssertionError('PokerKit chip conservation failure')
    net_bb=table['payoffs'][seat]/2
    # The canonical position is 1 for the button, which is fixed to seat zero.
    reinforcement=player.finish_hand(net_bb,int(seat==game.hand.button),learning,terminal_reward_override)
    weights_after=hashlib.sha256(player.brain.weights.tobytes()).hexdigest()
    if not learning and weights_after!=weights_before:raise AssertionError('Frozen poker hand changed weights')
    return {'deal_seed':deal_seed,'neural_seat':seat,'opponent':opponent_name,'opponent_version':opponent_version,
        'curriculum':curriculum,'neural_return_bb':net_bb,'action_counts':counts.tolist(),
        'neural_decisions':decisions,'public_terminal':table,'private_hand_checkpoint':game.serialize(),
        'learning':learning,'phase':phase,'learning_mode':player.mode,'optimization':player.optimization,'teacher_connected':learning and distilled,
        'terminal_reinforcement':reinforcement,'frozen_opponent':other.finish_hand() if snapshot else None,
        'weights_before_sha256':weights_before,
        'weights_after_sha256':weights_after,'target_permutation_seed':target_permutation_seed,'teacher_split':teacher_split,
        'teacher_targets_delivered':sum(row['teaching_after_commit'] is not None for row in decisions)}
