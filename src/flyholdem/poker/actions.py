"""Five deterministic wagers; all legality comes from PokerKit."""
from math import ceil

NAMES = ('Fold', 'Check / call', 'Raise ½ pot', 'Raise pot', 'All-in')


def translations(state):
    if state.actor_index is None:
        return [None] * 5
    actor = state.actor_index
    call = state.checking_or_calling_amount or 0
    actions = [None] * 5
    # Open-folding is deliberately disabled.
    if call > 0 and state.can_fold():
        actions[0] = ('fold', 0)
    if state.can_check_or_call():
        actions[1] = ('call', call)
    lo = state.min_completion_betting_or_raising_to_amount
    hi = state.max_completion_betting_or_raising_to_amount
    if lo is not None and hi is not None:
        # P includes all current bets; C is the additional call amount.
        # raise_to = own_street_bet + C + ceil(fraction * (P + C)).
        pot_after_call = state.total_pot_amount + call
        base = state.bets[actor] + call
        for action, fraction in ((2, .5), (3, 1.0), (4, None)):
            target = hi if fraction is None else min(hi, max(lo, base + ceil(fraction * pot_after_call)))
            if state.can_complete_bet_or_raise_to(target):
                actions[action] = ('raise', target)
    seen = set()
    for i, action in enumerate(actions):
        if action is not None:
            if action in seen:
                actions[i] = None
            else:
                seen.add(action)
    return actions
