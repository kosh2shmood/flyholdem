"""Verify observed game histories by reconstructing the canonical rules engine."""
from .engine import Hand


def replay_hand(seed, button, actions, stacks=(40,40), deck=None):
    hand=Hand(seed,button,stacks,deck)
    frames=[hand.view()]
    for action in actions:
        hand.act(action)
        frames.append(hand.view())
    return hand,frames
