"""Conventional-teacher-only public stack potential; no fly reward shaping.

For undiscounted finite hands, adding Phi(next)-Phi(current), with terminal Phi=0,
telescopes to a state-dependent constant. Action value ordering is unchanged.
"""
import math

VERSION='own-stack-potential-v1'


def transition_reward(raw_reward,previous_stack,next_stack,starting_stack):
    if (type(starting_stack) is not int or starting_stack<=0 or type(previous_stack) is not int
            or previous_stack<0 or next_stack is not None and (type(next_stack) is not int or next_stack<0)
            or not math.isfinite(raw_reward)):
        raise ValueError('Finite reward and public integer stack counts required')
    before=(previous_stack-starting_stack)/starting_stack
    after=0.0 if next_stack is None else (next_stack-starting_stack)/starting_stack
    shaped=float(raw_reward)+after-before
    return {'raw_reward':float(raw_reward),'previous_potential':before,'next_potential':after,
            'shaped_reward':shaped,'terminal':next_stack is None,'discount':1.0,'schema':VERSION}
