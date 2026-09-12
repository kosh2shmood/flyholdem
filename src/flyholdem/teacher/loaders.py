"""Explicit conventional policy kinds; never imported by frozen fly inference."""
import json
from pathlib import Path


def sampling(record):
    if record['schema']=='teacher-final-regret-policy-v1':return 'frozen-normalized-positive-final-regrets'
    if record['schema']=='teacher-external-regret-policy-v1':return 'frozen-reach-weighted-regret-average-fixed-population'
    if record['schema']=='teacher-average-policy-v1':return 'frozen-average-policy-probabilities'
    if record['schema']=='teacher-best-response-policy-v1':return 'frozen-equal-mixture-of-legal-greedy-Q-policies'
    if record['schema']=='teacher-potential-boundary-policy-v1':return 'frozen-Q-mixture-with-exact-stack-potential-fold-boundary'
    raise ValueError('Unsupported full-hand conventional policy kind')


def load_policy(path):
    record=json.loads((Path(path)/'manifest.json').read_text())
    sampling(record)
    if record['schema']=='teacher-external-regret-policy-v1':
        from .regret_policy import load_policy as load
    elif record['schema']=='teacher-final-regret-policy-v1':
        from .regret_current_policy import load_policy as load
    elif record['schema']=='teacher-average-policy-v1':
        from .policy import load_policy as load
    elif record['schema']=='teacher-potential-boundary-policy-v1':
        from .potential_policy import load_policy as load
    else:
        from .q_policy import load_policy as load
    return load(path)
