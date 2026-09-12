"""Explicit conventional policy kinds; never imported by frozen fly inference."""
import json
from pathlib import Path


def sampling(record):
    if record['schema']=='teacher-average-policy-v1':return 'frozen-average-policy-probabilities'
    if record['schema']=='teacher-best-response-policy-v1':return 'frozen-equal-mixture-of-legal-greedy-Q-policies'
    raise ValueError('Unsupported full-hand conventional policy kind')


def load_policy(path):
    record=json.loads((Path(path)/'manifest.json').read_text())
    sampling(record)
    if record['schema']=='teacher-average-policy-v1':
        from .policy import load_policy as load
    else:
        from .q_policy import load_policy as load
    return load(path)
