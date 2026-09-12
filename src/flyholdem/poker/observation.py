import hashlib
import json


def canonical_bytes(observation):
    return json.dumps(observation, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def information_id(observation):
    return hashlib.sha256(canonical_bytes(observation)).hexdigest()
