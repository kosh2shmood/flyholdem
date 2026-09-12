"""Fixed symbolic population code. No state object or opponent data accepted."""
import hashlib
import numpy as np

CARDS = [r + s for r in '23456789TJQKA' for s in 'cdhs']
SCALARS = ('pot', 'own_stack', 'opponent_stack', 'contribution', 'to_call', 'min_raise', 'max_raise', 'spr')
CHANNELS = ([f'hole:{c}' for c in CARDS] + [f'board:{c}' for c in CARDS]
    + [f'street:{i}' for i in range(4)] + ['position:BB', 'position:button']
    + [f'{name}:rbf:{i}' for name in SCALARS for i in range(5)]
    + [f'history:{slot}:{item}' for slot in range(8) for item in
       ['other', 'self', 'fold', 'call', 'half-pot', 'pot', 'all-in', 'small', 'medium', 'large']]
    + [f'legal:{i}' for i in range(5)] + ['begin-decision', 'begin-hand'])


def encode(obs):
    x = np.zeros(len(CHANNELS), dtype='<f8')
    for card in obs['hole']:
        x[CARDS.index(card)] = 1
    for card in obs['board']:
        x[52 + CARDS.index(card)] = 1
    x[104 + obs['street']] = 1
    x[108 + obs['position']] = 1
    for i, name in enumerate(SCALARS):
        value = obs[name] / (10 if name == 'spr' else 80)
        x[110 + i*5:115 + i*5] = np.exp(-((value - np.linspace(0, 1, 5)) / .25)**2)
    offset = 150
    for slot, h in enumerate(obs['history'][-8:]):
        start = offset + slot*10
        x[start + h['actor']] = 1
        x[start + 2 + h['action']] = 1
        x[start + 7 + min(2, int(h['pot_fraction'] * 2))] = 1
    x[-7:-2] = obs['legal_mask']
    x[-2] = 1
    x[-1] = not obs['history']
    return x


def projection(n_inputs=96, seed=1729):
    # Balanced round-robin targets, then a public seeded cell permutation.
    order = np.random.default_rng(seed).permutation(n_inputs)
    return np.array([[order[(2*i) % n_inputs], order[(2*i+1) % n_inputs]]
                     for i in range(len(CHANNELS))], dtype=np.int32)


def stimulus(x, mapping, n=126):
    drive = np.zeros(n)
    np.add.at(drive, mapping.ravel(), np.repeat(x, 2))
    drive[:96] = np.minimum(3.5, drive[:96] * 1.5)
    return drive


def encoded_hash(x):
    return hashlib.sha256(x.tobytes()).hexdigest()
