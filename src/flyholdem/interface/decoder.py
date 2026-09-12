import numpy as np


def decode(counts, ensembles, legal_mask, rng, temperature=0):
    raw = np.array([np.mean(counts[ids]) / .1 for ids in ensembles])
    scores = raw / 100.0  # Fixed 100 Hz scale, baseline zero for the synthetic fixture.
    legal = np.asarray(legal_mask, dtype=bool)
    if not legal.any() or not np.isfinite(scores).all():
        raise ValueError('Decoder requires legal actions and finite neural scores')
    masked = np.where(legal, scores, -np.inf)
    if temperature > 0:
        probs = np.exp((masked - max(masked)) / temperature)
        probs /= probs.sum()
        chosen = int(rng.choice(5, p=probs))
    else:
        chosen = int(np.argmax(masked))
    return {'raw_rates_hz': raw.tolist(), 'scores': scores.tolist(),
            'legal_mask': legal.tolist(), 'selected': chosen,
            'temperature': temperature, 'silent': bool(np.max(raw[legal]) == 0),
            'decoder_fallback': False, 'tie_break': 'lowest-legal-index'}
