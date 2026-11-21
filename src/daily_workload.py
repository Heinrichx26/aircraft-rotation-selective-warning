import numpy as np


def match_daily_volume(days, reference, score):
    days = np.asarray(days)
    reference = np.asarray(reference)
    score = np.asarray(score, dtype=float)
    if days.ndim != 1 or reference.ndim != 1 or score.ndim != 1:
        raise ValueError('Supply one-dimensional daily groups, alerts and scores.')
    if not (len(days) == len(reference) == len(score)) or not len(days):
        raise ValueError('Daily groups, alerts and scores must have the same positive length.')
    if pd_missing(days).any() or not np.isfinite(score).all():
        raise ValueError('Supply complete daily groups and finite scores.')
    if not np.isin(reference, [False, True, 0, 1]).all():
        raise ValueError('Reference alerts must be binary.')
    reference = reference.astype(bool)
    matched = np.zeros(len(days), dtype=bool)
    for group in np.unique(days):
        indices = np.flatnonzero(days == group)
        volume = int(reference[indices].sum())
        order = np.argsort(-score[indices], kind='stable')
        matched[indices[order[:volume]]] = True
        if int(matched[indices].sum()) != volume:
            raise AssertionError('The comparator must match the daily reference volume.')
    return matched


def pd_missing(values):
    if np.issubdtype(values.dtype, np.number):
        return ~np.isfinite(values)
    return np.asarray([value is None for value in values], dtype=bool)
