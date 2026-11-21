import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def interval(sums):
    valid = sums[:, 0] > 0
    values = 100 * sums[valid, 1] / sums[valid, 0]
    return {'valid': int(valid.sum()), 'interval_pp': np.quantile(values, [.025, .975]).tolist() if len(values) else None}


def compare(frame, days, repeats=1000, seed=42):
    required = ['day', 'aircraft', 'delay_minutes', 'routed_alert', 'tree_alert']
    if days < 3 or repeats < 1 or frame.empty or frame[required].isna().any().any():
        raise ValueError('Provide complete paired observations and at least three calendar days.')
    if not (frame.day.between(1, days) & frame.day.mod(1).eq(0)).all():
        raise ValueError('Days must be integer calendar positions in the observation period.')
    if not frame[['routed_alert', 'tree_alert']].isin([0, 1]).all().all():
        raise ValueError('Use binary alert indicators.')
    if not (np.isfinite(frame.delay_minutes) & frame.delay_minutes.ge(0)).all() or frame.delay_minutes.sum() == 0:
        raise ValueError('Provide nonnegative qualifying minutes and a positive total.')
    delta = frame.delay_minutes * (frame.routed_alert.astype(int) - frame.tree_alert.astype(int))
    work = frame.assign(delta=delta)
    columns = ['delay_minutes', 'delta']
    daily = work.groupby('day')[columns].sum().reindex(range(1, days + 1), fill_value=0).to_numpy()
    rng = np.random.default_rng(seed)
    results = {'denominator_minutes': float(frame.delay_minutes.sum()), 'difference_minutes': float(delta.sum()),
               'gain_pp': float(100 * delta.sum() / frame.delay_minutes.sum())}
    for width in (1, 3):
        starts = rng.integers(days, size=(repeats, (days + width - 1) // width))
        draw = ((starts[:, :, None] + np.arange(width)) % days).reshape(repeats, -1)[:, :days]
        results[f'days_{width}'] = interval(daily[draw].sum(axis=1))
    for column, cluster_seed in [('aircraft', seed), ('airport_pair', 314159)]:
        if column not in work:
            continue
        if work[column].isna().any():
            raise ValueError(f'Provide a {column} group for every connection.')
        clusters = work.groupby(column, sort=True)[columns].sum().to_numpy()
        rng = np.random.default_rng(cluster_seed)
        batches = []
        for start in range(0, repeats, 100):
            draw = rng.integers(len(clusters), size=(min(100, repeats - start), len(clusters)))
            batches.append(clusters[draw].sum(axis=1))
        results[column] = interval(np.concatenate(batches))
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Paired differences in captured delay shares, in percentage points.')
    parser.add_argument('connections', type=Path)
    parser.add_argument('--days', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(pd.read_csv(args.connections), args.days)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
