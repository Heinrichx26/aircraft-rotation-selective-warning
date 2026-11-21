import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from daily_workload import match_daily_volume
from paired_comparison import compare


def run(source):
    frame = pd.read_csv(source)
    required = {'day', 'aircraft', 'airport_pair', 'routed_alert', 'tree_score',
                'severity_score', 'delay_minutes'}
    if not required.issubset(frame.columns):
        raise ValueError(f'Missing columns: {sorted(required - set(frame.columns))}')
    if frame[list(required)].isna().any().any():
        raise ValueError('The constructed example requires complete records.')
    days = frame.day.to_numpy()
    reference = frame.routed_alert.to_numpy()
    rows = {}
    for name in ('tree', 'severity'):
        matched = match_daily_volume(days, reference, frame[f'{name}_score'].to_numpy())
        table = frame.assign(tree_alert=matched)
        paired = table[['day', 'aircraft', 'airport_pair', 'delay_minutes', 'routed_alert', 'tree_alert']]
        stats = compare(paired, int(frame.day.max()))
        routed_minutes = int(frame.loc[reference.astype(bool), 'delay_minutes'].sum())
        comparison_minutes = int(frame.loc[matched, 'delay_minutes'].sum())
        rows[name] = dict(alerts=[int(reference.sum()), int(matched.sum())],
                          false=[int(((reference == 1) & frame.delay_minutes.eq(0)).sum()),
                                 int((matched & frame.delay_minutes.eq(0).to_numpy()).sum())],
                          delay_minutes=[routed_minutes, comparison_minutes],
                          paired=stats)
    return dict(connections=len(frame), days=int(frame.day.max()), comparisons=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('connections', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    arguments = parser.parse_args()
    result = run(arguments.connections)
    arguments.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f"Compared {len(result['comparisons'])} rankings at identical daily alert volumes.")
