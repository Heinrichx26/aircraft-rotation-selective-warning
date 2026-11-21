import argparse
import csv
import json
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def timeline(rows):
    result = defaultdict(list)
    for row in rows:
        if row['event'] not in {'assignment','alert','departed','cancelled','diverted'}:
            raise ValueError('Unknown flight event.')
        if not row['service'] or (row['event'] in {'assignment','alert','departed'} and not row['aircraft']):
            raise ValueError('Supply service and aircraft identifiers for assignment, alert and departure records.')
        stamp = datetime.fromisoformat(row['time'])
        if stamp.utcoffset() is None:
            raise ValueError('Each event needs a time zone.')
        result[row['service']].append((stamp, row['event'], row['aircraft']))
    return result


def follow(rows):
    counts = dict(confirmed=0, departed=0, unchanged=0, reassigned=0, same_at_departure=0,
                  cancelled=0, diverted=0, pending=0, awaiting_confirmation=0)
    for events in timeline(rows).values():
        assignments = sorted((stamp, craft) for stamp, kind, craft in events if kind == 'assignment')
        times = [stamp for stamp, craft in assignments]
        if len(times) != len(set(times)):
            raise ValueError('Assignment revisions need distinct timestamps.')
        outcomes = [(stamp,kind,craft) for stamp,kind,craft in events if kind in {'departed','cancelled','diverted'}]
        if len(outcomes) > 1:
            raise ValueError('Supply one final outcome per service.')
        for stamp, kind, inbound in events:
            if kind != 'alert':
                continue
            known = bisect_right(times, stamp)
            if known == 0 or assignments[known-1][1] != inbound:
                counts['awaiting_confirmation'] += 1
                continue
            counts['confirmed'] += 1
            if not outcomes:
                counts['pending'] += 1
                continue
            end, outcome, final_aircraft = outcomes[0]
            if end < stamp:
                raise ValueError('Outcome precedes the alert.')
            counts[outcome] += 1
            if outcome == 'departed':
                if not final_aircraft:
                    raise ValueError('A completed departure needs its aircraft identifier.')
                retained = final_aircraft == inbound and all(craft == inbound for time,craft in assignments[known:bisect_right(times,end)])
                counts['unchanged' if retained else 'reassigned'] += 1
                counts['same_at_departure'] += int(final_aircraft == inbound)
    assert counts['confirmed'] == sum(counts[key] for key in ['departed','cancelled','diverted','pending'])
    assert counts['departed'] == counts['unchanged'] + counts['reassigned']
    assert counts['unchanged'] <= counts['same_at_departure'] <= counts['departed']
    counts['retention_among_departures'] = counts['unchanged']/counts['departed'] if counts['departed'] else None
    return counts


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Follow confirmed aircraft assignments from the alert time to departure.')
    parser.add_argument('events',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with args.events.open(encoding='utf-8',newline='') as source:
        counts=follow(csv.DictReader(source))
    args.output.write_text(json.dumps(counts,indent=2)+'\n',encoding='utf-8')
