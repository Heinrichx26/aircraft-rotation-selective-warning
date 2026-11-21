import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from expert_example import fit_and_score
from daily_workload import match_daily_volume
from paired_comparison import compare
from policy import alerts, calibrate, performance, scan
from public_linkage import read_archive


WINDOWS = (('01', '06'), ('06', '11'), ('11', '15'), ('15', '20'), ('20', '25'), ('25', '32'))


def connections(source):
    flights, valid, verified, _, _ = read_archive(source, 31, include_carrier=True)
    frame = flights.loc[valid & verified & flights.UniqueCarrier.notna()].sort_values(['TailNum', 'departed', 'id'])
    groups = frame.groupby('TailNum', sort=False)
    following = groups[['Origin', 'Dest', 'UniqueCarrier', 'planned', 'departed', 'DepDelay']].shift(-1).add_prefix('next_')
    linked = frame.join(following)
    minutes = lambda duration: (duration.dt.total_seconds() / 60).round()
    actual_ground_time = minutes(linked.next_departed - linked.landed)
    linked = linked.loc[linked.Dest.eq(linked.next_Origin) & actual_ground_time.between(1, 720)]
    slack = minutes(linked.next_planned - linked.landed).astype(int)
    return pd.DataFrame(dict(
        decision_time=linked.landed, outcome_time=linked.next_departed,
        aircraft=linked.TailNum, airport_pair=linked.next_Origin + '>' + linked.next_Dest,
        next_airline=linked.next_UniqueCarrier, next_dep_airport=linked.next_Origin, next_arr_airport=linked.next_Dest,
        route_key=linked.next_Origin + '>' + linked.next_Dest, decision_hour=linked.landed.dt.hour,
        decision_dow=linked.landed.dt.dayofweek, prev_dep_delay_min=linked.DepDelay,
        prev_arr_delay_min=linked.ArrDelay, scheduled_gap_min=minutes(linked.next_planned - linked.planned_arr),
        available_slack_min=slack, inherited_lb_min=(-slack).clip(lower=0), evidence_warning=slack.le(-15).astype(int),
        warning_target=linked.next_DepDelay.ge(15).astype(int), next_dep_delay_min=linked.next_DepDelay.astype(int)
    )).sort_values('decision_time', kind='stable')


def run(source, *, smoke=False):
    frame = connections(source)
    assert frame.index.is_unique and frame.outcome_time.gt(frame.decision_time).all()
    assert frame.loc[frame.evidence_warning.eq(1), 'warning_target'].eq(1).all()
    parts = []
    for index, (start, stop) in enumerate(WINDOWS):
        left = pd.Timestamp(2008, 1, int(start), tz='UTC')
        right = pd.Timestamp(2008, 1, 1, tz='UTC') + pd.Timedelta(days=int(stop) - 1)
        selected = frame.decision_time.ge(left) & frame.decision_time.lt(right)
        if index < len(WINDOWS) - 1:
            selected &= frame.outcome_time.lt(right)
        sample = frame.loc[selected].copy()
        if index in (1, 2):
            sample = sample.loc[sample.evidence_warning.eq(0)]
        if smoke or index in (1, 2):
            cap = 320 if smoke else {1: 32768, 2: 8192}[index]
            sample = sample.sample(min(cap, len(sample)), random_state=42).reset_index(drop=True)
        parts.append(sample)

    (selection, screening, test), train_size, gate_size, _ = fit_and_score(parts, threads=1 if smoke else 4)
    candidates = scan(selection, component_models_only=True)
    policies, records = calibrate(candidates, screening)
    frontier = []
    daily_volume_frontier = []
    for row in records.loc[records.method.isin(('hybrid', 'retrieval', 'tree'))].itertuples(index=False):
        decision = alerts(test, policies[row.method, row.budget_percent])
        outcome = performance(test, decision)
        frontier.append(dict(budget=int(row.budget_percent), policy=row.method, passed=bool(row.active),
                             screen_issued=int(row.calibration_alerts), screen_false=int(row.calibration_false),
                             issued=outcome['model_alerts'], false=outcome['model_false'],
                             model_minutes=outcome['model_minutes'], rule_minutes=outcome['rule_minutes']))
    for budget in range(4, 25):
        routed_policy = policies['hybrid', budget]
        if routed_policy is None:
            continue
        routed = alerts(test, routed_policy)
        eligible = test.loc[test.evidence_warning.eq(0)].reset_index(drop=True)
        routed_eligible = routed[test.evidence_warning.eq(0).to_numpy()]
        paired_rows = []
        for method, column in [('tree', 'tree_probability_calibrated'),
                               ('severity', 'aggregated_utility')]:
            comparator = match_daily_volume(eligible.decision_time.dt.day.to_numpy(),
                                            routed_eligible, eligible[column].to_numpy())
            delayed_minutes = np.where(eligible.warning_target.eq(1),
                                       eligible.next_dep_delay_min, 0)
            paired = pd.DataFrame(dict(
                day=eligible.decision_time.dt.day,
                aircraft=eligible.aircraft,
                airport_pair=eligible.airport_pair,
                delay_minutes=delayed_minutes,
                routed_alert=routed_eligible.astype(int),
                tree_alert=comparator.astype(int)))
            interval = compare(paired, 31)
            routed_minutes = int(delayed_minutes[routed_eligible].sum())
            comparator_minutes = int(delayed_minutes[comparator].sum())
            routed_false = int((routed_eligible & eligible.warning_target.eq(0).to_numpy()).sum())
            comparator_false = int((comparator & eligible.warning_target.eq(0).to_numpy()).sum())
            paired_rows.append(dict(method=method, alerts=int(routed_eligible.sum()),
                                    false=[routed_false, comparator_false],
                                    delay_minutes=[routed_minutes, comparator_minutes],
                                    difference_minutes=interval['difference_minutes'],
                                    capture_difference_pp=interval['gain_pp'],
                                    three_day_interval_pp=interval['days_3']['interval_pp']))
        daily_volume_frontier.append(dict(budget=budget, comparators=paired_rows))
    return dict(decision_clock='UTC', sample_sizes=list(map(len, parts)), expert_fit_rows=train_size, gate_fit_rows=gate_size,
                test_rule_alerts=int(test.evidence_warning.sum()), test_delay_minutes=int(test.next_dep_delay_min.where(test.warning_target.eq(1), 0).sum()),
                budgets=frontier, daily_volume_matched=daily_volume_frontier)


def main():
    parser = argparse.ArgumentParser(description='Run temporal model training and warning evaluation on the public January 2008 archive.')
    parser.add_argument('archive', type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output', required=True, type=Path)
    arguments = parser.parse_args()
    result = run(arguments.archive, smoke=arguments.smoke)
    arguments.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f"Scored {result['sample_sizes'][-1]:,} test connections.")


if __name__ == '__main__':
    main()
