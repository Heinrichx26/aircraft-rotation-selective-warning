import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from paired_comparison import compare
from policy import METHODS, alerts, bound, calibrate, counts, scan


def constructed_scores(size, generator):
    profile = generator.permutation(np.tile(np.arange(24), size // 24))
    delayed = (profile >= 8) & ~np.isin(profile, [12, 16, 20])
    delay = np.where(delayed, 15 + profile * 4 + (profile % 3) * 10, profile % 15)
    tree = 0.08 + 0.035 * profile
    retrieval = (profile + (profile % 5) * 2) / 32
    severity = 30 + (profile * 7 % 23) * 3
    return pd.DataFrame(dict(warning_target=delayed.astype(int), next_dep_delay_min=delay,
        evidence_warning=(profile == 23).astype(int), tree_probability_calibrated=tree, light=tree,
        warning_probability=tree, risk_aware_utility_score=tree * severity, available_slack_min=100 * (23 - profile) / 23,
        uncertainty_probability_entropy=-tree * np.log(tree) - (1 - tree) * np.log(1 - tree),
        aggregated_utility=retrieval, disagreement=np.abs(tree - retrieval), gate_retrieval_bonus=(profile % 7) / 6))


def match(selection, screening, ceiling, budget, method):
    score = METHODS[method]
    eligible_rows = selection.loc[selection.evidence_warning.eq(0)].sort_values(score, ascending=False)
    issued = np.arange(1, len(eligible_rows) + 1)
    false = eligible_rows.warning_target.eq(0).cumsum().to_numpy()
    minutes = eligible_rows.next_dep_delay_min.where(eligible_rows.warning_target.eq(1), 0).cumsum().to_numpy()
    distinct = ~eligible_rows[score].duplicated(keep='last').to_numpy()
    candidates = np.flatnonzero(distinct & (false <= ceiling))
    limits = [bound(false[index], issued[index]) for index in candidates]
    candidates = candidates[np.asarray(limits) <= budget / 100]
    if not len(candidates):
        return None, None
    if method == 'tree':
        best = candidates[-1]
    else:
        best = candidates[np.lexsort((candidates, false[candidates], -minutes[candidates]))[0]]
    policy = {'method': method, 'threshold': float(eligible_rows[score].iloc[best])}
    screen_issued, screen_false, _ = map(int, counts(alerts(screening, policy), screening))
    limit = bound(screen_false, screen_issued)
    active = bool(limit <= budget / 100)
    record = dict(threshold=policy['threshold'], selection_issued=int(issued[best]), selection_false=int(false[best]),
                  screening_issued=screen_issued, screening_false=screen_false,
                  screening_limit=float(limit) if np.isfinite(limit) else None, passed=active)
    return policy if active else None, record


def cohort_comparison(frame, routed, comparator, days):
    qualifying = frame.next_dep_delay_min.where(frame.warning_target.eq(1), 0)
    rows = frame[['day', 'aircraft', 'airport_pair']].assign(delay_minutes=qualifying,
        routed_alert=routed.astype(int), tree_alert=comparator.astype(int))
    return dict(connections=len(rows), routed=list(map(int, counts(routed, frame))),
                comparator=list(map(int, counts(comparator, frame))), paired=compare(rows, days))


def example(size):
    generator = np.random.default_rng(42)
    selection, screening, test = [constructed_scores(size, generator) for _ in range(3)]
    days = 6
    test = test.assign(day=generator.integers(1, days + 1, size), aircraft=generator.integers(0, 30, size),
                       airport_pair=generator.integers(0, 8, size))
    policies, _ = calibrate(scan(selection), screening)
    results = []
    for budget in (15, 18):
        routed = policies['hybrid', budget]
        if routed is None:
            results.append(dict(budget=budget, routed_policy_passed=False))
            continue
        ceiling = int(routed['false'])
        routed_alerts = alerts(test, routed)
        methods = {}
        for method in ('tree', 'empirical'):
            policy, record = match(selection, screening, ceiling, budget, method)
            comparator_alerts = alerts(test, policy)
            early = test.available_slack_min.ge(15).to_numpy()
            methods[method] = dict(screen=record, all_connections=cohort_comparison(test, routed_alerts, comparator_alerts, days),
                scheduled_lead_at_least_15=cohort_comparison(test.loc[early], routed_alerts[early], comparator_alerts[early], days))
        results.append(dict(budget=budget, routed_policy_passed=True, selection_false_alert_ceiling=ceiling, methods=methods))
    return dict(constructed_connections_per_period=size, budgets=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare separately screened policies on constructed flight connections.')
    parser.add_argument('--size', type=int, choices=[240, 2400], default=2400)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(example(args.size), indent=2, allow_nan=False) + '\n', encoding='utf-8')
