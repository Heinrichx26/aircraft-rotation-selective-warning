"""Selection, screening and monitoring of alert policies used in the article.

Selection evaluates every candidate policy on one period of outcomes. For each
method it keeps a frontier of candidates ordered by their one-sided selection
limit. Screening applies the fixed-sequence test along that frontier on an
independent later period, with a per-budget tail probability of 0.05 or a joint
tail probability of 0.05/21 across the 21 budgets. Operating ceilings also check
one-, three- and five-day block percentiles. Weekly checks pause model alerts
when a corrected limit exceeds the ceiling and select a replacement on the
preceding 30 days.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((ROOT / 'config' / 'article_protocol.json').read_text(encoding='utf-8'))
METHODS = {'tree': 'tree_probability_calibrated', 'light': 'light', 'probability': 'warning_probability',
           'empirical': 'risk_aware_utility_score', 'slack': 'available_slack_min', 'confidence': 'warning_probability'}
SYSTEMS = [*METHODS, 'retrieval', 'expert', 'hybrid']
OFFSETS = [np.inf if value == 'probability' else float(value) for value in PROTOCOL['tree_utility_offsets']]
FINE = np.unique(np.r_[PROTOCOL['candidate_quantiles'], np.linspace(.5, .999, PROTOCOL['single_score_points'])])
BUDGETS = range(PROTOCOL['budgets_percent'][0], PROTOCOL['budgets_percent'][1] + 1)
JOINT = 0.05 / len(BUDGETS)
CHECK_DAYS = tuple(PROTOCOL['monitoring']['check_days'])


def tree_utility(frame, offset):
    probability = frame.tree_expert_probability.to_numpy()
    if offset is None or not np.isfinite(offset):
        return probability
    return probability * (frame.tree_expert_severity.to_numpy() + offset)


def retrieval_utility(frame, weight, offset=0.0):
    base = (frame.aggregated_utility - weight * frame.disagreement).to_numpy()
    if offset is None or offset == 0:
        return base
    probability = frame.retrieval_probability.to_numpy()
    return probability if not np.isfinite(offset) else base + offset * probability


def limit(false, total, delta=0.05):
    """One-sided Clopper-Pearson upper limit with tail probability `delta`."""
    return float(beta.ppf(1 - delta, false + 1, total - false)) if 0 <= false < total else (1.0 if total else np.nan)


def limits(false, total, delta=0.05):
    false = np.asarray(false, dtype=float)
    total = np.asarray(total, dtype=float)
    out = np.where(total > 0, 1.0, np.nan)
    ok = (false >= 0) & (false < total)
    out[ok] = beta.ppf(1 - delta, false[ok] + 1, total[ok] - false[ok])
    return out


def thresholds(score):
    return np.unique(np.quantile(score, PROTOCOL['candidate_quantiles']))


def fine_thresholds(score):
    return np.unique(np.quantile(score, FINE))


def branch(score, region, target, minutes):
    rows = [(np.inf, 0, 0, 0)]
    if region.any():
        for threshold in thresholds(score[region]):
            mask = region & (score >= threshold)
            rows.append((threshold, mask.sum(), (mask & ~target).sum(), minutes[mask & target].sum()))
    return rows


def counts(mask, frame):
    target = frame.warning_target.to_numpy().astype(bool)
    return np.array([mask.sum(), (mask & ~target).sum(), frame.next_dep_delay_min.to_numpy()[mask & target].sum()], dtype='int64')


def single(score, target, minutes, grid=fine_thresholds):
    levels = grid(score)
    order = np.argsort(-score, kind='stable')
    cum_false = np.r_[0, np.cumsum(~target[order])]
    cum_minutes = np.r_[0, np.cumsum(np.where(target, minutes, 0)[order])]
    taken = np.searchsorted(-score[order], -levels, side='right')
    return levels, taken, cum_false[taken], cum_minutes[taken]


def inside_parts(tree, ret, region, target, minutes):
    rows = [(np.inf, np.inf, 0, 0, 0)]
    if region.any():
        for inner in np.r_[-np.inf, thresholds(tree[region])]:
            base = region & (tree >= inner)
            for cut in np.r_[-np.inf, thresholds(ret[region])]:
                mask = base & (ret >= cut)
                rows.append((inner, cut, mask.sum(), (mask & ~target).sum(), minutes[mask & target].sum()))
    return np.array(rows, dtype=float)


def scan(frame, systems=SYSTEMS):
    """Alert, false-alert and delay-minute counts of every candidate policy on `frame`."""
    work = frame.loc[frame.evidence_warning.eq(0)]
    target = work.warning_target.to_numpy().astype(bool)
    minutes = work.next_dep_delay_min.to_numpy().astype(float)
    blocks = []

    def add(method, n, false, mins, **policy):
        block = pd.DataFrame({'n': np.asarray(n, dtype='int64'), 'false': np.asarray(false, dtype='int64'),
                              'minutes': np.asarray(mins, dtype='int64'),
                              **{key: np.broadcast_to(value, np.shape(n)) for key, value in policy.items()}})
        block.insert(0, 'method', method)
        blocks.append(block)

    for method, col in METHODS.items():
        if method not in systems:
            continue
        score = work[col].to_numpy() * (-1 if method == 'slack' else 1)
        if method == 'confidence':
            for cutoff in PROTOCOL['entropy_cutoffs']:
                kept = np.where(work.uncertainty_probability_entropy.to_numpy() <= cutoff, score, -np.inf)
                levels = fine_thresholds(score)
                fired = [(kept >= level) for level in levels]
                add(method, [m.sum() for m in fired], [(m & ~target).sum() for m in fired],
                    [minutes[m & target].sum() for m in fired], threshold=levels, entropy=cutoff)
        else:
            levels, taken, false, mins = single(score, target, minutes)
            add(method, taken, false, mins, threshold=levels)
    if 'retrieval' in systems:
        for weight in PROTOCOL['disagreement_weights']:
            levels, taken, false, mins = single(retrieval_utility(work, weight), target, minutes)
            add('retrieval', taken, false, mins, threshold=levels, weight=weight, retrieval_offset=0.0)
    if 'expert' in systems:
        for offset in OFFSETS:
            levels, taken, false, mins = single(tree_utility(work, offset), target, minutes)
            add('expert', taken, false, mins, threshold=levels, offset=offset)
    if 'hybrid' in systems:
        gates = work.gate_retrieval_bonus.to_numpy()
        for cut in np.unique(np.quantile(gates, PROTOCOL['gate_quantiles'])):
            route = gates >= cut
            for offset in OFFSETS:
                tree = tree_utility(work, offset)
                outside = np.array(branch(tree, ~route, target, minutes), dtype=float)
                for weight in PROTOCOL['disagreement_weights']:
                    inside = inside_parts(tree, retrieval_utility(work, weight), route, target, minutes)
                    n = outside[:, 1][:, None] + inside[:, 2][None, :]
                    keep = n.ravel() > 0
                    pick = lambda values: values.ravel()[keep]
                    add('hybrid', pick(n), pick(outside[:, 2][:, None] + inside[:, 3][None, :]),
                        pick(outside[:, 3][:, None] + inside[:, 4][None, :]), gate=cut,
                        threshold=pick(np.repeat(outside[:, 0][:, None], len(inside), 1)),
                        inner_threshold=pick(np.repeat(inside[:, 0][None, :], len(outside), 0)),
                        retrieval_threshold=pick(np.repeat(inside[:, 1][None, :], len(outside), 0)),
                        weight=weight, offset=offset, retrieval_offset=0.0)
    candidates = pd.concat(blocks, ignore_index=True)
    candidates = candidates.loc[candidates.n > 0].reset_index(drop=True)
    candidates['upper'] = limits(candidates.false, candidates.n, PROTOCOL['selection']['selection_limit_tail'])
    return candidates


def alerts(frame, row):
    """Model alerts of one policy; the timing rule covers the remaining connections."""
    if row is None:
        return np.zeros(len(frame), dtype=bool)
    method = row['method']
    if method in METHODS:
        value = frame[METHODS[method]].to_numpy() * (-1 if method == 'slack' else 1)
        fired = value >= row['threshold']
        if method == 'confidence':
            fired &= frame.uncertainty_probability_entropy.to_numpy() <= row['entropy']
    elif method == 'expert':
        fired = tree_utility(frame, row.get('offset', np.inf)) >= row['threshold']
    else:
        ret = retrieval_utility(frame, row['weight'], row.get('retrieval_offset', 0.0))
        if method == 'retrieval':
            fired = ret >= row['threshold']
        else:
            route = frame.gate_retrieval_bonus.to_numpy() >= row['gate']
            tree = tree_utility(frame, row.get('offset', np.inf))
            inner = row.get('inner_threshold', -np.inf)
            fired = np.where(route, (ret >= row['retrieval_threshold']) & (tree >= inner), tree >= row['threshold'])
    return fired & frame.evidence_warning.eq(0).to_numpy()


def frontier(candidates, method, cap=None):
    """Candidates ordered by selection limit, kept only when delay minutes strictly increase."""
    part = candidates.loc[candidates.method == method]
    if cap is not None:
        part = part.loc[part.upper <= cap]
    part = part.sort_values(['upper', 'minutes', 'false'], ascending=[True, False, True], kind='stable')
    best, keep = -1, []
    for index, minutes in zip(part.index, part.minutes.to_numpy()):
        if minutes > best:
            keep.append(index)
            best = minutes
    return part.loc[keep]


def fixed_sequence(candidates, screen, method='hybrid', budgets=BUDGETS, delta=0.05):
    """Screen the frontier in order and stop at the first limit above each budget."""
    path = frontier(candidates, method, max(budgets) / 100)
    rows = [row.dropna().to_dict() for _, row in path.iterrows()]
    checks = [counts(alerts(screen, row), screen) for row in rows]
    result = {}
    for per in budgets:
        usable = int((path.upper <= per / 100).sum())
        passed = 0
        while passed < usable and limit(checks[passed][1], checks[passed][0], delta) <= per / 100:
            passed += 1
        row = rows[passed - 1] if passed else None
        total, false, minutes = checks[passed - 1] if passed else (0, 0, 0)
        result[per] = dict(policy=row, sequence_length=usable, sequence_passed=passed, screen_alerts=int(total),
                           screen_false=int(false), screen_minutes=int(minutes), screen_limit=limit(false, total, delta))
    return result


def block_percentile(frame, mask, width, seed=42, quantile=0.95, replicates=1000):
    """Percentile of the error fraction over circular blocks of `width` operating days."""
    target = frame.warning_target.to_numpy(dtype=bool)
    daily = pd.DataFrame({'day': frame.landed.dt.normalize(), 'n': mask, 'false': mask & ~target}).groupby('day').sum()
    days = len(daily)
    if days == 0:
        return np.nan
    generator = np.random.default_rng(seed)
    draw = ((generator.integers(days, size=(replicates, (days + width - 1) // width))[:, :, None] + np.arange(width)) % days)
    draw = draw.reshape(replicates, -1)[:, :days]
    n = daily.n.to_numpy()[draw].sum(1)
    false = daily['false'].to_numpy()[draw].sum(1)
    valid = n > 0
    return float(np.quantile(false[valid] / n[valid], quantile)) if valid.any() else np.nan


def ceiling_policy(selection, screen, ceiling, delta=JOINT, candidates=None):
    """Budget point admitted under an operating ceiling with the most selection delay minutes."""
    candidates = scan(selection, systems=['hybrid']) if candidates is None else candidates
    path = frontier(candidates, 'hybrid', max(BUDGETS) / 100)
    rows = [row.dropna().to_dict() for _, row in path.iterrows()]
    checks = [counts(alerts(screen, row), screen) for row in rows]
    best = None
    for point in BUDGETS:
        usable = int((path.upper <= point / 100).sum())
        passed = 0
        while passed < usable and limit(checks[passed][1], checks[passed][0], delta) <= point / 100:
            passed += 1
        if not passed:
            continue
        row = rows[passed - 1]
        total, false, _ = checks[passed - 1]
        mask = alerts(screen, row)
        worst = max([limit(false, total, delta)] + [block_percentile(screen, mask, width) for width in PROTOCOL['screening']['block_days']])
        if worst <= ceiling / 100 and (best is None or (row['minutes'], -row['false']) > (best[1]['minutes'], -best[1]['false'])):
            best = (point, row, worst)
    return best


def performance(frame, model):
    """Alert counts, false alerts and captured share of qualifying delay minutes."""
    target = frame.warning_target.to_numpy(dtype=bool)
    rule = frame.evidence_warning.to_numpy(dtype=bool)
    delay = np.where(target, frame.next_dep_delay_min, 0)
    fired = rule | model
    ratio = lambda top, bottom: top / bottom if bottom else np.nan
    return dict(connections=len(frame), model_alerts=int(model.sum()), model_false=int((model & ~target).sum()),
                model_error=ratio((model & ~target).sum(), model.sum()), rule_alerts=int(rule.sum()),
                alerts_per_thousand=ratio(1000 * fired.sum(), len(frame)), qualifying_minutes=int(delay.sum()),
                rule_minutes=int(delay[rule].sum()), model_minutes=int(delay[model].sum()),
                captured_share=ratio(delay[fired].sum(), delay.sum()))


def weekly_replay(data, pool, policy, ceiling, months, check_days=CHECK_DAYS, tail=0.05 / 4, window_days=30):
    """Weekly pause checks and 30-day replacement for one operating ceiling.

    `data` holds the scored replay months, `pool` every scored record available for
    replacement (the selection and screening periods and the replay months), and
    `months` a list of (year, month) pairs.
    """
    target = data.warning_target.to_numpy(dtype=bool)
    landed = data.landed.to_numpy()
    departed = data.next_departed.to_numpy()
    start = pd.Timestamp(*months[0], 1)
    issued = np.zeros(len(data), dtype=bool)
    events = []
    half = pd.Timedelta(days=window_days // 2)
    for year, month in months:
        for day in check_days:
            check = pd.Timestamp(year, month, day)
            left = max(start, pd.Timestamp(year, month, 1))
            live = (landed >= np.datetime64(start)) & (landed < np.datetime64(check))
            fired = alerts(data, policy) & live if policy is not None else np.zeros(len(data), dtype=bool)
            seen = fired & (landed >= np.datetime64(left)) & (departed < np.datetime64(check))
            n, k = int(seen.sum()), int((seen & ~target).sum())
            upper = float(beta.ppf(1 - tail, k + 1, max(n - k, 1))) if n > 0 else np.nan
            paused = policy is None or (n > 0 and upper > ceiling / 100)
            if not paused:
                continue
            issued |= fired
            middle = check - half
            selection = pool.loc[(pool.landed >= check - 2 * half) & (pool.landed < middle) & (pool.next_departed < middle)]
            screen = pool.loc[(pool.landed >= middle) & (pool.landed < check) & (pool.next_departed < check)]
            clock = time.monotonic()
            found = ceiling_policy(selection, screen, ceiling) if len(selection) and len(screen) else None
            events.append(dict(ceiling=ceiling, check=str(check.date()), checked_alerts=n, false_alerts=k,
                               upper=upper, replacement_point=None if found is None else found[0],
                               seconds=round(time.monotonic() - clock, 3)))
            policy = None if found is None else found[1]
            start = check
    if policy is not None:
        issued |= alerts(data, policy) & (landed >= np.datetime64(start))
    monthly = []
    for year, month in months:
        part = ((data.landed.dt.year == year) & (data.landed.dt.month == month)).to_numpy()
        result = performance(data.loc[part], issued[part])
        monthly.append(dict(ceiling=ceiling, year=year, month=month, **result))
    return issued, events, monthly


def review_groups(frame):
    return {'all': np.ones(len(frame), dtype=bool),
            'at_least_15_minutes': frame.available_slack_min.ge(15).to_numpy(),
            'at_least_25_minutes': frame.available_slack_min.ge(25).to_numpy()}


def daily_block_percentiles(frame, fired, quantiles, block_days, seed=42, replicates=1000):
    days = pd.date_range(frame.landed.min().normalize(), frame.landed.max().normalize())
    daily = pd.DataFrame(dict(day=frame.landed.dt.normalize(), alerts=fired, errors=fired & frame.warning_target.eq(0)))
    grouped = daily.groupby('day').sum().reindex(days, fill_value=0).to_numpy()
    size = len(days)
    generator = np.random.default_rng(seed)
    starts = generator.integers(0, size, size=(replicates, int(np.ceil(size / block_days))))
    indices = ((starts[:, :, None] + np.arange(block_days)) % size).reshape(replicates, -1)[:, :size]
    draws = grouped[indices].sum(axis=1)
    valid = draws[:, 0] > 0
    return (np.quantile(draws[valid, 1] / draws[valid, 0], quantiles)).tolist() if valid.any() else None


def window_policies(candidates, selection, screening, ceilings=(15, 18), budgets=BUDGETS, window=25):
    """Review-window policies screened on three nested lead-time groups with tail 0.05/63."""
    comparisons = len(budgets) * 3
    in_window = selection.available_slack_min.ge(window).to_numpy()
    delay = selection.warning_target.eq(1).to_numpy()
    screened = []
    for budget in budgets:
        eligible = candidates.loc[(candidates.method == 'hybrid') & candidates.upper.le(budget / 100)]
        eligible = eligible.sort_values(['minutes', 'false'], ascending=[False, True], kind='stable')
        if eligible.empty:
            continue
        policy = eligible.iloc[0].dropna().to_dict()
        fired = alerts(screening, policy)
        groups = {}
        for name, group in review_groups(screening).items():
            issued = int((fired & group).sum())
            false = int((fired & group & screening.warning_target.eq(0).to_numpy()).sum())
            part = screening.loc[group].reset_index(drop=True)
            blocks = [daily_block_percentiles(part, fired[group], [.95], width)[0] if issued else 1.0
                      for width in PROTOCOL['screening']['block_days']]
            groups[name] = dict(alerts=issued, false=false, joint_limit=limit(false, issued, 0.05 / comparisons) if issued else 1.0,
                                block_95th=max(blocks))
        chosen = alerts(selection, policy) & in_window
        screened.append(dict(budget=budget, policy=policy, groups=groups,
                             selection_window_minutes=int(selection.loc[chosen & delay, 'next_dep_delay_min'].sum())))
    result = {}
    for ceiling in ceilings:
        admitted = [row for row in screened if all(group['alerts'] and group['joint_limit'] <= ceiling / 100
                                                   and group['block_95th'] <= ceiling / 100 for group in row['groups'].values())]
        result[ceiling] = max(admitted, key=lambda row: (row['selection_window_minutes'], -row['budget']), default=None)
    return result


def window_checks(frame, fired, ceiling, check_dates, tail=0.05 / 12):
    """Weekly checks of a review-window policy on its three lead-time groups."""
    rows = []
    for check in check_dates:
        mature = (frame.landed.lt(check) & frame.next_departed.lt(check)).to_numpy()
        groups = {}
        exceeds = False
        for name, mask in review_groups(frame).items():
            observed = fired & mask & mature
            issued = int(observed.sum())
            false = int((observed & frame.warning_target.eq(0).to_numpy()).sum())
            upper = limit(false, issued, tail) if issued else None
            groups[name] = dict(alerts=issued, false=false, upper=upper)
            exceeds |= upper is not None and upper > ceiling / 100
        rows.append(dict(check=str(pd.Timestamp(check).date()), groups=groups, paused=bool(exceeds)))
        if exceeds:
            break
    return rows
