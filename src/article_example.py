"""End-to-end run of the article pipeline on a constructed calendar.

The constructed network has fictional carriers, airports, aircraft and flight
numbers that repeat a daily rotation pattern. The run builds decision-time
features, fits both experts, the probability calibration and the gate, selects
and screens policies at all 21 budgets, replays weekly checks over two months
and screens the review-window policies. Tree counts and training epochs are
reduced so that the run finishes in a few minutes on a desktop computer.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import policy_screen as ps
from article_experts import ArticleModels
from article_features import add_live_state, attach_flight_memory, flight_events, flight_operations

ROOT = Path(__file__).resolve().parents[1]
CALENDAR = {'history': ('2030-01-01', '2030-01-22'), 'experts': ('2030-01-22', '2030-02-19'),
            'gate_and_calibration': ('2030-02-19', '2030-03-05'), 'selection': ('2030-03-05', '2030-03-19'),
            'screening': ('2030-03-19', '2030-04-01'), 'replay': ('2030-04-01', '2030-06-01')}
REPLAY_MONTHS = [(2030, 4), (2030, 5)]
LEGS = 5


def constructed_flights(first, last, aircraft, seed=7):
    """Daily rotations of fictional aircraft with propagated delay."""
    generator = np.random.default_rng(seed)
    airports = np.array([f'Port{index:02d}' for index in range(10)])
    weights = np.array([.22, .16, .12, .10, .10, .08, .07, .06, .05, .04])
    carriers = np.array(['CarrierA', 'CarrierB', 'CarrierC'])
    block = generator.integers(55, 150, size=(10, 10))
    block = (block + block.T) // 2
    stops = np.zeros((aircraft, LEGS + 1), dtype=int)
    for plane in range(aircraft):
        stops[plane, 0] = generator.choice(10, p=weights)
        for leg in range(LEGS):
            following = generator.choice(10, p=weights)
            while following == stops[plane, leg]:
                following = generator.choice(10, p=weights)
            stops[plane, leg + 1] = following
    first_departure = generator.integers(360, 540, size=aircraft)
    turn = generator.integers(35, 65, size=(aircraft, LEGS))
    chronic = generator.normal(1, 9, size=(aircraft, LEGS))
    carrier_of = carriers[np.arange(aircraft) % 3]
    rows = []
    for day in pd.date_range(first, last, inclusive='left'):
        start = generator.integers(0, 24, size=10)
        strength = np.where(generator.random(10) < 0.3, generator.exponential(40, size=10), 0.0)
        carrier_day = dict(zip(carriers, generator.normal(0, 3, size=3)))
        planned = first_departure.astype(float)
        previous_arrival = None
        for leg in range(LEGS):
            origin, destination = stops[:, leg], stops[:, leg + 1]
            minutes = block[origin, destination]
            hour = (planned // 60) % 24
            disturbance = strength[origin] * ((hour >= start[origin]) & (hour < start[origin] + 5))
            noise = generator.gamma(1.3, 7.5, size=aircraft) - 5.0
            effect = np.array([carrier_day[carrier] for carrier in carrier_of])
            ready = planned + noise + disturbance + chronic[:, leg] + effect + 0.15 * (hour - 12).clip(min=0)
            if previous_arrival is not None:
                ready = np.maximum(ready, previous_arrival + 25)
            departed = np.maximum(ready, planned - 5).round()
            arrival_hour = ((departed + minutes) // 60) % 24
            landing_delay = strength[destination] * 0.4 * ((arrival_hour >= start[destination]) & (arrival_hour < start[destination] + 5))
            landed = (departed + minutes + generator.normal(0, 5, size=aircraft) + landing_delay).round()
            for plane in range(aircraft):
                rows.append((f'{day:%Y%m%d}-{plane:02d}-{leg}', f'REG{plane:02d}', carrier_of[plane],
                             f'{carrier_of[plane][-1]}{100 + plane * LEGS + leg}', airports[origin[plane]],
                             airports[destination[plane]], day + pd.Timedelta(minutes=float(planned[plane])),
                             day + pd.Timedelta(minutes=float(departed[plane])),
                             day + pd.Timedelta(minutes=float(planned[plane] + minutes[plane])),
                             day + pd.Timedelta(minutes=float(landed[plane])), plane, leg, day))
            previous_arrival = landed
            planned = planned + minutes + turn[:, leg]
    return pd.DataFrame(rows, columns=['id', 'reg', 'airline', 'flight', 'dep', 'arr', 'planned', 'departed', 'planned_arr',
                                       'landed', 'plane', 'leg', 'day'])


def connections(flights, tau=15):
    """Same-aircraft connections from each landing to the next departure."""
    inbound = flights.loc[flights.leg < LEGS - 1].reset_index(drop=True)
    onward = flights.set_index(['day', 'plane', 'leg']).loc[list(zip(inbound.day, inbound.plane, inbound.leg + 1))].reset_index()
    minutes = lambda later, earlier: ((later - earlier).dt.total_seconds() // 60).astype(int)
    frame = pd.DataFrame({
        'link_key': inbound.id, 'reg': inbound.reg, 'landed': inbound.landed, 'next_departed': onward.departed,
        'next_id': onward.id, 'next_flight': onward.flight, 'next_airline': onward.airline,
        'next_dep_airport': onward.dep, 'next_arr_airport': onward.arr,
        'prev_dep_delay_min': minutes(inbound.departed, inbound.planned),
        'prev_arr_delay_min': minutes(inbound.landed, inbound.planned_arr),
        'scheduled_gap_min': minutes(onward.planned, inbound.planned_arr),
        'available_slack_min': minutes(onward.planned, inbound.landed),
        'next_dep_delay_min': minutes(onward.departed, onward.planned)})
    frame['inherited_lb_min'] = (-frame.available_slack_min).clip(lower=0)
    frame['decision_hour'] = frame.landed.dt.hour
    frame['decision_dow'] = frame.landed.dt.dayofweek
    frame['decision_month'] = frame.landed.dt.strftime('%Y%m')
    frame['decision_day'] = frame.landed.dt.day
    frame['route_key'] = frame.next_dep_airport + '>' + frame.next_arr_airport
    frame['warning_target'] = (frame.next_dep_delay_min >= tau).astype(int)
    frame['evidence_warning'] = (frame.inherited_lb_min >= tau).astype(int)
    return frame.sort_values(['landed', 'link_key']).reset_index(drop=True)


def period(frame, name, outcomes=True):
    left, right = (pd.Timestamp(value) for value in CALENDAR[name])
    mask = frame.landed.ge(left) & frame.landed.lt(right)
    if outcomes:
        mask &= frame.next_departed.lt(right)
    return frame.loc[mask].reset_index(drop=True)


def settings_for_example():
    settings = json.loads((ROOT / 'config' / 'article_models.json').read_text(encoding='utf-8'))
    settings['retrieval']['epochs'] = 3
    for name in ('xgboost', 'lightgbm'):
        settings[name]['n_estimators'] = 60
    settings['severity']['xgboost_quantile']['n_estimators'] = 60
    settings['catboost']['iterations'] = 60
    return settings


def scored(models, frame):
    uncertain = frame.loc[frame.evidence_warning.eq(0)]
    scores = models.score(uncertain)
    return frame.merge(scores[['link_key'] + SCORED], on='link_key', how='left', validate='one_to_one')


def rounded(value):
    if isinstance(value, dict):
        return {str(key): rounded(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else round(float(value), 6)
    return value


SCORED = ['gate_retrieval_bonus', 'aggregated_utility', 'disagreement', 'retrieval_probability', 'tree_probability_raw',
          'tree_probability_calibrated', 'light', 'warning_probability', 'tree_excess', 'tail_severity_pred',
          'tree_expert_probability', 'tree_expert_severity', 'risk_aware_utility_score', 'uncertainty_probability_entropy']


def run(aircraft, smoke=False, seed=7):
    last = '2030-04-02' if smoke else CALENDAR['replay'][1]
    flights = constructed_flights(CALENDAR['history'][0], last, aircraft, seed)
    events = flight_events(flights)
    operations = flight_operations(flights)
    cohort = connections(flights)
    prepared = attach_flight_memory(add_live_state(cohort, events), operations)
    history = period(prepared, 'history')
    train = period(prepared, 'experts').query('evidence_warning == 0').reset_index(drop=True)
    april = period(prepared, 'gate_and_calibration').query('evidence_warning == 0').reset_index(drop=True)
    models = ArticleModels(settings_for_example(), tau=15, threads=1).fit(history, train, april)
    selection = scored(models, period(prepared, 'selection'))
    screening = scored(models, period(prepared, 'screening'))
    fixture = screening.loc[screening.evidence_warning.eq(0)].head(200).copy()
    flipped = fixture.assign(warning_target=1 - fixture.warning_target, next_dep_delay_min=500 - fixture.next_dep_delay_min)
    change = np.abs(models.score(flipped)[SCORED].to_numpy() - models.score(fixture)[SCORED].to_numpy()).max()
    candidates = ps.scan(selection)
    card = models.card()
    card.pop('seconds')
    output = dict(calendar=CALENDAR, aircraft=aircraft, seed=seed, connections={name: len(period(prepared, name, False)) for name in CALENDAR},
                  timing_rule_share=float(prepared.evidence_warning.mean()), expert_fit_rows=len(train), gate_fit_rows=len(april),
                  model_card=card, label_change_max=float(change),
                  candidates={method: int((candidates.method == method).sum()) for method in ps.SYSTEMS})
    per_budget = ps.fixed_sequence(candidates, screening, 'hybrid', delta=0.05)
    joint = ps.fixed_sequence(candidates, screening, 'hybrid', delta=ps.JOINT)
    output['budgets'] = [dict(budget=budget, sequence_length=row['sequence_length'], sequence_passed=row['sequence_passed'],
                              screen_alerts=row['screen_alerts'], screen_false=row['screen_false'], screen_limit=row['screen_limit'],
                              joint_passed=joint[budget]['sequence_passed'], joint_limit=joint[budget]['screen_limit'])
                         for budget, row in per_budget.items()]
    if smoke:
        return output
    replay = scored(models, prepared.loc[prepared.landed.ge(pd.Timestamp(CALENDAR['replay'][0]))
                                         & prepared.landed.lt(pd.Timestamp(CALENDAR['replay'][1]))].reset_index(drop=True))
    year, month = REPLAY_MONTHS[0]
    first = replay.loc[replay.landed.dt.year.eq(year) & replay.landed.dt.month.eq(month)].reset_index(drop=True)
    output['first_month'] = {budget: ps.performance(first, ps.alerts(first, per_budget[budget]['policy'])) for budget in (15, 18)}
    pool = pd.concat([selection, screening, replay], ignore_index=True)
    output['ceilings'] = {}
    for ceiling in (15, 18):
        found = ps.ceiling_policy(selection, screening, ceiling, candidates=candidates.loc[candidates.method == 'hybrid'])
        _, events_log, monthly = ps.weekly_replay(replay, pool, None if found is None else found[1], ceiling, REPLAY_MONTHS)
        output['ceilings'][ceiling] = dict(start_point=None if found is None else found[0],
                                           start_worst_limit=None if found is None else found[2],
                                           pauses=[{key: value for key, value in event.items() if key != 'seconds'} for event in events_log],
                                           monthly=monthly)
    windows = ps.window_policies(candidates, selection, screening)
    output['review_window'] = {}
    for ceiling, row in windows.items():
        if row is None:
            output['review_window'][ceiling] = None
            continue
        fired = ps.alerts(first, row['policy'])
        in_window = first.available_slack_min.ge(25).to_numpy()
        output['review_window'][ceiling] = dict(
            budget=row['budget'], groups=row['groups'], first_month=ps.performance(first.loc[in_window], fired[in_window]),
            weekly_checks=ps.window_checks(first, fired, ceiling, [pd.Timestamp(year, month, day) for day in ps.CHECK_DAYS]))
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--aircraft', type=int, default=500)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = rounded(run(args.aircraft, args.smoke, args.seed))
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('connections', 'expert_fit_rows', 'gate_fit_rows', 'label_change_max')}, indent=2))
