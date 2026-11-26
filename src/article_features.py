"""Decision-time features used by the article pipeline.

Two feature groups are added to each connection at the upstream landing time:

* operating state: departures, arrivals and departure backlog observed in the
  60 minutes before landing at the onward origin, onward destination and onward
  carrier, and route departures over 180 minutes;
* flight-number memory: outcomes of the same onward flight number that took off
  in the 28 days before the decision day, with smoothed rates for sparse numbers.

Every statistic uses events time-stamped strictly before the decision time.
"""
import numpy as np
import pandas as pd

WINDOW = 60
ROUTE_WINDOW = 180
LIVE_FEATURES = [
    'live_dep_count', 'live_dep_late_share', 'live_dep_mean_delay',
    'live_backlog', 'live_backlog_share',
    'live_arr_late_share', 'live_dest_dep_late_share', 'live_dest_arr_late_share',
    'live_airline_late_share', 'live_route_late_share',
]

WINDOW_DAYS = 28
LEVELS = [0.50, 0.70, 0.85, 0.90, 0.95]
STATS = ['flight_rate'] + [f'flight_q{int(level * 100)}' for level in LEVELS] + ['flight_log_support', 'flight_volatility']
FLIGHT_HISTORY = ['hist_flight_count', 'hist_flight_warning_rate', 'hist_flight_avg_delay_min']
MINUTE_SCALE = 60.0


def flight_events(flights):
    """Return the event table used by the operating-state features.

    `flights` holds one row per flight with columns dep, arr, airline, planned,
    departed, planned_arr and landed.
    """
    frame = flights.loc[flights.planned.notna(), ['dep', 'arr', 'airline', 'planned', 'departed', 'planned_arr', 'landed']].copy()
    frame['dep_delay'] = (frame.departed - frame.planned).dt.total_seconds() / 60
    frame['arr_delay'] = (frame.landed - frame.planned_arr).dt.total_seconds() / 60
    return frame


def _minutes(series):
    return pd.to_datetime(series).to_numpy(dtype='datetime64[ns]').astype('int64') / 6e10


def _window(events, key, stamp, values, queries, qkey, width):
    events = events.loc[events[stamp].notna() & events[key].notna()]
    out = {'count': np.zeros(len(queries))}
    out.update({name: np.zeros(len(queries)) for name in values})
    qtime = _minutes(queries.landed)
    qgroups = queries.groupby(queries[qkey].to_numpy()).indices
    for group, part in events.groupby(key):
        rows = qgroups.get(group)
        if rows is None:
            continue
        times = _minutes(part[stamp])
        order = np.argsort(times, kind='stable')
        times = times[order]
        hi = np.searchsorted(times, qtime[rows], 'left')
        lo = np.searchsorted(times, qtime[rows] - width, 'left')
        out['count'][rows] = hi - lo
        for name, column in values.items():
            cum = np.concatenate([[0.0], np.cumsum(column(part).to_numpy(dtype=float)[order])])
            out[name][rows] = cum[hi] - cum[lo]
    return out


def _backlog(events, queries, width):
    sched = events.loc[events.dep.notna(), ['dep', 'planned', 'departed']]
    qtime = _minutes(queries.landed)
    counts = np.zeros(len(queries))
    pending = np.zeros(len(queries))
    qgroups = queries.groupby(queries.next_dep_airport.to_numpy()).indices
    for group, part in sched.groupby('dep'):
        rows = qgroups.get(group)
        if rows is None:
            continue
        plan = _minutes(part.planned)
        gone = _minutes(part.departed.fillna(pd.Timestamp('2100-01-01')))
        order = np.argsort(plan, kind='stable')
        plan, gone = plan[order], gone[order]
        times = qtime[rows]
        hi = np.searchsorted(plan, times, 'left')
        lo = np.searchsorted(plan, times - width, 'left')
        counts[rows] = hi - lo
        span = int((hi - lo).max())
        if span == 0:
            continue
        for start in range(0, len(rows), 20000):
            block = slice(start, start + 20000)
            index = lo[block, None] + np.arange(span)[None, :]
            valid = index < hi[block, None]
            index = np.minimum(index, len(plan) - 1)
            pending[rows[block]] = ((gone[index] >= times[block, None]) & valid).sum(axis=1)
    return counts, pending


def _share(part, total):
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(total > 0, part / np.maximum(total, 1), np.nan)


def add_live_state(frame, events, tau=15):
    """Append the ten operating-state features observed before each upstream landing."""
    work = frame.copy()
    work['_route'] = work.next_dep_airport.astype(str) + '>' + work.next_arr_airport.astype(str)
    late = lambda part: (part.dep_delay >= tau).astype(float)
    positive = lambda part: part.dep_delay.clip(lower=0)
    arrival_late = lambda part: (part.arr_delay >= tau).astype(float)
    departures = events.loc[events.departed.notna()].copy()
    departures['route'] = departures.dep.astype(str) + '>' + departures.arr.astype(str)
    arrivals = events.loc[events.landed.notna()]
    origin = _window(departures, 'dep', 'departed', {'late': late, 'delay': positive}, work, 'next_dep_airport', WINDOW)
    work['live_dep_count'] = origin['count']
    work['live_dep_late_share'] = _share(origin['late'], origin['count'])
    work['live_dep_mean_delay'] = _share(origin['delay'], origin['count'])
    scheduled, pending = _backlog(events, work, WINDOW)
    work['live_backlog'] = pending
    work['live_backlog_share'] = _share(pending, scheduled)
    inbound = _window(arrivals, 'arr', 'landed', {'late': arrival_late}, work, 'next_dep_airport', WINDOW)
    work['live_arr_late_share'] = _share(inbound['late'], inbound['count'])
    destination_departures = _window(departures, 'dep', 'departed', {'late': late}, work, 'next_arr_airport', WINDOW)
    work['live_dest_dep_late_share'] = _share(destination_departures['late'], destination_departures['count'])
    destination_arrivals = _window(arrivals, 'arr', 'landed', {'late': arrival_late}, work, 'next_arr_airport', WINDOW)
    work['live_dest_arr_late_share'] = _share(destination_arrivals['late'], destination_arrivals['count'])
    carrier = _window(departures, 'airline', 'departed', {'late': late}, work, 'next_airline', WINDOW)
    work['live_airline_late_share'] = _share(carrier['late'], carrier['count'])
    route = _window(departures, 'route', 'departed', {'late': late}, work, '_route', ROUTE_WINDOW)
    work['live_route_late_share'] = _share(route['late'], route['count'])
    return work.drop(columns='_route')


def flight_operations(flights):
    """Return flight-number operations (flight, departed, delay in minutes) for the memory."""
    frame = flights.loc[flights.flight.notna() & flights.planned.notna() & flights.departed.notna(), ['flight', 'planned', 'departed']].copy()
    frame['delay'] = ((frame.departed - frame.planned).dt.total_seconds() // 60).astype(int)
    frame['day'] = frame.departed.dt.normalize()
    return frame[['flight', 'departed', 'delay', 'day']]


def _window_stats(operations, day, tau, smoothing):
    part = operations.loc[(operations.day < day) & (operations.day >= day - pd.Timedelta(days=WINDOW_DAYS))]
    late = (part.delay >= tau).astype(float)
    excess = (part.delay - tau).clip(lower=0).astype(float)
    work = pd.DataFrame({'flight': part.flight.to_numpy(), 'late': late.to_numpy(), 'excess': excess.to_numpy(),
                         'delay': part.delay.astype(float).to_numpy()})
    grouped = work.groupby('flight').agg(n=('late', 'size'), late=('late', 'sum'), delay=('delay', 'sum'),
                                         volatility=('excess', 'std'))
    positive = work.loc[work.excess > 0]
    for level in LEVELS:
        grouped[f'flight_q{int(level * 100)}'] = positive.groupby('flight').excess.quantile(level)
    grouped = grouped.fillna(0.0)
    rate = float(work.late.mean()) if len(work) else 0.0
    mean_delay = float(work.delay.mean()) if len(work) else 0.0
    grouped['flight_rate'] = grouped.late / grouped.n
    grouped['flight_log_support'] = np.log1p(grouped.n)
    grouped['flight_volatility'] = grouped.volatility
    grouped['hist_flight_count'] = grouped.n
    grouped['hist_flight_warning_rate'] = (grouped.late + smoothing * rate) / (grouped.n + smoothing)
    grouped['hist_flight_avg_delay_min'] = (grouped.delay + smoothing * mean_delay) / (grouped.n + smoothing)
    grouped['day'] = day
    return grouped[['day'] + STATS + FLIGHT_HISTORY].reset_index(), rate, mean_delay


def attach_flight_memory(frame, operations, tau=15, smoothing=20.0):
    """Append flight-number memory observed before each connection's decision day.

    `frame` needs `next_flight` (onward flight number) and `landed`.
    """
    work = frame.copy()
    work['memory_day'] = pd.to_datetime(work.landed).dt.normalize()
    rows, priors = [], []
    for day in sorted(work.memory_day.dropna().unique()):
        stats, rate, mean_delay = _window_stats(operations, pd.Timestamp(day), tau, smoothing)
        rows.append(stats)
        priors.append((pd.Timestamp(day), rate, mean_delay))
    table = pd.concat(rows, ignore_index=True)
    prior = pd.DataFrame(priors, columns=['day', 'prior_rate', 'prior_delay'])
    work = work.merge(table, left_on=['memory_day', 'next_flight'], right_on=['day', 'flight'], how='left').drop(columns=['day', 'flight'])
    work = work.merge(prior, left_on='memory_day', right_on='day', how='left').drop(columns='day')
    work['flight_exact'] = work.hist_flight_count.notna().astype(np.float32)
    work[STATS] = work[STATS].fillna(0.0)
    work['hist_flight_count'] = work.hist_flight_count.fillna(0.0)
    work['hist_flight_warning_rate'] = work.hist_flight_warning_rate.fillna(work.prior_rate)
    work['hist_flight_avg_delay_min'] = work.hist_flight_avg_delay_min.fillna(work.prior_delay)
    return work.drop(columns=['memory_day', 'prior_rate', 'prior_delay'])


def stat_array(frame):
    values = frame[STATS].to_numpy(dtype=np.float32).copy()
    values[:, 1:6] /= MINUTE_SCALE
    values[:, 7] /= MINUTE_SCALE
    return values
