import argparse
import hashlib
import json
from pathlib import Path

import airportsdata
import numpy as np
import pandas as pd


def read_archive(source, days, *, include_carrier=False):
    with source.open('rb') as handle:
        if hashlib.file_digest(handle,'md5').hexdigest()!='a73e97aa143a3e21db3bf0cdfba68775':
            raise ValueError('Archive integrity failure')
    fields='Month DayofMonth CRSDepTime CRSArrTime DepTime ArrTime DepDelay ArrDelay ActualElapsedTime CRSElapsedTime TailNum Origin Dest Cancelled Diverted'.split()
    if include_carrier:
        fields.append('UniqueCarrier')
    parts=[]
    for chunk in pd.read_csv(source,usecols=fields,chunksize=250000,dtype={'TailNum':'string'}):
        if days < 31:
            selected = chunk.Month.eq(1) & chunk.DayofMonth.le(days + 1)
        else:
            selected = chunk.Month.eq(1) | (chunk.Month.eq(2) & chunk.DayofMonth.eq(1))
        parts.append(chunk.loc[selected])
    flights=pd.concat(parts).copy()
    flights['id']=flights.index
    airports=airportsdata.load('IATA')
    flights['day']=pd.to_datetime(dict(year=2008,month=flights.Month,day=flights.DayofMonth))
    if days == 31 and not flights.day.max() == pd.Timestamp('2008-02-01'):
        raise ValueError('Full January reconstruction requires February 1 for successor lookup.')
    flights['zone']=flights.Origin.map({key:record['tz'] for key,record in airports.items()})
    flights['planned']=pd.Series(pd.NaT,index=flights.index,dtype='datetime64[ns, UTC]')
    for zone,group in flights.dropna(subset=['zone']).groupby('zone'):
        local=group.day+pd.to_timedelta((group.CRSDepTime//100)*60+group.CRSDepTime%100,unit='m')
        flights.loc[group.index,'planned']=local.dt.tz_localize(zone).dt.tz_convert('UTC')
    for col,origin,delta in [('planned_arr','planned','CRSElapsedTime'),('departed','planned','DepDelay'),('landed','departed','ActualElapsedTime')]:
        flights[col]=flights[origin]+pd.to_timedelta(flights[delta],unit='m')
    complete=(flights.Cancelled==0)&(flights.Diverted==0)&flights.departed.notna()&flights.landed.notna()
    measured=(flights.landed-flights.planned_arr).dt.total_seconds()/60
    verified=complete&measured.eq(flights.ArrDelay)
    tail=flights.TailNum.notna()&~flights.TailNum.isin(['0','000000','UNKNOW','UNKNOWN'])
    valid=tail&flights.planned.notna()
    return flights, valid, verified, complete, tail


def reconstruct(source, days):
    flights, valid, verified, complete, tail = read_archive(source, days)
    total=len(flights)
    basis=flights.loc[valid&verified&flights.Month.eq(1)&flights.DayofMonth.le(days)].copy()
    chosen={}
    outcomes={}
    for order in ['realized','scheduled']:
        actual=order=='realized'
        frame=flights.loc[valid&(verified if actual else True)].sort_values(['TailNum','departed' if actual else 'planned','id'])
        nxt=frame.groupby('TailNum')[['id','Origin','planned','departed','Cancelled','Diverted']].shift(-1).add_prefix('next_')
        pair=frame.join(nxt)
        gap=(pair.next_departed-pair.landed if actual else pair.next_planned-pair.planned_arr).dt.total_seconds()/60
        pair=pair.loc[pair.id.isin(basis.id)&pair.Dest.eq(pair.next_Origin)&gap.between(0,720)].copy()
        warn=(pair.landed-pair.next_planned).dt.total_seconds()/60>=15
        observed=pair.next_departed.notna()&pair.next_Cancelled.eq(0)&pair.next_Diverted.eq(0)
        window=(pair.next_departed-pair.landed).dt.total_seconds()/60
        outcomes[order]=dict(candidates=len(pair),evaluable=int(observed.sum()),rule=int(warn.sum()),observed_rule=int((warn&observed).sum()),
            nonpositive_window=int(((window<=0)&observed).sum()),median_window=float(window[observed].median()))
        cancelled=pair.next_Cancelled.eq(1)
        diverted=pair.next_Diverted.eq(1)&~cancelled
        other=~(observed|cancelled|diverted)
        outcomes[order].update(cancelled=int(cancelled.sum()),diverted=int(diverted.sum()),other_unobserved=int(other.sum()))
        assert int(observed.sum()+cancelled.sum()+diverted.sum()+other.sum())==len(pair)
        chosen[order]=pair.set_index('id').next_id
    sets=basis.set_index('id')[['DayofMonth']].join(pd.DataFrame(chosen))
    same=sets.realized.eq(sets.scheduled)
    both=sets.realized.notna()&sets.scheduled.notna()
    categories=np.select([same,both,sets.realized.notna(),sets.scheduled.notna()],['same','different','realized_only','scheduled_only'],default='neither')
    counts=pd.Series(categories).value_counts().to_dict()
    assert sum(counts.values())==len(basis) and outcomes['realized']['evaluable']==outcomes['realized']['candidates']
    assert outcomes['realized']['candidates']==counts.get('same',0)+counts.get('different',0)+counts.get('realized_only',0)
    summary=dict(days=days,input_with_next_day=total,unknown_origin_zone=int(flights.zone.isna().sum()),absent_tail=int((~tail).sum()),
        incomplete_or_clock_mismatch=int((complete&~verified).sum()),eligible_upstream=len(basis),first_day_boundary_upstreams=int(sets.DayofMonth.eq(1).sum()),coverage=counts,outcomes=outcomes)
    return summary


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Reconstruct onward flight connections from the 2008 US airline on-time archive.')
    parser.add_argument('archive', type=Path, help='2008.csv.bz2 from Data Expo 2009')
    parser.add_argument('--days', type=int, choices=[2, 14, 31], default=14)
    parser.add_argument('--output', type=Path, required=True, help='Destination for aggregate counts in JSON format')
    args = parser.parse_args()
    result = reconstruct(args.archive, args.days)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f"Eligible inbound flights: {result['eligible_upstream']:,}")
