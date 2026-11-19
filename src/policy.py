import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta


SPEC = json.loads((Path(__file__).resolve().parents[1] / 'data' / 'grid.json').read_text(encoding='utf-8'))


METHODS={'tree':'tree_probability_calibrated','light':'light','probability':'warning_probability','empirical':'risk_aware_utility_score','slack':'available_slack_min','confidence':'warning_probability'}


SYSTEMS=[*METHODS,'retrieval','hybrid']


def bound(false,total):
    return float(beta.ppf(.95,false+1,total-false)) if 0<=false<total else (1.0 if total else np.nan)


def thresholds(score):
    return np.unique(np.quantile(score,SPEC['candidate_quantiles']))


def counts(mask,frame):
    target=frame.warning_target.to_numpy().astype(bool)
    return np.array([mask.sum(),(mask&~target).sum(),frame.next_dep_delay_min.to_numpy()[mask&target].sum()],dtype='int64')


def scan(frame, *, component_models_only=False):
    work=frame.loc[frame.evidence_warning.eq(0)]
    rows=[]
    def add(method,values,**policy):
        total,false,minutes=values
        rows.append(dict(method=method,n=int(total),false=int(false),minutes=int(minutes),**policy))
    for method,col in METHODS.items():
        if component_models_only and method != "tree":
            continue
        score=work[col].to_numpy()*(-1 if method=='slack' else 1)
        for threshold in thresholds(score):
            mask=score>=threshold
            if method=='confidence':
                for ent in [.2,.3,.4,.5,.6,.7,.8]:
                    add(method,counts(mask&(work.uncertainty_probability_entropy.to_numpy()<=ent),work),threshold=threshold,entropy=ent)
            else:
                add(method,counts(mask,work),threshold=threshold)
    tree=work.tree_probability_calibrated.to_numpy()
    for weight in SPEC['disagreement_weights']:
        ret=(work.aggregated_utility-weight*work.disagreement).to_numpy()
        for threshold in thresholds(ret):
            add('retrieval',counts(ret>=threshold,work),threshold=threshold,weight=weight)
        gates=work.gate_retrieval_bonus.to_numpy()
        for cut in np.unique(np.quantile(gates,SPEC['gate_quantiles'])):
            route=gates>=cut
            tree_parts=[(thr,counts((~route)&(tree>=thr),work)) for thr in thresholds(tree)]
            ret_parts=[(thr,counts(route&(ret>=thr),work)) for thr in thresholds(ret)]
            for (th_tree,part_tree),(th_ret,part_ret) in product(tree_parts,ret_parts):
                add('hybrid',part_tree+part_ret,gate=cut,threshold=th_tree,retrieval_threshold=th_ret,weight=weight)
    candidates=pd.DataFrame(rows)
    candidates['upper']=[bound(false,total) for false,total in zip(candidates.false,candidates.n)]
    return candidates


def alerts(frame,row):
    if row is None:
        return np.zeros(len(frame),dtype=bool)
    method=row['method']
    if method in METHODS:
        value=frame[METHODS[method]].to_numpy()*(-1 if method=='slack' else 1)
        fired=value>=row['threshold']
        if method=='confidence':
            fired &= frame.uncertainty_probability_entropy.to_numpy()<=row['entropy']
    else:
        ret=(frame.aggregated_utility-row['weight']*frame.disagreement).to_numpy()
        if method=='retrieval':
            fired=ret>=row['threshold']
        else:
            route=frame.gate_retrieval_bonus.to_numpy()>=row['gate']
            fired=np.where(route,ret>=row['retrieval_threshold'],frame.tree_probability_calibrated.to_numpy()>=row['threshold'])
    return fired & frame.evidence_warning.eq(0).to_numpy()


def calibrate(candidates,calibration):
    policies={}
    records=[]
    for method in SYSTEMS:
        for per in range(4,25):
            eligible=candidates.loc[(candidates.method==method)&(candidates.upper<=per/100)].sort_values(['minutes','false'],ascending=[False,True],kind='stable')
            row=None if eligible.empty else eligible.iloc[0].dropna().to_dict()
            total,false,_=counts(alerts(calibration,row),calibration)
            upper=bound(false,total)
            active=bool(upper<=per/100)
            policies[method,per]=row if active else None
            records.append({**(row or {}),'method':method,'budget_percent':per,'calibration_alerts':total,'calibration_false':false,'calibration_upper':upper,'active':active,'selection_feasible':row is not None})
    return policies,pd.DataFrame(records)


def ratio(top,bottom):
    return top/bottom if bottom else np.nan


def performance(frame,model):
    target=frame.warning_target.to_numpy(dtype=bool)
    rule=frame.evidence_warning.to_numpy(dtype=bool)
    delay=np.where(target,frame.next_dep_delay_min,0)
    fired=rule|model
    base=len(frame)
    return dict(candidates=base,model_alerts=int(model.sum()),model_false=int((model&~target).sum()),model_risk=ratio((model&~target).sum(),model.sum()),
        rule_alerts=int(rule.sum()),rule_false=int((rule&~target).sum()),all_risk=ratio((fired&~target).sum(),fired.sum()),
        false_per_thousand=ratio(1000*(fired&~target).sum(),base),alerts_per_thousand=ratio(1000*fired.sum(),base),
        abstain=int((~fired).sum()),positive_minutes=int(delay.sum()),rule_minutes=int(delay[rule].sum()),model_minutes=int(delay[model].sum()),
        captured_share=ratio(delay[fired].sum(),delay.sum()))
