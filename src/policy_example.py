import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from policy import alerts, calibrate, performance, scan


def synthetic_scores(size, generator):
    signal = generator.uniform(size=size)
    probability = 1 / (1 + np.exp(-12 * (signal - .52)))
    delayed = generator.random(size) < probability
    minutes = np.where(delayed, generator.integers(15, 125, size), generator.integers(0, 15, size))
    tree = np.clip(probability + generator.normal(0, .12, size), .001, .999)
    retrieval = signal + generator.normal(0, .10, size)
    return pd.DataFrame(dict(warning_target=delayed.astype(int), next_dep_delay_min=minutes,
        evidence_warning=(delayed & (generator.random(size) < .04)).astype(int),
        tree_probability_calibrated=tree, light=signal, warning_probability=tree,
        risk_aware_utility_score=tree*50, available_slack_min=100*(1-signal),
        uncertainty_probability_entropy=-tree*np.log(tree)-(1-tree)*np.log(1-tree),
        aggregated_utility=retrieval, disagreement=np.abs(tree-retrieval),
        gate_retrieval_bonus=generator.uniform(size=size)))


def evaluate(size):
    generator = np.random.default_rng(42)
    selection, screening, test = [synthetic_scores(size, generator) for period in range(3)]
    candidates = scan(selection)
    policies, screens = calibrate(candidates, screening)
    output = []
    for screen in screens.query("method=='hybrid'").itertuples():
        policy = policies['hybrid', screen.budget_percent]
        mask = alerts(test, policy)
        metrics = performance(test, mask)
        assert metrics['model_alerts'] + metrics['rule_alerts'] + metrics['abstain'] == size
        assert policy is not None or metrics['model_alerts'] == 0
        output.append(dict(budget_percent=screen.budget_percent, selection_feasible=bool(screen.selection_feasible),
            selected_thresholds={field:float(getattr(screen,field)) for field in ['threshold','retrieval_threshold','gate','weight']} if screen.selection_feasible else None,
            screening_alerts=screen.calibration_alerts, screening_false=screen.calibration_false,
            screening_upper=float(screen.calibration_upper) if np.isfinite(screen.calibration_upper) else None,
            passed=bool(screen.active), test_model_alerts=metrics['model_alerts'], test_false=metrics['model_false'],
            test_captured_minutes=metrics['rule_minutes']+metrics['model_minutes'], test_delay_minutes=metrics['positive_minutes'],
            test_alerts_per_thousand=metrics['alerts_per_thousand']))
    return dict(rows_per_period=size, hybrid_candidates=int(candidates.method.eq('hybrid').sum()), budgets=output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Evaluate threshold policies on three synthetic score samples.')
    parser.add_argument('--size',type=int,choices=[240,2400],default=2400)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.write_text(json.dumps(evaluate(args.size),indent=2,allow_nan=False)+'\n',encoding='utf-8')
