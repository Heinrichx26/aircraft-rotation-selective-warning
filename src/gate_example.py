import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import logit

from gate import (_apply_hybrid_model, _apply_standardization, _apply_tree_calibrator,
                  _fit_hybrid_params, _fit_standardization_stats, _fit_tree_calibrator)
from policy import alerts, calibrate, performance, scan
from policy_example import synthetic_scores


FIELDS = ['support_log','weighted_fallback','disagreement_raw','tree_logit_calibrated',
          'tree_uncertainty_raw','retrieval_utility_raw']


def generate(size, generator):
    frame = synthetic_scores(size, generator)
    frame['tree_probability_raw'] = frame.tree_probability_calibrated
    frame['tree_logit_raw'] = logit(frame.tree_probability_raw)
    frame['tree_uncertainty_raw'] = frame.tree_probability_raw * (1-frame.tree_probability_raw)
    frame['retrieval_utility_raw'] = frame.aggregated_utility
    frame['disagreement_raw'] = frame.disagreement
    frame['support_log'] = np.log1p(generator.poisson(5,size=size))
    frame['weighted_fallback'] = generator.beta(1,5,size=size)
    return frame


def run(size):
    generator = np.random.default_rng(42)
    fit, selection, screening, test = [generate(size,generator) for period in range(4)]
    fit = fit.loc[fit.evidence_warning.eq(0)].copy()
    scale = _fit_tree_calibrator(fit)
    fit = _apply_tree_calibrator(fit,scale)
    statistics = _fit_standardization_stats(fit,FIELDS)
    cutoff = fit.support_log.quantile(.60)
    fit = _apply_standardization(fit,statistics,cutoff)
    reference = fit.tree_probability_raw.ge(.85)
    parameters = _fit_hybrid_params(fit,budget_target_warning=reference.mean(),
                                    budget_target_false=reference[fit.warning_target.eq(0)].mean())
    if not np.isfinite(parameters).all() or not np.any(parameters):
        raise ValueError('Gate fit did not converge')

    def score(frame):
        scaled = _apply_standardization(_apply_tree_calibrator(frame,scale),statistics,cutoff)
        return _apply_hybrid_model(scaled,parameters)

    candidates = scan(score(selection))
    policies, _ = calibrate(candidates,score(screening))
    scored = score(test)
    swapped = test.assign(warning_target=1-test.warning_target,next_dep_delay_min=0)
    assert np.array_equal(score(swapped).gate_retrieval_bonus,scored.gate_retrieval_bonus)
    budgets = []
    for budget in range(4,25):
        policy = policies['hybrid',budget]
        metrics = performance(scored,alerts(scored,policy))
        budgets.append(dict(budget=budget,passed=policy is not None,issued=metrics['model_alerts'],
                            false=metrics['model_false'],captured_minutes=metrics['model_minutes']+metrics['rule_minutes']))
    return dict(rows_per_period=size,gate_fit_rows=len(fit),budgets=budgets)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size',type=int,choices=[240,2400],default=2400)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.write_text(json.dumps(run(args.size),indent=2)+'\n')
