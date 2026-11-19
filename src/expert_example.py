import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from scipy.special import logit
from xgboost import XGBClassifier

import experts
from gate import (_apply_hybrid_model, _apply_standardization, _apply_tree_calibrator,
                  _fit_hybrid_params, _fit_standardization_stats, _fit_tree_calibrator)
from gate_example import FIELDS
from policy import alerts, calibrate, performance, scan


def connections(size, generator):
    airports = np.array(['Port' + str(index) for index in range(12)])
    origin = generator.integers(0, len(airports), size)
    destination = (origin + generator.integers(1, len(airports), size)) % len(airports)
    slack = generator.integers(-24, 100, size)
    arrival_delay = generator.integers(-12, 90, size)
    delay = np.maximum(-slack, 32 - slack * 0.45 + arrival_delay * 0.15 + generator.normal(0, 7, size)).round()
    return pd.DataFrame(dict(next_airline=np.where(origin % 2, 'CarrierA', 'CarrierB'),
        next_dep_airport=airports[origin], next_arr_airport=airports[destination],
        route_key=np.char.add(np.char.add(airports[origin], '>'), airports[destination]),
        decision_hour=generator.integers(0, 24, size), decision_dow=generator.integers(0, 7, size),
        prev_dep_delay_min=arrival_delay + generator.integers(-8, 9, size), prev_arr_delay_min=arrival_delay,
        scheduled_gap_min=slack + arrival_delay, available_slack_min=slack, inherited_lb_min=np.maximum(-slack, 0),
        evidence_warning=(-slack >= 15).astype(int), warning_target=(delay >= 15).astype(int), next_dep_delay_min=delay))


def run(size):
    torch.manual_seed(42)
    torch.set_num_threads(1)
    generator = np.random.default_rng(42)
    history, train, gate_fit, selection, screening, test = [connections(size, generator) for _ in range(6)]
    settings = json.loads((Path(__file__).resolve().parents[1] / 'data' / 'expert_settings.json').read_text(encoding='utf-8'))
    config = SimpleNamespace(warning_delay_threshold_min=15, csrmg=SimpleNamespace(**settings['retrieval']))
    preprocessor = experts.fit_csrmg_preprocessor(history, config)
    tables, priors = experts.build_history_tables(history, settings['history_smoothing'])
    train = train.loc[train.evidence_warning.eq(0)].reset_index(drop=True)
    network = experts.CSRMGNet(preprocessor, config)
    optimizer = torch.optim.AdamW(network.parameters(), lr=config.csrmg.learning_rate, weight_decay=config.csrmg.weight_decay)
    encoded = experts.arrays(train, preprocessor)
    for epoch in range(config.csrmg.epochs):
        for indices in torch.randperm(len(train)).split(config.csrmg.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = experts._compute_training_loss(network, {key:value[indices] for key,value in encoded.items()}, config)
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
    categories = {key:sorted(train[key].unique()) for key in experts.CATEGORICAL_FEATURES}

    def features(frame):
        enriched = experts.add_history_features(frame, tables, priors)
        columns = [key for key in experts.BASE_MODEL_FEATURES + experts.HISTORY_MODEL_FEATURES if key != 'next_distance_km']
        return enriched[columns].assign(**{key:pd.Categorical(enriched[key], categories=values) for key,values in categories.items()})

    tree = XGBClassifier(**settings['tree'], random_state=42, n_jobs=1).fit(features(train), train.warning_target)

    def predict(frame):
        utility, disagreement, support, fallback = experts.neural_scores(frame, preprocessor, network, config).T
        probability = tree.predict_proba(features(frame))[:, 1].clip(1e-6, 1-1e-6)
        return frame.assign(aggregated_utility=utility, disagreement=disagreement, retrieval_utility_raw=utility,
            disagreement_raw=disagreement, support_log=support, weighted_fallback=fallback, tree_probability_raw=probability,
            tree_logit_raw=logit(probability), tree_uncertainty_raw=probability*(1-probability))

    fit = predict(gate_fit.loc[gate_fit.evidence_warning.eq(0)].reset_index(drop=True))
    scale = _fit_tree_calibrator(fit)
    fit = _apply_tree_calibrator(fit, scale)
    stats = _fit_standardization_stats(fit, FIELDS)
    cutoff = fit.support_log.quantile(.60)
    fit = _apply_standardization(fit, stats, cutoff)
    reference = fit.tree_probability_raw.ge(.85)
    parameters = _fit_hybrid_params(fit, budget_target_warning=reference.mean(), budget_target_false=reference[fit.warning_target.eq(0)].mean())
    assert np.isfinite(parameters).all() and np.any(parameters)

    def score(frame):
        calibrated = _apply_tree_calibrator(predict(frame), scale)
        return _apply_hybrid_model(_apply_standardization(calibrated, stats, cutoff), parameters)

    candidates = scan(score(selection), component_models_only=True)
    policies, _ = calibrate(candidates, score(screening))
    final = score(test)
    changed = score(test.assign(warning_target=1-test.warning_target, next_dep_delay_min=0))
    for column in ['gate_retrieval_bonus','aggregated_utility','disagreement','tree_probability_calibrated']:
        np.testing.assert_allclose(final[column], changed[column], atol=1e-6, rtol=1e-6)
    budgets = []
    for budget in range(4,25):
        policy = policies['hybrid',budget]
        outcomes = performance(final, alerts(final, policy))
        budgets.append(dict(budget=budget, passed=policy is not None, issued=outcomes['model_alerts'],
                            false=outcomes['model_false'], captured_minutes=outcomes['model_minutes']+outcomes['rule_minutes']))
    return dict(connections_per_sample=size, expert_fit_rows=len(train), gate_fit_rows=len(fit), budgets=budgets)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', choices=[320, 2400], type=int, default=2400)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(run(args.size), indent=2) + '\n')
