"""Experts, probability calibration and routing gate of the article pipeline.

The retrieval expert extends the six-family network of `experts.py` with a
flight-number family read from the daily flight-number memory. The tree expert
averages XGBoost, LightGBM and CatBoost probabilities, calibrates the mean with a
logistic layer fitted on the April sample and multiplies it by a severity
estimate from two quantile regressors. The gate of `gate.py` routes each
unresolved connection between the two experts.
"""
import copy
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from torch import nn
from xgboost import XGBClassifier, XGBRegressor

import experts
import gate as routing
from article_features import FLIGHT_HISTORY, LIVE_FEATURES, STATS, stat_array

CONTEXT_FEATURES = ['next_airline', 'next_dep_airport', 'next_arr_airport', 'prev_dep_delay_min', 'prev_arr_delay_min',
                    'scheduled_gap_min', 'decision_hour', 'decision_dow']
RULE_FEATURES = ['available_slack_min', 'inherited_lb_min']
MEMORY_PREFIXES = ['hist_airline', 'hist_dep_airport', 'hist_arr_airport', 'hist_route', 'hist_hour', 'hist_airline_dep']
MEMORY_FEATURES = [f'{prefix}_{name}' for prefix in MEMORY_PREFIXES for name in ('warning_rate', 'avg_delay_min', 'count', 'reliability')]
TREE = [name for name in experts.BASE_MODEL_FEATURES + experts.HISTORY_MODEL_FEATURES if name != 'next_distance_km'] + LIVE_FEATURES + FLIGHT_HISTORY
REFERENCE = CONTEXT_FEATURES + RULE_FEATURES + MEMORY_FEATURES + LIVE_FEATURES + FLIGHT_HISTORY
NUMERIC = experts.NUMERIC_CONTEXT_COLUMNS + LIVE_FEATURES
GATE_INPUTS = ['support_log', 'weighted_fallback', 'disagreement_raw', 'tree_logit_calibrated', 'tree_uncertainty_raw', 'retrieval_utility_raw']
SCORE_COLUMNS = ['gate_retrieval_bonus', 'aggregated_utility', 'disagreement', 'retrieval_probability', 'tree_probability_raw',
                 'tree_probability_calibrated', 'light', 'warning_probability', 'tree_excess', 'tail_severity_pred',
                 'tree_expert_probability', 'tree_expert_severity', 'risk_aware_utility_score', 'uncertainty_probability_entropy']
DEVICE = torch.device('cpu')


def safe_logit(values):
    clipped = np.clip(np.asarray(values, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


class FlightHead(nn.Module):
    """Retrieval family built from the flight-number memory."""

    def __init__(self, query_dim, hidden_dim, num_quantiles):
        super().__init__()
        stat_dim = len(STATS)
        self.local_mlp = nn.Sequential(nn.Linear(query_dim + stat_dim + 2, hidden_dim), nn.ReLU(),
                                       nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.prob_head = nn.Linear(hidden_dim, 1)
        self.quantile_head = nn.Linear(hidden_dim, num_quantiles)
        self.gate_head = nn.Linear(hidden_dim + query_dim + 2, 1)

    def forward(self, query_repr, stats, exact):
        exact = exact.unsqueeze(-1)
        support = stats[:, STATS.index('flight_log_support')].unsqueeze(-1)
        hidden = self.local_mlp(torch.cat([query_repr, stats, support, exact], dim=-1))
        probability = torch.sigmoid(self.prob_head(hidden)).squeeze(-1)
        quantiles = torch.cumsum(F.softplus(self.quantile_head(hidden)), dim=-1)
        return {'probability': probability, 'quantiles': quantiles, 'utility': probability * quantiles.mean(dim=-1),
                'gate_logit': self.gate_head(torch.cat([query_repr, hidden, support, exact], dim=-1)).squeeze(-1),
                'avg_support': support.squeeze(-1), 'exact_flag': exact.squeeze(-1)}


class FlightRetrievalNet(experts.CSRMGNet):
    """Six historical families of `experts.CSRMGNet` plus the flight-number family."""

    def __init__(self, preprocessor, config, use_flight=True):
        super().__init__(preprocessor, config)
        self.use_flight = bool(use_flight)
        self.flight_head = FlightHead(int(config.csrmg.query_hidden_dim), int(config.csrmg.family_hidden_dim),
                                      len(self.quantile_levels))

    def forward(self, numeric, categorical, candidate_indices, exact_atom_index, score_lambda, flight_stats, flight_exact):
        query = self._encode_query(numeric, categorical)
        outputs = [self.family_heads[name](query_repr=query, candidate_indices=candidate_indices[name],
                                           exact_atom_index=exact_atom_index[name]) for name in self.family_names]
        if self.use_flight:
            outputs.append(self.flight_head(query, flight_stats, flight_exact))
        probability = torch.stack([out['probability'] for out in outputs], dim=-1)
        quantiles = torch.stack([out['quantiles'] for out in outputs], dim=1)
        utility = torch.stack([out['utility'] for out in outputs], dim=-1)
        support = torch.stack([out['avg_support'] for out in outputs], dim=-1)
        exact = torch.stack([out['exact_flag'] for out in outputs], dim=-1)
        logits = torch.stack([out['gate_logit'] for out in outputs], dim=-1)
        logits = logits + self.gate_support_bias * support + self.gate_exact_bias * exact
        gates = torch.softmax(logits, dim=-1)
        global_probability = (gates * probability).sum(-1)
        global_quantiles = (gates.unsqueeze(-1) * quantiles).sum(1)
        aggregated = (gates * utility).sum(-1)
        disagreement = (gates * (utility - aggregated.unsqueeze(-1)).pow(2)).sum(-1)
        return {'global_probability': global_probability, 'global_quantiles': global_quantiles,
                'aggregated_utility': aggregated, 'disagreement': disagreement,
                'score': aggregated - float(score_lambda) * disagreement, 'family_gates': gates,
                'family_supports': support, 'family_exact': exact}


def run_config(settings, tau):
    return SimpleNamespace(warning_delay_threshold_min=tau, history_smoothing=settings['history_smoothing'],
                           csrmg=SimpleNamespace(**settings['retrieval']))


def fit_preprocessor(history, config):
    """Numeric scaling, category codes and family memory banks from the history period."""
    means = {col: float(history[col].mean()) for col in NUMERIC}
    stds = {col: max(1e-6, float(history[col].std())) if history[col].std() > 1e-6 else 1.0 for col in NUMERIC}
    vocab = {col: {value: index + 1 for index, value in enumerate(sorted(history[col].unique()))}
             for col in experts.CATEGORICAL_FEATURES}
    return experts.CSRMGPreprocessor(NUMERIC, experts.CATEGORICAL_CONTEXT_COLUMNS.copy(), means, stds, vocab,
                                     experts.build_memory_banks(history, config), config.warning_delay_threshold_min)


def arrays(frame, preprocessor):
    split = experts.transform_csrmg_split(frame, preprocessor)
    data = {'numeric': split.numeric_array, 'warning_target': split.warning_target, 'excess_delay': split.excess_delay}
    for prefix, group in [('cat', split.categorical_arrays), ('cand', split.candidate_indices), ('exact', split.exact_atom_index)]:
        data.update({prefix + '::' + name: value for name, value in group.items()})
    data['flight_stats'] = stat_array(frame)
    data['flight_exact'] = frame.flight_exact.to_numpy(dtype=np.float32)
    return {name: torch.as_tensor(value, device=DEVICE) for name, value in data.items()}


def split_batch(batch):
    numeric, cats, candidates, exact, target, excess = experts._batch_to_inputs(batch)
    return numeric, cats, candidates, exact, target, excess, batch['flight_stats'], batch['flight_exact']


def training_loss(model, batch, config):
    numeric, cats, candidates, exact, target, excess, stats, flag = split_batch(batch)
    run = lambda values: model(values, cats, candidates, exact, config.csrmg.score_lambda, stats, flag)
    output = run(numeric)
    weight = 1.0 + float(config.csrmg.tail_weight_rho) * excess
    loss = F.binary_cross_entropy(output['global_probability'].clamp(1e-6, 1 - 1e-6), target, weight=weight)
    positive = target > 0
    if positive.any():
        loss = loss + float(config.csrmg.loss_dist_weight) * experts._pinball_loss(
            output['global_quantiles'][positive], excess[positive], model.quantile_levels)
    loss = loss + float(config.csrmg.loss_rank_weight) * experts._pairwise_rank_loss(
        output['score'], excess, int(config.csrmg.rank_pairs_per_batch))
    slack_up, lb_up = model.monotonic_counterfactuals(numeric)
    mono = F.relu(run(slack_up)['score'] - output['score']).mean() + F.relu(output['score'] - run(lb_up)['score']).mean()
    return loss + float(config.csrmg.loss_mono_weight) * mono


def capture(score, target, minutes, shares=(0.10, 0.15, 0.20)):
    """Mean share of qualifying delay minutes among the top 10%, 15% and 20% of scores."""
    delay = np.where(target > 0, minutes, 0.0)
    order = np.argsort(-score, kind='stable')
    cum = np.cumsum(delay[order])
    total = max(delay.sum(), 1.0)
    return float(np.mean([cum[max(int(share * len(score)) - 1, 0)] / total for share in shares]))


def neural_scores(frame, preprocessor, model, config):
    """Retrieval utility, disagreement, log support, fallback weight and probability."""
    chunks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(frame), 16384):
            piece = frame.iloc[start:start + 16384]
            data = arrays(piece, preprocessor)
            for pos in range(0, len(piece), 1024):
                numeric, cats, candidates, exact, _, _, stats, flag = split_batch({key: val[pos:pos + 1024] for key, val in data.items()})
                pred = model(numeric, cats, candidates, exact, config.csrmg.score_lambda, stats, flag)
                gate = pred['family_gates']
                chunks.append(torch.stack([pred['aggregated_utility'], pred['disagreement'],
                                           torch.log1p((gate * pred['family_supports']).sum(1).clip(min=0)),
                                           1 - (gate * pred['family_exact']).sum(1), pred['global_probability']], dim=1).cpu().numpy())
    return np.vstack(chunks)


def add_comparison_scores(frame, tau, lambdas):
    """Empirical severity score and probability entropy used by two comparison methods."""
    work = frame.copy()
    raw = work.warning_probability.astype(float).fillna(0.0)
    probability = raw.clip(1e-6, 1 - 1e-6)
    severity = work.tail_severity_pred.astype(float).fillna(0.0)
    entropy = -(probability * np.log(probability) + (1 - probability) * np.log(1 - probability)) / np.log(2.0)
    reliability = work[[f'{prefix}_reliability' for prefix in MEMORY_PREFIXES]].astype(float).fillna(0.0)
    scarcity = 1.0 - reliability.mean(axis=1).clip(lower=0.0, upper=1.0)
    boundary = 1.0 / (1.0 + work.available_slack_min.astype(float).fillna(0.0).abs() / max(float(tau), 1.0))
    work['uncertainty_probability_entropy'] = entropy
    work['risk_aware_utility_score'] = (raw * severity - lambdas['lambda_probability'] * severity * entropy
                                        - lambdas['lambda_memory'] * scarcity - lambdas['lambda_rule'] * boundary)
    return work


def ensemble_raw(frame):
    return ((frame.tree_probability_raw + frame.light + frame.warning_probability) / 3).to_numpy().clip(1e-6, 1 - 1e-6)


def routing_view(frame):
    """Gate inputs computed on the three-model mean probability."""
    view = frame.copy()
    raw = ensemble_raw(frame)
    view['tree_probability_raw'] = raw
    view['tree_logit_raw'] = np.log(raw / (1 - raw))
    view['tree_uncertainty_raw'] = raw * (1 - raw)
    return view


class ArticleModels:
    """Fit on history, expert and April samples; score any later connection."""

    def __init__(self, settings, tau=15, threads=1):
        self.settings = settings
        self.tau = tau
        self.threads = threads
        self.config = run_config(settings, tau)
        self.times = {}

    def _features(self, frame):
        return frame[TREE].assign(**{col: pd.Categorical(frame[col], categories=self.categories[col]) for col in self.categories})

    def fit(self, history, train, april):
        """`history` carries operating-state features; `train` and `april` also carry flight memory."""
        settings, tau, threads = self.settings, self.tau, self.threads
        clock = time.monotonic()
        self.tables, self.priors = experts.build_history_tables(history, settings['history_smoothing'])
        self.preprocessor = fit_preprocessor(history, self.config)
        self.times['memory_seconds'] = time.monotonic() - clock
        start = time.monotonic()
        torch.manual_seed(settings['retrieval']['seed'])
        torch.set_num_threads(threads)
        self.network = FlightRetrievalNet(self.preprocessor, self.config)
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.config.csrmg.learning_rate,
                                      weight_decay=self.config.csrmg.weight_decay)
        data = arrays(train, self.preprocessor)
        best, curve = (-1.0, None, 0), []
        for epoch in range(int(settings['retrieval']['epochs'])):
            self.network.train()
            for index in torch.randperm(len(train)).split(int(self.config.csrmg.batch_size)):
                optimizer.zero_grad(set_to_none=True)
                loss = training_loss(self.network, {key: val[index] for key, val in data.items()}, self.config)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite retrieval loss')
                loss.backward()
                optimizer.step()
            value = capture(neural_scores(april, self.preprocessor, self.network, self.config)[:, 0],
                            april.warning_target.to_numpy(), april.next_dep_delay_min.to_numpy())
            curve.append(round(value, 5))
            if value > best[0]:
                best = (value, copy.deepcopy(self.network.state_dict()), epoch + 1)
        self.network.load_state_dict(best[1])
        self.times.update(neural_seconds=time.monotonic() - start, selected_epoch=best[2], april_capture=curve)
        train = experts.add_history_features(train, self.tables, self.priors)
        self.categories = {col: sorted(train[col].unique()) for col in experts.CATEGORICAL_FEATURES}
        start = time.monotonic()
        self.xgb = XGBClassifier(**settings['xgboost'], n_jobs=threads).fit(self._features(train), train.warning_target)
        self.light = LGBMClassifier(**settings['lightgbm'], n_jobs=threads, verbosity=-1).fit(self._features(train), train.warning_target)
        options = dict(**settings['catboost'], cat_features=experts.CATEGORICAL_FEATURES, thread_count=threads,
                       allow_writing_files=False, verbose=False)
        self.cat = CatBoostClassifier(**options).fit(train[REFERENCE], train.warning_target)
        positive = train.loc[train.warning_target == 1]
        alpha = settings['severity']['catboost_quantile_alpha']
        self.severity = CatBoostRegressor(loss_function=f'Quantile:alpha={alpha}', **options).fit(positive[REFERENCE], positive.next_dep_delay_min)
        self.excess = XGBRegressor(**settings['severity']['xgboost_quantile'], n_jobs=threads).fit(
            self._features(positive), (positive.next_dep_delay_min - tau).clip(lower=0))
        self.times['tree_fits_seconds'] = time.monotonic() - start
        start = time.monotonic()
        april = self._raw_scores(april)
        self.scale = routing._fit_tree_calibrator(april)
        april = routing._apply_tree_calibrator(april, self.scale)
        self.ensemble = LogisticRegression(C=1e6).fit(safe_logit(ensemble_raw(april))[:, None], april.warning_target)
        self.route_scale = routing._fit_tree_calibrator(routing_view(april))
        view = routing._apply_tree_calibrator(routing_view(april), self.route_scale)
        self.stats = routing._fit_standardization_stats(view, GATE_INPUTS)
        self.cutoff = float(view.support_log.quantile(settings['gate']['dense_support_quantile']))
        view = routing._apply_standardization(view, self.stats, self.cutoff)
        reference = view.tree_probability_raw >= settings['gate']['reference_raw_probability']
        columns = [col for col in view if col.endswith('_z')] + ['warning_target', 'dense_support_mask', 'delay_weight', 'tree_probability_calibrated']
        self.reference = dict(count=int(reference.sum()), connections=len(view),
                              below_threshold=int(view.warning_target.eq(0).sum()),
                              below_threshold_count=int((reference & view.warning_target.eq(0)).sum()))
        self.params = routing._fit_hybrid_params(view[columns], budget_target_warning=float(reference.mean()),
                                                 budget_target_false=float(reference[view.warning_target == 0].mean()))
        if not np.isfinite(self.params).all() or np.all(self.params == 0):
            raise ValueError('The gate optimizer returned no fit')
        self.times['gate_seconds'] = time.monotonic() - start
        return self

    def _raw_scores(self, frame):
        numeric = neural_scores(frame, self.preprocessor, self.network, self.config)
        work = experts.add_history_features(frame, self.tables, self.priors)
        (work['aggregated_utility'], work['disagreement'], work['support_log'], work['weighted_fallback'],
         work['retrieval_probability']) = numeric.T
        features = self._features(work)
        work['tree_probability_raw'] = self.xgb.predict_proba(features)[:, 1].clip(1e-6, 1 - 1e-6)
        work['tree_excess'] = np.maximum(self.excess.predict(features), 0).reshape(len(work), -1).mean(axis=1)
        work['tree_logit_raw'] = safe_logit(work.tree_probability_raw)
        work['tree_uncertainty_raw'] = work.tree_probability_raw * (1 - work.tree_probability_raw)
        work['disagreement_raw'] = work.disagreement
        work['retrieval_utility_raw'] = work.aggregated_utility
        work['light'] = self.light.predict_proba(features)[:, 1]
        work['warning_probability'] = self.cat.predict_proba(work[REFERENCE])[:, 1]
        work['tail_severity_pred'] = np.maximum(self.severity.predict(work[REFERENCE]), 0)
        return add_comparison_scores(work, self.tau, self.settings['comparison_scores']['empirical_severity'])

    def score(self, frame):
        """Return the scored-record columns for connections without a timing alert."""
        work = routing._apply_tree_calibrator(self._raw_scores(frame), self.scale)
        routed = routing._apply_standardization(routing._apply_tree_calibrator(routing_view(work), self.route_scale),
                                                self.stats, self.cutoff)
        work['gate_retrieval_bonus'] = routing._apply_hybrid_model(routed, self.params).gate_retrieval_bonus.to_numpy()
        work['tree_expert_probability'] = self.ensemble.predict_proba(safe_logit(ensemble_raw(work))[:, None])[:, 1]
        work['tree_expert_severity'] = (work.tree_excess + np.maximum(work.tail_severity_pred - self.tau, 0)) / 2
        return work

    def card(self):
        """Model metadata: calibration and gate coefficients, fitting sizes and times."""
        return dict(selected_epoch=self.times['selected_epoch'], april_capture=self.times['april_capture'],
                    xgboost_calibration=[float(v) for v in self.scale],
                    routing_calibration=[float(v) for v in self.route_scale],
                    probability_calibration=dict(coefficient=float(self.ensemble.coef_[0, 0]), intercept=float(self.ensemble.intercept_[0])),
                    gate_parameters=[float(v) for v in self.params], dense_support_cutoff=self.cutoff,
                    gate_standardization={key: [float(mean), float(std)] for key, (mean, std) in self.stats.items()},
                    gate_reference=self.reference, tree_features=len(TREE), catboost_features=len(REFERENCE),
                    seconds={key: round(float(value), 3) for key, value in self.times.items() if key.endswith('seconds')})
