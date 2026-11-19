import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit


def _fit_tree_calibrator(train_frame: pd.DataFrame) -> np.ndarray:
    x = pd.to_numeric(train_frame["tree_logit_raw"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    y = pd.to_numeric(train_frame["warning_target"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    def objective(params: np.ndarray) -> float:
        log_scale, bias = params
        logits = np.exp(log_scale) * x + bias
        probability = np.clip(expit(logits), 1e-6, 1.0 - 1e-6)
        loss = -(y * np.log(probability) + (1.0 - y) * np.log(1.0 - probability)).mean()
        regularization = 0.0015 * float(np.square(params).sum())
        return float(loss + regularization)

    best_value = float("inf")
    best_params = np.zeros(2, dtype=float)
    for x0 in (
        np.array([0.0, 0.0], dtype=float),
        np.array([np.log(1.2), 0.0], dtype=float),
        np.array([np.log(0.8), 0.0], dtype=float),
    ):
        result = minimize(objective, x0=x0, method="L-BFGS-B")
        if result.success and float(result.fun) < best_value:
            best_value = float(result.fun)
            best_params = result.x.astype(float, copy=True)
    return best_params


def _apply_tree_calibrator(frame: pd.DataFrame, params: np.ndarray) -> pd.DataFrame:
    work = frame.copy()
    log_scale, bias = params
    work["tree_logit_calibrated"] = (
        np.exp(float(log_scale))
        * pd.to_numeric(work["tree_logit_raw"], errors="coerce").fillna(0.0)
        + float(bias)
    )
    work["tree_probability_calibrated"] = expit(
        pd.to_numeric(work["tree_logit_calibrated"], errors="coerce").fillna(0.0)
    )
    return work


def _fit_standardization_stats(frame: pd.DataFrame, columns: list[str]) -> dict[str, tuple[float, float]]:
    stats: dict[str, tuple[float, float]] = {}
    for col in columns:
        values = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
        mean = float(values.mean())
        std = float(values.std())
        stats[col] = (mean, std if std > 1e-6 else 1.0)
    return stats


def _apply_standardization(
    frame: pd.DataFrame,
    stats: dict[str, tuple[float, float]],
    dense_support_cutoff: float,
) -> pd.DataFrame:
    work = frame.copy()
    for col, (mean, std) in stats.items():
        work[f"{col}_z"] = (pd.to_numeric(work[col], errors="coerce").fillna(mean) - mean) / std
    positive_delay = pd.to_numeric(work["next_dep_delay_min"], errors="coerce").fillna(0.0).clip(lower=0.0)
    positive_mean = float(positive_delay[positive_delay > 0.0].mean()) if (positive_delay > 0.0).any() else 1.0
    if positive_mean <= 1e-6:
        positive_mean = 1.0
    work["delay_weight"] = positive_delay / positive_mean
    work["dense_support_mask"] = (
        pd.to_numeric(work["support_log"], errors="coerce").fillna(0.0) >= float(dense_support_cutoff)
    ) | (
        pd.to_numeric(work["weighted_fallback"], errors="coerce").fillna(0.0) <= 1e-9
    )
    return work


def _hybrid_components(frame: pd.DataFrame, params: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    (
        gate_bias,
        log_gate_support,
        log_gate_fallback,
        log_gate_tree_margin,
        log_gate_tree_uncertainty,
        log_bonus_scale,
        bonus_offset,
        log_retrieval_disagreement,
        log_threshold_support,
        log_threshold_fallback,
        log_threshold_disagreement,
        log_global_disagreement,
    ) = params
    support = frame["support_log_z"].to_numpy(dtype=float)
    fallback = frame["weighted_fallback_z"].to_numpy(dtype=float)
    disagreement = frame["disagreement_raw_z"].to_numpy(dtype=float)
    tree_margin = frame["tree_logit_calibrated_z"].to_numpy(dtype=float)
    tree_uncertainty = frame["tree_uncertainty_raw_z"].to_numpy(dtype=float)
    retrieval_raw = frame["retrieval_utility_raw_z"].to_numpy(dtype=float)

    gate = expit(
        float(gate_bias)
        - np.exp(float(log_gate_support)) * support
        + np.exp(float(log_gate_fallback)) * fallback
        - np.exp(float(log_gate_tree_margin)) * tree_margin
        + np.exp(float(log_gate_tree_uncertainty)) * tree_uncertainty
    )
    retrieval_adjusted = retrieval_raw - np.exp(float(log_retrieval_disagreement)) * disagreement
    bonus = np.maximum(retrieval_adjusted - tree_margin - float(bonus_offset), 0.0)
    hybrid_base = tree_margin + np.exp(float(log_bonus_scale)) * gate * bonus
    effective_score = (
        hybrid_base
        + np.exp(float(log_threshold_support)) * support
        - np.exp(float(log_threshold_fallback)) * fallback
        - np.exp(float(log_threshold_disagreement)) * disagreement
        - np.exp(float(log_global_disagreement)) * tree_uncertainty
    )
    return gate, effective_score, expit(effective_score)


def _hybrid_objective(
    params: np.ndarray,
    frame: pd.DataFrame,
    *,
    budget_target_warning: float,
    budget_target_false: float,
) -> float:
    gate, score, probability = _hybrid_components(frame, params)
    probability = np.clip(probability, 1e-6, 1.0 - 1e-6)
    y = frame["warning_target"].to_numpy(dtype=float)
    dense_mask = frame["dense_support_mask"].to_numpy(dtype=bool)
    delay_weight = frame["delay_weight"].to_numpy(dtype=float)
    tree_prob = np.clip(
        frame["tree_probability_calibrated"].to_numpy(dtype=float),
        1e-6,
        1.0 - 1e-6,
    )

    base_loss = -(
        (1.0 + 0.5 * delay_weight) * y * np.log(probability)
        + (1.0 - y) * np.log(1.0 - probability)
    ).mean()
    if dense_mask.any():
        dense_prob = probability[dense_mask]
        dense_tree = tree_prob[dense_mask]
        distill_loss = -(
            dense_tree * np.log(dense_prob)
            + (1.0 - dense_tree) * np.log(1.0 - dense_prob)
        ).mean()
    else:
        distill_loss = 0.0

    negative_mask = y <= 0.0
    negative_prob = probability[negative_mask] if negative_mask.any() else probability
    budget_loss = (
        float((probability.mean() - float(budget_target_warning)) ** 2)
        + 2.0 * float((negative_prob.mean() - float(budget_target_false)) ** 2)
    )

    epsilon = 0.25
    support_up = frame.copy()
    support_up["support_log_z"] = support_up["support_log_z"] + float(epsilon)
    fallback_up = frame.copy()
    fallback_up["weighted_fallback_z"] = fallback_up["weighted_fallback_z"] + float(epsilon)
    gate_support_up, _, _ = _hybrid_components(support_up, params)
    gate_fallback_up, fallback_score_up, _ = _hybrid_components(fallback_up, params)
    mono_loss = float(
        np.maximum(gate_support_up - gate, 0.0).mean()
        + np.maximum(score - fallback_score_up, 0.0).mean()
        + np.maximum(gate - gate_fallback_up, 0.0).mean() * 0.2
    )

    regularization = 0.0025 * float(np.square(params).sum())
    return float(base_loss + 0.45 * distill_loss + 0.20 * budget_loss + 0.10 * mono_loss + regularization)


def _fit_hybrid_params(
    train_frame: pd.DataFrame,
    *,
    budget_target_warning: float,
    budget_target_false: float,
) -> np.ndarray:
    best_value = float("inf")
    best_params = np.zeros(12, dtype=float)
    initial_points = [
        np.zeros(12, dtype=float),
        np.array([-0.4, 0.1, 0.1, 0.1, -0.1, -0.3, 0.0, -0.2, 0.0, 0.1, 0.1, -0.1], dtype=float),
        np.array([0.2, -0.2, 0.2, 0.1, 0.1, -0.1, -0.1, -0.4, 0.1, 0.2, 0.2, 0.0], dtype=float),
    ]
    for x0 in initial_points:
        result = minimize(
            lambda current: _hybrid_objective(
                current,
                train_frame,
                budget_target_warning=float(budget_target_warning),
                budget_target_false=float(budget_target_false),
            ),
            x0=x0,
            method="L-BFGS-B",
            options={"maxiter": 300},
        )
        if result.success and float(result.fun) < best_value:
            best_value = float(result.fun)
            best_params = result.x.astype(float, copy=True)
    return best_params


def _apply_hybrid_model(frame: pd.DataFrame, params: np.ndarray) -> pd.DataFrame:
    work = frame.copy()
    gate, score, probability = _hybrid_components(work, params)
    work["gate_retrieval_bonus"] = gate
    work["hybrid_score"] = score
    work["hybrid_probability"] = probability
    return work
