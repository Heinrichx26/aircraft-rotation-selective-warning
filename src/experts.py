from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

DEVICE = torch.device("cpu")


FAMILY_SPECS: OrderedDict[str, list[str]] = OrderedDict(
    [
        ("airline", ["next_airline"]),
        ("dep_airport", ["next_dep_airport"]),
        ("arr_airport", ["next_arr_airport"]),
        ("route", ["route_key"]),
        ("hour", ["decision_hour"]),
        ("airline_dep", ["next_airline", "next_dep_airport"]),
    ]
)


CATEGORICAL_CONTEXT_COLUMNS = [
    "next_airline",
    "next_dep_airport",
    "next_arr_airport",
]


NUMERIC_CONTEXT_COLUMNS = [
    "prev_dep_delay_min",
    "prev_arr_delay_min",
    "scheduled_gap_min",
    "available_slack_min",
    "inherited_lb_min",
    "decision_hour",
    "decision_dow",
]


@dataclass(frozen=True)
class FamilyAtomBank:
    family_name: str
    key_columns: list[str]
    atom_table: pd.DataFrame
    key_to_index: pd.DataFrame
    stats_array: np.ndarray
    neighbor_indices: np.ndarray
    fallback_indices: np.ndarray


@dataclass(frozen=True)
class MemoryBankBundle:
    family_banks: dict[str, FamilyAtomBank]
    stat_feature_names: list[str]
    quantile_levels: list[float]
    warning_delay_threshold_min: int


@dataclass(frozen=True)
class CSRMGPreprocessor:
    numeric_columns: list[str]
    categorical_columns: list[str]
    numeric_mean: dict[str, float]
    numeric_std: dict[str, float]
    categorical_vocab: dict[str, dict[str, int]]
    memory_banks: MemoryBankBundle
    warning_delay_threshold_min: int


@dataclass(frozen=True)
class EncodedSplit:
    frame: pd.DataFrame
    numeric_array: np.ndarray
    categorical_arrays: dict[str, np.ndarray]
    candidate_indices: dict[str, np.ndarray]
    exact_atom_index: dict[str, np.ndarray]
    warning_target: np.ndarray
    excess_delay: np.ndarray


@dataclass(frozen=True)
class CSRMGVariantSpec:
    name: str = "full"
    use_family_gate: bool = True
    use_disagreement_penalty: bool = True
    use_retrieval_atoms: bool = True
    use_flat_summary_head: bool = False
    exact_only_lookup: bool = False
    active_families: tuple[str, ...] | None = None


class FamilyRetrievalHead(nn.Module):
    def __init__(
        self,
        num_atoms: int,
        atom_stats: np.ndarray,
        query_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        num_quantiles: int,
        temperature: float,
        support_beta: float,
        exact_match_bonus: float,
        use_retrieval_atoms: bool = True,
    ) -> None:
        super().__init__()
        stat_dim = int(atom_stats.shape[1]) if atom_stats.ndim == 2 and atom_stats.size else 0
        safe_num_atoms = max(int(num_atoms), 1)
        safe_stats = atom_stats.astype(np.float32, copy=True)
        if safe_stats.size == 0:
            safe_stats = np.zeros((safe_num_atoms, max(stat_dim, 1)), dtype=np.float32)
            stat_dim = safe_stats.shape[1]
        self.atom_embedding = nn.Embedding(safe_num_atoms, embedding_dim)
        self.query_projection = nn.Linear(query_dim, embedding_dim)
        self.local_mlp = nn.Sequential(
            nn.Linear(query_dim + embedding_dim + stat_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.prob_head = nn.Linear(hidden_dim, 1)
        self.quantile_head = nn.Linear(hidden_dim, num_quantiles)
        self.gate_head = nn.Linear(hidden_dim + query_dim + 2, 1)
        self.temperature = float(max(temperature, 1e-3))
        self.support_beta = float(support_beta)
        self.exact_match_bonus = float(exact_match_bonus)
        self.use_retrieval_atoms = bool(use_retrieval_atoms)
        self.embedding_dim = int(embedding_dim)
        self.stat_dim = int(stat_dim)
        self.log_support_idx = max(stat_dim - 2, 0)
        self.register_buffer("atom_stats", torch.tensor(safe_stats, dtype=torch.float32))

    def forward(
        self,
        query_repr: torch.Tensor,
        candidate_indices: torch.Tensor,
        exact_atom_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        candidate_indices = candidate_indices.clamp_min(0)
        candidate_embed = self.atom_embedding(candidate_indices)
        candidate_stats = self.atom_stats[candidate_indices]
        exact_mask = (
            candidate_indices.eq(exact_atom_index.unsqueeze(1))
            & exact_atom_index.unsqueeze(1).ge(0)
        )
        log_support = candidate_stats[..., self.log_support_idx]

        if self.use_retrieval_atoms:
            projected_query = F.normalize(self.query_projection(query_repr), dim=-1).unsqueeze(1)
            normalized_atom = F.normalize(candidate_embed, dim=-1)
            similarity = (projected_query * normalized_atom).sum(dim=-1)
            attention_logits = (
                similarity / self.temperature
                + self.support_beta * log_support
                + self.exact_match_bonus * exact_mask.float()
            )
            attention = torch.softmax(attention_logits, dim=-1)
            pooled_embed = torch.sum(attention.unsqueeze(-1) * candidate_embed, dim=1)
            pooled_stats = torch.sum(attention.unsqueeze(-1) * candidate_stats, dim=1)
        else:
            batch_size = int(candidate_indices.shape[0])
            row_index = torch.arange(batch_size, device=query_repr.device)
            attention = torch.zeros_like(candidate_indices, dtype=query_repr.dtype)
            exact_position = torch.argmax(exact_mask.float(), dim=-1)
            exact_available = exact_atom_index.ge(0)
            fallback_position = torch.zeros_like(exact_position)
            selected_position = torch.where(exact_available, exact_position, fallback_position)
            attention[row_index, selected_position] = 1.0
            pooled_stats = candidate_stats[row_index, selected_position]
            pooled_embed = torch.zeros(
                (batch_size, self.embedding_dim),
                dtype=query_repr.dtype,
                device=query_repr.device,
            )

        memory_repr = torch.cat([pooled_embed, pooled_stats], dim=-1)
        avg_support = pooled_stats[..., self.log_support_idx].unsqueeze(-1)
        exact_flag = exact_atom_index.ge(0).float().unsqueeze(-1)
        summary_vector = torch.cat([pooled_stats, avg_support, exact_flag], dim=-1)

        local_input = torch.cat([query_repr, memory_repr, avg_support, exact_flag], dim=-1)
        local_hidden = self.local_mlp(local_input)
        probability = torch.sigmoid(self.prob_head(local_hidden)).squeeze(-1)
        quantile_increments = F.softplus(self.quantile_head(local_hidden))
        quantiles = torch.cumsum(quantile_increments, dim=-1)
        expected_excess = quantiles.mean(dim=-1)
        utility = probability * expected_excess
        gate_logit = self.gate_head(
            torch.cat([query_repr, local_hidden, avg_support, exact_flag], dim=-1)
        ).squeeze(-1)

        return {
            "attention": attention,
            "memory_repr": memory_repr,
            "avg_support": avg_support.squeeze(-1),
            "exact_flag": exact_flag.squeeze(-1),
            "summary_vector": summary_vector,
            "local_hidden": local_hidden,
            "probability": probability,
            "quantiles": quantiles,
            "expected_excess": expected_excess,
            "utility": utility,
            "gate_logit": gate_logit,
        }


class CSRMGNet(nn.Module):
    def __init__(
        self,
        preprocessor: CSRMGPreprocessor,
        config: RunConfig,
        variant: CSRMGVariantSpec | None = None,
    ) -> None:
        super().__init__()
        self.variant = variant or CSRMGVariantSpec()
        if self.variant.active_families:
            self.family_names = [
                family for family in FAMILY_SPECS.keys() if family in set(self.variant.active_families)
            ]
        else:
            self.family_names = list(FAMILY_SPECS.keys())
        self.numeric_columns = preprocessor.numeric_columns
        self.quantile_levels = preprocessor.memory_banks.quantile_levels
        self.available_slack_index = self.numeric_columns.index("available_slack_min")
        self.inherited_lb_index = self.numeric_columns.index("inherited_lb_min")
        self.available_slack_delta = float(
            config.csrmg.mono_slack_delta / preprocessor.numeric_std["available_slack_min"]
        )
        self.inherited_lb_delta = float(
            config.csrmg.mono_lb_delta / preprocessor.numeric_std["inherited_lb_min"]
        )
        self.gate_support_bias = float(config.csrmg.gate_support_bias)
        self.gate_exact_bias = float(config.csrmg.gate_exact_bias)

        embedding_dim = int(config.csrmg.embedding_dim)
        query_hidden_dim = int(config.csrmg.query_hidden_dim)
        family_hidden_dim = int(config.csrmg.family_hidden_dim)

        self.context_embeddings = nn.ModuleDict()
        total_cat_dim = 0
        for col in preprocessor.categorical_columns:
            vocab_size = max(len(preprocessor.categorical_vocab[col]) + 1, 2)
            emb_dim = min(embedding_dim, max(4, vocab_size // 8))
            self.context_embeddings[col] = nn.Embedding(vocab_size, emb_dim)
            total_cat_dim += emb_dim

        self.query_encoder = nn.Sequential(
            nn.Linear(len(preprocessor.numeric_columns) + total_cat_dim, query_hidden_dim),
            nn.ReLU(),
            nn.Linear(query_hidden_dim, query_hidden_dim),
            nn.ReLU(),
        )

        self.family_heads = nn.ModuleDict()
        for family_name, family_bank in preprocessor.memory_banks.family_banks.items():
            self.family_heads[family_name] = FamilyRetrievalHead(
                num_atoms=len(family_bank.atom_table),
                atom_stats=family_bank.stats_array,
                query_dim=query_hidden_dim,
                embedding_dim=embedding_dim,
                hidden_dim=family_hidden_dim,
                num_quantiles=len(self.quantile_levels),
                temperature=config.csrmg.temperature,
                support_beta=config.csrmg.support_beta,
                exact_match_bonus=config.csrmg.exact_match_bonus,
                use_retrieval_atoms=self.variant.use_retrieval_atoms,
            )
        if self.variant.use_flat_summary_head:
            summary_dim = len(preprocessor.memory_banks.stat_feature_names) + 2
            flat_hidden_dim = max(query_hidden_dim, family_hidden_dim)
            self.flat_summary_mlp = nn.Sequential(
                nn.Linear(query_hidden_dim + len(self.family_names) * summary_dim, flat_hidden_dim),
                nn.ReLU(),
                nn.Linear(flat_hidden_dim, flat_hidden_dim),
                nn.ReLU(),
            )
            self.flat_prob_head = nn.Linear(flat_hidden_dim, 1)
            self.flat_quantile_head = nn.Linear(flat_hidden_dim, len(self.quantile_levels))

    def _encode_query(
        self,
        numeric: torch.Tensor,
        categorical: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        cat_parts = [self.context_embeddings[col](categorical[col]) for col in CATEGORICAL_CONTEXT_COLUMNS]
        query_input = torch.cat([numeric] + cat_parts, dim=-1)
        return self.query_encoder(query_input)

    def forward(
        self,
        numeric: torch.Tensor,
        categorical: dict[str, torch.Tensor],
        candidate_indices: dict[str, torch.Tensor],
        exact_atom_index: dict[str, torch.Tensor],
        score_lambda: float,
    ) -> dict[str, torch.Tensor]:
        query_repr = self._encode_query(numeric, categorical)

        family_outputs: dict[str, dict[str, torch.Tensor]] = {}
        gate_logits: list[torch.Tensor] = []
        probabilities: list[torch.Tensor] = []
        quantiles: list[torch.Tensor] = []
        utilities: list[torch.Tensor] = []
        supports: list[torch.Tensor] = []
        exact_flags: list[torch.Tensor] = []
        summary_vectors: list[torch.Tensor] = []

        for family_name in self.family_names:
            output = self.family_heads[family_name](
                query_repr=query_repr,
                candidate_indices=candidate_indices[family_name],
                exact_atom_index=exact_atom_index[family_name],
            )
            family_outputs[family_name] = output
            gate_logits.append(output["gate_logit"])
            probabilities.append(output["probability"])
            quantiles.append(output["quantiles"])
            utilities.append(output["utility"])
            supports.append(output["avg_support"])
            exact_flags.append(output["exact_flag"])
            summary_vectors.append(output["summary_vector"])

        family_probability_tensor = torch.stack(probabilities, dim=-1)
        family_quantile_tensor = torch.stack(quantiles, dim=1)
        family_utility_tensor = torch.stack(utilities, dim=-1)
        family_support_tensor = torch.stack(supports, dim=-1)
        family_exact_tensor = torch.stack(exact_flags, dim=-1)
        if self.variant.use_flat_summary_head:
            batch_size = int(query_repr.shape[0])
            flat_input = torch.cat([query_repr] + summary_vectors, dim=-1)
            flat_hidden = self.flat_summary_mlp(flat_input)
            global_probability = torch.sigmoid(self.flat_prob_head(flat_hidden)).squeeze(-1)
            quantile_increments = F.softplus(self.flat_quantile_head(flat_hidden))
            global_quantiles = torch.cumsum(quantile_increments, dim=-1)
            expected_excess = global_quantiles.mean(dim=-1)
            aggregated_utility = global_probability * expected_excess
            disagreement = torch.zeros_like(aggregated_utility)
            family_gates = torch.full(
                (batch_size, len(self.family_names)),
                1.0 / max(len(self.family_names), 1),
                dtype=query_repr.dtype,
                device=query_repr.device,
            )
        else:
            gate_logits_tensor = torch.stack(gate_logits, dim=-1)
            if abs(self.gate_support_bias) > 1e-12:
                gate_logits_tensor = gate_logits_tensor + self.gate_support_bias * family_support_tensor
            if abs(self.gate_exact_bias) > 1e-12:
                gate_logits_tensor = gate_logits_tensor + self.gate_exact_bias * family_exact_tensor
            if self.variant.use_family_gate:
                family_gates = torch.softmax(gate_logits_tensor, dim=-1)
            else:
                family_gates = torch.full_like(
                    gate_logits_tensor,
                    1.0 / max(len(self.family_names), 1),
                )
            global_probability = torch.sum(family_gates * family_probability_tensor, dim=-1)
            global_quantiles = torch.sum(
                family_gates.unsqueeze(-1) * family_quantile_tensor,
                dim=1,
            )
            expected_excess = global_quantiles.mean(dim=-1)
            aggregated_utility = torch.sum(family_gates * family_utility_tensor, dim=-1)
            disagreement = torch.sum(
                family_gates * (family_utility_tensor - aggregated_utility.unsqueeze(-1)).pow(2),
                dim=-1,
            )
        if self.variant.use_disagreement_penalty:
            score = aggregated_utility - float(score_lambda) * disagreement
        else:
            score = aggregated_utility

        return {
            "global_probability": global_probability,
            "global_quantiles": global_quantiles,
            "expected_excess": expected_excess,
            "aggregated_utility": aggregated_utility,
            "disagreement": disagreement,
            "score": score,
            "family_gates": family_gates,
            "family_probabilities": family_probability_tensor,
            "family_quantiles": family_quantile_tensor,
            "family_utilities": family_utility_tensor,
            "family_supports": family_support_tensor,
            "family_exact": family_exact_tensor,
            "family_outputs": family_outputs,
        }

    def monotonic_counterfactuals(
        self,
        numeric: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        slack_up = numeric.clone()
        slack_up[:, self.available_slack_index] = (
            slack_up[:, self.available_slack_index] + self.available_slack_delta
        )
        lb_up = numeric.clone()
        lb_up[:, self.inherited_lb_index] = (
            lb_up[:, self.inherited_lb_index] + self.inherited_lb_delta
        )
        return slack_up, lb_up


def _batch_to_inputs(batch: dict[str, torch.Tensor]) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
]:
    numeric = batch["numeric"]
    categorical = {
        col: batch[f"cat::{col}"]
        for col in CATEGORICAL_CONTEXT_COLUMNS
    }
    candidates = {
        family: batch[f"cand::{family}"]
        for family in FAMILY_SPECS.keys()
    }
    exact_atom_index = {
        family: batch[f"exact::{family}"]
        for family in FAMILY_SPECS.keys()
    }
    warning_target = batch["warning_target"]
    excess_delay = batch["excess_delay"]
    return numeric, categorical, candidates, exact_atom_index, warning_target, excess_delay


def _pinball_loss(
    predicted_quantiles: torch.Tensor,
    target_excess: torch.Tensor,
    quantile_levels: list[float],
) -> torch.Tensor:
    if predicted_quantiles.numel() == 0:
        return predicted_quantiles.sum() * 0.0
    residual = target_excess.unsqueeze(-1) - predicted_quantiles
    quantile_tensor = torch.tensor(
        quantile_levels,
        dtype=predicted_quantiles.dtype,
        device=predicted_quantiles.device,
    ).view(1, -1)
    loss = torch.maximum(
        quantile_tensor * residual,
        (quantile_tensor - 1.0) * residual,
    )
    return loss.mean()


def _pairwise_rank_loss(
    scores: torch.Tensor,
    excess_delay: torch.Tensor,
    max_pairs: int,
) -> torch.Tensor:
    if scores.numel() < 2:
        return scores.sum() * 0.0
    positive_idx = torch.where(excess_delay > 0)[0]
    lower_idx = torch.where(excess_delay <= 0)[0]

    if len(positive_idx) == 0 or len(lower_idx) == 0:
        order = torch.argsort(excess_delay)
        if len(order) < 2:
            return scores.sum() * 0.0
        pair_count = min(max_pairs, len(order) // 2)
        high_idx = order[-pair_count:]
        low_idx = order[:pair_count]
    else:
        pair_count = min(max_pairs, len(positive_idx), len(lower_idx))
        perm_high = torch.randperm(len(positive_idx), device=scores.device)[:pair_count]
        perm_low = torch.randperm(len(lower_idx), device=scores.device)[:pair_count]
        high_idx = positive_idx[perm_high]
        low_idx = lower_idx[perm_low]

    score_margin = scores[high_idx] - scores[low_idx]
    weight = torch.abs(excess_delay[high_idx] - excess_delay[low_idx]).detach()
    return (weight * F.softplus(-score_margin)).mean()


def _compute_training_loss(
    model: CSRMGNet,
    batch: dict[str, torch.Tensor],
    config: RunConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    numeric, categorical, candidates, exact_atom_index, warning_target, excess_delay = _batch_to_inputs(batch)
    output = model(
        numeric=numeric,
        categorical=categorical,
        candidate_indices=candidates,
        exact_atom_index=exact_atom_index,
        score_lambda=config.csrmg.score_lambda,
    )

    alpha = 1.0 + float(config.csrmg.tail_weight_rho) * excess_delay
    classification_loss = F.binary_cross_entropy(
        output["global_probability"].clamp(1e-6, 1 - 1e-6),
        warning_target,
        weight=alpha,
        reduction="mean",
    )

    positive_mask = warning_target > 0
    if positive_mask.any():
        distribution_loss = _pinball_loss(
            predicted_quantiles=output["global_quantiles"][positive_mask],
            target_excess=excess_delay[positive_mask],
            quantile_levels=model.quantile_levels,
        )
    else:
        distribution_loss = output["global_quantiles"].sum() * 0.0

    rank_loss = _pairwise_rank_loss(
        scores=output["score"],
        excess_delay=excess_delay,
        max_pairs=int(config.csrmg.rank_pairs_per_batch),
    )

    slack_up, lb_up = model.monotonic_counterfactuals(numeric)
    slack_output = model(
        numeric=slack_up,
        categorical=categorical,
        candidate_indices=candidates,
        exact_atom_index=exact_atom_index,
        score_lambda=config.csrmg.score_lambda,
    )
    lb_output = model(
        numeric=lb_up,
        categorical=categorical,
        candidate_indices=candidates,
        exact_atom_index=exact_atom_index,
        score_lambda=config.csrmg.score_lambda,
    )
    monotonic_loss = (
        F.relu(slack_output["score"] - output["score"]).mean()
        + F.relu(output["score"] - lb_output["score"]).mean()
    )

    total_loss = (
        classification_loss
        + float(config.csrmg.loss_dist_weight) * distribution_loss
        + float(config.csrmg.loss_rank_weight) * rank_loss
        + float(config.csrmg.loss_mono_weight) * monotonic_loss
    )
    components = {
        "loss_total": float(total_loss.detach().cpu()),
        "loss_cls": float(classification_loss.detach().cpu()),
        "loss_dist": float(distribution_loss.detach().cpu()),
        "loss_rank": float(rank_loss.detach().cpu()),
        "loss_mono": float(monotonic_loss.detach().cpu()),
    }
    return total_loss, components


def _excess_delay(frame: pd.DataFrame, tau: float) -> pd.Series:
    delay = pd.to_numeric(frame["next_dep_delay_min"], errors="coerce").fillna(0.0)
    return (delay - float(tau)).clip(lower=0.0)


def _atom_label_from_frame(frame: pd.DataFrame, key_columns: list[str]) -> pd.Series:
    if len(key_columns) == 1:
        return frame[key_columns[0]].astype(str)
    return frame[key_columns].astype(str).agg("|".join, axis=1)


def _build_family_atom_bank(
    train_pairs: pd.DataFrame,
    family_name: str,
    key_columns: list[str],
    quantile_levels: list[float],
    tau: int,
    neighbor_k: int,
) -> FamilyAtomBank:
    work = train_pairs[key_columns + ["warning_target", "next_dep_delay_min"]].copy()
    work["excess_delay"] = _excess_delay(work, tau=tau)

    grouped = (
        work.groupby(key_columns, dropna=False)
        .agg(
            support_count=("warning_target", "size"),
            positive_count=("warning_target", "sum"),
            warning_rate=("warning_target", "mean"),
            volatility=("excess_delay", "std"),
        )
        .reset_index()
    )
    grouped["warning_rate"] = pd.to_numeric(grouped["warning_rate"], errors="coerce").fillna(0.0)
    grouped["volatility"] = pd.to_numeric(grouped["volatility"], errors="coerce").fillna(0.0)

    positive = work[work["excess_delay"] > 0].copy()
    for quantile in quantile_levels:
        col = f"q_{str(quantile).replace('.', '_')}"
        if positive.empty:
            grouped[col] = 0.0
            continue
        quantiles = (
            positive.groupby(key_columns, dropna=False)["excess_delay"]
            .quantile(float(quantile))
            .reset_index(name=col)
        )
        grouped = grouped.merge(quantiles, on=key_columns, how="left")
        grouped[col] = pd.to_numeric(grouped[col], errors="coerce").fillna(0.0)

    grouped["log_support"] = np.log1p(pd.to_numeric(grouped["support_count"], errors="coerce").fillna(0.0))
    grouped["atom_label"] = _atom_label_from_frame(grouped, key_columns)
    grouped["atom_index"] = np.arange(len(grouped), dtype=np.int64)

    stat_feature_names = ["warning_rate"] + [
        f"q_{str(quantile).replace('.', '_')}" for quantile in quantile_levels
    ] + ["log_support", "volatility"]
    stats_array = grouped[stat_feature_names].to_numpy(dtype=np.float32, copy=True)

    n_neighbors = min(max(int(neighbor_k), 1), max(len(grouped), 1))
    if len(grouped) == 0:
        neighbor_indices = np.zeros((0, n_neighbors), dtype=np.int64)
        fallback_indices = np.zeros((n_neighbors,), dtype=np.int64)
    elif len(grouped) == 1:
        neighbor_indices = np.zeros((1, n_neighbors), dtype=np.int64)
        fallback_indices = np.zeros((n_neighbors,), dtype=np.int64)
    else:
        nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
        nn_model.fit(stats_array)
        _, indices = nn_model.kneighbors(stats_array)
        neighbor_indices = indices.astype(np.int64, copy=False)
        support_sorted = grouped.sort_values(
            ["support_count", "warning_rate"],
            ascending=[False, False],
        )["atom_index"].to_numpy(dtype=np.int64)
        fallback_indices = support_sorted[:n_neighbors]
        if len(fallback_indices) < n_neighbors:
            fallback_indices = np.pad(
                fallback_indices,
                (0, n_neighbors - len(fallback_indices)),
                mode="edge",
            )

    key_to_index = grouped[key_columns + ["atom_index", "atom_label"]].copy()
    return FamilyAtomBank(
        family_name=family_name,
        key_columns=key_columns,
        atom_table=grouped.copy(),
        key_to_index=key_to_index,
        stats_array=stats_array,
        neighbor_indices=neighbor_indices,
        fallback_indices=fallback_indices,
    )


def build_memory_banks(
    train_pairs: pd.DataFrame,
    config: RunConfig,
) -> MemoryBankBundle:
    family_banks: dict[str, FamilyAtomBank] = {}
    for family_name, key_columns in FAMILY_SPECS.items():
        family_banks[family_name] = _build_family_atom_bank(
            train_pairs=train_pairs,
            family_name=family_name,
            key_columns=key_columns,
            quantile_levels=config.csrmg.quantile_levels,
            tau=config.warning_delay_threshold_min,
            neighbor_k=config.csrmg.atom_neighbor_k,
        )
    stat_feature_names = ["warning_rate"] + [
        f"q_{str(quantile).replace('.', '_')}" for quantile in config.csrmg.quantile_levels
    ] + ["log_support", "volatility"]
    return MemoryBankBundle(
        family_banks=family_banks,
        stat_feature_names=stat_feature_names,
        quantile_levels=config.csrmg.quantile_levels,
        warning_delay_threshold_min=int(config.warning_delay_threshold_min),
    )


def fit_csrmg_preprocessor(
    train_pairs: pd.DataFrame,
    config: RunConfig,
) -> CSRMGPreprocessor:
    numeric_mean: dict[str, float] = {}
    numeric_std: dict[str, float] = {}
    for col in NUMERIC_CONTEXT_COLUMNS:
        values = pd.to_numeric(train_pairs[col], errors="coerce").fillna(0.0)
        numeric_mean[col] = float(values.mean())
        std = float(values.std())
        numeric_std[col] = std if std > 1e-6 else 1.0

    categorical_vocab: dict[str, dict[str, int]] = {}
    for col in CATEGORICAL_CONTEXT_COLUMNS:
        values = train_pairs[col].astype(str).fillna("__UNK__").unique().tolist()
        categorical_vocab[col] = {value: idx + 1 for idx, value in enumerate(sorted(values))}

    return CSRMGPreprocessor(
        numeric_columns=NUMERIC_CONTEXT_COLUMNS.copy(),
        categorical_columns=CATEGORICAL_CONTEXT_COLUMNS.copy(),
        numeric_mean=numeric_mean,
        numeric_std=numeric_std,
        categorical_vocab=categorical_vocab,
        memory_banks=build_memory_banks(train_pairs, config),
        warning_delay_threshold_min=int(config.warning_delay_threshold_min),
    )


def _encode_numeric_frame(frame: pd.DataFrame, preprocessor: CSRMGPreprocessor) -> np.ndarray:
    arrays: list[np.ndarray] = []
    for col in preprocessor.numeric_columns:
        values = pd.to_numeric(frame[col], errors="coerce").fillna(preprocessor.numeric_mean[col])
        arrays.append(((values - preprocessor.numeric_mean[col]) / preprocessor.numeric_std[col]).to_numpy(dtype=np.float32))
    return np.column_stack(arrays).astype(np.float32, copy=False)


def _encode_categorical_frame(
    frame: pd.DataFrame,
    preprocessor: CSRMGPreprocessor,
) -> dict[str, np.ndarray]:
    encoded: dict[str, np.ndarray] = {}
    for col in preprocessor.categorical_columns:
        vocab = preprocessor.categorical_vocab[col]
        encoded[col] = (
            frame[col]
            .astype(str)
            .map(vocab)
            .fillna(0)
            .astype(np.int64)
            .to_numpy(copy=True)
        )
    return encoded


def _encode_family_candidates(
    frame: pd.DataFrame,
    family_bank: FamilyAtomBank,
    exact_only_lookup: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    merged = frame[family_bank.key_columns].merge(
        family_bank.key_to_index,
        on=family_bank.key_columns,
        how="left",
    )
    exact_atom_index = pd.to_numeric(merged["atom_index"], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
    if len(family_bank.atom_table) == 0:
        candidates = np.zeros((len(frame), 1), dtype=np.int64)
        return candidates, exact_atom_index
    safe_index = exact_atom_index.copy()
    missing_mask = safe_index < 0
    safe_index[missing_mask] = 0
    candidates = family_bank.neighbor_indices[safe_index].copy()
    if missing_mask.any():
        candidates[missing_mask] = family_bank.fallback_indices
    if exact_only_lookup:
        global_atom = int(family_bank.fallback_indices[0]) if len(family_bank.fallback_indices) else 0
        exact_rows = exact_atom_index >= 0
        if exact_rows.any():
            candidates[exact_rows] = np.repeat(
                exact_atom_index[exact_rows].reshape(-1, 1),
                candidates.shape[1],
                axis=1,
            )
        if missing_mask.any():
            candidates[missing_mask] = global_atom
    return candidates.astype(np.int64, copy=False), exact_atom_index


def transform_csrmg_split(
    frame: pd.DataFrame,
    preprocessor: CSRMGPreprocessor,
    variant: CSRMGVariantSpec | None = None,
) -> EncodedSplit:
    variant = variant or CSRMGVariantSpec()
    candidate_indices: dict[str, np.ndarray] = {}
    exact_atom_index: dict[str, np.ndarray] = {}
    for family_name, family_bank in preprocessor.memory_banks.family_banks.items():
        candidates, exact_idx = _encode_family_candidates(
            frame,
            family_bank,
            exact_only_lookup=bool(variant.exact_only_lookup),
        )
        candidate_indices[family_name] = candidates
        exact_atom_index[family_name] = exact_idx

    return EncodedSplit(
        frame=frame.reset_index(drop=True).copy(),
        numeric_array=_encode_numeric_frame(frame, preprocessor),
        categorical_arrays=_encode_categorical_frame(frame, preprocessor),
        candidate_indices=candidate_indices,
        exact_atom_index=exact_atom_index,
        warning_target=frame["warning_target"].astype(np.float32).to_numpy(copy=True),
        excess_delay=_excess_delay(frame, preprocessor.warning_delay_threshold_min).to_numpy(
            dtype=np.float32,
            copy=True,
        ),
    )


HISTORY_SPECS = [
    ("hist_airline", ["next_airline"]),
    ("hist_dep_airport", ["next_dep_airport"]),
    ("hist_arr_airport", ["next_arr_airport"]),
    ("hist_route", ["route_key"]),
    ("hist_hour", ["decision_hour"]),
    ("hist_airline_dep", ["next_airline", "next_dep_airport"]),
]


def build_history_tables(
    source_pairs: pd.DataFrame,
    smoothing: float,
) -> tuple[dict[str, pd.DataFrame], dict[str, float]]:
    global_warning_rate = float(source_pairs["warning_target"].mean())
    global_delay_mean = float(source_pairs["next_dep_delay_min"].mean())
    priors = {
        "global_warning_rate": global_warning_rate,
        "global_delay_mean": global_delay_mean,
        "smoothing": smoothing,
    }

    tables: dict[str, pd.DataFrame] = {}
    for prefix, group_cols in HISTORY_SPECS:
        grouped = (
            source_pairs.groupby(group_cols, dropna=False)
            .agg(
                hist_count=("warning_target", "size"),
                hist_warning_sum=("warning_target", "sum"),
                hist_delay_sum=("next_dep_delay_min", "sum"),
            )
            .reset_index()
        )
        support_reference = float(grouped["hist_count"].quantile(0.9)) if len(grouped) else 1.0
        support_reference = max(support_reference, 1.0)
        grouped[f"{prefix}_reliability"] = (
            np.log1p(grouped["hist_count"]) / np.log1p(support_reference)
        ).clip(lower=0.0, upper=1.0)
        base_warning_rate = (
            grouped["hist_warning_sum"] + smoothing * global_warning_rate
        ) / (grouped["hist_count"] + smoothing)
        base_avg_delay = (
            grouped["hist_delay_sum"] + smoothing * global_delay_mean
        ) / (grouped["hist_count"] + smoothing)
        grouped[f"{prefix}_count"] = grouped["hist_count"]
        grouped[f"{prefix}_warning_rate"] = (
            grouped[f"{prefix}_reliability"] * base_warning_rate
            + (1.0 - grouped[f"{prefix}_reliability"]) * global_warning_rate
        )
        grouped[f"{prefix}_avg_delay_min"] = (
            grouped[f"{prefix}_reliability"] * base_avg_delay
            + (1.0 - grouped[f"{prefix}_reliability"]) * global_delay_mean
        )
        tables[prefix] = grouped[
            group_cols
            + [
                f"{prefix}_count",
                f"{prefix}_reliability",
                f"{prefix}_warning_rate",
                f"{prefix}_avg_delay_min",
            ]
        ].copy()
    return tables, priors


def add_history_features(
    target_pairs: pd.DataFrame,
    history_tables: dict[str, pd.DataFrame],
    priors: dict[str, float],
) -> pd.DataFrame:
    enriched = target_pairs.copy()
    for prefix, group_cols in HISTORY_SPECS:
        enriched = enriched.merge(history_tables[prefix], on=group_cols, how="left")
        enriched[f"{prefix}_count"] = (
            pd.to_numeric(enriched[f"{prefix}_count"], errors="coerce").fillna(0.0)
        )
        enriched[f"{prefix}_reliability"] = pd.to_numeric(
            enriched[f"{prefix}_reliability"],
            errors="coerce",
        ).fillna(0.0)
        enriched[f"{prefix}_warning_rate"] = pd.to_numeric(
            enriched[f"{prefix}_warning_rate"],
            errors="coerce",
        ).fillna(priors["global_warning_rate"])
        enriched[f"{prefix}_avg_delay_min"] = pd.to_numeric(
            enriched[f"{prefix}_avg_delay_min"],
            errors="coerce",
        ).fillna(priors["global_delay_mean"])
    return enriched


BASE_MODEL_FEATURES = [
    "next_airline",
    "next_dep_airport",
    "next_arr_airport",
    "prev_dep_delay_min",
    "prev_arr_delay_min",
    "scheduled_gap_min",
    "available_slack_min",
    "inherited_lb_min",
    "next_distance_km",
    "decision_hour",
    "decision_dow",
]


HISTORY_MODEL_FEATURES = [
    "hist_airline_warning_rate",
    "hist_airline_avg_delay_min",
    "hist_dep_airport_warning_rate",
    "hist_dep_airport_avg_delay_min",
    "hist_arr_airport_warning_rate",
    "hist_route_warning_rate",
    "hist_route_avg_delay_min",
    "hist_route_count",
    "hist_hour_warning_rate",
    "hist_airline_dep_warning_rate",
]


CATEGORICAL_FEATURES = ["next_airline", "next_dep_airport", "next_arr_airport"]


def arrays(frame, pp):
    split=transform_csrmg_split(frame,pp)
    data={'numeric':split.numeric_array,'warning_target':split.warning_target,'excess_delay':split.excess_delay}
    for prefix,group in [('cat',split.categorical_arrays),('cand',split.candidate_indices),('exact',split.exact_atom_index)]:
        data.update({prefix+'::'+name:value for name,value in group.items()})
    return {name:torch.as_tensor(value,device=DEVICE) for name,value in data.items()}


def neural_scores(frame,pp,model,config):
    chunks=[]
    model.eval()
    with torch.no_grad():
        for start in range(0,len(frame),16384):
            piece=frame.iloc[start:start+16384]
            data=arrays(piece,pp)
            for pos in range(0,len(piece),1024):
                numeric,cats,candidates,exact,_,_=_batch_to_inputs({key:val[pos:pos+1024] for key,val in data.items()})
                pred=model(numeric,cats,candidates,exact,config.csrmg.score_lambda)
                gate=pred['family_gates']
                chunks.append(torch.stack([pred['aggregated_utility'],pred['disagreement'],
                    torch.log1p((gate*pred['family_supports']).sum(1).clip(min=0)),1-(gate*pred['family_exact']).sum(1)],dim=1).cpu().numpy())
    return np.vstack(chunks)
