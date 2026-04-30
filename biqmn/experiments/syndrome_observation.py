from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


SYNDROME_LABELS = ("00", "01", "10", "11")


@dataclass(frozen=True)
class SyndromeObservationConfig:
    observation_ratio: float = 1.0
    noise_prob: float = 0.0
    ambiguity_level: float = 0.0
    measurement_error_prob: float = 0.0
    reset_error_prob: float = 0.0
    consistency_threshold: float = 0.9


def _clip_unit_interval(value: float) -> float:
    return float(min(1.0, max(0.0, float(value))))


def _syndrome_label(raw: Any) -> str:
    bits = [ch for ch in str(raw) if ch in "01"]
    if len(bits) >= 2:
        return "".join(bits[:2])
    return str(raw)


def _normalize_prior(probabilities: dict[str, float]) -> dict[str, float]:
    prior = {label: 0.0 for label in SYNDROME_LABELS}
    for raw_key, value in probabilities.items():
        label = _syndrome_label(raw_key)
        if label in prior:
            prior[label] += max(0.0, float(value))
    total = sum(prior.values())
    if total <= 0.0:
        prior["00"] = 1.0
        total = 1.0
    return {label: float(value / total) for label, value in prior.items()}


def _effective_flip_prob(cfg: SyndromeObservationConfig) -> float:
    keep_prob = (
        (1.0 - _clip_unit_interval(cfg.noise_prob))
        * (1.0 - _clip_unit_interval(cfg.measurement_error_prob))
        * (1.0 - _clip_unit_interval(cfg.reset_error_prob))
    )
    return float(1.0 - keep_prob)


def _is_clean_observation(cfg: SyndromeObservationConfig) -> bool:
    return bool(
        _clip_unit_interval(cfg.observation_ratio) >= 1.0
        and _clip_unit_interval(cfg.noise_prob) <= 0.0
        and _clip_unit_interval(cfg.ambiguity_level) <= 0.0
        and _clip_unit_interval(cfg.measurement_error_prob) <= 0.0
        and _clip_unit_interval(cfg.reset_error_prob) <= 0.0
    )


def _sample_hidden_label(prior: dict[str, float], rng: np.random.Generator) -> str:
    labels = list(SYNDROME_LABELS)
    probs = [float(prior[label]) for label in labels]
    return str(rng.choice(labels, p=np.asarray(probs, dtype=float)))


def _sample_observed_label(
    true_label: str,
    *,
    cfg: SyndromeObservationConfig,
    rng: np.random.Generator,
) -> tuple[str, bool, bool, bool]:
    ratio = _clip_unit_interval(cfg.observation_ratio)
    flip_prob = _effective_flip_prob(cfg)
    observed_bits: list[str] = []
    is_partial = False
    is_corrupted = False
    for bit in str(true_label):
        if rng.random() > ratio:
            observed_bits.append("?")
            is_partial = True
            is_corrupted = True
            continue
        out_bit = str(bit)
        if rng.random() < flip_prob:
            out_bit = "1" if out_bit == "0" else "0"
            is_corrupted = True
        observed_bits.append(out_bit)
    observed_label = "".join(observed_bits)
    is_ambiguous = False
    if "?" not in observed_label and rng.random() < _clip_unit_interval(cfg.ambiguity_level):
        observed_label = "amb_eq" if observed_label[0] == observed_label[1] else "amb_neq"
        is_ambiguous = True
        is_corrupted = True
    return observed_label, is_partial, is_corrupted, is_ambiguous


def _observation_likelihood(
    true_label: str,
    observed_label: str,
    *,
    cfg: SyndromeObservationConfig,
) -> float:
    ratio = _clip_unit_interval(cfg.observation_ratio)
    flip_prob = _effective_flip_prob(cfg)
    ambiguity = _clip_unit_interval(cfg.ambiguity_level)

    def concrete_prob(concrete_label: str) -> float:
        prob = 1.0
        for hidden_bit, out_bit in zip(str(true_label), str(concrete_label)):
            if out_bit == "?":
                prob *= 1.0 - ratio
            else:
                bit_prob = 1.0 - flip_prob if out_bit == hidden_bit else flip_prob
                prob *= ratio * bit_prob
        return float(prob)

    if observed_label in {"amb_eq", "amb_neq"}:
        concrete_labels = ("00", "11") if observed_label == "amb_eq" else ("01", "10")
        return float(sum(concrete_prob(label) for label in concrete_labels) * ambiguity)

    likelihood = concrete_prob(observed_label)
    if "?" not in observed_label:
        likelihood *= 1.0 - ambiguity
    return float(likelihood)


def _posterior_no_error_probability(
    prior: dict[str, float],
    observed_label: str,
    *,
    cfg: SyndromeObservationConfig,
) -> float:
    numer = float(prior["00"] * _observation_likelihood("00", observed_label, cfg=cfg))
    denom = 0.0
    for label in SYNDROME_LABELS:
        denom += float(prior[label] * _observation_likelihood(label, observed_label, cfg=cfg))
    if denom <= 0.0:
        return 0.0
    return float(numer / denom)


def _information_fraction(observed_label: str) -> float:
    if observed_label in {"amb_eq", "amb_neq"}:
        return 0.5
    known = sum(1 for bit in str(observed_label) if bit in {"0", "1"})
    return float(known / 2.0)


def _observed_no_error_support(true_no_error_prob: float, observed_label: str) -> float:
    info_fraction = _information_fraction(observed_label)
    if observed_label == "amb_eq":
        compatibility = 0.5
    elif observed_label == "amb_neq":
        compatibility = 0.0
    else:
        chars = [ch for ch in str(observed_label) if ch in {"0", "1", "?"}]
        compatibility = 1.0 if all(ch in {"0", "?"} for ch in chars) else 0.0
    return float(_clip_unit_interval(true_no_error_prob) * info_fraction * compatibility)


def observe_syndrome_statistics(
    stats: dict[str, Any],
    *,
    cfg: SyndromeObservationConfig,
    rng: np.random.Generator,
) -> dict[str, Any]:
    # NOTE:
    # For imperfect syndrome experiments we distinguish between:
    #   (1) posterior_no_error_probability: Bayesian posterior for "no error"
    #       conditioned on the observed syndrome symbol, and
    #   (2) observed_no_error_support: a conservative policy-facing support
    #       score that is additionally down-weighted by information loss and
    #       by explicit incompatibility with the no-error label.
    #
    # The policy layer uses the conservative support quantity. For backward
    # compatibility the aggregate field mean_no_error_probability is kept as an
    # alias of mean_observed_no_error_support.
    if _is_clean_observation(cfg):
        observed_rows: list[dict[str, Any]] = []
        observed_labels: list[str] = []
        for item in stats.get("per_slice", []):
            tau_index = int(item.get("tau_index", len(observed_rows)))
            prior = _normalize_prior(dict(item.get("probabilities", {})))
            dominant_label = max(prior, key=prior.get)
            observed_rows.append(
                {
                    "tau_index": tau_index,
                    "true_label": str(dominant_label),
                    "observed_label": str(dominant_label),
                    "posterior_no_error_probability": float(item.get("no_error_probability", prior["00"])),
                    "observed_no_error_probability": float(item.get("no_error_probability", prior["00"])),
                    "is_partial": False,
                    "is_corrupted": False,
                    "is_ambiguous": False,
                    "information_fraction": 1.0,
                }
            )
            observed_labels.append(str(dominant_label))
        label_counts: dict[str, int] = {}
        for label in observed_labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        dominant_label = None
        dominant_rate = 0.0
        if label_counts:
            dominant_label = max(label_counts, key=label_counts.get)
            dominant_rate = float(label_counts[dominant_label] / len(observed_labels))
        return {
            "dominant": dominant_label,
            "dominant_rate": float(dominant_rate),
            "per_slice": observed_rows,
            "mean_posterior_no_error_probability": float(stats.get("mean_no_error_probability", 0.0)),
            "mean_observed_no_error_support": float(stats.get("mean_no_error_probability", 0.0)),
            "mean_no_error_probability": float(stats.get("mean_no_error_probability", 0.0)),
            "min_no_error_probability": float(stats.get("min_no_error_probability", 0.0)),
            "syndrome_corruption_rate": 0.0,
            "syndrome_information_loss": 0.0,
            "is_syndrome_partial": False,
            "is_syndrome_corrupted": False,
            "is_syndrome_ambiguous": False,
            "observation_ratio": 1.0,
            "noise_prob": 0.0,
            "ambiguity_level": 0.0,
            "measurement_error_prob": 0.0,
            "reset_error_prob": 0.0,
            "consistency_threshold": _clip_unit_interval(cfg.consistency_threshold),
        }

    observed_rows: list[dict[str, Any]] = []
    observed_labels: list[str] = []
    posterior_values: list[float] = []
    posterior_no_error: list[float] = []
    information_fractions: list[float] = []
    corruption_flags: list[bool] = []
    partial_flags: list[bool] = []
    ambiguous_flags: list[bool] = []

    for item in stats.get("per_slice", []):
        tau_index = int(item.get("tau_index", len(observed_rows)))
        prior = _normalize_prior(dict(item.get("probabilities", {})))
        true_label = _sample_hidden_label(prior, rng)
        observed_label, is_partial, is_corrupted, is_ambiguous = _sample_observed_label(
            true_label,
            cfg=cfg,
            rng=rng,
        )
        posterior = _posterior_no_error_probability(prior, observed_label, cfg=cfg)
        info_fraction = _information_fraction(observed_label)
        observed_no_error = _observed_no_error_support(prior["00"], observed_label)
        observed_rows.append(
            {
                "tau_index": tau_index,
                "true_label": str(true_label),
                "observed_label": str(observed_label),
                "posterior_no_error_probability": float(posterior),
                "observed_no_error_probability": float(observed_no_error),
                "is_partial": bool(is_partial),
                "is_corrupted": bool(is_corrupted),
                "is_ambiguous": bool(is_ambiguous),
                "information_fraction": float(info_fraction),
            }
        )
        observed_labels.append(str(observed_label))
        posterior_values.append(float(posterior))
        posterior_no_error.append(float(observed_no_error))
        information_fractions.append(float(info_fraction))
        corruption_flags.append(bool(is_corrupted))
        partial_flags.append(bool(is_partial))
        ambiguous_flags.append(bool(is_ambiguous))

    label_counts: dict[str, int] = {}
    for label in observed_labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    dominant_label = None
    dominant_rate = 0.0
    if label_counts:
        dominant_label = max(label_counts, key=label_counts.get)
        dominant_rate = float(label_counts[dominant_label] / len(observed_labels))

    mean_no_error = float(np.mean(posterior_no_error)) if posterior_no_error else 0.0
    mean_posterior = float(np.mean(posterior_values)) if posterior_values else 0.0
    min_no_error = float(np.min(posterior_no_error)) if posterior_no_error else 0.0
    info_loss = 1.0 - (float(np.mean(information_fractions)) if information_fractions else 0.0)
    corruption_rate = float(np.mean(corruption_flags)) if corruption_flags else 0.0

    return {
        "dominant": dominant_label,
        "dominant_rate": float(dominant_rate),
        "per_slice": observed_rows,
        "mean_posterior_no_error_probability": float(mean_posterior),
        "mean_observed_no_error_support": float(mean_no_error),
        "mean_no_error_probability": float(mean_no_error),
        "min_no_error_probability": float(min_no_error),
        "syndrome_corruption_rate": float(corruption_rate),
        "syndrome_information_loss": float(info_loss),
        "is_syndrome_partial": bool(any(partial_flags)),
        "is_syndrome_corrupted": bool(any(corruption_flags)),
        "is_syndrome_ambiguous": bool(any(ambiguous_flags)),
        "observation_ratio": _clip_unit_interval(cfg.observation_ratio),
        "noise_prob": _clip_unit_interval(cfg.noise_prob),
        "ambiguity_level": _clip_unit_interval(cfg.ambiguity_level),
        "measurement_error_prob": _clip_unit_interval(cfg.measurement_error_prob),
        "reset_error_prob": _clip_unit_interval(cfg.reset_error_prob),
        "consistency_threshold": _clip_unit_interval(cfg.consistency_threshold),
    }
