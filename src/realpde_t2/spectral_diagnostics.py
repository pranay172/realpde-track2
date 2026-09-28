"""Train-side mean/fluctuation and spatial-spectrum diagnostics for Track 2.

All functions operate on physical-space fields with layout ``(N, T, H, W, C)``.
The official TKE functional is temporal variance at each spatial point; it is
therefore reported separately from the spatial radial spectrum of the centered
fluctuation fields.  The latter uses an rFFT with explicit Hermitian weighting
and is normalized so its summed energy equals the mean spatial fluctuation
energy (Parseval), in physical velocity-squared units.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


MEASURED_CHANNELS = ("u", "v")


def _physical_uv(field: np.ndarray) -> np.ndarray:
    value = np.asarray(field, dtype=np.float64)
    if value.ndim != 5 or value.shape[-1] < len(MEASURED_CHANNELS):
        raise ValueError(f"expected (N,T,H,W,C>=2), got {value.shape}")
    if value.shape[0] < 1 or value.shape[1] < 1 or value.shape[2] < 1 or value.shape[3] < 1:
        raise ValueError(f"field has an empty non-channel dimension: {value.shape}")
    if not np.all(np.isfinite(value[..., :2])):
        raise ValueError("u/v field contains non-finite values")
    return value[..., :2]


def _region(field: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return field
    region_mask = np.asarray(mask, dtype=bool)
    if region_mask.shape != field.shape[2:4]:
        raise ValueError(
            f"region mask shape {region_mask.shape} != spatial shape {field.shape[2:4]}"
        )
    if not np.any(region_mask):
        raise ValueError("region mask is empty")
    # Boolean indexing deliberately flattens H/W into an unambiguous list of cells.
    return field[:, :, region_mask, :]


def temporal_error_sums(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    region_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Return exact temporal mean/fluctuation SSE components for u and v.

    For every sample and spatial cell, ``error = mean_t(error) + centered(error)``
    is orthogonal.  Consequently ``total_sse = mean_sse + fluctuation_sse`` up
    to floating point roundoff.  Sums are not averaged so results can be added
    exactly across batches or trajectory partitions.
    """
    pred = _region(_physical_uv(prediction), region_mask)
    tgt = _region(_physical_uv(target), region_mask)
    if pred.shape != tgt.shape:
        raise ValueError(f"prediction/target shape mismatch: {pred.shape}/{tgt.shape}")
    error = pred - tgt
    temporal_mean_error = error.mean(axis=1)
    centered_error = error - temporal_mean_error[:, None, ...]
    field_axes = tuple(range(error.ndim - 1))
    mean_axes = tuple(range(temporal_mean_error.ndim - 1))
    total = np.sum(error**2, axis=field_axes, dtype=np.float64)
    mean = float(error.shape[1]) * np.sum(
        temporal_mean_error**2, axis=mean_axes, dtype=np.float64
    )
    fluctuation = np.sum(centered_error**2, axis=field_axes, dtype=np.float64)
    return {
        "total_sse_by_channel": total,
        "mean_sse_by_channel": mean,
        "fluctuation_sse_by_channel": fluctuation,
    }


def tke_components(field: np.ndarray) -> np.ndarray:
    """Return 0.5 temporal-variance components, shape ``(N,H,W,2)``.

    Summing the last axis exactly gives the organizer's TKE map.  Keeping the
    channel components allows a useful u/v diagnosis without redefining TKE.
    """
    value = _physical_uv(field)
    centered = value - value.mean(axis=1, keepdims=True)
    return 0.5 * np.mean(centered**2, axis=1, dtype=np.float64)


def tke_map_moments(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    region_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Sufficient TKE-map sums for energy, amplitude, and Rel-L2 summaries."""
    pred = tke_components(prediction)
    tgt = tke_components(target)
    if region_mask is not None:
        mask = np.asarray(region_mask, dtype=bool)
        if mask.shape != pred.shape[1:3] or not np.any(mask):
            raise ValueError("invalid TKE region mask")
        pred = pred[:, mask, :]
        tgt = tgt[:, mask, :]
    axes = tuple(range(pred.ndim - 1))
    pred_map = pred.sum(axis=-1)
    tgt_map = tgt.sum(axis=-1)
    return {
        "prediction_energy_by_channel": np.sum(pred, axis=axes, dtype=np.float64),
        "target_energy_by_channel": np.sum(tgt, axis=axes, dtype=np.float64),
        "prediction_squared_norm_by_channel": np.sum(pred**2, axis=axes, dtype=np.float64),
        "target_squared_norm_by_channel": np.sum(tgt**2, axis=axes, dtype=np.float64),
        "difference_squared_norm_by_channel": np.sum((pred - tgt) ** 2, axis=axes, dtype=np.float64),
        # The official TKE map is TKE_u + TKE_v.  Preserve its cross-term for
        # combined map norms instead of summing the two component norm squares.
        "prediction_squared_norm_official_map": np.sum(pred_map**2, dtype=np.float64),
        "target_squared_norm_official_map": np.sum(tgt_map**2, dtype=np.float64),
        "difference_squared_norm_official_map": np.sum(
            (pred_map - tgt_map) ** 2, dtype=np.float64
        ),
    }


def radial_frequency_grid(height: int, width: int) -> np.ndarray:
    """Return rFFT radial frequencies in cycles per grid cell, shape ``(H,Wf)``."""
    if height < 2 or width < 2:
        raise ValueError(f"spatial grid must be at least 2x2, got {(height, width)}")
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.rfftfreq(width)[None, :]
    return np.sqrt(fy**2 + fx**2)


def _rfft_hermitian_weights(width: int) -> np.ndarray:
    """Multipliers recovering full-FFT energy from an rFFT along x."""
    width_frequency = width // 2 + 1
    weights = np.full(width_frequency, 2.0, dtype=np.float64)
    weights[0] = 1.0
    if width % 2 == 0:
        weights[-1] = 1.0
    return weights


def spatial_radial_spectrum(
    field: np.ndarray,
    *,
    radial_bin_count: int,
    band_edges: Sequence[float],
) -> dict[str, np.ndarray]:
    """Compute per-window radial energy spectra of temporal fluctuations.

    ``radial_energy`` has shape ``(N, 2, radial_bin_count)`` and stores the
    sum of energy of all modes in a radial bin, averaged over the 20 temporal
    frames.  Its sum over bins equals ``mean_{t,h,w}(u'²)`` or ``v'²``.  The
    zero spatial mode is retained in the fine bins but excluded from the named
    low/mid/high band energies, avoiding a spatial-mean contribution in the
    high-vs-low route gate.
    """
    value = _physical_uv(field)
    if radial_bin_count < 1:
        raise ValueError("radial_bin_count must be positive")
    edges = np.asarray(tuple(float(edge) for edge in band_edges), dtype=np.float64)
    if edges.ndim != 1 or len(edges) != 4 or edges[0] != 0.0 or np.any(np.diff(edges) <= 0):
        raise ValueError("band_edges must be four strictly increasing values starting at 0")
    height, width = value.shape[2:4]
    radius = radial_frequency_grid(height, width)
    max_radius = float(radius.max())
    if not np.isclose(edges[-1], max_radius, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"last band edge {edges[-1]} must equal rFFT max radius {max_radius}"
        )

    centered = value - value.mean(axis=1, keepdims=True)
    transformed = np.fft.rfft2(centered, axes=(2, 3))
    scale = float(height * width) ** 2
    power = np.abs(transformed) ** 2 / scale
    power *= _rfft_hermitian_weights(width)[None, None, None, :, None]

    # Equal-width fine bins including the maximum-frequency corner in the last bin.
    fine_edges = np.linspace(0.0, max_radius, radial_bin_count + 1, dtype=np.float64)
    fine_index = np.minimum(
        (radius / max_radius * radial_bin_count).astype(np.int64), radial_bin_count - 1
    )
    radial_energy = np.zeros(
        (value.shape[0], len(MEASURED_CHANNELS), radial_bin_count), dtype=np.float64
    )
    fine_counts = np.zeros(radial_bin_count, dtype=np.int64)
    for index in range(radial_bin_count):
        mode_mask = fine_index == index
        fine_counts[index] = int(np.count_nonzero(mode_mask))
        if fine_counts[index]:
            radial_energy[:, :, index] = np.sum(
                power[:, :, mode_mask, :], axis=(1, 2), dtype=np.float64
            ) / float(value.shape[1])

    band_energy = np.zeros((value.shape[0], len(MEASURED_CHANNELS), 3), dtype=np.float64)
    band_counts = np.zeros(3, dtype=np.int64)
    # Exclude the DC spatial mode. The temporal centering does not necessarily
    # make this spatial mode zero, so silently putting it in "low" would blur
    # a spatial-spectrum diagnosis.
    positive_radius = radius > 0.0
    for index, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mode_mask = positive_radius & (radius > lo) & (radius <= hi)
        band_counts[index] = int(np.count_nonzero(mode_mask))
        if band_counts[index] == 0:
            raise ValueError(f"spectral band {index} contains no rFFT modes")
        band_energy[:, :, index] = np.sum(
            power[:, :, mode_mask, :], axis=(1, 2), dtype=np.float64
        ) / float(value.shape[1])
    zero_energy = power[:, :, radius == 0.0, :].sum(axis=(1, 2), dtype=np.float64) / float(
        value.shape[1]
    )
    return {
        "radial_energy": radial_energy,
        "radial_bin_edges": fine_edges,
        "radial_bin_mode_counts": fine_counts,
        "band_energy": band_energy,
        "band_edges": edges,
        "band_mode_counts": band_counts,
        "zero_mode_energy": zero_energy,
    }


def log_spectrum_rmse(
    prediction_spectrum: np.ndarray,
    target_spectrum: np.ndarray,
    *,
    epsilon: float,
) -> np.ndarray:
    """Per-window RMSE between log radial-bin energies over u/v and bins."""
    pred = np.asarray(prediction_spectrum, dtype=np.float64)
    tgt = np.asarray(target_spectrum, dtype=np.float64)
    if pred.shape != tgt.shape or pred.ndim != 3 or pred.shape[1] != 2:
        raise ValueError(f"expected matching (N,2,bins) spectra, got {pred.shape}/{tgt.shape}")
    if epsilon <= 0 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be finite and positive")
    if np.any(pred < 0) or np.any(tgt < 0):
        raise ValueError("spectral energy must be non-negative")
    delta = np.log(pred + epsilon) - np.log(tgt + epsilon)
    return np.sqrt(np.mean(delta**2, axis=(1, 2), dtype=np.float64))


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Finite Pearson r, or NaN when a correlation is undefined."""
    left = np.asarray(x, dtype=np.float64).reshape(-1)
    right = np.asarray(y, dtype=np.float64).reshape(-1)
    if left.size != right.size or left.size < 2:
        raise ValueError("correlation requires matching arrays with at least two values")
    if not (np.all(np.isfinite(left)) and np.all(np.isfinite(right))):
        raise ValueError("correlation inputs must be finite")
    left = left - left.mean()
    right = right - right.mean()
    denom = float(np.sqrt(np.sum(left**2) * np.sum(right**2)))
    return float(np.sum(left * right) / denom) if denom > 0.0 else float("nan")


def average_ranks(value: np.ndarray) -> np.ndarray:
    """Average-tie ranks, equivalent to the usual Spearman convention."""
    data = np.asarray(value, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(data)):
        raise ValueError("ranks require finite values")
    order = np.argsort(data, kind="mergesort")
    sorted_data = data[order]
    ranks = np.empty(data.size, dtype=np.float64)
    start = 0
    while start < data.size:
        stop = start + 1
        while stop < data.size and sorted_data[stop] == sorted_data[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho using average ranks; NaN for constant inputs."""
    return pearson_correlation(average_ranks(x), average_ranks(y))


def summarize_error_sums(sums: dict[str, np.ndarray]) -> dict[str, Any]:
    """Convert accumulated temporal decomposition sums to JSON-safe summaries."""
    total = np.asarray(sums["total_sse_by_channel"], dtype=np.float64)
    mean = np.asarray(sums["mean_sse_by_channel"], dtype=np.float64)
    fluctuation = np.asarray(sums["fluctuation_sse_by_channel"], dtype=np.float64)
    if total.shape != (2,) or mean.shape != (2,) or fluctuation.shape != (2,):
        raise ValueError("expected exactly u/v temporal decomposition sums")

    def one(total_sse: float, mean_sse: float, fluct_sse: float) -> dict[str, float]:
        closure = total_sse - mean_sse - fluct_sse
        return {
            "total_sse": float(total_sse),
            "mean_sse": float(mean_sse),
            "fluctuation_sse": float(fluct_sse),
            "mean_share": float(mean_sse / total_sse) if total_sse > 0 else float("nan"),
            "fluctuation_share": float(fluct_sse / total_sse)
            if total_sse > 0
            else float("nan"),
            "closure_abs_sse": float(abs(closure)),
            "closure_relative": float(abs(closure) / total_sse) if total_sse > 0 else float("nan"),
        }

    return {
        "combined": one(float(total.sum()), float(mean.sum()), float(fluctuation.sum())),
        "channels": {
            channel: one(float(total[index]), float(mean[index]), float(fluctuation[index]))
            for index, channel in enumerate(MEASURED_CHANNELS)
        },
    }


def summarize_tke_moments(moments: dict[str, np.ndarray]) -> dict[str, Any]:
    """Convert accumulated TKE-map sufficient statistics to JSON-safe ratios."""
    names = (
        "prediction_energy_by_channel",
        "target_energy_by_channel",
        "prediction_squared_norm_by_channel",
        "target_squared_norm_by_channel",
        "difference_squared_norm_by_channel",
    )
    values = {name: np.asarray(moments[name], dtype=np.float64) for name in names}
    if any(value.shape != (2,) for value in values.values()):
        raise ValueError("expected exactly u/v TKE moments")

    official_names = (
        "prediction_squared_norm_official_map",
        "target_squared_norm_official_map",
        "difference_squared_norm_official_map",
    )
    official = {name: float(moments[name]) for name in official_names}
    if not all(np.isfinite(value) and value >= 0.0 for value in official.values()):
        raise ValueError("official combined TKE-map moments must be finite and non-negative")

    def one(
        *,
        pred_energy: float,
        target_energy: float,
        pred_norm_sq: float,
        target_norm_sq: float,
        diff_norm_sq: float,
    ) -> dict[str, float]:
        energy_ratio = pred_energy / target_energy if target_energy > 0 else float("nan")
        amplitude_ratio = (
            np.sqrt(pred_norm_sq / target_norm_sq) if target_norm_sq > 0 else float("nan")
        )
        rel_l2 = np.sqrt(diff_norm_sq / target_norm_sq) if target_norm_sq > 0 else float("nan")
        return {
            "prediction_energy": pred_energy,
            "target_energy": target_energy,
            "energy_ratio": float(energy_ratio),
            "prediction_l2_amplitude": float(np.sqrt(pred_norm_sq)),
            "target_l2_amplitude": float(np.sqrt(target_norm_sq)),
            "amplitude_ratio": float(amplitude_ratio),
            "map_rel_l2_global": float(rel_l2),
        }

    return {
        "combined": one(
            pred_energy=float(values["prediction_energy_by_channel"].sum()),
            target_energy=float(values["target_energy_by_channel"].sum()),
            pred_norm_sq=official["prediction_squared_norm_official_map"],
            target_norm_sq=official["target_squared_norm_official_map"],
            diff_norm_sq=official["difference_squared_norm_official_map"],
        ),
        "channels": {
            channel: one(
                pred_energy=float(values["prediction_energy_by_channel"][index]),
                target_energy=float(values["target_energy_by_channel"][index]),
                pred_norm_sq=float(values["prediction_squared_norm_by_channel"][index]),
                target_norm_sq=float(values["target_squared_norm_by_channel"][index]),
                diff_norm_sq=float(values["difference_squared_norm_by_channel"][index]),
            )
            for index, channel in enumerate(MEASURED_CHANNELS)
        },
    }


def summarize_spectrum(
    prediction_sum: np.ndarray,
    target_sum: np.ndarray,
    *,
    samples: int,
    radial_bin_edges: np.ndarray,
    radial_bin_mode_counts: np.ndarray,
    prediction_band_sum: np.ndarray,
    target_band_sum: np.ndarray,
    band_edges: np.ndarray,
    band_mode_counts: np.ndarray,
    prediction_zero_sum: np.ndarray,
    target_zero_sum: np.ndarray,
) -> dict[str, Any]:
    """Summarize accumulated radial energy in physical units."""
    if samples < 1:
        raise ValueError("spectrum summary requires at least one sample")
    pred = np.asarray(prediction_sum, dtype=np.float64)
    tgt = np.asarray(target_sum, dtype=np.float64)
    pred_band = np.asarray(prediction_band_sum, dtype=np.float64)
    tgt_band = np.asarray(target_band_sum, dtype=np.float64)
    if pred.shape != tgt.shape or pred.shape[0] != 2:
        raise ValueError("spectrum sums must have matching u/v shape")
    if pred_band.shape != (2, 3) or tgt_band.shape != (2, 3):
        raise ValueError("band sums must have shape (2,3)")

    def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        result = np.full(np.asarray(numerator).shape, np.nan, dtype=np.float64)
        np.divide(numerator, denominator, out=result, where=denominator > 0)
        return result

    average_pred = pred / float(samples)
    average_tgt = tgt / float(samples)
    average_pred_band = pred_band / float(samples)
    average_tgt_band = tgt_band / float(samples)
    labels = ("low", "mid", "high")

    def bands_for(index: int | None) -> dict[str, dict[str, float]]:
        numerator = (
            np.sum(average_pred_band, axis=0)
            if index is None
            else average_pred_band[index]
        )
        denominator = (
            np.sum(average_tgt_band, axis=0)
            if index is None
            else average_tgt_band[index]
        )
        ratios = safe_ratio(numerator, denominator)
        return {
            label: {
                "prediction_energy": float(numerator[position]),
                "target_energy": float(denominator[position]),
                "prediction_target_ratio": float(ratios[position]),
            }
            for position, label in enumerate(labels)
        }

    def bins_for(index: int | None) -> dict[str, list[float]]:
        numerator = np.sum(average_pred, axis=0) if index is None else average_pred[index]
        denominator = np.sum(average_tgt, axis=0) if index is None else average_tgt[index]
        return {
            "prediction_energy": [float(value) for value in numerator],
            "target_energy": [float(value) for value in denominator],
            "prediction_target_ratio": [float(value) for value in safe_ratio(numerator, denominator)],
        }

    average_zero_pred = np.asarray(prediction_zero_sum, dtype=np.float64) / float(samples)
    average_zero_tgt = np.asarray(target_zero_sum, dtype=np.float64) / float(samples)
    return {
        "frequency_units": "cycles per grid cell",
        "energy_units": "physical velocity squared, mean over temporal frames and windows",
        "zero_mode_excluded_from_named_bands": True,
        "radial_bin_edges": [float(value) for value in radial_bin_edges],
        "radial_bin_mode_counts": [int(value) for value in radial_bin_mode_counts],
        "radial_bins": {
            "combined": bins_for(None),
            "channels": {channel: bins_for(index) for index, channel in enumerate(MEASURED_CHANNELS)},
        },
        "band_edges": [float(value) for value in band_edges],
        "band_mode_counts": [int(value) for value in band_mode_counts],
        "bands": {
            "combined": bands_for(None),
            "channels": {channel: bands_for(index) for index, channel in enumerate(MEASURED_CHANNELS)},
        },
        "zero_mode": {
            "combined": {
                "prediction_energy": float(average_zero_pred.sum()),
                "target_energy": float(average_zero_tgt.sum()),
                "prediction_target_ratio": float(
                    safe_ratio(np.asarray([average_zero_pred.sum()]), np.asarray([average_zero_tgt.sum()]))[0]
                ),
            },
            "channels": {
                channel: {
                    "prediction_energy": float(average_zero_pred[index]),
                    "target_energy": float(average_zero_tgt[index]),
                    "prediction_target_ratio": float(
                        safe_ratio(
                            np.asarray([average_zero_pred[index]]),
                            np.asarray([average_zero_tgt[index]]),
                        )[0]
                    ),
                }
                for index, channel in enumerate(MEASURED_CHANNELS)
            },
        },
    }


def describe_values(value: np.ndarray) -> dict[str, float]:
    """Compact deterministic descriptive statistics for finite per-window values."""
    data = np.asarray(value, dtype=np.float64).reshape(-1)
    if data.size < 1 or not np.all(np.isfinite(data)):
        raise ValueError("summary requires a non-empty finite vector")
    return {
        "count": int(data.size),
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "p10": float(np.quantile(data, 0.10)),
        "p90": float(np.quantile(data, 0.90)),
        "minimum": float(np.min(data)),
        "maximum": float(np.max(data)),
    }
