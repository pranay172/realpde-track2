from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "realpde_t2_starting_kit_v6"))

from realpde_t2.spectral_diagnostics import (  # noqa: E402
    average_ranks,
    log_spectrum_rmse,
    spatial_radial_spectrum,
    spearman_correlation,
    summarize_tke_moments,
    temporal_error_sums,
    tke_components,
    tke_map_moments,
)

import scoring  # noqa: E402


class SpectralDiagnosticTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(7)
        self.shape = (3, 5, 8, 10, 3)

    def test_temporal_mean_fluctuation_decomposition_is_exact(self) -> None:
        target = self.rng.normal(size=self.shape)
        prediction = target + self.rng.normal(scale=0.2, size=self.shape)
        sums = temporal_error_sums(prediction, target)
        np.testing.assert_allclose(
            sums["total_sse_by_channel"],
            sums["mean_sse_by_channel"] + sums["fluctuation_sse_by_channel"],
            rtol=0.0,
            atol=1e-10,
        )
        expected = np.sum((prediction[..., :2] - target[..., :2]) ** 2, axis=(0, 1, 2, 3))
        np.testing.assert_allclose(sums["total_sse_by_channel"], expected)

    def test_tke_components_match_official_kinetic_energy(self) -> None:
        field = self.rng.normal(size=self.shape)
        ours = tke_components(field).sum(axis=-1)
        official = scoring.kinetic_energy(field[..., :2])
        np.testing.assert_allclose(ours, official, rtol=0.0, atol=1e-12)

    def test_combined_tke_map_summary_preserves_official_cross_term(self) -> None:
        target = self.rng.normal(size=self.shape)
        prediction = 0.6 * target + self.rng.normal(scale=0.1, size=self.shape)
        summary = summarize_tke_moments(tke_map_moments(prediction, target))["combined"]
        prediction_map = scoring.kinetic_energy(prediction[..., :2])
        target_map = scoring.kinetic_energy(target[..., :2])
        expected_amplitude = np.linalg.norm(prediction_map) / np.linalg.norm(target_map)
        expected_rel_l2 = np.linalg.norm(prediction_map - target_map) / np.linalg.norm(target_map)
        self.assertAlmostEqual(summary["amplitude_ratio"], expected_amplitude)
        self.assertAlmostEqual(summary["map_rel_l2_global"], expected_rel_l2)

    def test_rfft_spectrum_obeys_parseval_and_has_deterministic_bands(self) -> None:
        field = self.rng.normal(size=self.shape)
        rmax = float(np.sqrt(0.5**2 + 0.5**2))
        output = spatial_radial_spectrum(
            field,
            radial_bin_count=8,
            band_edges=[0.0, 0.2, 0.4, rmax],
        )
        centered = field[..., :2] - field[..., :2].mean(axis=1, keepdims=True)
        expected_energy = np.mean(centered**2, axis=(1, 2, 3))
        np.testing.assert_allclose(
            output["radial_energy"].sum(axis=-1), expected_energy, rtol=1e-12, atol=1e-12
        )
        self.assertEqual(output["radial_energy"].shape, (3, 2, 8))
        self.assertEqual(int(output["band_mode_counts"].sum()), 8 * 6 - 1)
        self.assertTrue(np.all(output["band_mode_counts"] > 0))

    def test_zero_spectra_have_finite_zero_log_error(self) -> None:
        zeros = np.zeros(self.shape, dtype=np.float64)
        rmax = float(np.sqrt(0.5**2 + 0.5**2))
        spectrum = spatial_radial_spectrum(
            zeros, radial_bin_count=6, band_edges=[0.0, 0.2, 0.4, rmax]
        )["radial_energy"]
        errors = log_spectrum_rmse(spectrum, spectrum, epsilon=1e-12)
        np.testing.assert_allclose(errors, np.zeros(self.shape[0]))
        self.assertTrue(np.all(np.isfinite(errors)))

    def test_spearman_uses_average_tie_ranks(self) -> None:
        values = np.asarray([4.0, 1.0, 1.0, 3.0])
        np.testing.assert_allclose(average_ranks(values), [4.0, 1.5, 1.5, 3.0])
        self.assertAlmostEqual(spearman_correlation(values, values), 1.0)


if __name__ == "__main__":
    unittest.main()
