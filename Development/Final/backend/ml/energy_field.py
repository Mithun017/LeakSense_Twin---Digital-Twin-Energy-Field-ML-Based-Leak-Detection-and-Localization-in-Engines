"""
LeakSense Twin — Energy Field Detector
Computes inter-sensor correlation manifold and detects distortions indicating leaks.
"""

import numpy as np
import joblib


# Primary physical channels for energy field computation
CHANNELS = ['MAF', 'MAP_boost', 'MAP_cac_out', 'T_cac_out',
            'T_exh_manifold', 'dP_dpf']

# Thermodynamic weighting matrix
# Higher weight to pressure-flow relationships (most sensitive to leaks)
THERMO_WEIGHTS = np.array([
    [1.0, 0.9, 0.8, 0.3, 0.2, 0.1],   # MAF row
    [0.9, 1.0, 0.95, 0.5, 0.3, 0.2],   # MAP_boost row
    [0.8, 0.95, 1.0, 0.6, 0.3, 0.2],   # MAP_cac_out row
    [0.3, 0.5, 0.6, 1.0, 0.7, 0.3],    # T_cac_out row
    [0.2, 0.3, 0.3, 0.7, 1.0, 0.5],    # T_exh_manifold row
    [0.1, 0.2, 0.2, 0.3, 0.5, 1.0],    # dP_dpf row
])

# Zone mapping from most disrupted sensor
ZONE_MAP = {
    'MAF': 'Zone 1 (Intake)',
    'MAP_boost': 'Zone 2 (Charge Air — pre-CAC)',
    'MAP_cac_out': 'Zone 2 (Charge Air — post-CAC)',
    'T_cac_out': 'Zone 2 (Charge Air — manifold)',
    'T_exh_manifold': 'Zone 3 (Exhaust — manifold)',
    'dP_dpf': 'Zone 3 (Exhaust — aftertreatment)',
}

# Global deviation threshold for leak detection
THRESHOLD_GLOBAL = 3.0


def compute_energy_field(sensor_window: np.ndarray) -> np.ndarray:
    """
    Compute the energy field (weighted correlation matrix) from a sensor window.

    Args:
        sensor_window: shape (T, 7) — T timesteps, 7 channels
                      [MAF, MAP_boost, MAP_cac_out, T_cac_out,
                       T_exh_manifold, dP_dpf, RPM]

    Returns:
        energy_field of shape (6, 6) — normalized, weighted correlation matrix
    """
    # Step 1: Normalize by RPM (speed-corrected values)
    rpm = sensor_window[:, 6:7]
    rpm_safe = np.where(rpm < 100, 1800.0, rpm)
    normalized = sensor_window[:, :6] / (rpm_safe / 1800.0)

    # Step 2: Compute correlation matrix
    # Handle edge case of constant columns
    stds = np.std(normalized, axis=0)
    stds_safe = np.where(stds < 1e-10, 1.0, stds)
    normalized_z = (normalized - np.mean(normalized, axis=0)) / stds_safe

    corr_matrix = np.corrcoef(normalized_z.T)

    # Replace any NaN with 0 (happens if a channel is constant)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    # Step 3: Apply thermodynamic weighting
    energy_field = corr_matrix[:6, :6] * THERMO_WEIGHTS

    return energy_field


class EnergyFieldDetector:
    """
    Detects anomalies by comparing live energy fields to a healthy baseline.
    Uses Frobenius norm of z-score deviation and cosine similarity.
    """

    def __init__(self):
        self.healthy_field_mean = None
        self.healthy_field_std = None
        self.is_fitted = False

    def fit(self, healthy_windows: list):
        """
        Fit the detector on healthy engine data.

        Args:
            healthy_windows: list of arrays, each shape (T, 7)
        """
        fields = []
        for w in healthy_windows:
            if w.shape[0] >= 5:  # Need at least 5 timesteps
                fields.append(compute_energy_field(w))

        if len(fields) == 0:
            raise ValueError("No valid healthy windows provided")

        fields = np.array(fields)
        self.healthy_field_mean = np.mean(fields, axis=0)
        self.healthy_field_std = np.std(fields, axis=0) + 1e-8
        self.is_fitted = True

    def compute_deviation(self, live_window: np.ndarray) -> dict:
        """
        Compute energy field deviation from healthy baseline.

        Args:
            live_window: shape (T, 7) sensor data

        Returns:
            dict with deviation metrics and zone localization
        """
        if not self.is_fitted:
            # Return defaults if not fitted
            return {
                'energy_field': np.zeros((6, 6)).tolist(),
                'z_field': np.zeros((6, 6)).tolist(),
                'global_deviation_score': 0.0,
                'cosine_similarity': 1.0,
                'row_deviations': [0.0] * 6,
                'most_disrupted_sensor': 'N/A',
                'suspected_zone': 'Unknown',
                'leak_detected': False,
            }

        live_field = compute_energy_field(live_window)

        # Z-score deviation from healthy baseline
        z_field = (live_field - self.healthy_field_mean) / self.healthy_field_std

        # Global deviation score: Frobenius norm
        global_deviation = float(np.linalg.norm(z_field, 'fro'))

        # Cosine similarity between live and healthy field
        flat_live = live_field.flatten()
        flat_healthy = self.healthy_field_mean.flatten()
        norm_product = np.linalg.norm(flat_live) * np.linalg.norm(flat_healthy) + 1e-8
        cos_sim = float(np.dot(flat_live, flat_healthy) / norm_product)

        # Per-sensor disruption
        row_deviations = np.linalg.norm(z_field, axis=1)
        most_disrupted_idx = int(np.argmax(row_deviations))
        most_disrupted_sensor = CHANNELS[most_disrupted_idx]

        return {
            'energy_field': live_field.tolist(),
            'z_field': z_field.tolist(),
            'global_deviation_score': global_deviation,
            'cosine_similarity': cos_sim,
            'row_deviations': row_deviations.tolist(),
            'most_disrupted_sensor': most_disrupted_sensor,
            'suspected_zone': ZONE_MAP.get(most_disrupted_sensor, 'Unknown'),
            'leak_detected': global_deviation > THRESHOLD_GLOBAL,
        }

    def get_field_features(self, live_window: np.ndarray) -> np.ndarray:
        """
        Extract flattened z-field features for ML input.
        Returns 36-dimensional feature vector (6×6 matrix flattened).
        """
        if not self.is_fitted:
            return np.zeros(36)

        live_field = compute_energy_field(live_window)
        z_field = (live_field - self.healthy_field_mean) / self.healthy_field_std
        return z_field.flatten()

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> 'EnergyFieldDetector':
        return joblib.load(path)
