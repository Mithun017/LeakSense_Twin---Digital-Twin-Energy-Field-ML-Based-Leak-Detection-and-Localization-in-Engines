"""
Energy Field Detector for LeakSense Twin
Computes the energy field as a mathematical construct that encodes the relationships
between sensor parameters in a healthy engine.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
import joblib
import logging

logger = logging.getLogger(__name__)

# Define the 7 primary physical channels from the Energy Field prompt
CHANNELS = ['MAF', 'MAP_boost', 'MAP_cac_out', 'T_cac_out',
            'T_exh_manifold', 'dP_dpf', 'RPM']

def compute_energy_field(sensor_window: np.ndarray) -> np.ndarray:
    """
    Compute the energy field from a sensor window

    Args:
        sensor_window: shape (T, 7) — T timesteps, 7 channels

    Returns:
        energy_field of shape (6, 6) — normalized correlation matrix
        with thermodynamic weighting (excluding RPM which is used for normalization)
    """
    # Step 1: Normalize by RPM (speed-corrected values)
    # Assuming columns are in the order of CHANNELS
    rpm_idx = CHANNELS.index('RPM')
    rpm = sensor_window[:, rpm_idx:rpm_idx+1]

    # Normalize the first 6 channels by RPM (corrected to 1800 RPM basis)
    # Avoid division by zero or very small RPM values
    rpm_safe = np.maximum(rpm, 1e-6)
    normalized = sensor_window[:, :6] / (rpm_safe / 1800.0)

    # Step 2: Compute correlation matrix (shape relationships, not magnitudes)
    corr_matrix = np.corrcoef(normalized.T)  # (6, 6) correlation

    # Step 3: Apply thermodynamic weighting
    # Higher weight to pressure-flow relationships (most sensitive to leaks)
    weights = np.array([
        [1.0, 0.9, 0.8, 0.3, 0.2, 0.1],  # MAF row
        [0.9, 1.0, 0.95, 0.5, 0.3, 0.2], # MAP_boost row
        [0.8, 0.95, 1.0, 0.6, 0.3, 0.2], # MAP_cac_out row
        [0.3, 0.5, 0.6, 1.0, 0.7, 0.3],  # T_cac_out row
        [0.2, 0.3, 0.3, 0.7, 1.0, 0.5],  # T_exh_manifold row
        [0.1, 0.2, 0.2, 0.3, 0.5, 1.0],  # dP_dpf row
    ])
    energy_field = corr_matrix * weights
    return energy_field  # shape (6, 6)

class EnergyFieldDetector(BaseEstimator):
    """
    Energy Field Detector for leak detection based on deviations from healthy baseline
    """

    def __init__(self):
        self.healthy_field_mean = None  # fitted from healthy data
        self.healthy_field_std = None
        self.is_fitted = False

    def fit(self, healthy_windows: list):
        """
        Fit the energy field detector on healthy data windows

        Args:
            healthy_windows: list of numpy arrays, each shape (T, 7)
        """
        if len(healthy_windows) == 0:
            raise ValueError("No healthy windows provided for fitting")

        logger.info(f"Fitting Energy Field Detector on {len(healthy_windows)} healthy windows")

        # Compute energy field for each healthy window
        fields = [compute_energy_field(w) for w in healthy_windows]
        fields_array = np.array(fields)  # shape (n_windows, 6, 6)

        # Compute mean and std across windows
        self.healthy_field_mean = np.mean(fields_array, axis=0)
        self.healthy_field_std = np.std(fields_array, axis=0) + 1e-8  # Avoid division by zero

        self.is_fitted = True
        logger.info("Energy Field Detector fitted successfully")
        return self

    def compute_deviation(self, live_window: np.ndarray) -> dict:
        """
        Compute deviation of live window from healthy baseline

        Args:
            live_window: numpy array of shape (T, 7)

        Returns:
            Dictionary with deviation metrics
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before computing deviation")

        # Compute live energy field
        live_field = compute_energy_field(live_window)

        # Z-score deviation from healthy baseline
        z_field = (live_field - self.healthy_field_mean) / self.healthy_field_std

        # Global deviation score: Frobenius norm of z-score matrix
        global_deviation = np.linalg.norm(z_field, 'fro')

        # Cosine similarity between live and healthy field (shape preservation)
        cos_sim = np.dot(live_field.flatten(), self.healthy_field_mean.flatten()) / \
                  (np.linalg.norm(live_field) * np.linalg.norm(self.healthy_field_mean) + 1e-8)

        # Zone-specific deviation: which row/column is most disrupted?
        row_deviations = np.linalg.norm(z_field, axis=1)  # per-sensor disruption
        most_disrupted_sensor_idx = np.argmax(row_deviations)
        most_disrupted_sensor = CHANNELS[most_disrupted_sensor_idx]  # First 6 channels

        # Zone mapping from most disrupted sensor
        zone_map = {
            'MAF': 'Zone 1 (Intake)',
            'MAP_boost': 'Zone 2 (Charge Air — pre-CAC)',
            'MAP_cac_out': 'Zone 2 (Charge Air — post-CAC)',
            'T_cac_out': 'Zone 2 (Charge Air — manifold)',
            'T_exh_manifold': 'Zone 3 (Exhaust — manifold)',
            'dP_dpf': 'Zone 3 (Exhaust — aftertreatment)',
        }

        suspected_zone = zone_map.get(most_disrupted_sensor, 'Unknown')

        # Determine if leak is detected (threshold would be calibrated)
        # For now, we'll return the score and let the calling code apply threshold
        leak_detected = global_deviation > 0.0  # Placeholder - actual threshold would be calibrated

        return {
            'energy_field': live_field,
            'z_field': z_field,
            'global_deviation_score': float(global_deviation),
            'cosine_similarity': float(cos_sim),
            'row_deviations': row_deviations.tolist(),
            'most_disrupted_sensor': most_disrupted_sensor,
            'suspected_zone': suspected_zone,
            'leak_detected': leak_detected  # This would be based on calibrated threshold
        }

    def save(self, filepath: str):
        """Save the detector to disk"""
        joblib.dump(self, filepath)
        logger.info(f"Energy Field Detector saved to {filepath}")

    @classmethod
    def load(cls, filepath: str):
        """Load the detector from disk"""
        detector = joblib.load(filepath)
        logger.info(f"Energy Field Detector loaded from {filepath}")
        return detector

def create_energy_field_features(sensor_data: pd.DataFrame, window_size: int = 30) -> pd.DataFrame:
    """
    Create energy field features from sensor data for use in ML models

    Args:
        sensor_data: DataFrame with sensor columns including those in CHANNELS
        window_size: Size of rolling window for energy field computation

    Returns:
        DataFrame with added energy field features (36 features representing flattened 6x6 z-field matrices)
    """
    # Check if we have all required columns
    missing_cols = [col for col in CHANNELS if col not in sensor_data.columns]
    if missing_cols:
        logger.warning(f"Missing columns for energy field feature creation: {missing_cols}")
        # Return original data with zero-filled energy field columns
        result = sensor_data.copy()
        # Add 36 zero columns for energy field features
        for i in range(36):
            result[f'energy_field_{i:02d}'] = 0.0
        return result

    # Extract sensor data in the order of CHANNELS
    sensor_values = sensor_data[CHANNELS].values
    n_samples = len(sensor_values)

    # Initialize energy field features (36 features per sample = flattened 6x6 matrix)
    energy_features = np.zeros((n_samples, 36))

    # Compute energy field for each window
    for i in range(n_samples):
        start_idx = max(0, i - window_size + 1)
        end_idx = i + 1
        window = sensor_values[start_idx:end_idx]

        if window.shape[0] == window_size:
            # We have a full window, compute energy field deviation
            # Note: This requires a fitted detector, but we'll compute the raw energy field
            # and return it. The caller should apply deviation computation if needed.
            try:
                # Compute raw energy field (correlation matrix)
                energy_field = compute_energy_field(window)  # shape (6, 6)
                # Flatten and store
                energy_features[i] = energy_field.flatten()
            except Exception as e:
                logger.warning(f"Could not compute energy field for window {i}: {e}")
                # Leave as zeros (already initialized)
        # else: not enough data for a full window, leave as zeros (already initialized)

    # Add energy field features to the data
    result = sensor_data.copy()
    for i in range(36):
        result[f'energy_field_{i:02d}'] = energy_features[:, i]

    return result

if __name__ == "__main__":
    # Example usage
    import sys
    import os

    # Add the data_generation directory to the path
    data_gen_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'data_generation')
    if data_gen_path not in sys.path:
        sys.path.append(data_gen_path)

    # Import SyntheticDataGenerator for example usage
    from synthetic_data_generator import SyntheticDataGenerator

    # Generate sample data
    generator = SyntheticDataGenerator()
    df = generator.generate_dataset(n_samples=1000, healthy_ratio=0.8, random_seed=42)

    # Filter steady-state data for fitting
    steady_state_mask = df['is_steady_state'] == True
    df_ss = df[steady_state_mask].copy()

    # Extract sensor data in the order of CHANNELS
    sensor_cols = [col for col in CHANNELS if col in df_ss.columns]
    sensor_data = df_ss[sensor_cols].values

    # Split into windows for fitting (using first 80% of steady-state data for healthy baseline)
    n_healthy = int(len(sensor_data) * 0.8)
    healthy_data = sensor_data[:n_healthy]

    # Create windows of 30 samples each
    window_size = 30
    healthy_windows = []
    for i in range(0, len(healthy_data) - window_size + 1, window_size):
        window = healthy_data[i:i+window_size]
        if len(window) == window_size:
            healthy_windows.append(window)

    # Create and fit detector
    detector = EnergyFieldDetector()
    detector.fit(healthy_windows)

    # Test on remaining data (including some faulty samples)
    test_data = sensor_data[n_healthy:]
    test_windows = []
    for i in range(0, len(test_data) - window_size + 1, window_size):
        window = test_data[i:i+window_size]
        if len(window) == window_size:
            test_windows.append(window)

    # Compute deviations for test windows
    deviations = []
    for window in test_windows[:10]:  # Just test first 10 windows
        deviation = detector.compute_deviation(window)
        deviations.append(deviation)
        print(f"Window deviation score: {deviation['global_deviation_score']:.3f}, "
              f"Suspected zone: {deviation['suspected_zone']}")

    # Save detector
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
    os.makedirs(model_dir, exist_ok=True)
    detector.save(os.path.join(model_dir, 'energy_field_detector.joblib'))