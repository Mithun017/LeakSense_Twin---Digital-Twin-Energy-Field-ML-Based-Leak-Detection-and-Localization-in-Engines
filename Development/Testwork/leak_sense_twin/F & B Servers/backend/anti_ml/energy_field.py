import numpy as np
import joblib

class EnergyFieldDetector:
    def __init__(self):
        self.CHANNELS = ['MAF', 'MAP_boost', 'MAP_cac_out', 'T_cac_out',
                         'T_exh_manifold', 'T_dpf_out', 'RPM']
        self.healthy_field_mean = None
        self.healthy_field_std = None
        self.THRESHOLD_GLOBAL = 2.5 # Adjusted based on Frobenius norm typical values

    def compute_energy_field(self, sensor_window: np.ndarray) -> np.ndarray:
        """
        sensor_window: shape (T, 7)
        """
        # Step 1: Normalize by RPM
        rpm = sensor_window[:, 6:7]
        normalized = sensor_window[:, :6] / (rpm / 1800.0 + 1e-8)
        
        # Step 2: Correlation matrix
        corr_matrix = np.corrcoef(normalized.T)
        
        # Step 3: Thermodynamic weighting (6x6)
        weights = np.array([
            [1.0, 0.9, 0.8, 0.3, 0.2, 0.1],
            [0.9, 1.0, 0.95, 0.5, 0.3, 0.2],
            [0.8, 0.95, 1.0, 0.6, 0.3, 0.2],
            [0.3, 0.5, 0.6, 1.0, 0.7, 0.3],
            [0.2, 0.3, 0.3, 0.7, 1.0, 0.5],
            [0.1, 0.2, 0.2, 0.3, 0.5, 1.0],
        ])
        
        energy_field = corr_matrix * weights
        return energy_field

    def fit(self, healthy_windows):
        """
        healthy_windows: list of np.ndarray
        """
        fields = [self.compute_energy_field(w) for w in healthy_windows]
        self.healthy_field_mean = np.mean(fields, axis=0)
        self.healthy_field_std = np.std(fields, axis=0) + 1e-8
        print("EnergyFieldDetector fitted.")

    def compute_deviation(self, live_window: np.ndarray) -> dict:
        live_field = self.compute_energy_field(live_window)
        
        # Z-score deviation
        z_field = (live_field - self.healthy_field_mean) / self.healthy_field_std
        
        # Global deviation (Frobenius norm)
        global_deviation = np.linalg.norm(z_field, 'fro')
        
        # Cosine similarity
        cos_sim = np.dot(live_field.flatten(), self.healthy_field_mean.flatten()) / \
                  (np.linalg.norm(live_field) * np.linalg.norm(self.healthy_field_mean) + 1e-8)
        
        # Zone-specific disruption
        row_deviations = np.linalg.norm(z_field, axis=1)
        most_disrupted_idx = np.argmax(row_deviations)
        most_disrupted_sensor = self.CHANNELS[most_disrupted_idx]
        
        zone_map = {
            'MAF': 'Zone 1 (Intake)',
            'MAP_boost': 'Zone 2 (Charge Air — pre-CAC)',
            'MAP_cac_out': 'Zone 2 (Charge Air — post-CAC)',
            'T_cac_out': 'Zone 2 (Charge Air — manifold)',
            'T_exh_manifold': 'Zone 3 (Exhaust — manifold)',
            'T_dpf_out': 'Zone 3 (Exhaust — aftertreatment)',
        }
        
        return {
            'energy_field': live_field.tolist(),
            'z_field': z_field.tolist(),
            'global_deviation_score': float(global_deviation),
            'cosine_similarity': float(cos_sim),
            'most_disrupted_sensor': most_disrupted_sensor,
            'suspected_zone': zone_map.get(most_disrupted_sensor, 'Unknown'),
            'leak_detected': bool(global_deviation > self.THRESHOLD_GLOBAL)
        }

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
