"""
LeakSense Twin — Zone 1: Intake Digital Twin
Physics: Mass continuity — predicts MAF from RPM + ambient conditions.
If actual MAF < predicted MAF by threshold: Zone 1 leak detected.
"""

import numpy as np
import joblib
from pathlib import Path


class IntakeTwinModel:
    """
    Physics-based Intake Zone Digital Twin for Cat C18.
    Predicts expected Mass Air Flow (MAF) using volumetric efficiency model.
    """

    # Cat C18 constants
    DISPLACEMENT_M3 = 18.1 / 1000.0   # m³ (18.1L)
    N_CYLINDERS = 6
    R_AIR = 287.0                      # J/(kg·K)
    VE_BASE = 0.92

    def __init__(self):
        """Initialize with default polynomial VE coefficients."""
        # Polynomial VE model coefficients (calibrated for C18)
        # VE peaks around 1400 RPM
        self.ve_coeffs = {
            'base': 0.92,
            'rpm_peak': 1400.0,
            'quadratic_coeff': -1.2e-5 / 1e6,
        }
        self.correction_factor = 1.0  # Online learning correction

    def ve_model(self, rpm: float) -> float:
        """
        Volumetric efficiency as a function of RPM.
        VE peaks around 1400 RPM for the C18 due to intake tuning.
        """
        base = self.ve_coeffs['base']
        peak = self.ve_coeffs['rpm_peak']
        coeff = self.ve_coeffs['quadratic_coeff']
        ve = base + coeff * (rpm - peak) ** 2
        return np.clip(ve, 0.75, 0.98)

    def predict_maf(self, rpm: float, map_ambient_kpa: float = 101.325,
                    t_ambient_k: float = 298.15, fuel_qty_mg: float = 120.0) -> float:
        """
        Predict expected Mass Air Flow (MAF) in kg/h.

        Physics: MAF = VE × V_cyl × N_cyl × (RPM/2/60) × ρ_air × 3600
        """
        ve = self.ve_model(rpm)
        # Convert pressure to Pa
        p_ambient_pa = map_ambient_kpa * 1000.0
        # Air density
        rho_air = p_ambient_pa / (self.R_AIR * t_ambient_k)
        # Volume per cylinder per cycle (m³)
        vol_per_cyl = self.DISPLACEMENT_M3 / self.N_CYLINDERS
        # Power strokes per second (4-stroke: 1 per 2 revolutions)
        n_strokes_per_sec = rpm / 2.0 / 60.0
        # MAF in kg/s → convert to kg/h
        maf_kgs = ve * vol_per_cyl * self.N_CYLINDERS * n_strokes_per_sec * rho_air
        maf_kgh = maf_kgs * 3600.0
        # Apply online learning correction
        return maf_kgh * self.correction_factor

    def predict(self, rpm, map_ambient=101.325, t_ambient_k=298.15, fuel_qty=120.0):
        """Convenience method returning predicted MAF."""
        return self.predict_maf(rpm, map_ambient, t_ambient_k, fuel_qty)

    def compute_residual(self, actual_maf: float, rpm: float,
                         map_ambient: float = 101.325,
                         t_ambient_k: float = 298.15,
                         fuel_qty: float = 120.0) -> float:
        """Compute residual: actual - predicted. Negative = possible leak."""
        predicted = self.predict_maf(rpm, map_ambient, t_ambient_k, fuel_qty)
        return actual_maf - predicted

    def update_online(self, actual_maf: float, rpm: float,
                      map_ambient: float = 101.325,
                      t_ambient_k: float = 298.15,
                      alpha: float = 0.01):
        """
        Online learning via Exponentially Weighted Average.
        Slowly adapts the correction factor to handle engine aging.
        """
        predicted = self.predict_maf(rpm, map_ambient, t_ambient_k)
        if predicted > 0:
            ratio = actual_maf / predicted
            self.correction_factor = (1 - alpha) * self.correction_factor + alpha * ratio

    def save(self, path: str):
        """Serialize model to disk."""
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> 'IntakeTwinModel':
        """Load model from disk."""
        return joblib.load(path)
