"""
Intake Zone Digital Twin Model
Physics: Mass continuity. Predict MAF from RPM + ambient conditions.
If actual MAF < predicted MAF by threshold: Zone 1 leak detected.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import logging

logger = logging.getLogger(__name__)

# Engine constants (Cat C18)
DISPLACEMENT = 18.1  # liters
N_CYLINDERS = 6
COMPRESSION_RATIO = 16.3
R_AIR = 287  # J/kg·K
CP_AIR = 1005  # J/kg·K
GAMMA = 1.4
CAC_EFFECTIVENESS = 0.88  # nominal
TURBO_EFF_COMPRESSOR = 0.78  # nominal isentropic
TURBO_EFF_TURBINE = 0.82  # nominal isentropic
VE_BASE = 0.92  # volumetric efficiency at rated

class IntakeTwinModel(BaseEstimator, RegressorMixin):
    """
    Physics-informed digital twin for Intake Zone (Airflow meter → Compressor inlet)
    Predicts MAF based on RPM and ambient conditions
    """

    def __init__(self):
        self.ve_coefficients = None
        self.residual_model = None
        self.is_fitted = False

    def ve_model(self, rpm: np.ndarray) -> np.ndarray:
        """
        Volumetric efficiency model as function of RPM
        Polynomial fit: VE peaks around 1400 RPM for C18
        """
        if self.ve_coefficients is None:
            # Default polynomial fit: VE peaks around 1400 RPM for C18
            return 0.92 - 1.2e-5 * (rpm - 1400)**2 / 1e6
        else:
            # Use fitted coefficients
            return np.polyval(self.ve_coefficients, rpm)

    def predict_maf_physics(self, rpm: np.ndarray, map_ambient: np.ndarray,
                           t_ambient_k: np.ndarray, fuel_qty_mg: np.ndarray) -> np.ndarray:
        """
        Physics-based MAF prediction using mass continuity

        Args:
            rpm: Engine speed in RPM
            map_ambient: Ambient pressure in kPa
            t_ambient_k: Ambient temperature in Kelvin
            fuel_qty_mg: Fuel injection quantity in mg/stroke

        Returns:
            Predicted MAF in kg/h
        """
        # Volumetric efficiency correction with RPM
        ve = self.ve_model(rpm)

        # Air mass per cycle
        n_strokes = rpm / 2 / 60  # 4-stroke: power stroke every 2 revolutions
        vol_per_cycle = (DISPLACEMENT / 1000) / N_CYLINDERS  # m³ per cylinder
        rho_air = map_ambient * 1000 / (R_AIR * t_ambient_k)  # kg/m³
        maf_pred = ve * vol_per_cycle * N_CYLINDERS * n_strokes * rho_air * 3600  # kg/h

        return maf_pred

    def fit(self, X: pd.DataFrame, y: np.ndarray = None):
        """
        Fit the intake twin model

        Args:
            X: DataFrame with columns ['RPM', 'MAP_intake', 'T_intake', 'fuel_qty']
            y: Target MAF values (if None, uses X['MAF'])
        """
        if y is None:
            y = X['MAF'].values

        # Extract features
        rpm = X['RPM'].values
        map_intake = X['MAP_intake'].values
        t_intake = X['T_intake'].values
        fuel_qty = X['fuel_qty'].values

        # Convert intake temperature to Kelvin
        t_ambient_k = t_intake + 273.15

        # Physics-based prediction
        maf_physics = self.predict_maf_physics(rpm, map_intake, t_ambient_k, fuel_qty)

        # Calculate residuals (actual - physics prediction)
        residuals = y - maf_physics

        # Fit residual correction model
        # Features for residual model: RPM, MAP_intake, T_intake, fuel_qty
        residual_features = np.column_stack([rpm, map_intake, t_intake, fuel_qty])

        # Create pipeline for residual model
        self.residual_model = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('regressor', GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            ))
        ])

        self.residual_model.fit(residual_features, residuals)

        # Optional: Fit VE coefficients if we want to optimize them
        # For now, we'll keep the default VE model
        self.is_fitted = True
        logger.info("Intake Twin model fitted successfully")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict MAF using physics + residual correction

        Args:
            X: DataFrame with columns ['RPM', 'MAP_intake', 'T_intake', 'fuel_qty']

        Returns:
            Predicted MAF values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Extract features
        rpm = X['RPM'].values
        map_intake = X['MAP_intake'].values
        t_intake = X['T_intake'].values
        fuel_qty = X['fuel_qty'].values

        # Convert intake temperature to Kelvin
        t_ambient_k = t_intake + 273.15

        # Physics-based prediction
        maf_physics = self.predict_maf_physics(rpm, map_intake, t_ambient_k, fuel_qty)

        # Residual correction
        residual_features = np.column_stack([rpm, map_intake, t_intake, fuel_qty])
        residual_correction = self.residual_model.predict(residual_features)

        # Final prediction
        maf_pred = maf_physics + residual_correction

        return maf_pred

    def predict_residual(self, X: pd.DataFrame) -> np.ndarray:
        """
        Calculate residuals (actual - predicted) for anomaly detection

        Args:
            X: DataFrame with columns ['RPM', 'MAP_intake', 'T_intake', 'fuel_qty', 'MAF']

        Returns:
            Residual values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        maf_actual = X['MAF'].values
        maf_pred = self.predict(X)
        residuals = maf_actual - maf_pred

        return residuals

    def save(self, filepath: str):
        """Save the model to disk"""
        joblib.dump(self, filepath)
        logger.info(f"Intake Twin model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str):
        """Load the model from disk"""
        model = joblib.load(filepath)
        logger.info(f"Intake Twin model loaded from {filepath}")
        return model

if __name__ == "__main__":
    # Example usage
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'data_generation'))

    from synthetic_data_generator import SyntheticDataGenerator

    # Generate sample data
    generator = SyntheticDataGenerator()
    df = generator.generate_dataset(n_samples=1000, healthy_ratio=0.8, random_seed=42)

    # Filter steady-state data
    steady_state_mask = df['is_steady_state'] == True
    df_ss = df[steady_state_mask].copy()

    # Prepare features and target
    feature_cols = ['RPM', 'MAP_intake', 'T_intake', 'fuel_qty']
    X = df_ss[feature_cols]
    y = df_ss['MAF'].values

    # Create and fit model
    model = IntakeTwinModel()
    model.fit(X, y)

    # Make predictions
    maf_pred = model.predict(X)
    residuals = model.predict_residual(df_ss)

    print(f"Intake Twin Model Example:")
    print(f"  Number of samples: {len(X)}")
    print(f"  Mean absolute residual: {np.mean(np.abs(residuals)):.3f} kg/h")
    print(f"  Std of residuals: {np.std(residuals):.3f} kg/h")

    # Save model
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, 'intake_twin.joblib'))