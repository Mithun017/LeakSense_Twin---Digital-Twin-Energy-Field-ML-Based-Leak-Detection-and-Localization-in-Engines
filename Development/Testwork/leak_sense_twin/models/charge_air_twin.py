"""
Charge Air System Digital Twin Model
Physics: Turbocharger compressor map + CAC heat exchanger model.
Predicts: MAP_boost, T_cac_out, MAP_intake
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

class ChargeAirTwinModel(BaseEstimator, RegressorMixin):
    """
    Physics-informed digital twin for Charge Air System (Compressor → CAC → Intake manifold)
    Predicts MAP_boost, T_cac_out, MAP_intake based on MAF, RPM, and ambient conditions
    """

    def __init__(self):
        self.boost_model = None
        self.t_boost_model = None
        self.t_cac_out_model = None
        self.map_cac_out_model = None
        self.map_intake_model = None
        self.is_fitted = False

    def compressor_map(self, corr_flow: np.ndarray, corr_speed: np.ndarray) -> np.ndarray:
        """
        Simplified compressor map: Pressure ratio as function of corrected flow and speed
        In practice, this would be fitted from GT-Power simulation or published maps
        Using a simplified quadratic surface: PR = c00 + c10*Q + c01*N + c20*Q² + c11*Q*N + c02*N²
        """
        # Placeholder coefficients - would be fitted from actual data
        c00, c10, c01, c20, c11, c02 = 0.7, 0.1, 0.05, -0.05, 0.02, -0.03
        pr = c00 + c10*corr_flow + c01*corr_speed + c20*corr_flow**2 + c11*corr_flow*corr_speed + c02*corr_speed**2
        # Ensure reasonable bounds
        pr = np.clip(pr, 1.0, 4.0)
        return pr

    def cac_pressure_drop(self, maf: np.ndarray) -> np.ndarray:
        """
        Pressure drop across CAC: Darcy-Weisbach, dP proportional to mass_flow²
        Calibrated for C18 CAC
        """
        return 0.0012 * maf**2  # kPa

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None):
        """
        Fit the charge air twin model

        Args:
            X: DataFrame with columns ['MAF', 'RPM', 'T_intake', 'MAP_intake']
            y: DataFrame with target columns ['MAP_boost', 'T_boost', 'T_cac_out', 'MAP_intake']
               If None, uses X columns with same names
        """
        if y is None:
            y = X[['MAP_boost', 'T_boost', 'T_cac_out', 'MAP_intake']].copy()

        # Extract features
        maf_actual = X['MAF'].values
        rpm = X['RPM'].values
        t_intake = X['T_intake'].values
        map_intake = X['MAP_intake'].values

        # Convert intake temperature to Kelvin
        t_ambient_k = t_intake + 273.15
        map_ambient = map_intake + 2.0  # Approximate ambient pressure (small pressure drop assumed)

        # Calculate corrected flow and speed for compressor map
        corrected_flow = maf_actual * np.sqrt(t_ambient_k / 288.15) / (map_ambient / 101.325)
        corrected_speed = rpm / np.sqrt(t_ambient_k / 288.15)

        # Physics-based predictions
        # Pressure ratio from compressor map
        pr = self.compressor_map(corrected_flow, corrected_speed)
        map_boost_physics = pr * map_ambient

        # Compressor outlet temperature (isentropic + efficiency)
        t_boost_physics = t_ambient_k * (1 + (pr**((GAMMA-1)/GAMMA) - 1) / TURBO_EFF_COMPRESSOR)

        # CAC heat exchanger
        t_cac_out_physics = t_boost_physics - CAC_EFFECTIVENESS * (t_boost_physics - t_ambient_k)

        # Pressure drop across CAC
        map_cac_out_physics = map_boost_physics - self.cac_pressure_drop(maf_actual)

        # Intake manifold pressure (small pressure drop across intake plumbing)
        map_intake_physics = map_cac_out_physics - 0.5  # kPa, approximate pressure drop

        # Prepare targets
        y_map_boost = y['MAP_boost'].values
        y_t_boost = y['T_boost'].values + 273.15  # Convert to Kelvin for consistency
        y_t_cac_out = y['T_cac_out'].values + 273.15  # Convert to Kelvin for consistency
        y_map_intake = y['MAP_intake'].values

        # Calculate residuals
        res_map_boost = y_map_boost - map_boost_physics
        res_t_boost = y_t_boost - t_boost_physics
        res_t_cac_out = y_t_cac_out - t_cac_out_physics
        res_map_intake = y_map_intake - map_intake_physics

        # Features for residual models
        residual_features = np.column_stack([maf_actual, rpm, t_intake, map_intake])

        # Create and fit residual models for each output
        self.boost_model = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('regressor', GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            ))
        ])

        self.t_boost_model = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('regressor', GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            ))
        ])

        self.t_cac_out_model = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('regressor', GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            ))
        ])

        self.map_intake_model = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('regressor', GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            ))
        ])

        # Fit all models
        self.boost_model.fit(residual_features, res_map_boost)
        self.t_boost_model.fit(residual_features, res_t_boost)
        self.t_cac_out_model.fit(residual_features, res_t_cac_out)
        self.map_intake_model.fit(residual_features, res_map_intake)

        self.is_fitted = True
        logger.info("Charge Air Twin model fitted successfully")
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict charge air system parameters using physics + residual correction

        Args:
            X: DataFrame with columns ['MAF', 'RPM', 'T_intake', 'MAP_intake']

        Returns:
            DataFrame with predictions ['MAP_boost_pred', 'T_boost_pred', 'T_cac_out_pred', 'MAP_intake_pred']
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Extract features
        maf_actual = X['MAF'].values
        rpm = X['RPM'].values
        t_intake = X['T_intake'].values
        map_intake = X['MAP_intake'].values

        # Convert intake temperature to Kelvin
        t_ambient_k = t_intake + 273.15
        map_ambient = map_intake + 2.0  # Approximate ambient pressure

        # Calculate corrected flow and speed for compressor map
        corrected_flow = maf_actual * np.sqrt(t_ambient_k / 288.15) / (map_ambient / 101.325)
        corrected_speed = rpm / np.sqrt(t_ambient_k / 288.15)

        # Physics-based predictions
        # Pressure ratio from compressor map
        pr = self.compressor_map(corrected_flow, corrected_speed)
        map_boost_physics = pr * map_ambient

        # Compressor outlet temperature (isentropic + efficiency)
        t_boost_physics = t_ambient_k * (1 + (pr**((GAMMA-1)/GAMMA) - 1) / TURBO_EFF_COMPRESSOR)

        # CAC heat exchanger
        t_cac_out_physics = t_boost_physics - CAC_EFFECTIVENESS * (t_boost_physics - t_ambient_k)

        # Pressure drop across CAC
        map_cac_out_physics = map_boost_physics - self.cac_pressure_drop(maf_actual)

        # Intake manifold pressure (small pressure drop across intake plumbing)
        map_intake_physics = map_cac_out_physics - 0.5  # kPa, approximate pressure drop

        # Residual correction
        residual_features = np.column_stack([maf_actual, rpm, t_intake, map_intake])

        res_map_boost = self.boost_model.predict(residual_features)
        res_t_boost = self.t_boost_model.predict(residual_features)
        res_t_cac_out = self.t_cac_out_model.predict(residual_features)
        res_map_intake = self.map_intake_model.predict(residual_features)  # New model for MAP_intake

        # Final predictions
        map_boost_pred = map_boost_physics + res_map_boost
        t_boost_pred = t_boost_physics + res_t_boost
        t_cac_out_pred = t_cac_out_physics + res_t_cac_out
        map_intake_pred = map_intake_physics + res_map_intake

        # Convert temperatures back to Celsius
        t_boost_pred_c = t_boost_pred - 273.15
        t_cac_out_pred_c = t_cac_out_pred - 273.15

        # Return as DataFrame
        predictions = pd.DataFrame({
            'MAP_boost_pred': map_boost_pred,
            'T_boost_pred': t_boost_pred_c,
            'T_cac_out_pred': t_cac_out_pred_c,
            'MAP_intake_pred': map_intake_pred
        }, index=X.index)

        return predictions

    def predict_residuals(self, X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate residuals (actual - predicted) for anomaly detection

        Args:
            X: DataFrame with features ['MAF', 'RPM', 'T_intake', 'MAP_intake']
            y: DataFrame with actual values ['MAP_boost', 'T_boost', 'T_cac_out', 'MAP_intake']

        Returns:
            DataFrame with residuals
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        y_pred = self.predict(X)

        # Ensure column names match
        residuals = pd.DataFrame({
            'res_MAP_boost': y['MAP_boost'].values - y_pred['MAP_boost_pred'].values,
            'res_T_boost': y['T_boost'].values - y_pred['T_boost_pred'].values,
            'res_T_cac_out': y['T_cac_out'].values - y_pred['T_cac_out_pred'].values,
            'res_MAP_intake': y['MAP_intake'].values - y_pred['MAP_cac_out_pred'].values  # Note: predicting intake from charge air model
        }, index=X.index)

        return residuals

    def save(self, filepath: str):
        """Save the model to disk"""
        joblib.dump(self, filepath)
        logger.info(f"Charge Air Twin model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str):
        """Load the model from disk"""
        model = joblib.load(filepath)
        logger.info(f"Charge Air Twin model loaded from {filepath}")
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

    # Prepare features and targets
    feature_cols = ['MAF', 'RPM', 'T_intake', 'MAP_intake']
    target_cols = ['MAP_boost', 'T_boost', 'T_cac_out', 'MAP_intake']  # Note: we predict MAP_intake from charge air model

    X = df_ss[feature_cols]
    y = df_ss[target_cols]

    # Create and fit model
    model = ChargeAirTwinModel()
    model.fit(X, y)

    # Make predictions
    y_pred = model.predict(X)
    residuals = model.predict_residuals(X, y)

    print(f"Charge Air Twin Model Example:")
    print(f"  Number of samples: {len(X)}")
    for col in ['MAP_boost', 'T_boost', 'T_cac_out', 'MAP_intake']:
        res_col = f'res_{col}'
        mae = np.mean(np.abs(residuals[res_col]))
        std = np.std(residuals[res_col])
        print(f"  {col}: MAE = {mae:.3f}, Std = {std:.3f}")

    # Save model
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, 'charge_air_twin.joblib'))