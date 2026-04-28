"""
Exhaust Zone Digital Twin Model
Physics: Energy balance. Predicts exhaust temperatures and back-pressure.
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

class ExhaustTwinModel(BaseEstimator, RegressorMixin):
    """
    Physics-informed digital twin for Exhaust Zone (Manifold → Turbo turbine → DOC → DPF → SCR)
    Predicts exhaust temperatures and back-pressure based on MAF, fuel quantity, RPM, and CAC outlet temperature
    """

    def __init__(self):
        self.t_exh_manifold_model = None
        self.t_post_turbine_model = None
        self.dP_dpf_model = None
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None):
        """
        Fit the exhaust twin model

        Args:
            X: DataFrame with columns ['MAF', 'fuel_qty', 'RPM', 'T_cac_out']
            y: DataFrame with target columns ['T_exh_manifold', 'T_post_turbine', 'dP_dpf']
               If None, uses X columns with same names (assuming they exist)
        """
        if y is None:
            # Assume target columns exist in X
            y = X[['T_exh_manifold', 'T_post_turbine', 'dP_dpf']].copy()

        # Extract features
        maf_actual = X['MAF'].values
        fuel_qty = X['fuel_qty'].values
        rpm = X['RPM'].values
        t_cac_out = X['T_cac_out'].values

        # Convert CAC outlet temperature to Kelvin
        t_cac_out_k = t_cac_out + 273.15

        # Physics-based predictions
        # Fuel mass flow
        fuel_mass_flow = fuel_qty * 1e-6 * rpm/2/60 * N_CYLINDERS  # kg/s
        air_mass_flow = maf_actual / 3600  # kg/s

        # Combustion temperature (simplified first law)
        lhv_diesel = 42800  # kJ/kg
        combustion_eff = 0.97
        t_exh_manifold_physics = t_cac_out_k + (fuel_mass_flow * lhv_diesel * 1000 * combustion_eff) / \
                                ((air_mass_flow + fuel_mass_flow) * CP_AIR)

        # Turbine expansion (simplified pressure ratio)
        pr_turbine = 0.5 + 0.0001 * rpm  # Simplified turbine pressure ratio
        t_post_turbine_physics = t_exh_manifold_physics * (1 - TURBO_EFF_TURBINE * (1 - pr_turbine**(-(GAMMA-1)/GAMMA)))

        # DPF pressure drop model (simplified)
        dP_dpf_physics = 0.5 + 0.0003 * maf_actual  # kPa, increases with flow

        # Prepare targets (convert to Kelvin for consistency where needed)
        y_t_exh_manifold = y['T_exh_manifold'].values + 273.15  # Convert to Kelvin
        y_t_post_turbine = y['T_post_turbine'].values + 273.15  # Convert to Kelvin
        y_dP_dpf = y['dP_dpf'].values  # Already in kPa

        # Calculate residuals
        res_t_exh_manifold = y_t_exh_manifold - t_exh_manifold_physics
        res_t_post_turbine = y_t_post_turbine - t_post_turbine_physics
        res_dP_dpf = y_dP_dpf - dP_dpf_physics

        # Features for residual models
        residual_features = np.column_stack([maf_actual, fuel_qty, rpm, t_cac_out])

        # Create and fit residual models for each output
        self.t_exh_manifold_model = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('regressor', GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            ))
        ])

        self.t_post_turbine_model = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('regressor', GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            ))
        ])

        self.dP_dpf_model = Pipeline([
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
        self.t_exh_manifold_model.fit(residual_features, res_t_exh_manifold)
        self.t_post_turbine_model.fit(residual_features, res_t_post_turbine)
        self.dP_dpf_model.fit(residual_features, res_dP_dpf)

        self.is_fitted = True
        logger.info("Exhaust Twin model fitted successfully")
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict exhaust system parameters using physics + residual correction

        Args:
            X: DataFrame with columns ['MAF', 'fuel_qty', 'RPM', 'T_cac_out']

        Returns:
            DataFrame with predictions ['T_exh_manifold_pred', 'T_post_turbine_pred', 'dP_dpf_pred']
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Extract features
        maf_actual = X['MAF'].values
        fuel_qty = X['fuel_qty'].values
        rpm = X['RPM'].values
        t_cac_out = X['T_cac_out'].values

        # Convert CAC outlet temperature to Kelvin
        t_cac_out_k = t_cac_out + 273.15

        # Physics-based predictions
        # Fuel mass flow
        fuel_mass_flow = fuel_qty * 1e-6 * rpm/2/60 * N_CYLINDERS  # kg/s
        air_mass_flow = maf_actual / 3600  # kg/s

        # Combustion temperature (simplified first law)
        lhv_diesel = 42800  # kJ/kg
        combustion_eff = 0.97
        t_exh_manifold_physics = t_cac_out_k + (fuel_mass_flow * lhv_diesel * 1000 * combustion_eff) / \
                                ((air_mass_flow + fuel_mass_flow) * CP_AIR)

        # Turbine expansion (simplified pressure ratio)
        pr_turbine = 0.5 + 0.0001 * rpm  # Simplified turbine pressure ratio
        t_post_turbine_physics = t_exh_manifold_physics * (1 - TURBO_EFF_TURBINE * (1 - pr_turbine**(-(GAMMA-1)/GAMMA)))

        # DPF pressure drop model (simplified)
        dP_dpf_physics = 0.5 + 0.0003 * maf_actual  # kPa, increases with flow

        # Residual correction
        residual_features = np.column_stack([maf_actual, fuel_qty, rpm, t_cac_out])

        res_t_exh_manifold = self.t_exh_manifold_model.predict(residual_features)
        res_t_post_turbine = self.t_post_turbine_model.predict(residual_features)
        res_dP_dpf = self.dP_dpf_model.predict(residual_features)

        # Final predictions (convert back to Celsius for temperature outputs)
        t_exh_manifold_pred = t_exh_manifold_physics + res_t_exh_manifold - 273.15
        t_post_turbine_pred = t_post_turbine_physics + res_t_post_turbine - 273.15
        dP_dpf_pred = dP_dpf_physics + res_dP_dpf

        # Return as DataFrame
        predictions = pd.DataFrame({
            'T_exh_manifold_pred': t_exh_manifold_pred,
            'T_post_turbine_pred': t_post_turbine_pred,
            'dP_dpf_pred': dP_dpf_pred
        }, index=X.index)

        return predictions

    def predict_residuals(self, X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate residuals (actual - predicted) for anomaly detection

        Args:
            X: DataFrame with features ['MAF', 'fuel_qty', 'RPM', 'T_cac_out']
            y: DataFrame with actual values ['T_exh_manifol', 'T_post_turbine', 'dP_dpf']

        Returns:
            DataFrame with residuals
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        y_pred = self.predict(X)

        # Ensure column names match
        residuals = pd.DataFrame({
            'res_T_exh_manifold': y['T_exh_manifold'].values - y_pred['T_exh_manifold_pred'].values,
            'res_T_post_turbine': y['T_post_turbine'].values - y_pred['T_post_turbine_pred'].values,
            'res_dP_dpf': y['dP_dpf'].values - y_pred['dP_dpf_pred'].values
        }, index=X.index)

        return residuals

    def save(self, filepath: str):
        """Save the model to disk"""
        joblib.dump(self, filepath)
        logger.info(f"Exhaust Twin model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str):
        """Load the model from disk"""
        model = joblib.load(filepath)
        logger.info(f"Exhaust Twin model loaded from {filepath}")
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
    feature_cols = ['MAF', 'fuel_qty', 'RPM', 'T_cac_out']
    target_cols = ['T_exh_manifold', 'T_post_turbine', 'dP_dpf']

    X = df_ss[feature_cols]
    y = df_ss[target_cols]

    # Create and fit model
    model = ExhaustTwinModel()
    model.fit(X, y)

    # Make predictions
    y_pred = model.predict(X)
    residuals = model.predict_residuals(X, y)

    print(f"Exhaust Twin Model Example:")
    print(f"  Number of samples: {len(X)}")
    for col in ['T_exh_manifold', 'T_post_turbine', 'dP_dpf']:
        res_col = f'res_{col}'
        mae = np.mean(np.abs(residuals[res_col]))
        std = np.std(residuals[res_col])
        print(f"  {col}: MAE = {mae:.3f}, Std = {std:.3f}")

    # Save model
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, 'exhaust_twin.joblib'))