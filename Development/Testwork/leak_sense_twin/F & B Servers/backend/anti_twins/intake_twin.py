import numpy as np
import joblib
import sys
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

class IntakeTwinModel:
    def __init__(self):
        self.DISPLACEMENT = 18.1
        self.N_CYLINDERS = 6
        self.R_AIR = 287
        self.scaler = StandardScaler()
        self.residual_corrector = Ridge(alpha=1.0) # LightGBM could be used too, using Ridge for simplicity/speed
        
    def _physics_maf(self, rpm, map_ambient, t_intake_k):
        # Polynomial fit: VE peaks around 1400 RPM for C18
        ve = 0.92 - 1.2e-5 * (rpm - 1400)**2 / 1e6
        vol_per_cycle = (self.DISPLACEMENT / 1000)
        rho_air = map_ambient * 1000 / (self.R_AIR * t_intake_k)
        maf_pred = ve * vol_per_cycle * (rpm / 120) * rho_air * 3600  # kg/h
        return maf_pred

    def fit(self, df_healthy):
        X = df_healthy[['RPM', 'MAP_intake', 'T_intake']].values
        y_true = df_healthy['MAF'].values
        
        # Calculate physics baseline
        y_phys = np.array([self._physics_maf(r, m, t + 273.15) for r, m, t in X])
        
        # Train corrector on residuals
        residuals = y_true - y_phys
        X_scaled = self.scaler.fit_transform(X)
        self.residual_corrector.fit(X_scaled, residuals)
        print("IntakeTwinModel fitted.")
        sys.stdout.flush()

    def predict(self, rpm, map_intake, t_intake):
        y_phys = self._physics_maf(rpm, map_intake, t_intake + 273.15)
        X = np.array([[rpm, map_intake, t_intake]])
        X_scaled = self.scaler.transform(X)
        y_corr = self.residual_corrector.predict(X_scaled)[0]
        return y_phys + y_corr

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
