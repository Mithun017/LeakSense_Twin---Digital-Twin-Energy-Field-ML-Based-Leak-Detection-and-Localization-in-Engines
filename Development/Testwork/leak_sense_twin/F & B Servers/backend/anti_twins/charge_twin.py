import numpy as np
import joblib
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

class ChargeAirTwinModel:
    def __init__(self):
        self.GAMMA = 1.4
        self.TURBO_EFF_COMPRESSOR = 0.78
        self.CAC_EFFECTIVENESS = 0.88
        self.scaler = StandardScaler()
        self.residual_corrector = Ridge(alpha=1.0)
        
    def _physics_predict(self, maf, rpm, t_intake, map_ambient):
        t_intake_k = t_intake + 273.15
        # Simplified pressure ratio model
        pr = 1.5 + (rpm - 1100) * 0.0015
        map_boost = pr * map_ambient
        
        # Compressor outlet temperature
        t_boost_k = t_intake_k * (1 + (pr**((self.GAMMA-1)/self.GAMMA) - 1) / self.TURBO_EFF_COMPRESSOR)
        t_boost = t_boost_k - 273.15
        
        # CAC heat exchanger
        t_cac_out = t_boost - self.CAC_EFFECTIVENESS * (t_boost - t_intake)
        
        # Pressure drop
        map_cac_out = map_boost - (0.0012 * maf**0.5) # Simplified
        
        return {
            'MAP_boost_pred': map_boost,
            'T_boost_pred': t_boost,
            'T_cac_out_pred': t_cac_out,
            'MAP_cac_out_pred': map_cac_out
        }

    def fit(self, df_healthy):
        X = df_healthy[['MAF', 'RPM', 'T_intake']].values
        y_true = df_healthy[['MAP_boost', 'T_boost', 'T_cac_out', 'MAP_cac_out']].values
        
        phys_preds = []
        for i in range(len(df_healthy)):
            row = df_healthy.iloc[i]
            phys_preds.append(list(self._physics_predict(row['MAF'], row['RPM'], row['T_intake'], 101.325).values()))
        
        phys_preds = np.array(phys_preds)
        residuals = y_true - phys_preds
        
        X_scaled = self.scaler.fit_transform(X)
        self.residual_corrector.fit(X_scaled, residuals)
        print("ChargeAirTwinModel fitted.")

    def predict(self, maf, rpm, t_intake, map_ambient=101.325):
        phys = self._physics_predict(maf, rpm, t_intake, map_ambient)
        X = np.array([[maf, rpm, t_intake]])
        X_scaled = self.scaler.transform(X)
        corr = self.residual_corrector.predict(X_scaled)[0]
        
        keys = ['MAP_boost_pred', 'T_boost_pred', 'T_cac_out_pred', 'MAP_cac_out_pred']
        for i, key in enumerate(keys):
            phys[key] += corr[i]
        return phys

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
