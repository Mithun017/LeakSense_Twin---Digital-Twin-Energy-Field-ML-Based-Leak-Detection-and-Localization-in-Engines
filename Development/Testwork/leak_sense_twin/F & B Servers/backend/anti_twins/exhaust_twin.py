import numpy as np
import joblib
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

class ExhaustTwinModel:
    def __init__(self):
        self.CP_AIR = 1005
        self.LHV_DIESEL = 42800
        self.N_CYLINDERS = 6
        self.TURBO_EFF_TURBINE = 0.82
        self.GAMMA = 1.4
        self.scaler = StandardScaler()
        self.residual_corrector = Ridge(alpha=1.0)

    def _physics_predict(self, maf, fuel_qty, rpm, t_cac_out):
        fuel_mass_flow = fuel_qty * 1e-6 * (rpm / 120) * self.N_CYLINDERS # kg/s
        air_mass_flow = maf / 3600 # kg/s
        
        combustion_eff = 0.97
        t_cac_out_k = t_cac_out + 273.15
        t_exh_manifold_k = t_cac_out_k + (fuel_mass_flow * self.LHV_DIESEL * 1000 * combustion_eff) / \
                          ((air_mass_flow + fuel_mass_flow) * self.CP_AIR)
        
        # Expansion (simplified)
        pr_turbine = 1.2 + (rpm / 2100) * 0.5
        t_post_turbine_k = t_exh_manifold_k * (1 - self.TURBO_EFF_TURBINE * (1 - pr_turbine**(-(self.GAMMA-1)/self.GAMMA)))
        
        return {
            'T_exh_manifold_pred': t_exh_manifold_k - 273.15,
            'T_post_turbine_pred': t_post_turbine_k - 273.15,
            'dP_dpf_pred': 5.0 + (maf/1000)**2 * 2.0 # simplified backpressure
        }

    def fit(self, df_healthy):
        X = df_healthy[['MAF', 'fuel_qty', 'RPM', 'T_cac_out']].values
        y_true = df_healthy[['T_exh_manifold', 'T_dpf_in', 'T_dpf_out']].values # Proxying for turbine/dpf preds
        
        phys_preds = []
        for i in range(len(df_healthy)):
            row = df_healthy.iloc[i]
            phys_preds.append(list(self._physics_predict(row['MAF'], row['fuel_qty'], row['RPM'], row['T_cac_out']).values()))
        
        phys_preds = np.array(phys_preds)
        residuals = y_true - phys_preds
        
        X_scaled = self.scaler.fit_transform(X)
        self.residual_corrector.fit(X_scaled, residuals)
        print("ExhaustTwinModel fitted.")

    def predict(self, maf, fuel_qty, rpm, t_cac_out):
        phys = self._physics_predict(maf, fuel_qty, rpm, t_cac_out)
        X = np.array([[maf, fuel_qty, rpm, t_cac_out]])
        X_scaled = self.scaler.transform(X)
        corr = self.residual_corrector.predict(X_scaled)[0]
        
        keys = ['T_exh_manifold_pred', 'T_post_turbine_pred', 'dP_dpf_pred']
        for i, key in enumerate(keys):
            phys[key] += corr[i]
        return phys

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
