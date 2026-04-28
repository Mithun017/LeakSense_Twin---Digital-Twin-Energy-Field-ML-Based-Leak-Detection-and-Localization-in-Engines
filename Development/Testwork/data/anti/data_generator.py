import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

class CatC18DataGenerator:
    def __init__(self):
        # Constants from blueprint
        self.DISPLACEMENT = 18.1  # liters
        self.N_CYLINDERS = 6
        self.R_AIR = 287  # J/kg·K
        self.CP_AIR = 1005  # J/kg·K
        self.GAMMA = 1.4
        self.MAP_AMBIENT = 101.325  # kPa
        self.T_AMBIENT = 25  # Celsius
        
    def generate_healthy_baseline(self, num_samples=40000):
        data = []
        start_time = datetime.now()
        
        # RPM range 1100-2100
        rpms = np.linspace(1100, 2100, 50)
        
        for i in range(num_samples):
            rpm = np.random.choice(rpms) + np.random.normal(0, 5)
            # Simplified fuel qty based on RPM (higher RPM -> more fuel)
            fuel_qty = 50 + (rpm - 1100) * 0.15 + np.random.normal(0, 1)
            
            # Physics-based sensor values
            # 1. Intake
            t_intake = self.T_AMBIENT + np.random.normal(0, 0.5)
            t_intake_k = t_intake + 273.15
            ve = 0.92 - 1.2e-5 * (rpm - 1400)**2 / 1e6
            rho_air = self.MAP_AMBIENT * 1000 / (self.R_AIR * t_intake_k)
            maf = ve * (self.DISPLACEMENT / 1000) * (rpm / 120) * rho_air * 3600 # kg/h
            
            # 2. Charge Air
            pr = 1.5 + (rpm - 1100) * 0.0015 + (fuel_qty - 50) * 0.01
            map_boost = self.MAP_AMBIENT * pr
            t_boost = t_intake_k * (pr**((self.GAMMA-1)/self.GAMMA)) - 273.15
            
            effectiveness = 0.88
            t_cac_out = t_boost - effectiveness * (t_boost - self.T_AMBIENT)
            map_cac_in = map_boost - 2
            map_cac_out = map_cac_in - 3
            map_intake = map_cac_out - 1
            
            # 3. Exhaust
            t_exh_manifold = t_cac_out + (fuel_qty * 8) + np.random.normal(0, 5)
            t_dpf_in = t_exh_manifold - 50
            t_dpf_out = t_dpf_in - 30
            
            # Add noise
            maf *= (1 + np.random.normal(0, 0.015))
            map_boost *= (1 + np.random.normal(0, 0.005))
            t_exh_manifold *= (1 + np.random.normal(0, 0.01))
            
            data.append({
                'timestamp': (start_time + timedelta(seconds=i)).isoformat(),
                'RPM': rpm,
                'MAF': maf,
                'MAP_intake': map_intake,
                'MAP_boost': map_boost,
                'MAP_cac_in': map_cac_in,
                'MAP_cac_out': map_cac_out,
                'T_intake': t_intake,
                'T_boost': t_boost,
                'T_cac_out': t_cac_out,
                'T_exh_manifold': t_exh_manifold,
                'T_dpf_in': t_dpf_in,
                'T_dpf_out': t_dpf_out,
                'fuel_qty': fuel_qty,
                'is_steady_state': 1,
                'leak_zone': 0,
                'leak_severity': 0
            })
            
        return pd.DataFrame(data)

    def inject_leaks(self, df, num_faults=10000):
        fault_data = []
        zones = [1, 2, 3, 4, 5]
        severities = [1, 2, 3] # 2%, 8%, 15%
        severity_map = {1: 0.02, 2: 0.08, 3: 0.15}
        
        for _ in range(num_faults):
            # Take a random healthy sample as base
            sample = df.iloc[np.random.randint(0, len(df))].copy()
            zone = np.random.choice(zones)
            severity = np.random.choice(severities)
            leak_frac = severity_map[severity]
            
            sample['leak_zone'] = zone
            sample['leak_severity'] = severity
            
            if zone == 1: # Intake (MAF meter to compressor)
                sample['MAF'] *= (1 - leak_frac)
                sample['MAP_boost'] *= (1 - leak_frac * 0.5)
            elif zone == 2: # Charge Air (Post-compressor to CAC)
                sample['MAP_boost'] *= (1 - leak_frac)
                sample['T_intake'] += (leak_frac * 20)
            elif zone == 3: # CAC to Intake Manifold
                sample['MAP_cac_out'] *= (1 - leak_frac)
                sample['MAP_intake'] *= (1 - leak_frac * 1.1)
            elif zone == 4: # Exhaust Manifold to Turbo
                sample['T_exh_manifold'] *= (1 - leak_frac * 0.5)
                sample['MAP_boost'] *= (1 - leak_frac * 0.3)
            elif zone == 5: # DPF/SCR
                # back pressure delta drops
                p_diff = sample['T_dpf_in'] - sample['T_dpf_out'] # pseudo-proxy
                sample['T_dpf_out'] += (leak_frac * 10)
                
            fault_data.append(sample)
            
        return pd.DataFrame(fault_data)

    def generate_all(self, output_path):
        print("Generating healthy data...")
        healthy_df = self.generate_healthy_baseline(5000)
        print("Injecting leaks...")
        fault_df = self.inject_leaks(healthy_df, 1000)
        
        full_df = pd.concat([healthy_df, fault_df]).sample(frac=1).reset_index(drop=True)
        full_df.to_csv(output_path, index=False)
        print(f"Dataset saved to {output_path}")
        return full_df

if __name__ == "__main__":
    gen = CatC18DataGenerator()
    os.makedirs("Development/Testing/Anti/data", exist_ok=True)
    gen.generate_all("Development/Testing/Anti/data/c18_sensor_data.csv")
