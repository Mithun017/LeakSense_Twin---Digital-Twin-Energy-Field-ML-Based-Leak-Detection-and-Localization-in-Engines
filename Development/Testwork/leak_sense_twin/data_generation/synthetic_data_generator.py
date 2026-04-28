"""
Synthetic Data Generator for LeakSense Twin
Generates realistic sensor data for Cat C18 diesel engine with leak injection capabilities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class SyntheticDataGenerator:
    """Generate synthetic sensor data for Cat C18 engine with leak simulation"""

    def __init__(self):
        # Engine constants from documentation
        self.DISPLACEMENT = 18.1  # liters
        self.N_CYLINDERS = 6
        self.COMPRESSION_RATIO = 16.3
        self.R_AIR = 287  # J/kg·K
        self.CP_AIR = 1005  # J/kg·K
        self.GAMMA = 1.4
        self.CAC_EFFECTIVENESS = 0.88  # nominal
        self.TURBO_EFF_COMPRESSOR = 0.78  # nominal isentropic
        self.TURBO_EFF_TURBINE = 0.82  # nominal isentropic
        self.VE_BASE = 0.92  # volumetric efficiency at rated

        # Sensor noise parameters
        self.PRESSURE_NOISE_SIGMA = 0.005  # 0.5%
        self.TEMPERATURE_NOISE_SIGMA = 0.01  # 1%
        self.FLOW_NOISE_SIGMA = 0.015  # 1.5%
        self.DRIFT_RATE = 0.001  # 0.1%/hour

        # Steady-state detection thresholds
        self.RPM_STD_THRESHOLD = 10  # RPM
        self.MAF_STD_THRESHOLD = 0.02  # 2% of mean
        self.FUEL_QTY_STD_THRESHOLD = 0.01  # 1% of mean

    def ve_model(self, rpm: float) -> float:
        """Volumetric efficiency model as function of RPM"""
        return 0.92 - 1.2e-5 * (rpm - 1400)**2 / 1e6

    def compute_healthy_baseline(self, rpm: float, fuel_qty_mg: float,
                                map_ambient: float = 101.325,
                                t_ambient_k: float = 298.15) -> Dict[str, float]:
        """
        Compute expected sensor readings for healthy engine at given operating point

        Args:
            rpm: Engine speed in RPM
            fuel_qty_mg: Fuel injection quantity in mg/stroke
            map_ambient: Ambient pressure in kPa (default: sea level)
            t_ambient_k: Ambient temperature in Kelvin (default: 25°C)

        Returns:
            Dictionary of expected sensor values
        """
        # Volumetric efficiency correction with RPM
        ve = self.ve_model(rpm)

        # Air mass per cycle
        n_strokes = rpm / 2 / 60  # 4-stroke: power stroke every 2 revolutions
        vol_per_cycle = (self.DISPLACEMENT / 1000) / self.N_CYLINDERS  # m³ per cylinder
        rho_air = map_ambient * 1000 / (self.R_AIR * t_ambient_k)  # kg/m³
        maf_pred = ve * vol_per_cycle * self.N_CYLINDERS * n_strokes * rho_air * 3600  # kg/h

        # Simplified boost pressure calculation (would normally come from compressor map)
        # Using a simplified quadratic approximation
        corrected_flow = maf_pred * np.sqrt(t_ambient_k / 288.15) / (map_ambient / 101.325)
        corrected_speed = rpm / np.sqrt(t_ambient_k / 288.15)

        # Simplified compressor map (placeholder - would be fitted from actual data)
        pr = 0.8 + 0.0002 * rpm + 0.0001 * fuel_qty_mg  # Simplified pressure ratio
        map_boost = pr * map_ambient

        # Compressor outlet temperature (isentropic + efficiency)
        t_boost = t_ambient_k * (1 + (pr**((self.GAMMA-1)/self.GAMMA) - 1) / self.TURBO_EFF_COMPRESSOR)

        # CAC heat exchanger
        t_cac_out = t_boost - self.CAC_EFFECTIVENESS * (t_boost - t_ambient_k)

        # Pressure drop across CAC: ~2-4 kPa typical
        map_cac_out = map_boost - 0.0012 * maf_pred**2  # kPa, calibrated for C18 CAC
        map_cac_in = map_boost  # Assuming negligible pressure drop before CAC

        # Intake manifold pressure (simplified)
        map_intake = map_cac_out - 0.5  # Small pressure drop across intake plumbing

        # Temperatures
        t_intake = t_ambient_k  # Simplified
        t_boost_k = t_boost  # Already in Kelvin

        # Exhaust temperature (simplified energy balance)
        fuel_mass_flow = fuel_qty_mg * 1e-6 * rpm/2/60 * self.N_CYLINDERS  # kg/s
        air_mass_flow = maf_pred / 3600  # kg/s
        lhv_diesel = 42800  # kJ/kg
        combustion_eff = 0.97
        t_exh_manifold = t_cac_out + (fuel_mass_flow * lhv_diesel * 1000 * combustion_eff) / \
                        ((air_mass_flow + fuel_mass_flow) * self.CP_AIR)

        # Post-turbine temperature (simplified)
        pr_turbine = 0.5 + 0.0001 * rpm  # Simplified turbine pressure ratio
        t_post_turbine = t_exh_manifold * (1 - self.TURBO_EFF_TURBINE * (1 - pr_turbine**(-(self.GAMMA-1)/self.GAMMA)))

        # DPF pressure drop (simplified)
        dP_dpf = 0.5 + 0.0003 * maf_pred  # kPa, increases with flow

        # Temperatures in Celsius for output
        t_exh_manifold_c = t_exh_manifold - 273.15
        t_post_turbine_c = t_post_turbine - 273.15
        t_dpf_in_c = t_post_turbine_c  # Simplified: same as post-turbine temp
        t_dpf_out_c = t_post_turbine_c - 50  # Simplified cooling across DPF

        return {
            'RPM': rpm,
            'MAF': maf_pred,
            'MAP_intake': map_intake,
            'MAP_boost': map_boost,
            'MAP_cac_in': map_cac_in,
            'MAP_cac_out': map_cac_out,
            'T_intake': t_intake - 273.15,
            'T_boost': t_boost_k - 273.15,
            'T_cac_out': t_cac_out - 273.15,
            'T_exh_manifold': t_exh_manifold_c,
            'T_post_turbine': t_post_turbine_c,
            'T_dpf_in': t_dpf_in_c,
            'T_dpf_out': t_dpf_out_c,
            'fuel_qty': fuel_qty_mg,
            'dP_dpf': dP_dpf
        }

    def inject_leak(self, healthy_values: Dict[str, float], leak_zone: int,
                   leak_severity: int) -> Dict[str, float]:
        """
        Inject leak effects into healthy sensor values

        Args:
            healthy_values: Dictionary of healthy sensor values
            leak_zone: Zone identifier (1-5) where 0=no leak
            leak_severity: Severity level (1=small, 2=medium, 3=large)

        Returns:
            Dictionary of sensor values with leak effects applied
        """
        if leak_zone == 0:
            return healthy_values.copy()

        # Leak severity mapping to flow loss fraction
        leak_fractions = {1: 0.02, 2: 0.08, 3: 0.15}  # small=2%, medium=8%, large=15% as per spec
        leak_fraction = leak_fractions.get(leak_severity, 0.0)

        # Copy healthy values to modify
        leaky_values = healthy_values.copy()

        # Apply leak effects based on zone
        if leak_zone == 1:  # Zone A: Airflow meter → Compressor inlet
            # MAF reading is lower than actual due to leak upstream of sensor
            actual_maf = healthy_values['MAF'] / (1 - leak_fraction)
            leaky_values['MAF'] = healthy_values['MAF']  # Sensor reads lower value
            # ECU compensates by increasing fuel, but we'll show the effect on boost
            leaky_values['MAP_boost'] *= (1 - leak_fraction * 0.8)  # Boost drops
            leaky_values['MAP_cac_out'] *= (1 - leak_fraction * 0.8)
            leaky_values['MAP_intake'] *= (1 - leak_fraction * 0.8)
            # Intake temperature increases slightly due to less dense air
            leaky_values['T_intake'] += leak_fraction * 15

        elif leak_zone == 2:  # Zone B: Post-compressor → CAC inlet
            # Leak is downstream of MAF sensor, so MAF reads correctly
            leaky_values['MAP_boost'] *= (1 - leak_fraction * 0.9)  # Boost drops
            leaky_values['MAP_cac_out'] *= (1 - leak_fraction * 0.9)
            leaky_values['MAP_intake'] *= (1 - leak_fraction * 0.85)  # Slightly less effect downstream
            # Temperature rises due to less dense charge
            leaky_values['T_intake'] += leak_fraction * 25  # Approximate
            # Boost temperature increases due to less mass flow
            leaky_values['T_boost'] += leak_fraction * 10

        elif leak_zone == 3:  # Zone C: CAC → Intake manifold
            leaky_values['MAP_cac_out'] *= (1 - leak_fraction * 0.9)
            leaky_values['MAP_intake'] *= (1 - leak_fraction * 0.9)
            # MAF appears normal but volumetric efficiency drops
            leaky_values['MAF'] = healthy_values['MAF']  # Sensor unchanged
            # Effective air flow into cylinders is reduced
            leaky_values['MAP_boost'] *= (1 - leak_fraction * 0.4)  # Effect upstream
            leaky_values['T_intake'] += leak_fraction * 20  # Intake temp increases

        elif leak_zone == 4:  # Zone D: Exhaust manifold → Turbo turbine
            leaky_values['T_exh_manifold'] -= leak_fraction * 120  # Temperature drop at leak
            leaky_values['T_dpf_in'] -= leak_fraction * 100
            leaky_values['T_dpf_out'] -= leak_fraction * 100
            # Turbo speed drops → cascading boost drop
            leaky_values['MAP_boost'] *= (1 - leak_fraction * 0.7)
            leaky_values['MAP_cac_out'] *= (1 - leak_fraction * 0.7)
            leaky_values['MAP_intake'] *= (1 - leak_fraction * 0.7)
            # Back-pressure changes
            leaky_values['dP_dpf'] *= (1 - leak_fraction * 0.5)

        elif leak_zone == 5:  # Zone E: DPF/SCR area
            leaky_values['dP_dpf'] *= (1 - leak_fraction * 0.8)  # Back-pressure delta drops
            leaky_values['T_dpf_out'] += leak_fraction * 40  # Temperature change
            # Some effect on upstream sensors due to back-pressure changes
            leaky_values['MAP_boost'] *= (1 - leak_fraction * 0.2)
            leaky_values['MAP_intake'] *= (1 - leak_fraction * 0.2)
            leaky_values['T_exh_manifold'] += leak_fraction * 15

        return leaky_values

    def add_noise_and_drift(self, values: Dict[str, float],
                           timestamp: float, base_time: float = 0.0) -> Dict[str, float]:
        """
        Add realistic sensor noise and drift

        Args:
            values: Dictionary of sensor values
            timestamp: Current timestamp in hours
            base_time: Reference timestamp for drift calculation

        Returns:
            Dictionary with noise and drift applied
        """
        noisy_values = values.copy()
        hours_elapsed = (timestamp - base_time) / 3600.0  # Convert to hours if needed

        # Sensor-specific noise and drift
        noise_params = {
            'MAF': ('flow', self.FLOW_NOISE_SIGMA),
            'MAP_intake': ('pressure', self.PRESSURE_NOISE_SIGMA),
            'MAP_boost': ('pressure', self.PRESSURE_NOISE_SIGMA),
            'MAP_cac_in': ('pressure', self.PRESSURE_NOISE_SIGMA),
            'MAP_cac_out': ('pressure', self.PRESSURE_NOISE_SIGMA),
            'T_intake': ('temperature', self.TEMPERATURE_NOISE_SIGMA),
            'T_boost': ('temperature', self.TEMPERATURE_NOISE_SIGMA),
            'T_cac_out': ('temperature', self.TEMPERATURE_NOISE_SIGMA),
            'T_exh_manifold': ('temperature', self.TEMPERATURE_NOISE_SIGMA),
            'T_dpf_in': ('temperature', self.TEMPERATURE_NOISE_SIGMA),
            'T_dpf_out': ('temperature', self.TEMPERATURE_NOISE_SIGMA),
            'fuel_qty': ('flow', self.FLOW_NOISE_SIGMA),
            'dP_dpf': ('pressure', self.PRESSURE_NOISE_SIGMA)
        }

        for sensor, (noise_type, sigma) in noise_params.items():
            if sensor in noisy_values:
                # Add Gaussian noise
                noise = np.random.normal(0, sigma * abs(noisy_values[sensor]))
                noisy_values[sensor] += noise

                # Add drift (linear over time)
                drift = self.DRIFT_RATE * hours_elapsed * noisy_values[sensor]
                noisy_values[sensor] += drift

                # Ensure physical bounds
                if 'MAP' in sensor or 'dP_dpf' in sensor:
                    noisy_values[sensor] = max(0.1, noisy_values[sensor])  # Positive pressure
                elif 'T_' in sensor:
                    noisy_values[sensor] = max(-50, min(1000, noisy_values[sensor]))  # Reasonable temp range
                elif 'MAF' in sensor or 'fuel_qty' in sensor:
                    noisy_values[sensor] = max(0, noisy_values[sensor])  # Positive flow

        # RPM typically very stable, add minimal noise
        if 'RPM' in noisy_values:
            rpm_noise = np.random.normal(0, 0.001 * abs(noisy_values['RPM']))  # 0.1% noise
            noisy_values['RPM'] += rpm_noise
            noisy_values['RPM'] = max(500, noisy_values['RPM'])  # Reasonable RPM range

        return noisy_values

    def is_steady_state(self, window_data: pd.DataFrame) -> bool:
        """
        Determine if a data window represents steady-state operation

        Args:
            window_data: DataFrame containing sensor readings over time window

        Returns:
            True if window is steady-state, False otherwise
        """
        if len(window_data) < 2:
            return False

        rpm_std = window_data['RPM'].std()
        maf_std = window_data['MAF'].std()
        fuel_std = window_data['fuel_qty'].std()

        maf_mean = window_data['MAF'].mean()
        fuel_mean = window_data['fuel_qty'].mean()

        # Avoid division by zero
        maf_threshold = self.MAF_STD_THRESHOLD * maf_mean if maf_mean > 0 else self.MAF_STD_THRESHOLD
        fuel_threshold = self.FUEL_QTY_STD_THRESHOLD * fuel_mean if fuel_mean > 0 else self.FUEL_QTY_STD_THRESHOLD

        return (rpm_std < self.RPM_STD_THRESHOLD and
                maf_std < maf_threshold and
                fuel_std < fuel_threshold)

    def generate_dataset(self, n_samples: int = 50000,
                        healthy_ratio: float = 0.8,
                        random_seed: Optional[int] = None) -> pd.DataFrame:
        """
        Generate complete synthetic dataset

        Args:
            n_samples: Total number of samples to generate
            healthy_ratio: Proportion of healthy samples (0.0-1.0)
            random_seed: Seed for reproducibility

        Returns:
            DataFrame with synthetic sensor data
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        logger.info(f"Generating dataset with {n_samples} samples ({healthy_ratio*100}% healthy)")

        # Calculate number of healthy and faulty samples
        n_healthy = int(n_samples * healthy_ratio)
        n_faulty = n_samples - n_healthy

        # Fault distribution across zones and severities
        n_zones = 5
        n_severities = 3
        samples_per_zone_severity = n_faulty // (n_zones * n_severities)
        remainder = n_faulty % (n_zones * n_severities)

        data_rows = []

        # Generate healthy baseline data
        logger.info("Generating healthy baseline data...")
        for i in range(n_healthy):
            # Random operating point within engine range
            rpm = np.random.uniform(1100, 2100)  # RPM range from spec
            fuel_qty = np.random.uniform(10, 70)  # Typical fuel quantity range (mg/stroke)

            # Generate healthy values
            healthy_values = self.compute_healthy_baseline(rpm, fuel_qty)

            # Add noise and drift
            timestamp = i * 0.1  # 100ms intervals
            noisy_values = self.add_noise_and_drift(healthy_values, timestamp)

            # Add metadata
            noisy_values.update({
                'timestamp': timestamp,
                'is_steady_state': True,  # Will be filtered later
                'leak_zone': 0,
                'leak_severity': 0
            })

            data_rows.append(noisy_values)

        # Generate faulty data
        logger.info("Generating faulty data with leak injections...")
        fault_idx = 0
        for zone in range(1, n_zones + 1):  # Zones 1-5
            for severity in range(1, n_severities + 1):  # Severities 1-3
                n_samples_this_type = samples_per_zone_severity
                if fault_idx < remainder:  # Distribute remainder
                    n_samples_this_type += 1

                logger.info(f"  Generating {n_samples_this_type} samples for Zone {zone}, Severity {severity}")

                for j in range(n_samples_this_type):
                    # Random operating point
                    rpm = np.random.uniform(1100, 2100)
                    fuel_qty = np.random.uniform(10, 70)

                    # Generate healthy baseline
                    healthy_values = self.compute_healthy_baseline(rpm, fuel_qty)

                    # Inject leak
                    leaky_values = self.inject_leak(healthy_values, zone, severity)

                    # Add noise and drift
                    timestamp = (n_healthy + fault_idx) * 0.1
                    noisy_values = self.add_noise_and_drift(leaky_values, timestamp)

                    # Add metadata
                    noisy_values.update({
                        'timestamp': timestamp,
                        'is_steady_state': True,  # Will be filtered later
                        'leak_zone': zone,
                        'leak_severity': severity
                    })

                    data_rows.append(noisy_values)
                    fault_idx += 1

        # Create DataFrame
        df = pd.DataFrame(data_rows)

        # Reorder columns for consistency
        column_order = [
            'timestamp', 'RPM', 'MAF', 'MAP_intake', 'MAP_boost', 'MAP_cac_in',
            'MAP_cac_out', 'T_intake', 'T_boost', 'T_cac_out', 'T_exh_manifold',
            'T_post_turbine', 'T_dpf_in', 'T_dpf_out', 'fuel_qty', 'dP_dpf',
            'is_steady_state', 'leak_zone', 'leak_severity'
        ]

        # Ensure all columns exist
        for col in column_order:
            if col not in df.columns:
                df[col] = 0.0 if col not in ['is_steady_state', 'leak_zone', 'leak_severity'] else 0

        df = df[column_order]

        logger.info(f"Dataset generation complete. Shape: {df.shape}")
        logger.info(f"Leak zone distribution:\n{df['leak_zone'].value_counts().sort_index()}")
        logger.info(f"Leak severity distribution:\n{df['leak_severity'].value_counts().sort_index()}")

        return df

def main():
    """Main function to demonstrate data generation"""
    import os

    # Create output directory
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize generator
    generator = SyntheticDataGenerator()

    # Generate dataset
    df = generator.generate_dataset(n_samples=50000, healthy_ratio=0.8, random_seed=42)

    # Save to CSV
    output_path = os.path.join(output_dir, "synthetic_engine_data.csv")
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")

    # Display basic statistics
    print("\nDataset Statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Healthy samples: {len(df[df['leak_zone'] == 0])}")
    print(f"Faulty samples: {len(df[df['leak_zone'] > 0])}")
    print(f"Features: {len(df.columns)}")

    # Show steady-state filtering example
    print("\nExample steady-state check:")
    # Simulate a window of data
    window_df = df.head(30)  # First 30 samples
    is_ss = generator.is_steady_state(window_df)
    print(f"First 30 samples steady-state: {is_ss}")
    print(f"RPM std: {window_df['RPM'].std():.2f}")
    print(f"MAF std: {window_df['MAF'].std():.2f} ({window_df['MAF'].std()/window_df['MAF'].mean()*100:.2f}% of mean)")
    print(f"Fuel qty std: {window_df['fuel_qty'].std():.2f} ({window_df['fuel_qty'].std()/window_df['fuel_qty'].mean()*100:.2f}% of mean)")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()