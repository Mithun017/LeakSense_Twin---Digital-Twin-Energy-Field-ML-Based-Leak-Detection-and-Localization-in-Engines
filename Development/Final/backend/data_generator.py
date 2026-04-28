"""
LeakSense Twin — Synthetic Data Generator
Generates 50,000+ training samples for Cat C18 engine:
  - 40,000 healthy baseline samples
  - 10,000 fault samples (5 zones × 3 severities)
Uses physics equations from C18 torque/power curve.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DISPLACEMENT, N_CYLINDERS, R_AIR, CP_AIR, GAMMA,
    CAC_EFFECTIVENESS, TURBO_EFF_COMPRESSOR, TURBO_EFF_TURBINE,
    VE_BASE, LHV_DIESEL, COMBUSTION_EFF, MAP_AMBIENT, T_AMBIENT_K,
    NOISE_CONFIG
)


def ve_model(rpm: float) -> float:
    """Volumetric efficiency as function of RPM — peaks at 1400 RPM."""
    return VE_BASE - 1.2e-5 * (rpm - 1400) ** 2 / 1e6


def generate_healthy_sample(rpm: float, ambient_variation: float = 0.0) -> dict:
    """
    Generate a single healthy-state sensor reading at a given RPM.
    Uses Cat C18 thermodynamic equations.
    """
    # Ambient conditions with slight variation
    T_ambient_K = T_AMBIENT_K + ambient_variation * 5.0  # ±5K variation
    P_ambient = MAP_AMBIENT + ambient_variation * 2.0     # ±2 kPa variation

    # Fuel quantity scales with RPM (from C18 fuel map)
    fuel_qty = 80.0 + (rpm - 1100) * 0.06  # mg/stroke — linear approximation
    fuel_qty = np.clip(fuel_qty, 60, 160)

    # --- Intake Zone ---
    ve = ve_model(rpm)
    rho_air = P_ambient * 1000.0 / (R_AIR * T_ambient_K)
    vol_per_cyl = (DISPLACEMENT / 1000.0) / N_CYLINDERS  # m³
    n_strokes = rpm / 2.0 / 60.0  # 4-stroke
    MAF = ve * vol_per_cyl * N_CYLINDERS * n_strokes * rho_air * 3600.0  # kg/h

    # --- Charge Air System ---
    # Corrected flow/speed for compressor map
    corr_flow = MAF * np.sqrt(T_ambient_K / 288.15) / (P_ambient / 101.325)
    corr_speed = rpm / np.sqrt(T_ambient_K / 288.15)

    # Pressure ratio (simplified compressor map for C18)
    PR = 1.0 + 0.0015 * corr_flow + 0.0008 * corr_speed - 2e-7 * corr_flow**2
    PR = np.clip(PR, 1.2, 4.0)

    MAP_boost = PR * P_ambient
    T_boost_K = T_ambient_K * (1.0 + (PR**((GAMMA-1)/GAMMA) - 1.0) / TURBO_EFF_COMPRESSOR)
    T_cac_out_K = T_boost_K - CAC_EFFECTIVENESS * (T_boost_K - T_ambient_K)

    # CAC pressure drop
    MAP_cac_in = MAP_boost - 0.5  # small duct loss
    MAP_cac_out = MAP_boost - 0.0012 * (MAF / 100.0)**2
    MAP_intake = MAP_cac_out - 1.5  # manifold runner loss

    # --- Exhaust Zone ---
    fuel_mass_flow = fuel_qty * 1e-6 * n_strokes * N_CYLINDERS  # kg/s
    air_mass_flow = MAF / 3600.0
    total_flow = air_mass_flow + fuel_mass_flow

    if total_flow > 0:
        delta_T = (fuel_mass_flow * LHV_DIESEL * 1000.0 * COMBUSTION_EFF) / (total_flow * CP_AIR)
        T_exh_manifold = T_cac_out_K + delta_T
    else:
        T_exh_manifold = T_cac_out_K + 400.0

    T_exh_manifold = np.clip(T_exh_manifold, 723.15, 1173.15)

    # Turbine
    PR_turbine = 1.5 + 1.2 * (rpm / 2100.0) + 0.8 * (MAF / 1200.0)
    PR_turbine = np.clip(PR_turbine, 1.2, 4.0)
    T_post_turbine = T_exh_manifold * (1.0 - TURBO_EFF_TURBINE * (1.0 - PR_turbine**(-(GAMMA-1)/GAMMA)))

    # DPF
    dP_dpf = 2.5 * (air_mass_flow ** 1.8)
    T_dpf_in = T_post_turbine - 30.0
    T_dpf_out = T_dpf_in - 50.0

    return {
        'RPM': rpm,
        'MAF': MAF,
        'MAP_intake': MAP_intake,
        'MAP_boost': MAP_boost,
        'MAP_cac_in': MAP_cac_in,
        'MAP_cac_out': MAP_cac_out,
        'T_intake': T_ambient_K - 273.15,      # °C
        'T_boost': T_boost_K - 273.15,          # °C
        'T_cac_out': T_cac_out_K - 273.15,      # °C
        'T_exh_manifold': T_exh_manifold - 273.15,  # °C
        'T_dpf_in': T_dpf_in - 273.15,
        'T_dpf_out': T_dpf_out - 273.15,
        'fuel_qty': fuel_qty,
        'T_post_turbine': T_post_turbine - 273.15,
        'dP_dpf': dP_dpf,
    }


def inject_leak(sample: dict, zone: int, severity: int) -> dict:
    """
    Inject a leak into a healthy sample.

    Args:
        sample: healthy sensor reading dict
        zone: 1-5 (leak zone)
        severity: 1=small(2%), 2=medium(8%), 3=large(15%)

    Returns:
        modified sample with leak effects
    """
    s = sample.copy()
    severity_map = {1: 0.02, 2: 0.08, 3: 0.15}
    leak_fraction = severity_map[severity]

    if zone == 1:
        # Zone A: Airflow meter → Compressor inlet
        # MAF drops, boost drops proportionally
        s['MAF'] *= (1.0 - leak_fraction)
        s['MAP_boost'] *= (1.0 - leak_fraction * 0.7)
        s['T_boost'] += leak_fraction * 5.0  # slight temp rise

    elif zone == 2:
        # Zone B: Post-compressor → CAC inlet
        # MAP_boost drops, MAF unchanged (leak downstream of sensor)
        s['MAP_boost'] *= (1.0 - leak_fraction * 0.8)
        s['MAP_cac_in'] *= (1.0 - leak_fraction * 0.9)
        s['T_intake'] += leak_fraction * 8.0  # less cooling

    elif zone == 3:
        # Zone C: CAC → Intake manifold
        s['MAP_cac_out'] *= (1.0 - leak_fraction * 0.85)
        s['MAP_intake'] *= (1.0 - leak_fraction * 0.9)
        s['T_cac_out'] += leak_fraction * 6.0

    elif zone == 4:
        # Zone D: Exhaust manifold → Turbo turbine
        s['T_exh_manifold'] *= (1.0 - leak_fraction * 0.3)
        s['MAP_boost'] *= (1.0 - leak_fraction * 0.5)  # cascading boost drop
        s['T_dpf_in'] *= (1.0 - leak_fraction * 0.2)

    elif zone == 5:
        # Zone E: DPF/SCR area
        s['dP_dpf'] *= (1.0 - leak_fraction * 0.6)
        s['T_dpf_out'] *= (1.0 - leak_fraction * 0.15)
        s['T_dpf_in'] *= (1.0 - leak_fraction * 0.1)

    return s


def add_noise(sample: dict) -> dict:
    """Add realistic Gaussian sensor noise to a sample."""
    s = sample.copy()
    nc = NOISE_CONFIG

    # Pressure sensors: 0.5% noise
    for key in ['MAP_intake', 'MAP_boost', 'MAP_cac_in', 'MAP_cac_out', 'dP_dpf']:
        if key in s:
            s[key] += np.random.normal(0, abs(s[key]) * nc['pressure_sigma_pct'])

    # Temperature sensors: 1% noise
    for key in ['T_intake', 'T_boost', 'T_cac_out', 'T_exh_manifold', 'T_dpf_in', 'T_dpf_out', 'T_post_turbine']:
        if key in s:
            s[key] += np.random.normal(0, max(abs(s[key]) * nc['temperature_sigma_pct'], 0.5))

    # Flow sensors: 1.5% noise
    if 'MAF' in s:
        s['MAF'] += np.random.normal(0, abs(s['MAF']) * nc['flow_sigma_pct'])

    # RPM noise (small)
    s['RPM'] += np.random.normal(0, 3.0)

    # Fuel qty noise
    s['fuel_qty'] += np.random.normal(0, abs(s['fuel_qty']) * 0.005)

    return s


def generate_dataset(n_healthy: int = 40000, n_fault_per_combo: int = 667,
                     seed: int = 42) -> pd.DataFrame:
    """
    Generate the complete training dataset.

    Args:
        n_healthy: number of healthy samples
        n_fault_per_combo: samples per zone×severity combination
        seed: random seed for reproducibility

    Returns:
        DataFrame with all samples, labels, and metadata
    """
    np.random.seed(seed)
    records = []

    # RPM range: 1100-2100 in steps, with random sub-sampling
    rpm_range = np.linspace(1100, 2100, 11)

    print("Generating healthy samples...")
    for i in range(n_healthy):
        rpm = np.random.choice(rpm_range) + np.random.uniform(-25, 25)
        rpm = np.clip(rpm, 1100, 2100)
        ambient_var = np.random.uniform(-1.0, 1.0)

        sample = generate_healthy_sample(rpm, ambient_var)
        sample = add_noise(sample)
        sample['leak_zone'] = 0
        sample['leak_severity'] = 0
        sample['is_steady_state'] = 1
        records.append(sample)

    # Fault samples: 5 zones × 3 severities
    print("Generating fault samples...")
    for zone in range(1, 6):
        for severity in range(1, 4):
            for i in range(n_fault_per_combo):
                rpm = np.random.choice(rpm_range) + np.random.uniform(-25, 25)
                rpm = np.clip(rpm, 1100, 2100)
                ambient_var = np.random.uniform(-1.0, 1.0)

                sample = generate_healthy_sample(rpm, ambient_var)
                sample = inject_leak(sample, zone, severity)
                sample = add_noise(sample)
                sample['leak_zone'] = zone
                sample['leak_severity'] = severity
                sample['is_steady_state'] = 1
                records.append(sample)

    df = pd.DataFrame(records)

    # Add timestamp column
    df['timestamp'] = pd.date_range(start='2026-01-01', periods=len(df), freq='1s')

    # Shuffle
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    print(f"Dataset generated: {len(df)} samples")
    print(f"  Healthy: {(df['leak_zone'] == 0).sum()}")
    print(f"  Fault:   {(df['leak_zone'] > 0).sum()}")
    for z in range(1, 6):
        print(f"    Zone {z}: {(df['leak_zone'] == z).sum()}")

    return df


def compute_features(df: pd.DataFrame,
                     intake_twin=None,
                     charge_twin=None,
                     exhaust_twin=None) -> np.ndarray:
    """
    Compute the 31-feature vector for each sample.

    Features:
      - 13 raw sensor features
      - 9 residual features from digital twins
      - 6 derived ratio features
      - 3 statistical features (simplified — rolling stats need time series)
    """
    n = len(df)

    # 13 raw sensor features
    raw_cols = ['RPM', 'MAF', 'MAP_intake', 'MAP_boost', 'MAP_cac_in', 'MAP_cac_out',
                'T_intake', 'T_boost', 'T_cac_out', 'T_exh_manifold', 'T_dpf_in',
                'T_dpf_out', 'fuel_qty']
    raw_features = df[raw_cols].values  # (n, 13)

    # 9 residual features from digital twins
    from twins.intake_twin import IntakeTwinModel
    from twins.charge_air_twin import ChargeAirTwinModel
    from twins.exhaust_twin import ExhaustTwinModel

    if intake_twin is None:
        intake_twin = IntakeTwinModel()
    if charge_twin is None:
        charge_twin = ChargeAirTwinModel()
    if exhaust_twin is None:
        exhaust_twin = ExhaustTwinModel()

    residuals = np.zeros((n, 9))
    for i in range(n):
        row = df.iloc[i]
        rpm = row['RPM']
        maf = row['MAF']
        t_intake_k = row['T_intake'] + 273.15
        fuel = row['fuel_qty']
        t_cac_k = row['T_cac_out'] + 273.15

        # Intake residual
        maf_pred = intake_twin.predict(rpm, MAP_AMBIENT, t_intake_k, fuel)
        residuals[i, 0] = maf - maf_pred

        # Charge air residuals
        ca = charge_twin.predict(maf, rpm, t_intake_k, MAP_AMBIENT)
        residuals[i, 1] = row['MAP_boost'] - ca['MAP_boost_pred']
        residuals[i, 2] = row['T_boost'] - (ca['T_boost_pred'] - 273.15)
        residuals[i, 3] = row['T_cac_out'] - (ca['T_cac_out_pred'] - 273.15)
        residuals[i, 4] = row['MAP_cac_out'] - ca['MAP_cac_out_pred']

        # Exhaust residuals
        ex = exhaust_twin.predict(maf, fuel, rpm, t_cac_k)
        residuals[i, 5] = row['T_exh_manifold'] - (ex['T_exh_manifold_pred'] - 273.15)
        t_post = row.get('T_post_turbine', 400)
        residuals[i, 6] = t_post - (ex['T_post_turbine_pred'] - 273.15)
        dP = row.get('dP_dpf', 0)
        residuals[i, 7] = dP - ex['dP_dpf_pred']
        residuals[i, 8] = row['MAP_intake'] - ca['MAP_intake_pred']

    # 6 derived ratio features
    ratios = np.zeros((n, 6))
    ratios[:, 0] = df['MAP_boost'].values / (MAP_AMBIENT + 1e-8)           # PR_compressor
    ratios[:, 1] = np.where(
        (df['T_boost'].values - df['T_intake'].values).abs() > 1e-3 if hasattr(df['T_boost'], 'abs') else np.abs(df['T_boost'].values - df['T_intake'].values) > 1e-3,
        (df['T_boost'].values - df['T_cac_out'].values) / (df['T_boost'].values - df['T_intake'].values + 1e-8),
        0.88
    )  # CAC effectiveness
    ratios[:, 2] = df['T_exh_manifold'].values / (df['T_cac_out'].values + 273.15 + 1e-8)  # T_exh ratio
    ratios[:, 3] = df['MAP_boost'].values / (df['MAF'].values + 1e-8)      # boost/MAF ratio
    ratios[:, 4] = df.get('dP_dpf', pd.Series(np.zeros(n))).values / (df['MAP_boost'].values + 1e-8)  # dpf_dp ratio
    # Air-fuel ratio
    fuel_flow = df['fuel_qty'].values * 1e-6 * (df['RPM'].values / 2 / 60) * 6
    ratios[:, 5] = (df['MAF'].values / 3600) / (fuel_flow + 1e-8)          # AFR

    # 3 statistical features (using per-sample noise as proxy for window stats)
    stats = np.zeros((n, 3))
    stats[:, 0] = np.abs(residuals[:, 0]) * 0.1  # MAF_std proxy
    stats[:, 1] = np.abs(residuals[:, 1]) * 0.1  # MAP_boost_std proxy
    stats[:, 2] = np.abs(residuals[:, 5]) * 0.1  # T_exh_std proxy

    # Concatenate: 13 + 9 + 6 + 3 = 31 features
    features = np.hstack([raw_features, residuals, ratios, stats])

    return features


if __name__ == "__main__":
    # Generate dataset and save
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)

    df = generate_dataset()
    df.to_csv(output_dir / "training_data.csv", index=False)
    print(f"Saved dataset to {output_dir / 'training_data.csv'}")
