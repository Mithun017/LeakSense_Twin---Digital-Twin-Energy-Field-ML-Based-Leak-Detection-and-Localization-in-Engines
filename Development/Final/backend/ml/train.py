"""
LeakSense Twin — Training Pipeline
Trains all ML models:
  1. Generate synthetic data
  2. Compute features via digital twins
  3. Train LeakSenseNet (binary detection)
  4. Train LeakLocalizationNet (zone classification)
  5. Train RF + GB ensemble models
  6. Fit Energy Field detector
  7. Save all models
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score
import joblib
import os
import sys
from pathlib import Path

# Setup paths — go up one level from ml/ to backend/
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

from config import ML_CONFIG, MAP_AMBIENT
from ml.models import LeakSenseNet, LeakLocalizationNet, FocalLoss
from ml.ensemble import LeakSenseEnsemble
from ml.energy_field import EnergyFieldDetector
from twins.intake_twin import IntakeTwinModel
from twins.charge_air_twin import ChargeAirTwinModel
from twins.exhaust_twin import ExhaustTwinModel
from data_generator import generate_dataset


def compute_features_fast(df: pd.DataFrame,
                          intake_twin: IntakeTwinModel,
                          charge_twin: ChargeAirTwinModel,
                          exhaust_twin: ExhaustTwinModel) -> np.ndarray:
    """Compute 31-feature vector for the dataset (optimized)."""
    n = len(df)

    # Raw sensor features (13)
    raw_cols = ['RPM', 'MAF', 'MAP_intake', 'MAP_boost', 'MAP_cac_in', 'MAP_cac_out',
                'T_intake', 'T_boost', 'T_cac_out', 'T_exh_manifold', 'T_dpf_in',
                'T_dpf_out', 'fuel_qty']
    raw_features = df[raw_cols].values

    # Residual features (9) — vectorized where possible
    residuals = np.zeros((n, 9))
    print("  Computing digital twin residuals...")
    for i in range(n):
        if i % 10000 == 0:
            print(f"    Processing sample {i}/{n}...")
        row = df.iloc[i]
        rpm = row['RPM']
        maf = row['MAF']
        t_intake_k = row['T_intake'] + 273.15
        fuel = row['fuel_qty']
        t_cac_k = row['T_cac_out'] + 273.15

        maf_pred = intake_twin.predict(rpm, MAP_AMBIENT, t_intake_k, fuel)
        residuals[i, 0] = maf - maf_pred

        ca = charge_twin.predict(maf, rpm, t_intake_k, MAP_AMBIENT)
        residuals[i, 1] = row['MAP_boost'] - ca['MAP_boost_pred']
        residuals[i, 2] = row['T_boost'] - (ca['T_boost_pred'] - 273.15)
        residuals[i, 3] = row['T_cac_out'] - (ca['T_cac_out_pred'] - 273.15)
        residuals[i, 4] = row['MAP_cac_out'] - ca['MAP_cac_out_pred']

        ex = exhaust_twin.predict(maf, fuel, rpm, t_cac_k)
        residuals[i, 5] = row['T_exh_manifold'] - (ex['T_exh_manifold_pred'] - 273.15)
        t_post = row.get('T_post_turbine', 400.0)
        residuals[i, 6] = t_post - (ex['T_post_turbine_pred'] - 273.15)
        dP = row.get('dP_dpf', 0.0)
        residuals[i, 7] = dP - ex['dP_dpf_pred']
        residuals[i, 8] = row['MAP_intake'] - ca['MAP_intake_pred']

    # Derived ratio features (6)
    ratios = np.zeros((n, 6))
    ratios[:, 0] = df['MAP_boost'].values / (MAP_AMBIENT + 1e-8)
    t_boost = df['T_boost'].values
    t_intake = df['T_intake'].values
    t_cac = df['T_cac_out'].values
    denom = np.abs(t_boost - t_intake) + 1e-8
    ratios[:, 1] = (t_boost - t_cac) / denom
    ratios[:, 2] = df['T_exh_manifold'].values / (t_cac + 273.15 + 1e-8)
    ratios[:, 3] = df['MAP_boost'].values / (df['MAF'].values + 1e-8)
    dP_vals = df['dP_dpf'].values if 'dP_dpf' in df.columns else np.zeros(n)
    ratios[:, 4] = dP_vals / (df['MAP_boost'].values + 1e-8)
    fuel_flow = df['fuel_qty'].values * 1e-6 * (df['RPM'].values / 120.0) * 6
    ratios[:, 5] = (df['MAF'].values / 3600.0) / (fuel_flow + 1e-8)

    # Statistical features (3)
    stats = np.zeros((n, 3))
    stats[:, 0] = np.abs(residuals[:, 0]) * 0.1
    stats[:, 1] = np.abs(residuals[:, 1]) * 0.1
    stats[:, 2] = np.abs(residuals[:, 5]) * 0.1

    features = np.hstack([raw_features, residuals, ratios, stats])
    return features


def train_leak_net(X_train_t, y_train_t, X_val_t, y_val_t,
                   input_dim: int, config: dict) -> LeakSenseNet:
    """Train the binary leak detection network."""
    model = LeakSenseNet(input_dim=input_dim, hidden_dim=config['hidden_dim'])
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(),
                            lr=config['learning_rate'],
                            weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['max_epochs'])

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_dl = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)

    best_f1 = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(config['max_epochs']):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_dl:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_binary = (val_pred > 0.5).float()
            f1 = f1_score(y_val_t.numpy(), val_binary.numpy(), zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 20 == 0:
            avg_loss = total_loss / len(train_dl)
            print(f"    Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Val F1: {f1:.4f} | Best: {best_f1:.4f}")

        if patience_counter >= config['patience']:
            print(f"    Early stopping at epoch {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    print(f"  LeakSenseNet final best F1: {best_f1:.4f}")
    return model


def train_loc_net(X_train_t, y_train_t, X_val_t, y_val_t,
                  input_dim: int, n_classes: int, config: dict) -> LeakLocalizationNet:
    """Train the zone localization network."""
    model = LeakLocalizationNet(input_dim=input_dim, n_classes=n_classes)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(),
                            lr=config['learning_rate'],
                            weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['max_epochs'])

    train_ds = TensorDataset(X_train_t, y_train_t)
    train_dl = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)

    best_acc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(config['max_epochs']):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_dl:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_labels = torch.argmax(val_pred, dim=1)
            acc = float((val_labels == y_val_t).float().mean())

        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 20 == 0:
            avg_loss = total_loss / len(train_dl)
            print(f"    Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Val Acc: {acc:.4f} | Best: {best_acc:.4f}")

        if patience_counter >= config['patience']:
            print(f"    Early stopping at epoch {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    print(f"  LocalizationNet final best accuracy: {best_acc:.4f}")
    return model


def fit_energy_field(df_healthy: pd.DataFrame) -> EnergyFieldDetector:
    """Fit the energy field detector on healthy data."""
    ef = EnergyFieldDetector()

    # Create sliding windows from healthy data
    ef_cols = ['MAF', 'MAP_boost', 'MAP_cac_out', 'T_cac_out',
               'T_exh_manifold', 'dP_dpf', 'RPM']

    # Ensure dP_dpf column exists
    if 'dP_dpf' not in df_healthy.columns:
        df_healthy = df_healthy.copy()
        df_healthy['dP_dpf'] = 0.0

    data = df_healthy[ef_cols].values
    window_size = 30
    windows = []
    for i in range(0, len(data) - window_size, window_size):
        windows.append(data[i:i+window_size])

    if len(windows) > 0:
        ef.fit(windows)
        print(f"  Energy Field fitted on {len(windows)} windows")
    else:
        print("  Warning: Not enough data for energy field fitting")

    return ef


def run_training():
    """Main training pipeline."""
    print("=" * 60)
    print("LeakSense Twin — Training Pipeline")
    print("=" * 60)

    # Create output directories
    models_dir = Path(BACKEND_DIR) / "models"
    data_dir = Path(BACKEND_DIR) / "data"
    models_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)

    # Step 1: Generate data
    print("\n[1/7] Generating synthetic training data...")
    data_file = data_dir / "training_data.csv"
    if data_file.exists():
        print("  Loading existing dataset...")
        df = pd.read_csv(data_file)
    else:
        df = generate_dataset()
        df.to_csv(data_file, index=False)

    # Fill missing dP_dpf
    if 'dP_dpf' not in df.columns:
        df['dP_dpf'] = 0.0
    if 'T_post_turbine' not in df.columns:
        df['T_post_turbine'] = 400.0

    # Step 2: Initialize digital twins
    print("\n[2/7] Initializing digital twin models...")
    intake_twin = IntakeTwinModel()
    charge_twin = ChargeAirTwinModel()
    exhaust_twin = ExhaustTwinModel()

    # Save digital twins
    intake_twin.save(str(models_dir / "intake_twin.joblib"))
    charge_twin.save(str(models_dir / "charge_air_twin.joblib"))
    exhaust_twin.save(str(models_dir / "exhaust_twin.joblib"))
    print("  Digital twins saved.")

    # Step 3: Compute features
    print("\n[3/7] Computing feature vectors (31 features)...")
    features = compute_features_fast(df, intake_twin, charge_twin, exhaust_twin)

    # Replace NaN/Inf
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # Labels
    y_binary = (df['leak_zone'] > 0).astype(np.float32).values
    y_zone = df['leak_zone'].astype(np.int64).values

    print(f"  Feature matrix shape: {features.shape}")
    print(f"  Binary labels — Healthy: {(y_binary==0).sum()}, Leak: {(y_binary==1).sum()}")

    # Step 4: Train/val/test split
    print("\n[4/7] Splitting data (70/15/15)...")
    X_train, X_temp, y_b_train, y_b_temp, y_z_train, y_z_temp = train_test_split(
        features, y_binary, y_zone, test_size=0.30, random_state=42, stratify=y_zone
    )
    X_val, X_test, y_b_val, y_b_test, y_z_val, y_z_test = train_test_split(
        X_temp, y_b_temp, y_z_temp, test_size=0.50, random_state=42, stratify=y_z_temp
    )

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)
    joblib.dump(scaler, str(models_dir / "scaler.joblib"))

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train_s)
    X_val_t = torch.FloatTensor(X_val_s)
    X_test_t = torch.FloatTensor(X_test_s)
    y_b_train_t = torch.FloatTensor(y_b_train)
    y_b_val_t = torch.FloatTensor(y_b_val)
    y_z_train_t = torch.LongTensor(y_z_train)
    y_z_val_t = torch.LongTensor(y_z_val)

    input_dim = X_train_s.shape[1]
    print(f"  Input dimension: {input_dim}")

    # Step 5: Train LeakSenseNet
    print("\n[5/7] Training LeakSenseNet (binary detector)...")
    config = ML_CONFIG.copy()
    leak_net = train_leak_net(X_train_t, y_b_train_t, X_val_t, y_b_val_t, input_dim, config)
    torch.save(leak_net.state_dict(), str(models_dir / "leak_sense_net.pth"))

    # Step 6: Train LocalizationNet
    print("\n[6/7] Training LeakLocalizationNet (zone classifier)...")
    loc_net = train_loc_net(X_train_t, y_z_train_t, X_val_t, y_z_val_t,
                            input_dim, config['n_classes'], config)
    torch.save(loc_net.state_dict(), str(models_dir / "leak_localization_net.pth"))

    # Step 7: Train ensemble
    print("\n[7/7] Training Ensemble (RF + GB)...")
    ensemble = LeakSenseEnsemble()
    ensemble.fit_sklearn_models(X_train, y_b_train, y_z_train)
    ensemble.leak_net = leak_net
    ensemble.loc_net = loc_net
    ensemble.scaler = scaler
    ensemble.save(str(models_dir / "ensemble.joblib"))

    # Fit Energy Field
    print("\n[BONUS] Fitting Energy Field detector...")
    df_healthy = df[df['leak_zone'] == 0]
    ef_detector = fit_energy_field(df_healthy)
    ef_detector.save(str(models_dir / "energy_field_detector.joblib"))

    # ── Evaluation ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION ON TEST SET")
    print("=" * 60)

    # Binary detection
    leak_net.eval()
    with torch.no_grad():
        test_pred_b = leak_net(X_test_t)
        test_binary = (test_pred_b > 0.5).float().numpy()

    print("\n Binary Detection (LeakSenseNet):")
    print(classification_report(y_b_test, test_binary,
                                target_names=['Healthy', 'Leak'], zero_division=0))

    # Zone localization
    loc_net.eval()
    with torch.no_grad():
        test_pred_z = loc_net(X_test_t)
        test_zones = torch.argmax(test_pred_z, dim=1).numpy()

    zone_names = ['Healthy', 'Zone 1', 'Zone 2', 'Zone 3', 'Zone 4', 'Zone 5']
    print("\n Zone Localization (LocalizationNet):")
    print(classification_report(y_z_test, test_zones,
                                target_names=zone_names, zero_division=0))

    # Ensemble evaluation
    print("\n Ensemble Evaluation:")
    ensemble_results = []
    for i in range(len(X_test_s)):
        feat_t = torch.FloatTensor([X_test_s[i]])
        feat_np = X_test_s[i:i+1]
        result = ensemble.predict(feat_t, feat_np)
        ensemble_results.append(result['confidence'])

    ensemble_binary = np.array([1.0 if c > 0.5 else 0.0 for c in ensemble_results])
    ens_f1 = f1_score(y_b_test, ensemble_binary, zero_division=0)
    print(f"  Ensemble F1: {ens_f1:.4f}")

    # Save config
    config_info = {
        'input_dim': input_dim,
        'n_classes': 6,
        'n_samples': len(df),
        'n_features': input_dim,
        'ensemble_f1': float(ens_f1),
    }
    joblib.dump(config_info, str(models_dir / "config.joblib"))

    print("\n" + "=" * 60)
    print("All models saved to:", models_dir)
    print("Training complete!")
    print("=" * 60)

    return config_info


if __name__ == "__main__":
    run_training()
