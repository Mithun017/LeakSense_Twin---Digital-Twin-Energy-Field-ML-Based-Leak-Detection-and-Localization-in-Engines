"""
Main Leak Detection System for LeakSense Twin
Integrates all components: data generation, digital twins, energy field, and ML models
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import joblib
import logging
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Import our custom components
from data_generation.synthetic_data_generator import SyntheticDataGenerator
from models.intake_twin import IntakeTwinModel
from models.charge_air_twin import ChargeAirTwinModel
from models.exhaust_twin import ExhaustTwinModel
from energy_field.energy_field_detector import EnergyFieldDetector, compute_energy_field
from ml.leak_sense_net import LeakSenseNet, LeakLocalizationNet, LeakSenseEnsemble, focal_loss

logger = logging.getLogger(__name__)

class LeakSenseDataset(Dataset):
    """PyTorch Dataset for LeakSense training data"""

    def __init__(self, features: np.ndarray, labels: np.ndarray, zone_labels: np.ndarray = None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.zone_labels = torch.LongTensor(zone_labels) if zone_labels is not None else None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.zone_labels is not None:
            return self.features[idx], self.labels[idx], self.zone_labels[idx]
        return self.features[idx], self.labels[idx]

def prepare_features_for_ml(df: pd.DataFrame,
                           intake_twin: IntakeTwinModel = None,
                           charge_air_twin: ChargeAirTwinModel = None,
                           exhaust_twin: ExhaustTwinModel = None,
                           energy_field_detector: EnergyFieldDetector = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare features for ML models including raw sensors, residuals, and energy field features

    Returns:
        Tuple of (features, leak_labels, zone_labels)
    """
    # Define feature columns as per ML Model Build Prompt
    raw_sensor_cols = ['RPM', 'MAF', 'MAP_intake', 'MAP_boost', 'MAP_cac_in', 'MAP_cac_out',
                      'T_intake', 'T_boost', 'T_cac_out', 'T_exh_manifold', 'T_dpf_in', 'T_dpf_out', 'fuel_qty']

    # Check which columns are available
    available_cols = [col for col in raw_sensor_cols if col in df.columns]
    missing_cols = [col for col in raw_sensor_cols if col not in df.columns]

    if missing_cols:
        logger.warning(f"Missing sensor columns: {missing_cols}")
        # Fill missing columns with zeros for now
        for col in missing_cols:
            df[col] = 0.0

    raw_features = df[raw_sensor_cols].values

    # Compute residuals from digital twins if available
    residual_features = []
    if intake_twin is not None and intake_twin.is_fitted:
        try:
            intake_features = df[['RPM', 'MAP_intake', 'T_intake', 'fuel_qty', 'MAF']]
            res_MAF = intake_twin.predict_residual(intake_features)
            residual_features.append(res_MAF.reshape(-1, 1))
        except Exception as e:
            logger.warning(f"Could not compute intake twin residuals: {e}")
            residual_features.append(np.zeros((len(df), 1)))

    if charge_air_twin is not None and charge_air_twin.is_fitted:
        try:
            charge_air_features = df[['MAF', 'RPM', 'T_intake', 'MAP_intake']]
            charge_air_pred = charge_air_twin.predict(charge_air_features)
            # We need actual values to compute residuals - using available columns
            actual_boost = df['MAP_boost'].values if 'MAP_boost' in df.columns else np.zeros(len(df))
            actual_t_boost = df['T_boost'].values if 'T_boost' in df.columns else np.zeros(len(df))
            actual_t_cac_out = df['T_cac_out'].values if 'T_cac_out' in df.columns else np.zeros(len(df))
            actual_map_intake = df['MAP_intake'].values if 'MAP_intake' in df.columns else np.zeros(len(df))

            res_MAP_boost = actual_boost - charge_air_pred['MAP_boost_pred'].values
            res_T_boost = actual_t_boost - charge_air_pred['T_boost_pred'].values
            res_T_cac_out = actual_t_cac_out - charge_air_pred['T_cac_out_pred'].values
            res_MAP_intake = actual_map_intake - charge_air_pred['MAP_intake_pred'].values

            residual_features.extend([
                res_MAP_boost.reshape(-1, 1),
                res_T_boost.reshape(-1, 1),
                res_T_cac_out.reshape(-1, 1),
                res_MAP_intake.reshape(-1, 1)
            ])
        except Exception as e:
            logger.warning(f"Could not compute charge air twin residuals: {e}")
            residual_features.extend([np.zeros((len(df), 1)) for _ in range(4)])

    if exhaust_twin is not None and exhaust_twin.is_fitted:
        try:
            exhaust_features = df[['MAF', 'fuel_qty', 'RPM', 'T_cac_out']]
            exhaust_pred = exhaust_twin.predict(exhaust_features)
            actual_t_exh = df['T_exh_manifold'].values if 'T_exh_manifold' in df.columns else np.zeros(len(df))
            actual_t_post = df['T_post_turbine'].values if 'T_post_turbine' in df.columns else np.zeros(len(df))
            actual_dP_dpf = df['dP_dpf'].values if 'dP_dpf' in df.columns else np.zeros(len(df))

            res_T_exh_manifold = actual_t_exh - exhaust_pred['T_exh_manifold_pred'].values
            res_T_post_turbine = actual_t_post - exhaust_pred['T_post_turbine_pred'].values
            res_dP_dpf = actual_dP_dpf - exhaust_pred['dP_dpf_pred'].values

            residual_features.extend([
                res_T_exh_manifold.reshape(-1, 1),
                res_T_post_turbine.reshape(-1, 1),
                res_dP_dpf.reshape(-1, 1)
            ])
        except Exception as e:
            logger.warning(f"Could not compute exhaust twin residuals: {e}")
            residual_features.extend([np.zeros((len(df), 1)) for _ in range(3)])

    # If we couldn't compute any residuals, create placeholder zeros
    if len(residual_features) == 0:
        # We expect 9 residual features based on the prompt
        residual_features = [np.zeros((len(df), 1)) for _ in range(9)]

    residual_features = np.hstack(residual_features) if residual_features else np.zeros((len(df), 0))

    # Compute energy field features if detector is available
    energy_features = np.zeros((len(df), 36))  # Default to zeros (6x6 flattened)
    if energy_field_detector is not None and energy_field_detector.is_fitted:
        try:
            sensor_cols_for_ef = ['MAF', 'MAP_boost', 'MAP_cac_out', 'T_cac_out',
                                 'T_exh_manifold', 'dP_dpf', 'RPM']
            # Check if we have all columns
            if all(col in df.columns for col in sensor_cols_for_ef):
                sensor_data = df[sensor_cols_for_ef].values
                window_size = 30
                # We'll compute for each sample i, the window ending at i (of size window_size) if available
                for i in range(len(sensor_data)):
                    start_idx = max(0, i - window_size + 1)
                    end_idx = i + 1
                    window = sensor_data[start_idx:end_idx]
                    if window.shape[0] == window_size:
                        # We have a full window
                        deviation = energy_field_detector.compute_deviation(window)
                        z_field = deviation['z_field']  # shape (6,6)
                        energy_features[i] = z_field.flatten()
                    # else: not enough data for a full window, leave as zeros (already initialized)
            else:
                logger.warning("Missing columns for energy field computation")
        except Exception as e:
            logger.warning(f"Could not compute energy field features: {e}")
            # energy_features remains zeros
    # else: detector not available or not fitted, energy_features remains zeros

    # Combine all features
    all_features = np.hstack([raw_features, residual_features, energy_features])

    # Create labels
    leak_labels = (df['leak_zone'] > 0).astype(int).values  # 1 if leak, 0 if healthy
    zone_labels = df['leak_zone'].values  # 0-5 for zones (0=no leak)

    logger.info(f"Prepared features shape: {all_features.shape}")
    logger.info(f"  Raw sensors: {raw_features.shape[1]}")
    logger.info(f"  Residuals: {residual_features.shape[1] if residual_features.size > 0 else 0}")
    logger.info(f"  Energy field: {energy_features.shape[1] if energy_features.size > 0 else 0}")
    logger.info(f"Leak label distribution: {np.bincount(leak_labels)}")
    logger.info(f"Zone label distribution: {np.bincount(zone_labels)}")

    # Log some statistics about the features to help debug
    if all_features.size > 0:
        logger.info(f"Feature stats - mean: {np.mean(all_features):.4f}, std: {np.std(all_features):.4f}")
        logger.info(f"Feature stats - min: {np.min(all_features):.4f}, max: {np.max(all_features):.4f}")

        # Check if there's separation between leak and non-leak in first few features
        if leak_labels.sum() > 0 and (leak_labels == 0).sum() > 0:
            leak_features = all_features[leak_labels == 1]
            healthy_features = all_features[leak_labels == 0]
            if leak_features.size > 0 and healthy_features.size > 0:
                leak_mean = np.mean(leak_features, axis=0)
                healthy_mean = np.mean(healthy_features, axis=0)
                diff = np.mean(np.abs(leak_mean - healthy_mean))
                logger.info(f"Mean absolute difference between leak/healthy features: {diff:.4f}")

    return all_features, leak_labels, zone_labels

def train_ml_models(X_train: np.ndarray, y_train_leak: np.ndarray, y_train_zone: np.ndarray,
                   X_val: np.ndarray, y_val_leak: np.ndarray, y_val_zone: np.ndarray,
                   input_dim: int) -> LeakSenseEnsemble:
    """
    Train the ML models for leak detection and localization
    """
    logger.info("Training ML models...")

    # Create datasets and data loaders
    train_dataset = LeakSenseDataset(X_train, y_train_leak, y_train_zone)
    val_dataset = LeakSenseDataset(X_val, y_val_leak, y_val_zone)

    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize models
    leak_net = LeakSenseNet(input_dim=input_dim)
    loc_net = LeakLocalizationNet(input_dim=input_dim, n_classes=6)
    ensemble = LeakSenseEnsemble(input_dim=input_dim)

    # Debug: check initial leak net output
    with torch.no_grad():
        # Take a small batch from training data
        debug_batch = torch.FloatTensor(X_train[:32])
        debug_output = leak_net(debug_batch)
        logger.info(f"Initial leak net output stats - min: {debug_output.min():.4f}, max: {debug_output.max():.4f}, mean: {debug_output.mean():.4f}")

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    leak_net.to(device)
    loc_net.to(device)

    # Loss functions
    leak_criterion = focal_loss(alpha=0.75, gamma=2.0)  # Weight leak class more (alpha=0.75 means 75% weight on leak class)
    zone_criterion = nn.CrossEntropyLoss()

    # Optimizers
    leak_optimizer = optim.AdamW(leak_net.parameters(), lr=1e-3, weight_decay=1e-4)
    loc_optimizer = optim.AdamW(loc_net.parameters(), lr=1e-3, weight_decay=1e-4)

    # Learning rate schedulers
    leak_scheduler = optim.lr_scheduler.CosineAnnealingLR(leak_optimizer, T_max=200)
    loc_scheduler = optim.lr_scheduler.CosineAnnealingLR(loc_optimizer, T_max=200)

    # Training loop
    epochs = 200
    best_val_f1 = 0.0
    patience_counter = 0
    patience = 20

    for epoch in range(epochs):
        # Training phase
        leak_net.train()
        loc_net.train()

        train_leak_loss = 0.0
        train_zone_loss = 0.0
        train_leak_correct = 0
        train_zone_correct = 0
        train_total = 0

        for batch_idx, (data, leak_target, zone_target) in enumerate(train_loader):
            data = data.to(device)
            leak_target = leak_target.to(device)
            zone_target = zone_target.to(device)

            # Leak detection training
            leak_optimizer.zero_grad()
            leak_output = leak_net(data)
            leak_loss = leak_criterion(leak_output, leak_target)
            leak_loss.backward()
            leak_optimizer.step()

            # Zone localization training
            loc_optimizer.zero_grad()
            zone_output = loc_net(data)
            zone_loss = zone_criterion(zone_output, zone_target)
            zone_loss.backward()
            loc_optimizer.step()

            # Statistics
            train_leak_loss += leak_loss.item()
            train_zone_loss += zone_loss.item()

            leak_pred = (leak_output > 0.5).float()
            train_leak_correct += (leak_pred == leak_target).sum().item()

            _, zone_pred = torch.max(zone_output.data, 1)
            train_zone_correct += (zone_pred == zone_target).sum().item()
            train_total += data.size(0)

        # Validation phase
        leak_net.eval()
        loc_net.eval()

        val_leak_loss = 0.0
        val_zone_loss = 0.0
        val_leak_correct = 0
        val_zone_correct = 0
        val_total = 0
        all_val_leak_preds = []
        all_val_leak_targets = []
        all_val_zone_preds = []
        all_val_zone_targets = []

        with torch.no_grad():
            for data, leak_target, zone_target in val_loader:
                data = data.to(device)
                leak_target = leak_target.to(device)
                zone_target = zone_target.to(device)

                leak_output = leak_net(data)
                zone_output = loc_net(data)

                val_leak_loss += leak_criterion(leak_output, leak_target).item()
                val_zone_loss += zone_criterion(zone_output, zone_target).item()

                leak_pred = (leak_output > 0.5).float()
                val_leak_correct += (leak_pred == leak_target).sum().item()

                _, zone_pred = torch.max(zone_output.data, 1)
                val_zone_correct += (zone_pred == zone_target).sum().item()
                val_total += data.size(0)

                # Store for F1 calculation
                all_val_leak_preds.append(leak_pred.cpu().numpy())
                all_val_leak_targets.append(leak_target.cpu().numpy())
                all_val_zone_preds.append(zone_pred.cpu().numpy())
                all_val_zone_targets.append(zone_target.cpu().numpy())

        # Calculate metrics
        train_leak_acc = train_leak_correct / train_total
        train_zone_acc = train_zone_correct / train_total
        val_leak_acc = val_leak_correct / val_total
        val_zone_acc = val_zone_correct / val_total

        # Calculate F1 scores
        from sklearn.metrics import f1_score
        val_leak_preds_all = np.concatenate(all_val_leak_preds)
        val_leak_targets_all = np.concatenate(all_val_leak_targets)
        val_zone_preds_all = np.concatenate(all_val_zone_preds)
        val_zone_targets_all = np.concatenate(all_val_zone_targets)

        val_leak_f1 = f1_score(val_leak_targets_all, val_leak_preds_all)
        val_zone_f1 = f1_score(val_zone_targets_all, val_zone_preds_all, average='weighted')

        # Update schedulers
        leak_scheduler.step()
        loc_scheduler.step()

        # Logging
        if (epoch + 1) % 20 == 0 or epoch == 0:
            logger.info(f'Epoch [{epoch+1}/{epochs}] '
                       f'Train Leak Loss: {train_leak_loss/len(train_loader):.4f}, Acc: {train_leak_acc:.4f} '
                       f'Train Zone Loss: {train_zone_loss/len(train_loader):.4f}, Acc: {train_zone_acc:.4f} '
                       f'Val Leak Loss: {val_leak_loss/len(val_loader):.4f}, Acc: {val_leak_acc:.4f}, F1: {val_leak_f1:.4f} '
                       f'Val Zone Loss: {val_zone_loss/len(val_loader):.4f}, Acc: {val_zone_acc:.4f}, F1: {val_zone_f1:.4f}')

        # Early stopping based on leak detection F1
        if val_leak_f1 > best_val_f1:
            best_val_f1 = val_leak_f1
            patience_counter = 0
            # Save best models
            torch.save(leak_net.state_dict(), 'best_leak_net.pth')
            torch.save(loc_net.state_dict(), 'best_loc_net.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f'Early stopping triggered after {epoch+1} epochs')
            break

    # Load best models if they exist
    best_leak_path = 'best_leak_net.pth'
    best_loc_path = 'best_loc_net.pth'
    if os.path.exists(best_leak_path) and os.path.exists(best_loc_path):
        leak_net.load_state_dict(torch.load(best_leak_path))
        loc_net.load_state_dict(torch.load(best_loc_path))
        # Clean up
        os.remove(best_leak_path)
        os.remove(best_loc_path)
    else:
        logger.info("Best model files not found, using the model from the last epoch")

    # Assign trained networks to ensemble
    ensemble.leak_net = leak_net
    ensemble.loc_net = loc_net

    logger.info(f"Training completed. Best validation leak F1: {best_val_f1:.4f}")

    return ensemble

def detect_leaks(model_dir: str, test_data_path: str = None, n_samples: int = 1000):
    """
    Main function to demonstrate the complete leak detection system
    """
    logger.info("Starting LeakSense Twin Leak Detection System Demo")

    # Create directories
    os.makedirs(model_dir, exist_ok=True)
    data_dir = os.path.join(os.path.dirname(model_dir), 'data')
    os.makedirs(data_dir, exist_ok=True)

    # Step 1: Generate or load data
    if test_data_path and os.path.exists(test_data_path):
        logger.info(f"Loading test data from {test_data_path}")
        df = pd.read_csv(test_data_path)
    else:
        logger.info("Generating synthetic data for demonstration")
        generator = SyntheticDataGenerator()
        df = generator.generate_dataset(n_samples=n_samples, healthy_ratio=0.8, random_seed=42)

        # Save generated data
        data_path = os.path.join(data_dir, 'synthetic_engine_data.csv')
        df.to_csv(data_path, index=False)
        logger.info(f"Generated data saved to {data_path}")

    # Step 2: Filter steady-state data for training digital twins
    logger.info("Filtering steady-state data...")
    # We'll use a simple approach: assume all data is steady-state for demo
    # In practice, we would use the is_steady_state method from the generator
    df_ss = df.copy()  # Using all data for simplicity in demo

    # Step 3: Train Digital Twin Models
    logger.info("Training Digital Twin Models...")

    # Use only healthy data for training digital twins
    healthy_data = df_ss[df_ss['leak_zone'] == 0]
    logger.info(f"Training digital twins on {len(healthy_data)} healthy samples")

    # Intake Twin
    intake_twin = IntakeTwinModel()
    intake_features = healthy_data[['RPM', 'MAP_intake', 'T_intake', 'fuel_qty']]
    intake_twin.fit(intake_features, healthy_data['MAF'].values)
    intake_twin.save(os.path.join(model_dir, 'intake_twin.joblib'))

    # Charge Air Twin
    charge_air_twin = ChargeAirTwinModel()
    charge_air_features = healthy_data[['MAF', 'RPM', 'T_intake', 'MAP_intake']]
    charge_air_targets = healthy_data[['MAP_boost', 'T_boost', 'T_cac_out', 'MAP_intake']]
    charge_air_twin.fit(charge_air_features, charge_air_targets)
    charge_air_twin.save(os.path.join(model_dir, 'charge_air_twin.joblib'))

    # Exhaust Twin
    exhaust_twin = ExhaustTwinModel()
    exhaust_features = healthy_data[['MAF', 'fuel_qty', 'RPM', 'T_cac_out']]
    exhaust_targets = healthy_data[['T_exh_manifold', 'T_post_turbine', 'dP_dpf']]
    exhaust_twin.fit(exhaust_features, exhaust_targets)
    exhaust_twin.save(os.path.join(model_dir, 'exhaust_twin.joblib'))

    # Step 4: Train Energy Field Detector
    logger.info("Training Energy Field Detector...")
    # Extract sensor data for energy field (using steady-state healthy data)
    healthy_data = df_ss[df_ss['leak_zone'] == 0]
    sensor_cols_for_ef = ['MAF', 'MAP_boost', 'MAP_cac_out', 'T_cac_out',
                         'T_exh_manifold', 'dP_dpf', 'RPM']

    # Check if we have all required columns
    available_ef_cols = [col for col in sensor_cols_for_ef if col in healthy_data.columns]
    if len(available_ef_cols) == len(sensor_cols_for_ef):
        healthy_sensor_data = healthy_data[available_ef_cols].values

        # Create windows for energy field fitting (overlapping windows)
        window_size = 30
        healthy_windows = []
        for i in range(len(healthy_sensor_data) - window_size + 1):
            window = healthy_sensor_data[i:i+window_size]
            if len(window) == window_size:
                healthy_windows.append(window)

        if len(healthy_windows) > 0:
            energy_field_detector = EnergyFieldDetector()
            energy_field_detector.fit(healthy_windows)
            energy_field_detector.save(os.path.join(model_dir, 'energy_field_detector.joblib'))
        else:
            logger.warning("Not enough healthy data windows for energy field detector")
            energy_field_detector = None
    else:
        logger.warning("Missing columns for energy field detector training")
        energy_field_detector = None

    # Step 5: Prepare Features for ML Models
    logger.info("Preparing features for ML models...")
    X, y_leak, y_zone = prepare_features_for_ml(
        df_ss,
        intake_twin=intake_twin,
        charge_air_twin=charge_air_twin,
        exhaust_twin=exhaust_twin,
        energy_field_detector=energy_field_detector
    )
    logger.info(f"Prepared features shape: {X.shape}")
    logger.info(f"Leak labels shape: {y_leak.shape}, sum: {y_leak.sum()} (leaky samples)")
    logger.info(f"Zone labels shape: {y_zone.shape}, unique values: {np.unique(y_zone)}")

    # Step 6: Split data for ML training
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_leak_train, y_leak_temp, y_zone_train, y_zone_temp = train_test_split(
        X, y_leak, y_zone, test_size=0.3, random_state=42, stratify=y_leak
    )
    X_val, X_test, y_leak_val, y_leak_test, y_zone_val, y_zone_test = train_test_split(
        X_temp, y_leak_temp, y_zone_temp, test_size=0.5, random_state=42, stratify=y_leak_temp
    )
    logger.info(f"Training set leak rate: {y_leak_train.mean():.3f}")
    logger.info(f"Validation set leak rate: {y_leak_val.mean():.3f}")
    logger.info(f"Test set leak rate: {y_leak_test.mean():.3f}")
    logger.info(f"Training zone distribution: {np.bincount(y_zone_train.astype(int))}")
    logger.info(f"Validation zone distribution: {np.bincount(y_zone_val.astype(int))}")
    logger.info(f"Test zone distribution: {np.bincount(y_zone_test.astype(int))}")

    logger.info(f"Data splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Step 7: Train ML Models
    input_dim = X_train.shape[1]
    ensemble = train_ml_models(X_train, y_leak_train, y_zone_train,
                              X_val, y_leak_val, y_zone_val,
                              input_dim)

    # Step 8: Evaluate on Test Set
    logger.info("Evaluating on test set...")
    # Use the trained networks from the ensemble
    leak_net = ensemble.leak_net
    loc_net = ensemble.loc_net

    # Convert to tensors for evaluation
    X_test_tensor = torch.FloatTensor(X_test)
    y_leak_test_tensor = torch.FloatTensor(y_leak_test)
    y_zone_test_tensor = torch.LongTensor(y_zone_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    leak_net.to(device)
    loc_net.to(device)

    leak_net.eval()
    loc_net.eval()

    with torch.no_grad():
        leak_outputs = leak_net(X_test_tensor.to(device))
        zone_outputs = loc_net(X_test_tensor.to(device))

        # Debug: print statistics of model outputs
        logger.info(f"Leak output stats - min: {leak_outputs.min():.4f}, max: {leak_outputs.max():.4f}, mean: {leak_outputs.mean():.4f}")
        logger.info(f"Leak output percentiles - 10th: {torch.quantile(leak_outputs, 0.1):.4f}, 50th: {torch.quantile(leak_outputs, 0.5):.4f}, 90th: {torch.quantile(leak_outputs, 0.9):.4f}")

        leak_preds = (leak_outputs > 0.5).float().cpu().numpy()
        _, zone_preds = torch.max(zone_outputs.data, 1)
        zone_preds = zone_preds.cpu().numpy()

        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

        leak_accuracy = accuracy_score(y_leak_test, leak_preds)
        leak_precision = precision_score(y_leak_test, leak_preds, zero_division=0)
        leak_recall = recall_score(y_leak_test, leak_preds, zero_division=0)
        leak_f1 = f1_score(y_leak_test, leak_preds, zero_division=0)

        zone_accuracy = accuracy_score(y_zone_test, zone_preds)
        zone_precision = precision_score(y_zone_test, zone_preds, average='weighted', zero_division=0)
        zone_recall = recall_score(y_zone_test, zone_preds, average='weighted', zero_division=0)
        zone_f1 = f1_score(y_zone_test, zone_preds, average='weighted', zero_division=0)

        logger.info("=== TEST SET RESULTS ===")
        logger.info(f"Leak Detection:")
        logger.info(f"  Accuracy: {leak_accuracy:.4f}")
        logger.info(f"  Precision: {leak_precision:.4f}")
        logger.info(f"  Recall: {leak_recall:.4f}")
        logger.info(f"  F1-Score: {leak_f1:.4f}")
        logger.info(f"Zone Localization:")
        logger.info(f"  Accuracy: {zone_accuracy:.4f}")
        logger.info(f"  Precision: {zone_precision:.4f}")
        logger.info(f"  Recall: {zone_recall:.4f}")
        logger.info(f"  F1-Score: {zone_f1:.4f}")

        # Detailed classification report for zones
        print("\nZone Classification Report:")
        print(classification_report(y_zone_test, zone_preds,
                                  target_names=['Healthy', 'Zone A', 'Zone B', 'Zone C', 'Zone D', 'Zone E'],
                                  zero_division=0))

    # Step 9: Save ML Models
    logger.info("Saving ML models...")
    torch.save(leak_net.state_dict(), os.path.join(model_dir, 'leak_sense_net.pth'))
    torch.save(loc_net.state_dict(), os.path.join(model_dir, 'leak_localization_net.pth'))

    # Step 10: Demonstrate Real-time Detection (simplified)
    logger.info("Demonstrating real-time leak detection...")

    # Take a few samples and show predictions
    demo_samples = min(10, len(X_test))
    X_demo = X_test[:demo_samples]
    y_leak_demo = y_leak_test[:demo_samples]
    y_zone_demo = y_zone_test[:demo_samples]

    X_demo_tensor = torch.FloatTensor(X_demo).to(device)

    with torch.no_grad():
        leak_probs = leak_net(X_demo_tensor)
        zone_probs = loc_net(X_demo_tensor)

        leak_preds_demo = (leak_probs > 0.5).float().cpu().numpy()
        _, zone_preds_demo = torch.max(zone_probs.data, 1)
        zone_preds_demo = zone_preds_demo.cpu().numpy()

        print("\n=== REAL-TIME DEMONSTRATION ===")
        print("Sample\tActual Leak\tPred Leak\tActual Zone\tPred Zone\tLeak Prob")
        print("-" * 70)
        zone_names = ['Healthy', 'Zone A', 'Zone B', 'Zone C', 'Zone D', 'Zone E']
        for i in range(demo_samples):
            actual_leak = "YES" if y_leak_demo[i] == 1 else "NO"
            pred_leak = "YES" if leak_preds_demo[i] == 1 else "NO"
            actual_zone = zone_names[y_zone_demo[i]]
            pred_zone = zone_names[zone_preds_demo[i]]
            leak_prob = leak_probs[i].cpu().item()
            print(f"{i+1}\t{actual_leak}\t\t{pred_leak}\t\t{actual_zone}\t\t{pred_zone}\t\t{leak_prob:.3f}")

    logger.info("LeakSense Twin Leak Detection System Demo Completed Successfully!")
    logger.info(f"All models saved to: {model_dir}")

    return {
        'intake_twin': intake_twin,
        'charge_air_twin': charge_air_twin,
        'exhaust_twin': exhaust_twin,
        'energy_field_detector': energy_field_detector,
        'leak_net': leak_net,
        'loc_net': loc_net,
        'test_metrics': {
            'leak_accuracy': leak_accuracy,
            'leak_precision': leak_precision,
            'leak_recall': leak_recall,
            'leak_f1': leak_f1,
            'zone_accuracy': zone_accuracy,
            'zone_precision': zone_precision,
            'zone_recall': zone_recall,
            'zone_f1': zone_f1
        }
    }

if __name__ == "__main__":
    # Configuration
    MODEL_DIR = "saved_models"

    # Run the complete leak detection system with fewer samples and epochs for quick testing
    results = detect_leaks(model_dir=MODEL_DIR, n_samples=5000)

    print("\n" + "="*60)
    print("LEAKSENSE TWIN LEAK DETECTION SYSTEM - DEMO COMPLETE")
    print("="*60)
    print(f"Leak Detection F1-Score: {results['test_metrics']['leak_f1']:.4f}")
    print(f"Zone Localization F1-Score: {results['test_metrics']['zone_f1']:.4f}")
    print(f"Models saved in: {MODEL_DIR}/")
    print("="*60)