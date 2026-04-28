import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
import sys

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'Development', 'Testing', 'Anti', 'backend'))

from twins.intake_twin import IntakeTwinModel
from twins.charge_twin import ChargeAirTwinModel
from twins.exhaust_twin import ExhaustTwinModel
from ml.energy_field import EnergyFieldDetector
from ml.models import LeakSenseNet, LeakLocalizationNet, SensorGNN
from ml.ensemble import LeakSenseEnsemble

def train_pipeline():
    data_path = "Development/Testing/Anti/data/c18_sensor_data.csv"
    if not os.path.exists(data_path):
        print("Data not found. Run data_generator.py first.")
        return

    df = pd.read_csv(data_path)
    healthy_df = df[df['leak_zone'] == 0].copy()
    
    # 1. Train Digital Twins
    print("Training Digital Twins...")
    sys.stdout.flush()
    intake_twin = IntakeTwinModel()
    intake_twin.fit(healthy_df)
    
    charge_twin = ChargeAirTwinModel()
    charge_twin.fit(healthy_df)
    
    exhaust_twin = ExhaustTwinModel()
    exhaust_twin.fit(healthy_df)
    
    # 2. Compute Residuals for all data
    print("Computing residuals...")
    sys.stdout.flush()
    def get_residuals(row):
        res_intake = row['MAF'] - intake_twin.predict(row['RPM'], row['MAP_intake'], row['T_intake'])
        
        c_preds = charge_twin.predict(row['MAF'], row['RPM'], row['T_intake'])
        res_boost = row['MAP_boost'] - c_preds['MAP_boost_pred']
        res_t_boost = row['T_boost'] - c_preds['T_boost_pred']
        res_t_cac = row['T_cac_out'] - c_preds['T_cac_out_pred']
        res_map_cac = row['MAP_cac_out'] - c_preds['MAP_cac_out_pred']
        
        e_preds = exhaust_twin.predict(row['MAF'], row['fuel_qty'], row['RPM'], row['T_cac_out'])
        res_t_exh = row['T_exh_manifold'] - e_preds['T_exh_manifold_pred']
        
        return pd.Series([res_intake, res_boost, res_t_boost, res_t_cac, res_map_cac, res_t_exh])

    res_cols = ['res_MAF', 'res_MAP_boost', 'res_T_boost', 'res_T_cac_out', 'res_MAP_cac_out', 'res_T_exh_manifold']
    df[res_cols] = df.apply(get_residuals, axis=1)
    
    # 3. Energy Field Fitting
    print("Fitting Energy Field Detector...")
    sys.stdout.flush()
    ef_detector = EnergyFieldDetector()
    # Create windows for fitting
    windows = []
    for i in range(0, 1000, 30):
        window = healthy_df.iloc[i:i+30][['MAF', 'MAP_boost', 'MAP_cac_out', 'T_cac_out', 'T_exh_manifold', 'T_dpf_out', 'RPM']].values
        if len(window) == 30:
            windows.append(window)
    ef_detector.fit(windows)
    
    # 4. Prepare ML Features
    print("Preparing ML features...")
    sys.stdout.flush()
    # Base features (13 sensors + 6 residuals) = 19
    # Plus Energy Field (36) = 55 (simplified from 67)
    
    # We'll use a simplified version for this demo
    base_cols = ['RPM', 'MAF', 'MAP_intake', 'MAP_boost', 'T_intake', 'T_boost', 'T_cac_out', 'T_exh_manifold', 'fuel_qty']
    ml_features = df[base_cols + res_cols].values
    labels_binary = (df['leak_zone'] > 0).astype(int).values
    labels_multi = df['leak_zone'].values
    
    X_train, X_test, y_train_bin, y_test_bin, y_train_multi, y_test_multi = train_test_split(
        ml_features, labels_binary, labels_multi, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Train PyTorch Models
    print("Training LeakSenseNet...")
    input_dim = X_train_scaled.shape[1]
    leak_net = LeakSenseNet(input_dim=input_dim)
    criterion_bin = nn.BCELoss()
    optimizer_bin = optim.Adam(leak_net.parameters(), lr=0.001)
    
    X_train_t = torch.FloatTensor(X_train_scaled)
    y_train_t = torch.FloatTensor(y_train_bin)
    
    for epoch in range(20):
        optimizer_bin.zero_grad()
        outputs = leak_net(X_train_t)
        loss = criterion_bin(outputs, y_train_t)
        loss.backward()
        optimizer_bin.step()
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            
    print("Training LeakLocalizationNet...")
    loc_net = LeakLocalizationNet(input_dim=input_dim, n_classes=6)
    criterion_multi = nn.CrossEntropyLoss()
    optimizer_multi = optim.Adam(loc_net.parameters(), lr=0.001)
    y_train_multi_t = torch.LongTensor(y_train_multi)
    
    for epoch in range(20):
        optimizer_multi.zero_grad()
        outputs = loc_net(X_train_t)
        loss = criterion_multi(outputs, y_train_multi_t)
        loss.backward()
        optimizer_multi.step()
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            
    # 6. Train Scikit-Learn Models
    print("Training RF and GB models...")
    rf = RandomForestClassifier(n_estimators=50)
    rf.fit(X_train_scaled, y_train_bin)
    
    gb = GradientBoostingClassifier(n_estimators=50)
    gb.fit(X_train_scaled, y_train_bin)
    
    # 7. Save Models
    print("Saving all models...")
    os.makedirs("Development/Testing/Anti/models", exist_ok=True)
    intake_twin.save("Development/Testing/Anti/models/intake_twin.joblib")
    charge_twin.save("Development/Testing/Anti/models/charge_twin.joblib")
    exhaust_twin.save("Development/Testing/Anti/models/exhaust_twin.joblib")
    ef_detector.save("Development/Testing/Anti/models/energy_field.joblib")
    joblib.dump(scaler, "Development/Testing/Anti/models/scaler.joblib")
    
    torch.save(leak_net.state_dict(), "Development/Testing/Anti/models/leak_net.pth")
    torch.save(loc_net.state_dict(), "Development/Testing/Anti/models/loc_net.pth")
    # GNN skipped for now or simplified
    
    ensemble = LeakSenseEnsemble(leak_net, loc_net, loc_net, rf, gb) # proxying GNN with loc_net for now
    ensemble.save("Development/Testing/Anti/models/ensemble.joblib")
    
    print("Pipeline training complete!")

if __name__ == "__main__":
    train_pipeline()
