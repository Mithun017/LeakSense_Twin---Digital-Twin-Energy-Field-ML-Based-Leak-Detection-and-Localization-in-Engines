"""
LeakSense Twin — FastAPI Backend Server
Unified API serving digital twin predictions, ML anomaly detection,
energy field analysis, and AI diagnostic advisor.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, Optional, List
import uvicorn
import joblib
import torch
import numpy as np
import os
import sys
import json
import asyncio
from collections import deque
from datetime import datetime
from pathlib import Path

# Path setup
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BACKEND_DIR)

from config import ZONE_NAMES, MAP_AMBIENT, EF_CHANNELS, ML_CONFIG
from twins.intake_twin import IntakeTwinModel
from twins.charge_air_twin import ChargeAirTwinModel
from twins.exhaust_twin import ExhaustTwinModel
from ml.models import LeakSenseNet, LeakLocalizationNet
from ml.ensemble import LeakSenseEnsemble
from ml.energy_field import EnergyFieldDetector

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="LeakSense Twin API",
    version="1.0.0",
    description="AI-powered leak detection & localization for Cat C18 diesel engines"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Model Paths ──────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(BACKEND_DIR, "models")

# ─── History Storage ──────────────────────────────────────────────────────────
prediction_history: List[Dict] = []
MAX_HISTORY = 500


# ─── Unified Predictor ────────────────────────────────────────────────────────
class UnifiedPredictor:
    """Main prediction engine combining digital twins, ML, and energy field."""

    def __init__(self):
        self.intake_twin = None
        self.charge_twin = None
        self.exhaust_twin = None
        self.ef_detector = None
        self.leak_net = None
        self.loc_net = None
        self.ensemble = None
        self.scaler = None
        self.config = None
        self.window = deque(maxlen=30)
        self.alert_buffer = deque(maxlen=3)  # Anti-flicker: 3 consecutive alerts
        self._load_models()

    def _load_models(self):
        """Load all trained models from disk."""
        try:
            # Digital Twins
            self.intake_twin = IntakeTwinModel.load(
                os.path.join(MODELS_DIR, "intake_twin.joblib"))
            self.charge_twin = ChargeAirTwinModel.load(
                os.path.join(MODELS_DIR, "charge_air_twin.joblib"))
            self.exhaust_twin = ExhaustTwinModel.load(
                os.path.join(MODELS_DIR, "exhaust_twin.joblib"))
            print("  [OK] Digital twins loaded")

            # Energy Field
            self.ef_detector = EnergyFieldDetector.load(
                os.path.join(MODELS_DIR, "energy_field_detector.joblib"))
            print("  [OK] Energy field detector loaded")

            # Scaler
            self.scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))

            # Config
            self.config = joblib.load(os.path.join(MODELS_DIR, "config.joblib"))
            input_dim = self.config.get('input_dim', 31)

            # Neural nets
            self.leak_net = LeakSenseNet(input_dim=input_dim)
            self.leak_net.load_state_dict(
                torch.load(os.path.join(MODELS_DIR, "leak_sense_net.pth"),
                           map_location='cpu', weights_only=True))
            self.leak_net.eval()

            self.loc_net = LeakLocalizationNet(input_dim=input_dim, n_classes=6)
            self.loc_net.load_state_dict(
                torch.load(os.path.join(MODELS_DIR, "leak_localization_net.pth"),
                           map_location='cpu', weights_only=True))
            self.loc_net.eval()
            print("  [OK] Neural networks loaded")

            # Ensemble
            self.ensemble = LeakSenseEnsemble.load(
                os.path.join(MODELS_DIR, "ensemble.joblib"))
            self.ensemble.leak_net = self.leak_net
            self.ensemble.loc_net = self.loc_net
            print("  [OK] Ensemble loaded")

            print("  [OK] All models loaded successfully!")

        except Exception as e:
            print(f"  [WARN] Model loading error: {e}")
            print("  [INFO] Run 'python ml/train.py' to train models first.")
            # Initialize default twins for demo mode
            self.intake_twin = IntakeTwinModel()
            self.charge_twin = ChargeAirTwinModel()
            self.exhaust_twin = ExhaustTwinModel()

    def predict(self, sensor_data: Dict) -> Dict:
        """
        Run full prediction pipeline:
        1. Extract sensor values
        2. Compute digital twin residuals
        3. Build feature vector
        4. Run ML ensemble
        5. Compute energy field
        6. Generate final result
        """
        # Extract sensor values with defaults
        rpm = float(sensor_data.get('RPM', 1800))
        maf = float(sensor_data.get('MAF', 850))
        map_intake = float(sensor_data.get('MAP_intake', 210))
        map_boost = float(sensor_data.get('MAP_boost', 215))
        map_cac_in = float(sensor_data.get('MAP_cac_in', 212))
        map_cac_out = float(sensor_data.get('MAP_cac_out', 205))
        t_intake = float(sensor_data.get('T_intake', 25))
        t_boost = float(sensor_data.get('T_boost', 120))
        t_cac_out = float(sensor_data.get('T_cac_out', 45))
        t_exh = float(sensor_data.get('T_exh_manifold', 550))
        t_dpf_in = float(sensor_data.get('T_dpf_in', 250))
        t_dpf_out = float(sensor_data.get('T_dpf_out', 200))
        fuel_qty = float(sensor_data.get('fuel_qty', 120))
        t_post_turbine = float(sensor_data.get('T_post_turbine', 400))
        dP_dpf = float(sensor_data.get('dP_dpf', 0.5))

        t_intake_k = t_intake + 273.15
        t_cac_k = t_cac_out + 273.15

        # ── Digital Twin Residuals ────────────────────────────────────────
        residuals = {}
        try:
            maf_pred = self.intake_twin.predict(rpm, MAP_AMBIENT, t_intake_k, fuel_qty)
            residuals['res_MAF'] = maf - maf_pred

            ca = self.charge_twin.predict(maf, rpm, t_intake_k, MAP_AMBIENT)
            residuals['res_MAP_boost'] = map_boost - ca['MAP_boost_pred']
            residuals['res_T_boost'] = t_boost - (ca['T_boost_pred'] - 273.15)
            residuals['res_T_cac_out'] = t_cac_out - (ca['T_cac_out_pred'] - 273.15)
            residuals['res_MAP_cac_out'] = map_cac_out - ca['MAP_cac_out_pred']
            residuals['res_MAP_intake'] = map_intake - ca['MAP_intake_pred']

            ex = self.exhaust_twin.predict(maf, fuel_qty, rpm, t_cac_k)
            residuals['res_T_exh_manifold'] = t_exh - (ex['T_exh_manifold_pred'] - 273.15)
            residuals['res_T_post_turbine'] = t_post_turbine - (ex['T_post_turbine_pred'] - 273.15)
            residuals['res_dP_dpf'] = dP_dpf - ex['dP_dpf_pred']
        except Exception as e:
            print(f"  Twin residual error: {e}")
            for key in ['res_MAF', 'res_MAP_boost', 'res_T_boost', 'res_T_cac_out',
                        'res_MAP_cac_out', 'res_T_exh_manifold', 'res_T_post_turbine',
                        'res_dP_dpf', 'res_MAP_intake']:
                residuals.setdefault(key, 0.0)

        # ── Build 31-feature vector ───────────────────────────────────────
        raw = [rpm, maf, map_intake, map_boost, map_cac_in, map_cac_out,
               t_intake, t_boost, t_cac_out, t_exh, t_dpf_in, t_dpf_out, fuel_qty]

        res = [residuals.get('res_MAF', 0), residuals.get('res_MAP_boost', 0),
               residuals.get('res_T_boost', 0), residuals.get('res_T_cac_out', 0),
               residuals.get('res_MAP_cac_out', 0), residuals.get('res_T_exh_manifold', 0),
               residuals.get('res_T_post_turbine', 0), residuals.get('res_dP_dpf', 0),
               residuals.get('res_MAP_intake', 0)]

        # Derived ratios
        pr_comp = map_boost / (MAP_AMBIENT + 1e-8)
        t_diff = abs(t_boost - t_intake) + 1e-8
        cac_eff = (t_boost - t_cac_out) / t_diff if t_diff > 1 else 0.88
        t_exh_ratio = t_exh / (t_cac_out + 273.15 + 1e-8)
        boost_maf = map_boost / (maf + 1e-8)
        dpf_ratio = dP_dpf / (map_boost + 1e-8)
        fuel_flow = fuel_qty * 1e-6 * (rpm / 120.0) * 6
        afr = (maf / 3600.0) / (fuel_flow + 1e-8)
        ratios = [pr_comp, cac_eff, t_exh_ratio, boost_maf, dpf_ratio, afr]

        # Stats (simplified)
        stats = [abs(res[0]) * 0.1, abs(res[1]) * 0.1, abs(res[5]) * 0.1]

        feature_vec = np.array([raw + res + ratios + stats], dtype=np.float32)
        feature_vec = np.nan_to_num(feature_vec, nan=0.0, posinf=0.0, neginf=0.0)

        # ── ML Prediction ─────────────────────────────────────────────────
        confidence = 0.5
        zone_idx = 0
        zone_probs = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        try:
            if self.scaler is not None:
                scaled = self.scaler.transform(feature_vec)
            else:
                scaled = feature_vec

            feat_tensor = torch.FloatTensor(scaled)

            if self.ensemble is not None:
                result = self.ensemble.predict(feat_tensor, scaled)
                confidence = result['confidence']
                zone_idx = result['suspected_zone_idx']
                zone_probs = result.get('zone_probabilities', zone_probs)
            elif self.leak_net is not None:
                with torch.no_grad():
                    confidence = float(self.leak_net(feat_tensor).item())
                    loc_probs = self.loc_net(feat_tensor).numpy()[0]
                    zone_idx = int(np.argmax(loc_probs))
                    zone_probs = loc_probs.tolist()
        except Exception as e:
            print(f"  ML prediction error: {e}")

        # ── Energy Field ──────────────────────────────────────────────────
        ef_data = {}
        self.window.append([maf, map_boost, map_cac_out, t_cac_out, t_exh, dP_dpf, rpm])
        if len(self.window) == 30 and self.ef_detector is not None:
            try:
                ef_data = self.ef_detector.compute_deviation(np.array(self.window))
            except Exception as e:
                print(f"  EF error: {e}")

        # ── Anti-flicker logic ────────────────────────────────────────────
        leak_detected = confidence > ML_CONFIG['leak_threshold']
        self.alert_buffer.append(leak_detected)
        confirmed_leak = all(self.alert_buffer) and len(self.alert_buffer) >= ML_CONFIG['consecutive_alerts']

        # ── Severity estimation ───────────────────────────────────────────
        if confirmed_leak:
            if confidence > 0.85:
                severity = "CRITICAL"
                flow_loss_pct = 12.0 + (confidence - 0.85) * 20
            elif confidence > 0.7:
                severity = "MEDIUM"
                flow_loss_pct = 5.0 + (confidence - 0.7) * 47
            else:
                severity = "SMALL"
                flow_loss_pct = 1.0 + (confidence - 0.5) * 20
        else:
            severity = "NONE"
            flow_loss_pct = 0.0

        # ── Recommended action ────────────────────────────────────────────
        action_map = {
            0: "System healthy - continue operation",
            1: "Inspect intake ducting between airflow meter and compressor inlet. Check hose clamps and silicone couplers.",
            2: "Check boost pipe connections between compressor outlet and CAC inlet. Inspect for cracked charge pipes.",
            3: "Inspect CAC-to-intake manifold connections. Check CAC end tanks and pipe clamps.",
            4: "Inspect exhaust manifold gaskets and V-band clamps. Check turbocharger turbine housing bolts.",
            5: "Inspect DPF/SCR connections. Check DOC mounting bolts and exhaust pipe V-bands.",
        }

        # ── Build response ────────────────────────────────────────────────
        ef_matrix = ef_data.get('energy_field', [[0]*6 for _ in range(6)])
        if isinstance(ef_matrix, np.ndarray):
            ef_matrix = ef_matrix.tolist()

        result = {
            "timestamp": datetime.now().isoformat(),
            "leak_detected": bool(confirmed_leak),
            "confidence": round(float(confidence), 4),
            "suspected_zone": ZONE_NAMES[zone_idx] if confirmed_leak else ZONE_NAMES[0],
            "suspected_zone_idx": int(zone_idx),
            "zone_probabilities": [round(float(p), 4) for p in zone_probs],
            "severity": severity,
            "flow_loss_pct": round(float(flow_loss_pct), 1),
            "go_no_go": "NO-GO" if confirmed_leak else "GO",
            "recommended_action": action_map.get(zone_idx if confirmed_leak else 0, ""),
            "residuals": {k: round(float(v), 3) for k, v in residuals.items()},
            "energy_field": {
                "matrix": ef_matrix,
                "global_deviation": round(float(ef_data.get('global_deviation_score', 0.0)), 4),
                "cosine_similarity": round(float(ef_data.get('cosine_similarity', 1.0)), 4),
                "most_disrupted_sensor": ef_data.get('most_disrupted_sensor', 'N/A'),
                "ef_suspected_zone": ef_data.get('suspected_zone', 'N/A'),
            },
            "sensors": {
                "RPM": round(rpm, 1),
                "MAF": round(maf, 2),
                "MAP_intake": round(map_intake, 2),
                "MAP_boost": round(map_boost, 2),
                "MAP_cac_in": round(map_cac_in, 2),
                "MAP_cac_out": round(map_cac_out, 2),
                "T_intake": round(t_intake, 2),
                "T_boost": round(t_boost, 2),
                "T_cac_out": round(t_cac_out, 2),
                "T_exh_manifold": round(t_exh, 2),
                "T_dpf_in": round(t_dpf_in, 2),
                "T_dpf_out": round(t_dpf_out, 2),
                "fuel_qty": round(fuel_qty, 2),
                "dP_dpf": round(dP_dpf, 3),
            },
        }

        # Store in history
        prediction_history.append(result)
        if len(prediction_history) > MAX_HISTORY:
            prediction_history.pop(0)

        return result


# ─── Global predictor ─────────────────────────────────────────────────────────
predictor: UnifiedPredictor = None


@app.on_event("startup")
async def startup():
    global predictor
    print("\n[INIT] Starting LeakSense Twin API...")
    predictor = UnifiedPredictor()
    print("[READY] LeakSense Twin API ready!\n")


# ─── REST Endpoints ───────────────────────────────────────────────────────────
@app.get("/api/health")
async def health_check():
    """System health check."""
    models_loaded = predictor is not None and predictor.intake_twin is not None
    return {
        "status": "healthy" if models_loaded else "degraded",
        "models_loaded": models_loaded,
        "version": "1.0.0",
        "engine": "Cat C18",
    }


@app.post("/api/predict")
async def predict_endpoint(data: Dict):
    """Run leak detection on sensor data."""
    if not predictor:
        raise HTTPException(503, "Models not loaded — run training first")
    try:
        return predictor.predict(data)
    except Exception as e:
        print(f"Prediction error: {e}")
        raise HTTPException(500, str(e))


@app.get("/api/health-field")
async def get_health_field():
    """Get current energy field matrix."""
    if predictor and predictor.ef_detector and predictor.ef_detector.is_fitted:
        return {
            "healthy_baseline": predictor.ef_detector.healthy_field_mean.tolist(),
            "channels": EF_CHANNELS[:6],
        }
    return {"healthy_baseline": [[0]*6]*6, "channels": EF_CHANNELS[:6]}


@app.get("/api/history")
async def get_history(limit: int = 50):
    """Get prediction history."""
    return prediction_history[-limit:]


@app.get("/api/zones")
async def get_zones():
    """Get zone definitions."""
    return {"zones": ZONE_NAMES}


@app.post("/api/simulate")
async def simulate_scenario(data: Dict):
    """
    Simulate a sensor scenario. Useful for testing.
    Accepts: { "rpm": 1800, "leak_zone": 0, "leak_severity": 0 }
    """
    from data_generator import generate_healthy_sample, inject_leak, add_noise

    rpm = float(data.get('rpm', 1800))
    zone = int(data.get('leak_zone', 0))
    severity = int(data.get('leak_severity', 0))

    sample = generate_healthy_sample(rpm)
    if zone > 0 and severity > 0:
        sample = inject_leak(sample, zone, severity)
    sample = add_noise(sample)

    # Run prediction on the simulated sample
    result = predictor.predict(sample)
    result['simulation'] = {
        'injected_zone': zone,
        'injected_severity': severity,
        'rpm': rpm,
    }
    return result


# ─── WebSocket Live Stream ────────────────────────────────────────────────────
@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    """Real-time prediction streaming at 1 Hz."""
    await ws.accept()
    print("  WebSocket client connected")
    try:
        while True:
            data = await ws.receive_json()
            result = predictor.predict(data)
            await ws.send_json(result)
    except WebSocketDisconnect:
        print("  WebSocket client disconnected")
    except Exception as e:
        print(f"  WebSocket error: {e}")


@app.websocket("/ws/demo")
async def websocket_demo(ws: WebSocket):
    """Demo mode: auto-generates and streams sensor data."""
    await ws.accept()
    print("  Demo WebSocket connected")

    from data_generator import generate_healthy_sample, inject_leak, add_noise

    rpm_range = [1200, 1400, 1600, 1800, 2000]
    cycle = 0

    try:
        while True:
            cycle += 1
            rpm = rpm_range[cycle % len(rpm_range)]

            # Inject leak every 30 seconds for 10 seconds
            zone = 0
            severity = 0
            if 30 <= (cycle % 50) < 40:
                zone = (cycle // 50 % 5) + 1
                severity = (cycle // 150 % 3) + 1

            sample = generate_healthy_sample(rpm + np.random.uniform(-20, 20))
            if zone > 0:
                sample = inject_leak(sample, zone, severity)
            sample = add_noise(sample)

            result = predictor.predict(sample)
            result['demo_info'] = {
                'cycle': cycle,
                'injected_zone': zone,
                'injected_severity': severity,
            }
            await ws.send_json(result)
            await asyncio.sleep(1.0)

    except WebSocketDisconnect:
        print("  Demo WebSocket disconnected")
    except Exception as e:
        print(f"  Demo error: {e}")


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
