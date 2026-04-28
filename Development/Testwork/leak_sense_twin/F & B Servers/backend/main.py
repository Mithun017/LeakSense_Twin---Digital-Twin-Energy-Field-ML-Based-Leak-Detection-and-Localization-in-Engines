"""
LeakSense Twin — Unified FastAPI Backend
Integrates:
  - Claude 57-feature physics models (intake, charge-air, exhaust digital twins + energy field)
  - Anti 15-feature ensemble (RF + GB + LeakSenseNet + LocalizationNet)
  - WebSocket live streaming endpoint
  - Dual-model blended predictions
"""

from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import joblib
import torch
import numpy as np
import pandas as pd
import os
import sys
from typing import Dict
from collections import deque
import asyncio
from ai_advisor import AIAdvisor

# ── Path Setup ──────────────────────────────────────────────────────────────
BACKEND_DIR   = os.path.dirname(os.path.abspath(__file__))
CLAUDE_ROOT   = os.path.abspath(os.path.join(BACKEND_DIR, "..", "..", ".."))
PACKAGE_ROOT  = os.path.join(CLAUDE_ROOT, "leak_sense_twin")

for p in [CLAUDE_ROOT, PACKAGE_ROOT, BACKEND_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Claude models ─────────────────────────────────────────────────────────
from leak_sense_twin.models.intake_twin    import IntakeTwinModel
from leak_sense_twin.models.charge_air_twin import ChargeAirTwinModel
from leak_sense_twin.models.exhaust_twin   import ExhaustTwinModel
from leak_sense_twin.energy_field.energy_field_detector import EnergyFieldDetector
from leak_sense_twin.ml.leak_sense_net     import LeakSenseNet, LeakLocalizationNet

# ── Anti models ────────────────────────────────────────────────────────────
from anti_twins.intake_twin  import IntakeTwinModel  as AntiIntakeTwin
from anti_twins.charge_twin  import ChargeAirTwinModel as AntiChargeTwin
from anti_twins.exhaust_twin import ExhaustTwinModel  as AntiExhaustTwin
from anti_ml.energy_field    import EnergyFieldDetector as AntiEFDetector
from anti_ml.models          import LeakSenseNet as AntiLeakNet, LeakLocalizationNet as AntiLocNet
from anti_ml.ensemble        import LeakSenseEnsemble

# ── Module Aliases for Joblib ───────────────────────────────────────────────
import anti_twins
import anti_ml
sys.modules['twins'] = anti_twins
sys.modules['ml']    = anti_ml

# ── App ───────────────────────────────────────────────────────────────────
app = FastAPI(title="LeakSense Twin Unified API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model Paths ───────────────────────────────────────────────────────────
CLAUDE_MODEL_DIR = os.path.join(CLAUDE_ROOT, "leak_sense_twin", "saved_models")
ANTI_MODEL_DIR   = os.path.join(BACKEND_DIR, "anti_models")

ZONE_NAMES = [
    "No Leak / Healthy",
    "Zone 1 — Intake",
    "Zone 2 — Charge Air",
    "Zone 3 — CAC / Manifold",
    "Zone 4 — Exhaust Manifold",
    "Zone 5 — DPF / SCR"
]

EF_CHANNELS = ['MAF', 'MAP_boost', 'MAP_cac_out', 'T_cac_out',
               'T_exh_manifold', 'T_dpf_out', 'RPM']


# ── Unified Predictor ─────────────────────────────────────────────────────
class UnifiedPredictor:
    def __init__(self):
        self._load_claude_models()
        self._load_anti_models()
        # Sliding windows for energy field computation
        self.window_claude = deque(maxlen=30)
        self.window_anti   = deque(maxlen=30)
        print("OK: All models loaded — LeakSense Twin Unified.")

    # ── Claude model loading ─────────────────────────────────────────────
    def _load_claude_models(self):
        try:
            self.c_intake  = IntakeTwinModel.load(os.path.join(CLAUDE_MODEL_DIR, "intake_twin.joblib"))
            self.c_charge  = ChargeAirTwinModel.load(os.path.join(CLAUDE_MODEL_DIR, "charge_air_twin.joblib"))
            self.c_exhaust = ExhaustTwinModel.load(os.path.join(CLAUDE_MODEL_DIR, "exhaust_twin.joblib"))
            self.c_ef      = EnergyFieldDetector.load(os.path.join(CLAUDE_MODEL_DIR, "energy_field_detector.joblib"))
            INPUT_DIM = 57
            self.c_leak_net = LeakSenseNet(input_dim=INPUT_DIM)
            self.c_leak_net.load_state_dict(torch.load(os.path.join(CLAUDE_MODEL_DIR, "leak_sense_net.pth")))
            self.c_loc_net  = LeakLocalizationNet(input_dim=INPUT_DIM, n_classes=6)
            self.c_loc_net.load_state_dict(torch.load(os.path.join(CLAUDE_MODEL_DIR, "leak_localization_net.pth")))
            self.c_leak_net.eval()
            self.c_loc_net.eval()
            print("  [OK] Claude physics models loaded (57-feat)")
        except Exception as e:
            print(f"  [ERROR] Claude models: {e}")
            self.c_intake = self.c_charge = self.c_exhaust = None

    # ── Anti model loading ────────────────────────────────────────────────
    def _load_anti_models(self):
        try:
            self.a_intake  = AntiIntakeTwin.load(os.path.join(ANTI_MODEL_DIR, "intake_twin.joblib"))
            self.a_charge  = AntiChargeTwin.load(os.path.join(ANTI_MODEL_DIR, "charge_twin.joblib"))
            self.a_exhaust = AntiExhaustTwin.load(os.path.join(ANTI_MODEL_DIR, "exhaust_twin.joblib"))
            self.a_ef      = AntiEFDetector.load(os.path.join(ANTI_MODEL_DIR, "energy_field.joblib"))
            self.a_scaler  = joblib.load(os.path.join(ANTI_MODEL_DIR, "scaler.joblib"))
            self.a_ensemble = joblib.load(os.path.join(ANTI_MODEL_DIR, "ensemble.joblib"))
            A_INPUT_DIM = 15
            self.a_leak_net = AntiLeakNet(input_dim=A_INPUT_DIM)
            self.a_leak_net.load_state_dict(torch.load(os.path.join(ANTI_MODEL_DIR, "leak_net.pth")))
            self.a_loc_net  = AntiLocNet(input_dim=A_INPUT_DIM, n_classes=6)
            self.a_loc_net.load_state_dict(torch.load(os.path.join(ANTI_MODEL_DIR, "loc_net.pth")))
            self.a_ensemble.leak_net = self.a_leak_net
            self.a_ensemble.loc_net  = self.a_loc_net
            self.a_leak_net.eval()
            self.a_loc_net.eval()
            print("  [OK] Anti ensemble models loaded (15-feat RF+GB+NN)")
        except Exception as e:
            print(f"  [ERROR] Anti models: {e}")
            self.a_ensemble = None

    # ── Core predict ──────────────────────────────────────────────────────
    def predict(self, sensor: Dict) -> Dict:
        # Sensor defaults
        rpm   = float(sensor.get('RPM', 1800))
        maf   = float(sensor.get('MAF', 850))
        map_i = float(sensor.get('MAP_intake', 210))
        t_i   = float(sensor.get('T_intake', 25))
        fuel  = float(sensor.get('fuel_qty', 120))
        t_exh = float(sensor.get('T_exh_manifold', 550))
        t_boost = float(sensor.get('T_boost', 120))
        t_cac   = float(sensor.get('T_cac_out', 45))
        map_boost = float(sensor.get('MAP_boost', 215))
        map_cac   = float(sensor.get('MAP_cac_out', 205))
        t_dpf_out = float(sensor.get('T_dpf_out', 200))
        t_dpf_in  = float(sensor.get('T_dpf_in', 250))
        map_cac_in = float(sensor.get('MAP_cac_in', 212))

        # ── Anti residuals (fast, scalar API) ─────────────────────────────
        anti_result = {}
        res_anti = [0.0]*6
        if self.a_ensemble:
            try:
                a_res_intake = maf - self.a_intake.predict(rpm, map_i, t_i)
                a_c = self.a_charge.predict(maf, rpm, t_i)
                a_res_boost    = map_boost  - a_c['MAP_boost_pred']
                a_res_t_boost  = t_boost    - a_c['T_boost_pred']
                a_res_t_cac    = t_cac      - a_c['T_cac_out_pred']
                a_res_map_cac  = map_cac    - a_c['MAP_cac_out_pred']
                a_e = self.a_exhaust.predict(maf, fuel, rpm, t_cac)
                a_res_t_exh = t_exh - a_e['T_exh_manifold_pred']
                res_anti = [a_res_intake, a_res_boost, a_res_t_boost, a_res_t_cac, a_res_map_cac, a_res_t_exh]

                base = [rpm, maf, map_i, map_boost, t_i, t_boost, t_cac, t_exh, fuel]
                feats_np = np.array([base + res_anti])
                feats_scaled = self.a_scaler.transform(feats_np)
                feats_t = torch.FloatTensor(feats_scaled)
                anti_result = self.a_ensemble.predict(feats_t, features_numpy=feats_scaled)
            except Exception as e:
                print(f"  Anti predict error: {e}")

        # ── Anti Energy Field ─────────────────────────────────────────────
        ef_data = {}
        self.window_anti.append([sensor.get(c, 0) for c in EF_CHANNELS])
        if len(self.window_anti) == 30 and self.a_ef:
            try:
                ef_data = self.a_ef.compute_deviation(np.array(self.window_anti))
            except Exception as e:
                print(f"  Anti EF error: {e}")

        # ── Claude 57-feat prediction ──────────────────────────────────────
        claude_conf = None
        claude_zone_idx = 0
        if self.c_intake:
            try:
                df = pd.DataFrame([{
                    'RPM': rpm, 'MAF': maf, 'MAP_intake': map_i, 'MAP_boost': map_boost,
                    'MAP_cac_in': map_cac_in, 'MAP_cac_out': map_cac, 'T_intake': t_i,
                    'T_boost': t_boost, 'T_cac_out': t_cac, 'T_exh_manifold': t_exh,
                    'T_dpf_in': t_dpf_in, 'T_dpf_out': t_dpf_out, 'fuel_qty': fuel,
                    'T_post_turbine': sensor.get('T_post_turbine', 400),
                    'dP_dpf': sensor.get('dP_dpf', 0.0)
                }])
                raw = df[['RPM','MAF','MAP_intake','MAP_boost','MAP_cac_in','MAP_cac_out',
                          'T_intake','T_boost','T_cac_out','T_exh_manifold','T_dpf_in',
                          'T_dpf_out','fuel_qty']].values[0]
                c_res_MAF   = self.c_intake.predict_residual(df[['RPM','MAP_intake','T_intake','fuel_qty','MAF']])[0]
                c_preds = self.c_charge.predict(df[['MAF','RPM','T_intake','MAP_intake']])
                c_res_boost = map_boost - c_preds['MAP_boost_pred'].values[0]
                c_res_t_b   = t_boost   - c_preds['T_boost_pred'].values[0]
                c_res_t_cac = t_cac     - c_preds['T_cac_out_pred'].values[0]
                c_res_map_c = map_i     - c_preds['MAP_intake_pred'].values[0]
                c_epreds = self.c_exhaust.predict(df[['MAF','fuel_qty','RPM','T_cac_out']])
                c_res_exh   = t_exh     - c_epreds['T_exh_manifold_pred'].values[0]
                c_res_tpost = sensor.get('T_post_turbine', 400) - c_epreds['T_post_turbine_pred'].values[0]
                c_res_dpf   = sensor.get('dP_dpf', 0.0) - c_epreds['dP_dpf_pred'].values[0]
                res_c = [c_res_MAF, c_res_boost, c_res_t_b, c_res_t_cac, c_res_map_c, c_res_exh, c_res_tpost, c_res_dpf]
                ef_c_feats = np.zeros(36)
                all_57 = np.concatenate([raw, res_c, ef_c_feats])
                t57 = torch.FloatTensor([all_57])
                with torch.no_grad():
                    claude_conf = float(self.c_leak_net(t57).item())
                    claude_zone_idx = int(torch.argmax(self.c_loc_net(t57)).item())
            except Exception as e:
                print(f"  Claude predict error: {e}")

        # ── Blend both models ──────────────────────────────────────────────
        anti_conf = float(anti_result.get('confidence', 0.5)) if anti_result else 0.5
        anti_zone = int(anti_result.get('suspected_zone_idx', 0)) if anti_result else 0

        if claude_conf is not None:
            blended_conf = 0.5 * claude_conf + 0.5 * anti_conf
            # Simple zone vote: prefer Anti if confident, else use Claude
            blended_zone = anti_zone if anti_conf > 0.6 else claude_zone_idx
        else:
            blended_conf = anti_conf
            blended_zone = anti_zone

        leak_detected = blended_conf > 0.5

        # Build energy field matrix for frontend heatmap
        ef_matrix = ef_data.get('energy_field', [[0]*6 for _ in range(6)])
        flat_ef = []
        for row in ef_matrix:
            flat_ef.extend(row)

        return {
            "leak_detected":    bool(leak_detected),
            "confidence":       round(blended_conf, 4),
            "anti_confidence":  round(anti_conf, 4),
            "claude_confidence": round(claude_conf, 4) if claude_conf is not None else None,
            "suspected_zone":   ZONE_NAMES[blended_zone] if leak_detected else ZONE_NAMES[0],
            "go_no_go":         "NO-GO" if leak_detected else "GO",
            "residuals": {
                "MAF":          round(float(res_anti[0]), 3),
                "Boost":        round(float(res_anti[1]), 3),
                "Exhaust_Temp": round(float(res_anti[5]), 3),
            },
            "energy_field": {
                "matrix":           flat_ef,
                "global_deviation": round(float(ef_data.get('global_deviation_score', 0.0)), 4),
                "cosine_similarity": round(float(ef_data.get('cosine_similarity', 1.0)), 4),
                "most_disrupted_sensor": ef_data.get('most_disrupted_sensor', 'N/A'),
                "ef_suspected_zone": ef_data.get('suspected_zone', 'N/A'),
            },
            "sensors": {
                "RPM": round(rpm, 1), "MAF": round(maf, 2),
                "MAP_intake": round(map_i, 2), "MAP_boost": round(map_boost, 2),
                "T_intake": round(t_i, 2), "T_boost": round(t_boost, 2),
                "T_cac_out": round(t_cac, 2), "T_exh_manifold": round(t_exh, 2),
                "fuel_qty": round(fuel, 2),
            }
        }


predictor: UnifiedPredictor = None
advisor: AIAdvisor = None

@app.on_event("startup")
async def startup():
    global predictor, advisor
    predictor = UnifiedPredictor()
    advisor = AIAdvisor()

@app.post("/api/predict")
async def predict_endpoint(data: Dict):
    if not predictor:
        raise HTTPException(503, "Models not loaded")
    try:
        return predictor.predict(data)
    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(500, str(e))

@app.post("/api/ai-hint")
async def ai_hint_endpoint(data: Dict):
    if not advisor:
        raise HTTPException(503, "AI Advisor not loaded")
    try:
        hint = advisor.generate_hint(data)
        return {"hint": hint}
    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(500, str(e))

@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_json()
            result = predictor.predict(data)
            await ws.send_json(result)
    except Exception:
        pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
