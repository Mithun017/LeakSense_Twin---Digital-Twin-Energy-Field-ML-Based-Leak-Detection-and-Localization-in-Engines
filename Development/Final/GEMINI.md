# LeakSense Twin — AI Knowledge Base & Implementation Documentation

## Project Overview

**LeakSense Twin** is an AI-powered, real-time, non-invasive system for detecting and localizing air/exhaust leaks in Cat C18 diesel engines during development testing — without requiring additional hardware. The system uses a combination of **physics-based Digital Twins**, **Energy Field Analysis**, and **Machine Learning** to achieve 85-95% leak detection accuracy with sub-2-second latency.

### Target Engine: Cat C18
- **Configuration**: Inline 6, 4-Stroke-Cycle Diesel
- **Displacement**: 18.1L | Bore: 145mm | Stroke: 183mm
- **Max Power**: 597 kW (800 bhp) @ 2100 RPM
- **Max Torque**: 3655 Nm @ 1400 RPM
- **Compression Ratio**: 16.3:1
- **Aspiration**: Twin-Turbocharged Aftercooled (DITA)
- **ECU**: ADEM A4 | Communication: SAE J1939

---

## System Architecture

```
                     ┌─────────────────────────────────┐
                     │     ECU / SAE J1939 CAN Bus     │
                     │   (13 Sensor Signals @ 1 Hz)    │
                     └──────────────┬──────────────────┘
                                    │
                     ┌──────────────▼──────────────────┐
                     │   Steady-State Detection Filter  │
                     │  (30s window, RPM/MAF/fuel std)  │
                     └──────────────┬──────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            │                       │                       │
   ┌────────▼────────┐   ┌─────────▼─────────┐   ┌────────▼────────┐
   │  INTAKE TWIN    │   │  CHARGE AIR TWIN  │   │  EXHAUST TWIN   │
   │  (Zone 1)       │   │  (Zone 2)         │   │  (Zone 3)       │
   │  MAF prediction │   │  Boost/CAC/MAP    │   │  T_exh/DPF/dP   │
   └────────┬────────┘   └─────────┬─────────┘   └────────┬────────┘
            │                       │                       │
            └───────────────────────┼───────────────────────┘
                                    │ Residuals (9 features)
                     ┌──────────────▼──────────────────┐
                     │      Feature Engineering        │
                     │  31 features = 13 raw + 9 res   │
                     │  + 6 ratios + 3 stats           │
                     └──────────────┬──────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            │                       │                       │
   ┌────────▼────────┐   ┌─────────▼─────────┐   ┌────────▼────────┐
   │  LeakSenseNet   │   │  LocalizationNet  │   │  RF + GB        │
   │  (Binary Det.)  │   │  (Zone Classif.)  │   │  (Ensemble)     │
   │  F1 = 0.886     │   │  Acc = 95.6%      │   │  F1 = 0.879     │
   └────────┬────────┘   └─────────┬─────────┘   └────────┬────────┘
            │                       │                       │
            └───────────────────────┼───────────────────────┘
                                    │ Weighted Ensemble Vote
                     ┌──────────────▼──────────────────┐
                     │      Energy Field Detector      │
                     │  6x6 correlation matrix + z-score│
                     │  Frobenius norm + cosine sim    │
                     └──────────────┬──────────────────┘
                                    │
                     ┌──────────────▼──────────────────┐
                     │    Anti-Flicker Decision Logic   │
                     │  3 consecutive alerts required   │
                     └──────────────┬──────────────────┘
                                    │
                     ┌──────────────▼──────────────────┐
                     │   GO / NO-GO + Zone + Severity   │
                     │   + Confidence + Recommended     │
                     │   Action + Energy Field Heatmap  │
                     └─────────────────────────────────┘
```

---

## What Was Implemented

### 1. Digital Twin Models (Physics-Based)

#### Zone 1 — Intake Twin (`twins/intake_twin.py`)
- **Physics**: Mass continuity equation
- **Equation**: `MAF = VE × V_cyl × N_cyl × (RPM/2/60) × ρ_air × 3600`
- **Volumetric Efficiency Model**: Polynomial fit peaking at 1400 RPM
- **Air Density**: `ρ = P_ambient / (R_air × T_ambient)`
- **Online Learning**: EWA correction factor for engine aging

#### Zone 2 — Charge Air Twin (`twins/charge_air_twin.py`)
- **Compressor Map**: Polynomial surface fit `PR = f(corrected_flow, corrected_speed)`
- **Boost Temperature**: Isentropic compression with efficiency correction
- **CAC Model**: `T_cac_out = T_boost - ε × (T_boost - T_ambient)` where ε = 0.88
- **Pressure Drop**: Darcy-Weisbach: `dP = 0.0012 × (MAF/100)²`

#### Zone 3 — Exhaust Twin (`twins/exhaust_twin.py`)
- **Combustion Temperature**: First law thermodynamics with LHV_diesel = 42,800 kJ/kg
- **Turbine Expansion**: Isentropic with η_turbine = 0.82
- **DPF Back-pressure**: `dP = 2.5 × (mass_flow)^1.8 × loading_factor`

### 2. ML Models

#### LeakSenseNet — Binary Detector (`ml/models.py`)
- **Architecture**: Cosine similarity → ReLU amplification → Sigmoid output
- **Formula**: `H = σ(ReLU(W_relu · normalize(W_cos · X)))`
- **Training**: Focal Loss (α=0.75, γ=2.0), AdamW optimizer, CosineAnnealingLR
- **Performance**: F1 = 0.886, Precision = 0.90, Recall = 0.87

#### LeakLocalizationNet — Zone Classifier (`ml/models.py`)
- **Architecture**: FC(128) → BN → FC(64) → BN → Softmax(6)
- **Training**: CrossEntropy with label smoothing (0.05)
- **Performance**: Top-1 Accuracy = 95.6%, best on Zone 2 & 3

#### Ensemble (`ml/ensemble.py`)
- **Combination**: LeakSenseNet(0.35) + LocalizationNet(0.25) + RF(0.20) + GB(0.20)
- **RF/GB**: 100 estimators, balanced class weights
- **Ensemble F1**: 0.879

### 3. Energy Field Detector (`ml/energy_field.py`)

The Energy Field is a mathematical construct encoding inter-sensor relationships in a healthy engine:

- **Computation**: RPM-normalized correlation matrix × thermodynamic weights
- **6×6 matrix** covering: MAF, MAP_boost, MAP_cac_out, T_cac_out, T_exh_manifold, dP_dpf
- **Deviation Detection**: Frobenius norm of z-score matrix
- **Zone Localization**: Most disrupted row identifies the fault zone
- **Healthy Baseline**: Fitted on 1,333 windows from 40,000 healthy samples

### 4. Synthetic Data Generator (`data_generator.py`)

- **50,005 total samples** (40,000 healthy + 10,005 fault)
- **5 fault zones × 3 severities** (2%, 8%, 15% flow loss)
- **RPM range**: 1100-2100 in 100 RPM steps
- **Noise**: Gaussian (0.5% pressure, 1% temp, 1.5% flow)
- **Steady-state filter**: 30s window with RPM/MAF/fuel stability thresholds

### 5. FastAPI Backend (`main.py`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | System health check |
| `/api/predict` | POST | Full leak detection pipeline |
| `/api/simulate` | POST | Simulate sensor scenario with leak injection |
| `/api/health-field` | GET | Get healthy energy field baseline |
| `/api/history` | GET | Prediction history (last 500) |
| `/api/zones` | GET | Zone definitions |
| `/ws/live` | WebSocket | Real-time prediction streaming |
| `/ws/demo` | WebSocket | Auto-demo with leak injection cycles |

### 6. React Dashboard (`frontend/`)

| Component | Description |
|-----------|-------------|
| **GoNoGoIndicator** | Large GO/NO-GO status with animated glow |
| **ConfidenceRing** | SVG circular gauge showing leak probability |
| **LeakAlertCard** | Zone, severity, flow loss %, recommended action |
| **SensorGrid** | Real-time display of all 14 sensor channels |
| **EnergyFieldHeatmap** | Live 6×6 heatmap with global deviation metrics |
| **ResidualsCard** | Digital twin residuals with deviation bars |
| **ZoneProbabilities** | Horizontal bar chart of per-zone probabilities |
| **EngineDiagram** | Zone map with leak highlighting |
| **HistoryTable** | Time-series of predictions |
| **DemoControls** | Start/stop demo, inject faults, select zones |

---

## How To Run

### Prerequisites
- Python 3.10+
- Node.js 18+
- pip packages: `pip install -r backend/requirements.txt`

### Quick Start

```bash
# 1. Train ML models (first time only, ~5 minutes)
cd backend
python ml/train.py

# 2. Start backend server
python main.py
# Server runs on http://localhost:8000

# 3. Start frontend (in new terminal)
cd frontend
npm install
node node_modules/vite/bin/vite.js --port 5173
# Dashboard at http://localhost:5173
```

### Using the Dashboard
1. Click **"Start Live Demo"** to begin auto-streaming with periodic leak injection
2. Use **zone/severity selectors** + **"Inject & Predict"** for manual testing
3. Watch the **Energy Field heatmap** distort during leak events
4. Monitor **Go/No-Go indicator** for confirmed leak decisions
5. Check **History** tab for prediction timeline

---

## Performance Metrics

| Metric | Target | Achieved |
|--------|--------|----------|
| Binary Detection F1 | > 0.89 | **0.886** |
| Binary Precision | > 0.90 | **0.90** |
| Binary Recall | > 0.88 | **0.87** |
| Zone Top-1 Accuracy | > 0.85 | **0.956** |
| Inference Latency | < 50ms | **< 10ms** |
| End-to-End Latency | < 2s | **< 500ms** |
| Dataset Size | > 50,000 | **50,005** |

---

## Technical Details

### Feature Engineering Pipeline (31 Features)

1. **Raw Sensors (13)**: RPM, MAF, MAP_intake, MAP_boost, MAP_cac_in, MAP_cac_out, T_intake, T_boost, T_cac_out, T_exh_manifold, T_dpf_in, T_dpf_out, fuel_qty
2. **Digital Twin Residuals (9)**: res_MAF, res_MAP_boost, res_T_boost, res_T_cac_out, res_MAP_cac_out, res_T_exh_manifold, res_T_post_turbine, res_dP_dpf, res_MAP_intake
3. **Derived Ratios (6)**: PR_compressor, CAC_effectiveness, T_exh_ratio, boost_to_maf_ratio, dpf_dp_ratio, air_fuel_ratio
4. **Statistical (3)**: MAF_std_proxy, MAP_boost_std_proxy, T_exh_std_proxy

### Anti-Flicker Logic
- Leak detection requires **3 consecutive positive predictions** to trigger NO-GO
- Prevents false alarms from transient sensor spikes
- Configurable via `ML_CONFIG['consecutive_alerts']`

### Severity Estimation
| Confidence Range | Severity | Estimated Flow Loss |
|-----------------|----------|-------------------|
| 0.50 - 0.70 | SMALL | 1-5% |
| 0.70 - 0.85 | MEDIUM | 5-12% |
| > 0.85 | CRITICAL | 12-15%+ |

### Leak Zone Mapping
| Zone | Location | Key Sensors |
|------|----------|-------------|
| Zone 1 | Airflow meter → Compressor inlet | MAF, RPM |
| Zone 2 | Compressor outlet → CAC inlet | MAP_boost, T_boost |
| Zone 3 | CAC → Intake manifold | MAP_cac_out, T_cac_out |
| Zone 4 | Exhaust manifold → Turbo turbine | T_exh_manifold |
| Zone 5 | DPF/SCR area | dP_dpf, T_dpf_out |

---

## File Structure

```
Development/Final/
├── backend/
│   ├── main.py                 # FastAPI server (REST + WebSocket)
│   ├── config.py               # Engine constants & ML hyperparameters
│   ├── data_generator.py       # 50K sample synthetic data generator
│   ├── requirements.txt        # Python dependencies
│   ├── twins/
│   │   ├── intake_twin.py      # Zone 1: MAF prediction (mass continuity)
│   │   ├── charge_air_twin.py  # Zone 2: Boost/CAC/manifold prediction
│   │   └── exhaust_twin.py     # Zone 3: Exhaust temp/pressure prediction
│   ├── ml/
│   │   ├── models.py           # LeakSenseNet + LocalizationNet + FocalLoss
│   │   ├── ensemble.py         # Weighted ensemble (NN + RF + GB)
│   │   ├── energy_field.py     # Correlation-based anomaly detector
│   │   └── train.py            # Complete training pipeline
│   ├── models/                 # Trained model weights (.pth, .joblib)
│   └── data/                   # Generated training data (.csv)
├── frontend/
│   ├── src/
│   │   ├── App.jsx             # React dashboard with all components
│   │   ├── main.jsx            # Entry point
│   │   └── index.css           # Premium dark glassmorphism design
│   ├── index.html              # SEO-optimized HTML
│   └── package.json            # Node dependencies
└── GEMINI.md                   # This documentation file
```

---

## Innovation Highlights

1. **Hybrid Physics-ML Architecture**: Digital twins provide physically grounded baselines; ML learns the residual patterns
2. **Energy Field Analysis**: Novel approach using inter-sensor correlation manifold distortion for anomaly detection
3. **Anti-Flicker Logic**: Industrial-grade decision stability preventing false alarms
4. **Severity Quantification**: Goes beyond binary detection to estimate % flow loss
5. **Online Learning (EWA)**: Models adapt to engine aging via exponentially weighted correction factors
6. **Real-Time WebSocket Streaming**: 1 Hz live predictions with sub-500ms latency

---

## Future Extensions (Not Yet Implemented)

- **Temporal Attention Transformer**: Multi-head self-attention over sensor sequences
- **Topological Data Analysis (TDA)**: Persistent homology on sensor manifold
- **Conformal Prediction**: Calibrated uncertainty intervals (MAPIE)
- **LLM RAG Chatbot**: Diagnostic assistant using Cat C18 service manual
- **Federated Learning**: Privacy-preserving training across test cells
- **Graph Neural Network**: Sensor dependency graph for enhanced localization

---

*Built with: Python 3.10 | FastAPI | PyTorch | scikit-learn | React | Vite*
*Last updated: 2026-04-28*
