<div align="center">

# 🔍 LeakSense Twin

### Digital Twin & Energy Field & ML Based Leak Detection and Localization in Engines

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![React](https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

---

**AI-powered, real-time, non-invasive leak detection and localization system for Cat C18 diesel engines — no additional hardware required.**

</div>

---

## 📋 Overview

**LeakSense Twin** is a production-ready diagnostic system that detects and localizes air/exhaust leaks in a **Caterpillar C18 diesel engine** (Inline-6, 18.1L, 597kW @ 2100 RPM, Twin-Turbocharged Aftercooled) during development testing in a test cell — using only existing ECU sensor data via SAE J1939.

The system combines three core technologies:

| Technology | Purpose |
|-----------|---------|
| 🏗️ **Physics-Based Digital Twins** | Predict healthy-state sensor values using thermodynamic equations |
| ⚡ **Energy Field Analysis** | Detect correlation-pattern distortions in inter-sensor relationships |
| 🧠 **Machine Learning Ensemble** | Classify leaks and localize to specific engine zones |

---

## 🎯 Key Features

- **85–96% leak detection accuracy** (F1: 0.886, Zone accuracy: 95.6%)
- **5-zone localization**: Intake → Charge Air → CAC/Manifold → Exhaust → DPF/SCR
- **3 severity levels**: Small (2%), Medium (8%), Large (15%) flow loss
- **Real-time operation** with < 500ms end-to-end latency
- **Go / No-Go indicator** for test cell operators
- **Live 6×6 Energy Field heatmap** updating at 1 Hz
- **WebSocket streaming** for continuous monitoring
- **Simulation mode** for fault injection testing
- **Zero raw data exposure** — only predictions leave the test cell

---

## 🏗️ System Architecture

```
ECU (SAE J1939 CAN Bus) → 13 Sensor Signals @ 1Hz
         │
    Steady-State Filter (30s window)
         │
    ┌─────┼─────┐
    │     │     │
 Intake  Charge  Exhaust     ← 3 Digital Twin Models
  Twin   Air     Twin           (Physics-based)
    │   Twin     │
    └─────┼─────┘
          │
   Feature Engineering (31 features)
          │
    ┌─────┼─────┐
    │     │     │
 LeakNet  LocNet  RF+GB     ← ML Ensemble
 (Binary) (Zone)  (Boost)      (Weighted Vote)
    │     │     │
    └─────┼─────┘
          │
   Energy Field Detector (6×6 correlation matrix)
          │
   Anti-Flicker Logic (3 consecutive alerts)
          │
   GO/NO-GO + Zone + Severity + Action
```

---

## 🔧 Target Engine — Cat C18 Specifications

| Parameter | Value |
|-----------|-------|
| Configuration | Inline 6, 4-Stroke-Cycle Diesel |
| Displacement | 18.1L (Bore: 145mm, Stroke: 183mm) |
| Max Power | 597 kW (800 bhp) @ 2100 RPM |
| Max Torque | 3655 Nm @ 1400 RPM |
| Compression Ratio | 16.3:1 |
| Aspiration | Twin-Turbocharged Aftercooled (DITA) |
| ECU | ADEM A4 |
| Communication | SAE J1939 |
| Fuel System | MEUI Direct Injection |

---

## 📁 Project Structure

```
LeakSense_Twin/
├── Development/
│   └── Final/                      # Production-ready system
│       ├── backend/
│       │   ├── main.py             # FastAPI server (REST + WebSocket)
│       │   ├── config.py           # Engine constants & ML config
│       │   ├── data_generator.py   # 50K synthetic data generator
│       │   ├── requirements.txt    # Python dependencies
│       │   ├── twins/
│       │   │   ├── intake_twin.py      # Zone 1 — Mass continuity model
│       │   │   ├── charge_air_twin.py  # Zone 2 — Compressor + CAC model
│       │   │   └── exhaust_twin.py     # Zone 3 — Combustion + turbine model
│       │   ├── ml/
│       │   │   ├── models.py       # LeakSenseNet + LocalizationNet
│       │   │   ├── ensemble.py     # Weighted ensemble (NN + RF + GB)
│       │   │   ├── energy_field.py # Correlation-based anomaly detector
│       │   │   └── train.py        # Complete training pipeline
│       │   └── models/             # Trained model weights (gitignored)
│       ├── frontend/
│       │   ├── src/
│       │   │   ├── App.jsx         # React dashboard components
│       │   │   ├── main.jsx        # Entry point
│       │   │   └── index.css       # Premium glassmorphism dark theme
│       │   └── index.html          # SEO-optimized HTML
│       ├── GEMINI.md               # AI knowledge base & documentation
│       └── start.bat               # One-click startup script
└── Readme.md                       # This file
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** with pip
- **Node.js 18+** with npm
- **PyTorch**, **scikit-learn**, **FastAPI** (installed via requirements)

### 1️⃣ Install Dependencies

```bash
# Backend
cd Development/Final/backend
pip install -r requirements.txt

# Frontend
cd ../frontend
npm install
```

### 2️⃣ Train ML Models (first time only, ~5 minutes)

```bash
cd Development/Final/backend
python ml/train.py
```

This generates 50,005 synthetic training samples, trains all models, and saves them to `models/`.

### 3️⃣ Run the System

**Option A — One-click (Windows):**
```bash
# Double-click start.bat in Development/Final/
```

**Option B — Manual:**
```bash
# Terminal 1 — Backend
cd Development/Final/backend
python main.py
# → http://localhost:8000

# Terminal 2 — Frontend
cd Development/Final/frontend
node node_modules/vite/bin/vite.js --port 5173
# → http://localhost:5173
```

### 4️⃣ Use the Dashboard

1. Click **"▶ Start Live Demo"** for auto-streaming with periodic leak injection
2. Use **zone/severity selectors** + **"🧪 Inject & Predict"** for manual testing
3. Watch the **Energy Field heatmap** distort during leak events
4. Monitor the **Go/No-Go indicator** for confirmed leak decisions

---

## 📊 Performance Metrics

| Metric | Target | Achieved |
|--------|--------|----------|
| Binary Detection F1 | > 0.89 | **0.886** |
| Precision | > 0.90 | **0.90** |
| Recall | > 0.88 | **0.87** |
| Zone Top-1 Accuracy | > 0.85 | **0.956** |
| Inference Latency | < 50ms | **< 10ms** |
| End-to-End Latency | < 2s | **< 500ms** |
| Training Dataset | > 50,000 | **50,005** |

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | `GET` | System health check |
| `/api/predict` | `POST` | Run leak detection on sensor data |
| `/api/simulate` | `POST` | Simulate scenario with fault injection |
| `/api/health-field` | `GET` | Get healthy energy field baseline |
| `/api/history` | `GET` | Prediction history |
| `/api/zones` | `GET` | Zone definitions |
| `/ws/live` | `WebSocket` | Real-time prediction streaming |
| `/ws/demo` | `WebSocket` | Auto-demo with leak cycles |

Full API docs at: `http://localhost:8000/docs`

---

## 🧠 Technical Highlights

### Digital Twins (Physics-Based)
- **Intake Twin**: MAF prediction via volumetric efficiency & mass continuity
- **Charge Air Twin**: Compressor map + CAC heat exchanger + Darcy-Weisbach
- **Exhaust Twin**: First-law combustion + isentropic turbine + DPF back-pressure

### ML Architecture
- **LeakSenseNet**: Cosine similarity → ReLU amplification → Sigmoid output (H = σ(ReLU(W·normalize(W·X))))
- **LocalizationNet**: FC(128) → BatchNorm → FC(64) → BatchNorm → Softmax(6 classes)
- **Ensemble**: LeakNet(0.35) + LocNet(0.25) + RandomForest(0.20) + GradientBoosting(0.20)
- **Focal Loss** for class imbalance handling

### Energy Field
- 6×6 weighted correlation matrix across pressure/temperature/flow channels
- Z-score deviation via Frobenius norm
- Cosine similarity for shape preservation detection
- Row-wise disruption analysis for zone localization

---

## 🛡️ Data Security

- All sensor data processed **in-memory** within the test cell network
- Only **predictions and metadata** are stored — never raw sensor values
- TLS 1.3 ready for all external communications
- Role-based access control architecture

---

## 📚 Documentation

See [`Development/Final/GEMINI.md`](Development/Final/GEMINI.md) for the complete AI knowledge base including:
- Full architecture diagrams
- Physics equations for each digital twin
- Feature engineering pipeline details
- Training configuration & hyperparameters
- Leak zone mapping tables

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.10, FastAPI, Uvicorn |
| ML Framework | PyTorch, scikit-learn |
| Digital Twins | NumPy, SciPy, Joblib |
| Frontend | React 19, Vite 8 |
| Styling | Vanilla CSS (glassmorphism dark theme) |
| Communication | REST API + WebSocket |
| Data | Pandas, synthetic C18 engine data |

---

<div align="center">

**Built for Caterpillar Cat C18 Engine Development Testing**

*LeakSense Twin — Detect. Localize. Decide.*

</div>
