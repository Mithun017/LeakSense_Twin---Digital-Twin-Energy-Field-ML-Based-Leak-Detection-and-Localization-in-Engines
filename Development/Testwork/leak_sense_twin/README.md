# LeakSense Twin - Digital Twin & Energy Field & ML Based Leak Detection and Localization in Engines

This project implements a real-time, non-invasive, AI-powered system that detects and localizes air/exhaust leaks in a Cat C18 diesel engine during development testing.

## Project Structure

```
leak_sense_twin/
├── data_generation/
│   └── synthetic_data_generator.py      # Generates synthetic engine data with leak injection
├── models/
│   ├── intake_twin.py                   # Intake Zone Digital Twin (MAF prediction)
│   ├── charge_air_twin.py               # Charge Air System Digital Twin (boost, temps)
│   └── exhaust_twin.py                  # Exhaust Zone Digital Twin (exhaust temps, back-pressure)
├── energy_field/
│   └── energy_field_detector.py         # Energy Field computation and deviation detection
├── ml/
│   └── leak_sense_net.py                # Neural network models (LeakSenseNet, LocalizationNet, GNN)
├── saved_models/                        # Directory for trained models (created during runtime)
├── data/                                # Directory for generated/loading data (created during runtime)
├── main_leak_detection_system.py        # Main pipeline integrating all components
└── requirements.txt                     # Python dependencies
```

## Features Implemented

1. **Synthetic Data Generator**: Creates realistic Cat C18 engine sensor data with configurable leak injection across 5 zones and 3 severity levels
2. **Three Physics-Informed Digital Twins**:
   - Intake Twin: Predicts MAF from RPM and ambient conditions
   - Charge Air Twin: Predicts boost pressure, temperatures using compressor map and CAC model
   - Exhaust Twin: Predicts exhaust temperatures and back-pressure using energy balance
3. **Energy Field Detector**: Computes inter-sensor relationships as a correlation manifold for leak detection
4. **ML Models**:
   - LeakSenseNet: Binary leak detection using ReLU-Sigmoid-Cosine architecture
   - LeakLocalizationNet: Multi-class zone localization
   - SensorGNN: Graph neural network for inter-sensor dependencies (simplified version)
5. **Complete Integration Pipeline**: Trains all components and evaluates performance

## Installation

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the complete leak detection system demonstration:

```bash
python main_leak_detection_system.py
```

This will:
1. Generate synthetic engine data (5000 samples by default)
2. Train all three digital twin models
3. Train the energy field detector
4. Prepare features for ML models (raw sensors + residuals + energy field features)
5. Train the neural network ensemble
6. Evaluate on test set and demonstrate real-time detection

## Model Outputs

Trained models are saved in the `saved_models/` directory:
- `intake_twin.joblib`
- `charge_air_twin.joblib`
- `exhaust_twin.joblib`
- `energy_field_detector.joblib`
- `leak_sense_net.pth`
- `leak_localization_net.pth`

## Performance Targets (from Project Requirements)

- Leak detection: Precision > 0.90, Recall > 0.88, F1 > 0.89
- Zone localization: Top-1 accuracy > 0.85, Top-2 accuracy > 0.95
- Latency: inference < 50ms per window on CPU
- Detect leaks with 85-95% accuracy
- Localize leak to specific zone and component group

## Notes

This is a demonstration implementation based on the project specifications. For production use, additional work would be needed including:
- Real sensor data integration
- More sophisticated compressor/turbine maps
- Advanced uncertainty quantification
- Production-optimized model serving
- Comprehensive validation against engine test data

## References

The implementation follows the specifications outlined in the Master Project Blueprint Prompt files, particularly:
- Data Sources & Collection Strategy
- Digital Twin Build Prompt
- ML Model Build Prompt
- Energy Field Build & Compute Prompt
- Evaluation & Validation Prompt