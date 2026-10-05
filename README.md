# ⚡ Digital Twin & RUL Prediction for DC Motor (Industry 4.0)

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-MLP-EE4C2C.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://streamlit.io/)
[![Simulink](https://img.shields.io/badge/MATLAB-Simulink-0076A8.svg)](https://www.mathworks.com/)

This repository presents an end-to-end **Digital Twin** system for **Predictive Maintenance (PdM)** of a Direct Current (DC) motor. By coupling a multi-physics Simulink simulation with a Deep Learning network in PyTorch, this system estimates the motor's **Remaining Useful Life (RUL)** in real time based on electrical and kinematic degradation trends.

---

## 🎯 Industrial Context & Real-World Connection

In modern smart manufacturing (Industry 4.0), unplanned downtime of rotating machinery leads to critical production losses, high repair costs, and safety hazards.

### The Physical Problem: Mechanical Bearing Wear
Over **40% of rotating machinery failures** originate from mechanical bearing degradation. Over time, friction increases between moving components ($b$).

* **Physical Chain Effect:** As friction ($b$) increases, the motor consumes significantly more electrical current ($i$) to maintain its steady-state operating speed ($\omega$).
* **The Digital Twin Solution:** Instead of installing expensive vibration sensors, this Digital Twin estimates health degradation **non-invasively** by monitoring standard voltage, current, and speed signals already present in motor drives.

---

## 🏗️ System Architecture

The project is built around 3 core modules:

```text
+------------------------------------+
| 1. Physics-Based Model (Simulink)  |  --> Simulates dynamic mechanical degradation (b)
+------------------------------------+
                  |
                  v
+------------------------------------+
| 2. Deep Learning Model (PyTorch)   |  --> RUL estimation & feature regression
+------------------------------------+
                  |
                  v
+------------------------------------+
| 3. Interactive Web Dashboard       |  --> Real-time monitoring & decision-making (Streamlit)
+------------------------------------+
```

1. **Physics-Based Simulation (Simulink):** Generates run-to-failure synthetic data over 100 operational cycles (10 s each) by drifting the friction coefficient $b$ from $0.0005$ to $0.0025\ \text{N·m·s/rad}$.
2. **Feature Engineering & AI (PyTorch):** Extracts health indicators (`current_ss`, `speed_ss`, `current_peak`, `current_rms`) and trains a Multi-Layer Perceptron (MLP) regressor to output a normalized RUL $[1.0 \rightarrow 0.0]$.
3. **Operational Dashboard (Streamlit):** Serves as a Human-Machine Interface (HMI) for operators to inspect real-time signals, track degradation curves, and schedule maintenance interventions before failure.

---

## 📊 Model Performance & Evaluation

The PyTorch MLP model was evaluated on an unseen test set (20% of cycles held out during training):

| Metric | Measured Value | Industrial Interpretation |
| :--- | :--- | :--- |
| **$R^2$ Score** | **0.9999** | Near-perfect alignment between predicted and actual degradation |
| **RMSE** | **0.0022** | Negligible error across all lifecycle phases |
| **MAE** | **0.0018** | Average error is less than **0.18%** of the total RUL |

> ⚠️ **Note:** The data is synthetic and noise-free, so these scores reflect an idealized setting. Performance on real sensor data is expected to be lower.

---

## 🚀 Installation & Usage

```bash
git clone https://github.com/safwanenadah-eng/DigitalTwin-DCMotor-RUL.git
cd DigitalTwin-DCMotor-RUL
pip install torch streamlit numpy pandas scipy matplotlib scikit-learn

python 01_eda.py
python 02_feature_engineering.py
python 03_train_model.py
python 04_evaluate_model.py
streamlit run app.py
```

---

## 📁 Repository Structure

```text
DigitalTwin-DCMotor-RUL/
├── data/
│   ├── motor_degradation_dataset.mat  # Raw time-series signals exported from Simulink
│   └── extracted_features.csv         # Physical features & health indicators
├── models/
│   └── rul_model.pth                  # Saved PyTorch model state dictionary
├── 01_eda.py                          # Exploratory Data Analysis & signal plotting
├── 02_feature_engineering.py          # Feature extraction (steady-state, RMS, peak)
├── 03_train_model.py                  # PyTorch MLP training & loss tracking
├── 04_evaluate_model.py               # Test set validation & trajectory plotting
├── app.py                             # Interactive Streamlit web dashboard
├── .gitignore                         # Excluded temporary files
└── README.md                          # Project documentation
```

---

## ⚠️ Limitations & Future Work

- Synthetic, noise-free data (100 cycles only)
- Single degradation mode (bearing friction)
- Next steps: add sensor noise, chronological train/test split, test on real datasets
