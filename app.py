import os
import scipy.io
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import streamlit as st
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Configuration de la page Streamlit
# ---------------------------------------------------------
st.set_page_config(
    page_title="Jumeau Numérique - Moteur CC",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Jumeau Numérique & Prédiction RUL du Moteur CC")
st.markdown("---")

# ---------------------------------------------------------
# Chargement des données et du modèle PyTorch
# ---------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_data():
    mat_path = os.path.join(script_dir, 'data', 'motor_degradation_dataset.mat')
    csv_path = os.path.join(script_dir, 'data', 'extracted_features.csv')
    
    mat_data = scipy.io.loadmat(mat_path)
    df_features = pd.read_csv(csv_path)
    
    return mat_data, df_features

mat_data, df_features = load_data()

# Architecture du modèle PyTorch
class RULPredictor(nn.Module):
    def __init__(self):
        super(RULPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        
    def forward(self, x):
        return self.network(x)

@st.cache_resource
def load_trained_model():
    model = RULPredictor()
    model_path = os.path.join(script_dir, 'models', 'rul_model.pth')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

model = load_trained_model()

# Préparation du scaler
X = df_features[['current_ss', 'speed_ss', 'current_peak', 'current_rms']].values
scaler = StandardScaler()
scaler.fit(X)

# ---------------------------------------------------------
# Barre latérale (Sidebar) : Contrôle interactif
# ---------------------------------------------------------
st.sidebar.header("🎛️ Panneau de Contrôle")
selected_cycle = st.sidebar.slider("Sélectionner un cycle de fonctionnement :", 1, 100, 1)

# Extraction des données du cycle sélectionné
cycle_idx = selected_cycle - 1
time = mat_data['time_vector'].flatten()
current_sig = mat_data['dataset_current'][:, cycle_idx]
speed_sig = mat_data['dataset_speed'][:, cycle_idx]

# Extrait des caractéristiques
row = df_features.iloc[cycle_idx]
features_val = np.array([[row['current_ss'], row['speed_ss'], row['current_peak'], row['current_rms']]])
features_scaled = scaler.transform(features_val)

# Inférence PyTorch
with torch.no_grad():
    input_tensor = torch.tensor(features_scaled, dtype=torch.float32)
    predicted_rul = model(input_tensor).item()

predicted_rul = float(np.clip(predicted_rul, 0.0, 1.0))
actual_rul = float(row['RUL'])

# ---------------------------------------------------------
# Affichage des Indicateurs Clés (KPIs)
# ---------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="RUL Réel", value=f"{actual_rul * 100:.1f} %")

with col2:
    delta_val = f"{(predicted_rul - actual_rul) * 100:+.2f} %"
    st.metric(label="RUL Prédit (IA)", value=f"{predicted_rul * 100:.1f} %", delta=delta_val)

with col3:
    st.metric(label="Courant Perm. (A)", value=f"{row['current_ss']:.4f} A")

with col4:
    st.metric(label="Vitesse Perm. (rad/s)", value=f"{row['speed_ss']:.2f} rad/s")

# Progression de santé
st.subheader("État de Santé Global du Moteur")
st.progress(predicted_rul)

st.markdown("---")

# ---------------------------------------------------------
# Graphiques interactifs
# ---------------------------------------------------------
col_left, col_right = st.columns(2)

with col_left:
    st.subheader(f"📊 Signaux Physiques (Cycle {selected_cycle})")
    fig, ax = plt.subplots(2, 1, figsize=(6, 5))
    
    # Courant
    ax[0].plot(time, current_sig, color='tab:blue')
    ax[0].set_ylabel("Courant (A)")
    ax[0].grid(True)
    
    # Vitesse
    ax[1].plot(time, speed_sig, color='tab:orange')
    ax[1].set_xlabel("Temps (s)")
    ax[1].set_ylabel("Vitesse (rad/s)")
    ax[1].grid(True)
    
    plt.tight_layout()
    st.pyplot(fig)

with col_right:
    st.subheader("📈 Trajectoire de Dégradation Temporelle")
    fig_traj, ax_traj = plt.subplots(figsize=(6, 4.3))
    
    ax_traj.plot(df_features['cycle'], df_features['RUL'], label='RUL Réel', color='green')
    
    # Calcul des prédictions globales
    X_all_scaled = scaler.transform(X)
    with torch.no_grad():
        all_pred = model(torch.tensor(X_all_scaled, dtype=torch.float32)).numpy().flatten()
    
    ax_traj.plot(df_features['cycle'], all_pred, label='RUL Prédit (IA)', color='orange', linestyle='--')
    ax_traj.axvline(x=selected_cycle, color='red', linestyle=':', label=f'Cycle Actuel ({selected_cycle})')
    
    ax_traj.set_xlabel("Cycle")
    ax_traj.set_ylabel("RUL Normalisé")
    ax_traj.grid(True)
    ax_traj.legend()
    
    plt.tight_layout()
    st.pyplot(fig_traj)