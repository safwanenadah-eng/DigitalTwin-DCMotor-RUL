import os
import scipy.io
import numpy as np
import pandas as pd

# 1. Chargement des données
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'data', 'motor_degradation_dataset.mat')
mat_data = scipy.io.loadmat(data_path)

current = mat_data['dataset_current']  # (108, 100)
speed = mat_data['dataset_speed']      # (108, 100)
rul = mat_data['dataset_rul'].flatten()  # (100,)

num_cycles = current.shape[1]
features_list = []

# 2. Boucle d'extraction des caractéristiques par cycle
for k in range(num_cycles):
    i_signal = current[:, k]
    w_signal = speed[:, k]
    
    # Courant et vitesse en régime permanent (20 derniers % du cycle)
    i_ss = np.mean(i_signal[-20:])
    w_ss = np.mean(w_signal[-20:])
    
    # Pic de courant au démarrage
    i_peak = np.max(i_signal)
    
    # Valeur RMS du courant (Root Mean Square)
    i_rms = np.sqrt(np.mean(i_signal**2))
    
    # Enregistrement des caractéristiques du cycle k
    features_list.append({
        'cycle': k + 1,
        'current_ss': i_ss,
        'speed_ss': w_ss,
        'current_peak': i_peak,
        'current_rms': i_rms,
        'RUL': rul[k]
    })

# 3. Création et sauvegarde du DataFrame Pandas
df_features = pd.DataFrame(features_list)

csv_path = os.path.join(script_dir, 'data', 'extracted_features.csv')
df_features.to_csv(csv_path, index=False)

print("--- Extraction terminée avec succès ! ---")
print(f"Tableau sauvegardé dans : {csv_path}")
print("\nAperçu des 5 premiers cycles (Moteur neuf) :")
print(df_features.head())
print("\nAperçu des 5 derniers cycles (Moteur usé) :")
print(df_features.tail())