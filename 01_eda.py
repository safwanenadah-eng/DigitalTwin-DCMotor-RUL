import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Chargement du fichier .mat avec chemin dynamique
# ---------------------------------------------------------
# Calcule automatiquement le chemin absolu vers le dossier 'data'
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'data', 'motor_degradation_dataset.mat')

print(f"Chargement du fichier depuis : {data_path}")
mat_data = scipy.io.loadmat(data_path)

# Extraction des variables
current = mat_data['dataset_current']    # Matrice (N_echantillons, 100)
speed = mat_data['dataset_speed']        # Matrice (N_echantillons, 100)
time = mat_data['time_vector'].flatten()   # Vecteur (N_echantillons,)
rul = mat_data['dataset_rul'].flatten()      # Vecteur (100,)

# ---------------------------------------------------------
# 2. Vérification des dimensions
# ---------------------------------------------------------
print("\n--- Inspection des données ---")
print(f"Vecteur Temps   : {time.shape} points")
print(f"Matrice Courant : {current.shape} (Lignes: temps, Colonnes: cycles)")
print(f"Matrice Vitesse : {speed.shape}")
print(f"Vecteur RUL     : {rul.shape} (de {rul[0]:.1f} à {rul[-1]:.1f})\n")

# ---------------------------------------------------------
# 3. Visualisation : Cycle 1 (Sain) vs Cycle 100 (Dégradé)
# ---------------------------------------------------------
plt.figure(figsize=(12, 5))

# Graphique du Courant (i)
plt.subplot(1, 2, 1)
plt.plot(time, current[:, 0], label=f'Cycle 1 (RUL={rul[0]:.1f})', color='green')
plt.plot(time, current[:, -1], label=f'Cycle 100 (RUL={rul[-1]:.1f})', color='red', linestyle='--')
plt.title("Évolution du Courant Statorique i(t)")
plt.xlabel("Temps (s)")
plt.ylabel("Courant (A)")
plt.grid(True)
plt.legend()

# Graphique de la Vitesse (omega)
plt.subplot(1, 2, 2)
plt.plot(time, speed[:, 0], label=f'Cycle 1 (RUL={rul[0]:.1f})', color='green')
plt.plot(time, speed[:, -1], label=f'Cycle 100 (RUL={rul[-1]:.1f})', color='red', linestyle='--')
plt.title("Évolution de la Vitesse de Rotation w(t)")
plt.xlabel("Temps (s)")
plt.ylabel("Vitesse (rad/s)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()