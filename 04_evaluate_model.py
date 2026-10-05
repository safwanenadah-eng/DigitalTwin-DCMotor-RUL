import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Chargement et Préparation des Données
# ---------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'data', 'extracted_features.csv')
df = pd.read_csv(csv_path)

X = df[['current_ss', 'speed_ss', 'current_peak', 'current_rms']].values
y = df['RUL'].values.reshape(-1, 1)

# Même découpage avec la même graine (random_state=42) pour isoler l'ensemble de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation identique
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

# ---------------------------------------------------------
# 2. Définition et Chargement du Modèle Entraîné
# ---------------------------------------------------------
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

model = RULPredictor()
model_path = os.path.join(script_dir, 'models', 'rul_model.pth')
model.load_state_dict(torch.load(model_path))
model.eval()  # Mode évaluation

# ---------------------------------------------------------
# 3. Prédictions sur l'ensemble de Test
# ---------------------------------------------------------
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)
    y_pred = y_pred_tensor.numpy()

# ---------------------------------------------------------
# 4. Calcul des Métriques
# ---------------------------------------------------------
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("--- Métriques d'Évaluation sur le Test Set ---")
print(f"RMSE (Root Mean Squared Error) : {rmse:.4f}")
print(f"MAE  (Mean Absolute Error)     : {mae:.4f}")
print(f"Score R² (Coefficient det.)    : {r2:.4f} (1.00 = Prédiction parfaite)")

# ---------------------------------------------------------
# 5. Visualisation : RUL Réel vs RUL Prédit
# ---------------------------------------------------------
plt.figure(figsize=(10, 5))

# Graphique 1 : Comparaison point par point
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, color='blue', alpha=0.7, label='Cycles de test')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Prédiction parfaite (Y = X)')
plt.title("RUL Réel vs RUL Prédit")
plt.xlabel("RUL Réel (Vérité Terrain)")
plt.ylabel("RUL Prédit par PyTorch")
plt.grid(True)
plt.legend()

# Graphique 2 : Prédiction sur l'ensemble des 100 cycles ordinés
X_all_scaled = scaler.transform(X)
X_all_tensor = torch.tensor(X_all_scaled, dtype=torch.float32)

with torch.no_grad():
    y_all_pred = model(X_all_tensor).numpy()

plt.subplot(1, 2, 2)
plt.plot(df['cycle'], df['RUL'], label='RUL Réel', color='green', linewidth=2)
plt.plot(df['cycle'], y_all_pred, label='RUL Prédit', color='orange', linestyle='--')
plt.title("Trajectoire de Santé sur 100 Cycles")
plt.xlabel("Numéro du Cycle")
plt.ylabel("RUL Normalisé")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()