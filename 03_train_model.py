import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Chargement et Préparation des Données
# ---------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'data', 'extracted_features.csv')
df = pd.read_csv(csv_path)

# Variables d'entrée (Features X) et cible (Target y)
X = df[['current_ss', 'speed_ss', 'current_peak', 'current_rms']].values
y = df['RUL'].values.reshape(-1, 1)

# Séparation Train (80%) / Test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation des features (Moyenne = 0, Ecart-type = 1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Conversion des données en Tenseurs PyTorch
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# ---------------------------------------------------------
# 2. Définition de l'Architecture du Réseau de Neurones
# ---------------------------------------------------------
class RULPredictor(nn.Module):
    def __init__(self):
        super(RULPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 16),   # Couche d'entrée (4 features) -> 16 neurones
            nn.ReLU(),          # Fonction d'activation non-linéaire
            nn.Linear(16, 8),   # Couche cachée (16 neurones) -> 8 neurones
            nn.ReLU(),
            nn.Linear(8, 1)     # Couche de sortie (1 neurone = RUL prédit)
        )
        
    def forward(self, x):
        return self.network(x)

model = RULPredictor()
criterion = nn.MSELoss()  # Erreur Quadratique Moyenne pour la régression
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ---------------------------------------------------------
# 3. Boucle d'Entraînement
# ---------------------------------------------------------
epochs = 300
train_losses = []

print("--- Début de l'Entraînement PyTorch ---")
for epoch in range(epochs):
    model.train()
    
    # Forward pass
    predictions = model(X_train_tensor)
    loss = criterion(predictions, y_train_tensor)
    
    # Backward pass & Optimisation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    train_losses.append(loss.item())
    
    if (epoch + 1) % 50 == 0:
        print(f"Époque [{epoch+1}/{epochs}] - Perte (Loss MSE) : {loss.item():.6f}")

# ---------------------------------------------------------
# 4. Sauvegarde du Modèle
# ---------------------------------------------------------
models_dir = os.path.join(script_dir, 'models')
os.makedirs(models_dir, exist_ok=True)
model_save_path = os.path.join(models_dir, 'rul_model.pth')
torch.save(model.state_dict(), model_save_path)
print(f"\nModèle entraîné et sauvegardé avec succès dans : {model_save_path}")

# ---------------------------------------------------------
# 5. Graphique de la Courbe d'Apprentissage (Loss)
# ---------------------------------------------------------
plt.figure(figsize=(8, 4))
plt.plot(train_losses, label='Train MSE Loss', color='blue')
plt.title("Courbe d'Apprentissage (Décroissance de l'erreur)")
plt.xlabel("Époques")
plt.ylabel("Perte (MSE)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()