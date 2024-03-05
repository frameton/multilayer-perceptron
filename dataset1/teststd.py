from sklearn.preprocessing import StandardScaler
import numpy as np

# Génération de données fictives
X = np.random.rand(100, 2)  # Fonctionne pour un ensemble de 100 échantillons avec 2 fonctionnalités

# Création d'un objet StandardScaler
scaler = StandardScaler()

# Application de la standardisation sur les données
X_scaled = scaler.fit_transform(X)

# Affichage des statistiques après la standardisation
print(X)
print(X_scaled)
print("Moyenne après standardisation:", X_scaled.mean(axis=0))
print("Écart-type après standardisation:", X_scaled.std(axis=0))