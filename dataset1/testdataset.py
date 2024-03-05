from sklearn.model_selection import train_test_split
import numpy as np

# Génération de données fictives
X = np.random.rand(100, 5)  # Fonctionne pour un ensemble de 100 échantillons avec 5 fonctionnalités
y = np.random.randint(2, size=100)  # Des étiquettes binaires aléatoires

# Division des données en ensembles d'entraînement, de validation et de test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)  # Division initiale en train (70%) et temp (30%)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # Division du reste en validation (15%) et test (15%)

# Affichage des tailles des ensembles
print("Taille de l'ensemble d'entraînement:", X_train.shape[0])
print("Taille de l'ensemble de validation:", X_val.shape[0])
print("Taille de l'ensemble de test:", X_test.shape[0])