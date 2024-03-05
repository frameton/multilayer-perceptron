from sklearn.model_selection import train_test_split
from tools import colors, load_csv, parse_csv, get_csv_object
import numpy as np
import pandas as pd


def split_csv_data(data):
    # Séparation des caractéristiques (X) et de la cible (y)
    X = data.drop(columns=['diagnostic'])
    y = data['diagnostic']

    # Division des données en ensembles d'entraînement, de validation et de test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)  # Division initiale en train (70%) et temp (30%)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # Division du reste en validation (15%) et test (15%)

    # Enregistrer les données d'entraînement dans un fichier CSV
    train_data = pd.concat([X_train, y_train], axis=1)  # Concaténer les caractéristiques et la cible
    train_data.to_csv('datasets/train_data.csv', index=False)

    # Enregistrer les données de validation dans un fichier CSV
    val_data = pd.concat([X_val, y_val], axis=1)
    val_data.to_csv('datasets/val_data.csv', index=False)

    # Enregistrer les données de test dans un fichier CSV
    test_data = pd.concat([X_test, y_test], axis=1)
    test_data.to_csv('datasets/test_data.csv', index=False)

    # Affichage des tailles des ensembles
    print("Taille de l'ensemble d'entraînement:", X_train.shape[0])
    print("Taille de l'ensemble de validation:", X_val.shape[0])
    print("Taille de l'ensemble de test:", X_test.shape[0])


if __name__ == "__main__":
    print(colors.clr.fg.yellow, "Spliting csv data...", colors.clr.reset)

    data = get_csv_object.get_no_parse()
    split_csv_data(data)

    print(colors.clr.fg.green, "Split success.", colors.clr.reset)

    print("")
    print("")
    exit(0)