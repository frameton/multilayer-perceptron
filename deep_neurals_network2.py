from sklearn.datasets import make_blobs, make_circles
from sklearn.metrics import accuracy_score
#from tools import colors, load_csv, parse_csv, get_csv_object
from sklearn.model_selection import train_test_split
from tools import colors, load_csv, parse_csv, get_csv_object
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

def log_loss(A, y):
    return 1 / len(y) * np.sum(-y * np.log(A) - (1 - y) * np.log(1 - A))

def initialisation(dimensions):
    
    parametres = {}
    C = len(dimensions)

    np.random.seed(1)

    for c in range(1, C):
        parametres['W' + str(c)] = np.random.randn(dimensions[c], dimensions[c - 1])
        parametres['b' + str(c)] = np.random.randn(dimensions[c], 1)

    return parametres


def forward_propagation(X, parametres):
  
  activations = {'A0': X}

  C = len(parametres) // 2

  for c in range(1, C + 1):

    Z = parametres['W' + str(c)].dot(activations['A' + str(c - 1)]) + parametres['b' + str(c)]
    # print("")
    # print("")
    # print(colors.clr.fg.red, Z[0][0], colors.clr.reset)
    # exit()
    activations['A' + str(c)] = 1 / (1 + np.exp(-Z))

  return activations


def back_propagation(y, parametres, activations):

  m = y.shape[1]
  C = len(parametres) // 2

  dZ = activations['A' + str(C)] - y
  gradients = {}

  for c in reversed(range(1, C + 1)):
    gradients['dW' + str(c)] = 1/m * np.dot(dZ, activations['A' + str(c - 1)].T)
    gradients['db' + str(c)] = 1/m * np.sum(dZ, axis=1, keepdims=True)
    if c > 1:
      dZ = np.dot(parametres['W' + str(c)].T, dZ) * activations['A' + str(c - 1)] * (1 - activations['A' + str(c - 1)])

  return gradients
def update(gradients, parametres, learning_rate):

    C = len(parametres) // 2

    for c in range(1, C + 1):
        parametres['W' + str(c)] = parametres['W' + str(c)] - learning_rate * gradients['dW' + str(c)]
        parametres['b' + str(c)] = parametres['b' + str(c)] - learning_rate * gradients['db' + str(c)]

    return parametres
def predict(X, parametres):
  activations = forward_propagation(X, parametres)
  C = len(parametres) // 2
  Af = activations['A' + str(C)]
  return Af >= 0.5
def deep_neural_network(X_train, y_train, X_test, y_test, X_val, y_val, hidden_layers = (16, 16, 16), learning_rate = 0.001, n_iter = 3000):
    
    # initialisation parametres
    dimensions = list(hidden_layers)
    dimensions_test = list(hidden_layers)
    dimensions_val = list(hidden_layers)
    dimensions.insert(0, X_train.shape[0])
    dimensions.append(y_train.shape[0])
    dimensions_test.insert(0, X_test.shape[0])
    dimensions_test.append(y_test.shape[0])
    dimensions_val.insert(0, X_test.shape[0])
    dimensions_val.append(y_test.shape[0])
    np.random.seed(1)
    parametres = initialisation(dimensions)
    parametres_test = initialisation(dimensions_test)
    parametres_val = initialisation(dimensions_val)

    # tableau numpy contenant les futures accuracy et log_loss
    training_history = np.zeros((int(n_iter), 2))
    test_history = np.zeros((int(n_iter), 2))
    val_history = np.zeros((int(n_iter), 2))

    C = len(parametres) // 2

    # gradient descent
    for i in tqdm(range(n_iter)):

        activations = forward_propagation(X_train, parametres)
        gradients = back_propagation(y_train, parametres, activations)
        parametres = update(gradients, parametres, learning_rate)
        Af = activations['A' + str(C)]

        # calcul du log_loss et de l'accuracy
        training_history[i, 0] = (log_loss(Af.flatten(), y_train.flatten()))
        y_pred = predict(X_train, parametres)
        training_history[i, 1] = (accuracy_score(y_train.flatten(), y_pred.flatten()))



        activations_test = forward_propagation(X_test, parametres_test)
        gradients_test = back_propagation(y_test, parametres_test, activations_test)
        parametres_test = update(gradients_test, parametres_test, learning_rate)
        Af_test = activations_test['A' + str(C)]

        # calcul du log_loss et de l'accuracy
        test_history[i, 0] = (log_loss(Af_test.flatten(), y_test.flatten()))
        y_pred_test = predict(X_test, parametres_test)
        test_history[i, 1] = (accuracy_score(y_test.flatten(), y_pred_test.flatten()))



        activations_val = forward_propagation(X_val, parametres_val)
        gradients_val = back_propagation(y_val, parametres_val, activations_val)
        parametres_val = update(gradients_val, parametres_val, learning_rate)
        Af_val = activations_val['A' + str(C)]

        # calcul du log_loss et de l'accuracy
        val_history[i, 0] = (log_loss(Af_val.flatten(), y_val.flatten()))
        y_pred_val = predict(X_val, parametres_val)
        val_history[i, 1] = (accuracy_score(y_val.flatten(), y_pred_val.flatten()))

    # Plot courbe d'apprentissage
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(training_history[:, 0], label='train loss')
    plt.plot(test_history[:, 0], label='test loss')
    plt.plot(val_history[:, 0], label='val loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(training_history[:, 1], label='train acc')
    plt.plot(test_history[:, 1], label='test acc')
    plt.plot(val_history[:, 1], label='val acc')
    plt.legend()
    plt.show()

    return training_history


def split_csv_data(data):
    # Séparation des caractéristiques (X) et de la cible (y)
    X = data.drop(columns=['diagnostic', 'id'])
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

    return X_train, X_test, y_train, y_test, X_val, y_val


if __name__ == "__main__":
    X, y = make_circles(n_samples=10, noise=0.1, factor=0.3, random_state=0)
    X = X.T
    y = y.reshape((1, y.shape[0]))

    X_train1 = X
    y_train1 = y
    
    
    print(colors.clr.fg.yellow, "Spliting csv data...", colors.clr.reset)

    data = get_csv_object.get()

    X_train, X_test, y_train, y_test, X_val, y_val = split_csv_data(data['data_std'])

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()

    X_train = X_train.T
    y_train = y_train.reshape((1, y_train.shape[0]))

    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    X_test = X_test.T
    y_test = y_test.reshape((1, y_test.shape[0]))

    X_val = X_val.to_numpy()
    y_val = y_val.to_numpy()

    X_val = X_val.T
    y_val = y_val.reshape((1, y_val.shape[0]))


    print(colors.clr.fg.green, "Split success.", colors.clr.reset)



    index = 0
    for i in y_train[0]:
      if i == 'B':
        y_train[0][index] = 0
      else:
        y_train[0][index] = 1
      index += 1

    index = 0
    for i in y_test[0]:
      if i == 'B':
        y_test[0][index] = 0
      else:
        y_test[0][index] = 1
      index += 1

    index = 0
    for i in y_val[0]:
      if i == 'B':
        y_val[0][index] = 0
      else:
        y_val[0][index] = 1
      index += 1

    y_train = np.int64(y_train)
    y_test = np.int64(y_test)
    y_val = np.int64(y_val)

    print('dimensions de X:', X_train1.shape)
    print('dimensions de y:', y_train1.shape)

    print('dimensions de X:', X_train.shape)
    print('dimensions de y:', y_train.shape)

    print(type(X_train1[0][0]))
    print(type(X_train[0][0]))

    print(type(y_train1[0][0]))
    print(type(y_train[0][0]))

   

    print(colors.clr.fg.yellow, X_train, colors.clr.reset)
    print(colors.clr.fg.red, y_train, colors.clr.reset)

    print(colors.clr.fg.yellow, X_train1, colors.clr.reset)
    print(colors.clr.fg.red, y_train1, colors.clr.reset)

    #exit(0)

    # plt.scatter(X[0, :], X[1, :], c=y, cmap='summer')
    # plt.show()

    # deep_neural_network(X_train1, y_train1, hidden_layers = (16, 16, 16), learning_rate = 0.1, n_iter = 3000)

    deep_neural_network(X_train, y_train, X_test, y_test, X_val, y_val, hidden_layers = (16, 16, 16), learning_rate = 0.1, n_iter = 3000)
    print("")
    print("")
    exit(0)