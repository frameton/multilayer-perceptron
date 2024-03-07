from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
#from tools import colors, load_csv, parse_csv, get_csv_object
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def model(X, W, b):
    Z = X.dot(W) + b
    A = 1 / (1 + np.exp(-Z))
    return A


def initialisation(X):
    W = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    return (W, b)


def log_loss(A, y):
    epsilon = 1e-15
    return 1 / len(y) * np.sum(-y * np.log(A + epsilon) - (1 - y) * np.log(1 - A + epsilon))


def gradients(A, X, y):
    dW = 1 / len(y) * np.dot(X.T, A - y)
    db = 1 / len(y) * np.sum(A - y)
    return (dW, db)


def update(dW, db, W, b, learning_rate):
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return (W, b)



def artificial_neurone(X_train, y_train, X_test, y_test, learning_rate=0.01, n_iter=100):
    #initialisation
    W, b = initialisation(X_train)

    train_loss = []
    train_acc = []

    test_loss = []
    test_acc = []

    for i in tqdm(range(n_iter)):
        A = model(X_train, W, b)

        if i % 10 == 0:

            train_loss.append(log_loss(A, y_train))
            y_pred = predict(X_train, W, b)
            train_acc.append(accuracy_score(y_train, y_pred))

            A_test = model(X_test, W, b)
            test_loss.append(log_loss(A_test, y_test))
            y_pred = predict(X_test, W, b)
            test_acc.append(accuracy_score(y_test, y_pred))

        dW, db = gradients(A, X_train, y_train)
        W, b = update(dW, db, W, b, learning_rate)


    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label="train loss")
    plt.plot(test_loss, label="test loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label="train acc")
    plt.plot(test_acc, label="test acc")
    plt.legend()
    plt.show()

    return (W, b)


def predict(X, W, b):
    A = model(X, W, b)
    return A >= 0.5


if __name__ == "__main__":
    
    X, y = make_blobs(n_samples=100, n_features=5, centers=2, random_state=0)
    y = y.reshape((y.shape[0], 1))

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)  # Division initiale en train (70%) et temp (30%)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # Division du reste en validation (15%) et test (15%)

    print("Dimensions de X:", X.shape)
    print("Dimensions de y:", y.shape)

    plt.scatter(X[:,0], X[:,1], c=y, cmap='summer')
    plt.show()

    W, b = artificial_neurone(X_train, y_train, X_test, y_test)

    plt.scatter(X[:,0], X[:,1], c=y, cmap='summer')
    plt.show()

    print("")
    print("")
    exit(0)