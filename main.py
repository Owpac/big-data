from IPython.display import display
import numpy as np
import pandas as pd
import abc
import matplotlib.pyplot as plt


def fit(x, y, alpha=0.00001, n_iterations=100):
    """
    C'est la fonction implémentant l'algorithme de la régression linéaire.

    parameters:
    X(DataFrame): Matrice représentant les exemples d'entrainement. Chaque ligne représente un exemple d'entrainement
    y(DataFrame): vecteur colonne représentant toutes les étiquettes des exemples d'entrainement
    """

    # Features (variables, attributs)
    # x = x

    # Vecteur cible (étiquettes)
    # y = y
    y = y.values.reshape(-1)

    # Créer une liste pour sauvegarder l'historique des coûts lors de la descente en gradient
    costs = []

    # Créer une Liste pour sauvegarder l'historique des paramètres theta lors de la descente en gradient
    thetas = []

    # Calculer le nombre d'exemples d'entrainement. Utiliser par exemple la fonction  "np.ma.size()"
    m = np.size(x, 0)

    # Calculer le nombre de features dans X (valeur de n dans le cours). Utiliser par exemple la fonction  "np.ma.size()"
    n = np.size(x, 1)

    # Générer le vecteur colonne theta initial selon le nombre de features dans X
    thetas = np.ones(n)

    for i in range(n_iterations):
        # Calculer la fonction hypothèse
        h = x.values @ thetas
        # h = pd.DataFrame(h)

        # Calculer l'erreur
        err = h - y

        # Calculer la fonction de coût J(theta)
        cost = np.sum(np.square(err)) / (2 * m)

        # Sauvegarder le coût actuel dans self.costs
        costs.append(cost)

        # Gradient descent: 1. Caclul de la dérivée (\nabla j(\theta))
        # deriv = (np.transpose(x.values) @ err) / m
        # deriv = np.sum(err) / m
        deriv = (np.transpose(x.values) @ err) / m

        # Gradient descent: 2. Mettre à jour les valeurs du vecteur theta
        thetas = thetas - alpha * deriv

        # Sauvegarder l'historique des theta
        # thetas.append(theta)

    return thetas, costs


def predict(x, thetas):
    # Implémenter la fonction hypothèse
    prediction = 0

    for i in range(len(x)):
        prediction += x[i] * thetas[i]

    return prediction


if __name__ == "__main__":
    # execute only if run as a script
    data = pd.read_csv('data.csv', delimiter=',')
    x = data.loc[:, ['median_income', 'total_rooms']]
    y = data.loc[:, ['median_house_value']]
    thetas, costs = fit(x, y)
    print(predict([8.3252, 8880.0], thetas))
