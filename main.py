from IPython.display import display
import numpy as np
import pandas as pd
import abc
import matplotlib.pyplot as plt


class Classifier(metaclass=abc.ABCMeta):
    def __init__(self, cls_type="Generic"):
        self.cls_type = cls_type

    def __str__(self):
        return "{} classifier".format(self.cls_type)

    def __repr__(self):
        return "{} classifier".format(self.cls_type)

    def _ipython_display_(self):
        return "{} classifier".format(self.cls_type)

    def __getattr__(self, name):
        print("{} don't exist in {} classifier".format(name, self.cls_type))

    @abc.abstractmethod
    def fit(self, x, y):
        """Méthode d'entrainement de l'algorithme."""

    @abc.abstractmethod
    def predict(self, x):
        """Fonction de prédiction pour une nouvelle entrée x."""


class LinearClassifier(Classifier):

    def __init__(self, alpha=0.05, n_iterations=1):
        Classifier.__init__(self, "Linear Regression")
        self.alpha = alpha
        self.steps = 0
        self.n_iterations = n_iterations
        self.x = []
        self.y = []
        self.thetas = []
        self.costs = []

    def fit(self, x, y):
        """
        C'est la fonction implémentant l'algorithme de la régression linéaire.

        parameters:
        X(DataFrame): Matrice représentant les exemples d'entrainement. Chaque ligne représente un exemple d'entrainement
        y(DataFrame): vecteur colonne représentant toutes les étiquettes des exemples d'entrainement
        """

        self.steps = 0

        # Features (variables, attributs)
        self.x = x

        # Vecteur cible (étiquettes)
        self.y = y

        # Créer une liste pour sauvegarder l'historique des coûts lors de la descente en gradient
        # self.costs = ?

        # Créer une Liste pour sauvegarder l'historique des paramètres theta lors de la descente en gradient
        # self.thetas = ?

        # Calculer le nombre d'exemples d'entrainement. Utiliser par exemple la fonction  "np.ma.size()"
        m = np.size(x, 0)

        # Calculer le nombre de features dans X (valeur de n dans le cours). Utiliser par exemple la fonction  "np.ma.size()"
        n = np.size(x, 1)

        # Générer le vecteur colonne theta initial selon le nombre de features dans X
        self.thetas = np.ones(n)
        pd.DataFrame(self.thetas).to_numpy()

        for i in range(self.n_iterations):
            # Calculer la fonction hypothèse
            display(self.x)
            display(self.thetas)
            h = self.x @ self.thetas
            pd.DataFrame({'h': h})
            # Calculer l'erreur
            display(h)
            display(self.y)
            err = np.subtract(h, self.y)

            # Calculer la fonction de coût J(theta)
            cost = np.sum(np.square(err)) / (2 * m)

            # Sauvegarder le coût actuel dans self.costs
            self.costs.append(cost)

            # Gradient descent: 1. Caclul de la dérivée (\nabla j(\theta))
            deriv = (np.transpose(x) @ err) / m

            # Gradient descent: 2. Mettre à jour les valeurs du vecteur theta
            theta = self.thetas[i] - self.alpha * deriv

            # Sauvegarder l'historique des theta
            self.thetas = theta

        # axes = plt.axes()
        # axes.grid()
        # plt.scatter(self.x, self.y, axes=axes)
        # plt.plot(self.x, self.thetas, c='r', axes=axes)
        # plt.show()

    def predict(self, x):
        # Implémenter la fonction hypothèse
        prediction = 0

        for i in range(len(x)):
            prediction += x[i] * self.thetas[i]

        return prediction


if __name__ == "__main__":
    # execute only if run as a script
    data = pd.read_csv('data.csv', delimiter=',')
    x = data.loc[:, ['median_income', 'total_rooms']]
    y = data.loc[:, ['median_house_value']]
    test = LinearClassifier()
    test.fit(x, y)
    print(test.predict([[8.3000, 880]]))
