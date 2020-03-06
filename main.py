from IPython.display import display
import numpy as np
import pandas as pd
import abc


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
    def fit():
        """Méthode d'entrainement de l'algorithme."""

    @abc.abstractmethod
    def predict():
        """Fonction de prédiction pour une nouvelle entrée x."""


if __name__ == "__main__":
    # execute only if run as a script
    Classifier()
