import random
from sklearn.neighbors import KNeighborsClassifier


class Individual_KNN():

    fitness = None

    hyperparam = {
        'n_neighbors': (1, 102),  # Número de vizinhos (mínimo: 1, máximo: 50)
        'weights': ['uniform', 'distance'],  # Tipo de ponderação
        'p': [1, 2]  # Distância de Minkowski: 1 (Manhattan) ou 2 (Euclidiana)
    }

    def __init__(self):
        """
        Inicializa um indivíduo com hiperparâmetros aleatórios dentro dos intervalos definidos.
        """
        self.hyperparam = {
            'n_neighbors': random.randint(*self.hyperparam['n_neighbors']),
            'weights': random.choice(self.hyperparam['weights']),
            'p': random.choice(self.hyperparam['p'])
        }

    def get_model(self):
        """
        Retorna um modelo KNN com os hiperparâmetros do indivíduo.
        """
        return KNeighborsClassifier(**self.hyperparam)
