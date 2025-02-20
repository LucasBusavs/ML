import random


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

    # def __init__(self, hyperparam, generation=0):
    #     self.generation = generation
    #     parent1 = None
    #     parent2 = None
