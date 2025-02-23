import random
from sklearn.neighbors import KNeighborsClassifier


class Individual_KNN():
    fitness = None
    parent1 = None
    parent2 = None

    hyperparam_dict = {
        'n_neighbors': (1, 102),  # Número de vizinhos (mínimo: 1, máximo: 101)
        'weights': ['uniform', 'distance'],  # Tipo de ponderação
        'p': [1, 2]  # Distância de Minkowski: 1 (Manhattan) ou 2 (Euclidiana)
    }

    def __init__(self):
        """
        Inicializa um indivíduo com hiperparâmetros aleatórios dentro dos intervalos definidos.
        """
        self.hyperparam = {
            'n_neighbors': random.randint(*self.hyperparam_dict['n_neighbors']),
            'weights': random.choice(self.hyperparam_dict['weights']),
            'p': random.choice(self.hyperparam_dict['p'])
        }

    def get_model(self):
        """
        Retorna um modelo KNN com os hiperparâmetros do indivíduo.
        """
        return KNeighborsClassifier(**self.hyperparam)

    # TODO: Implementação de chromossomo binário facilitaria a mutação

    def mutation(self, pMutation=0.02):
        """
        Realiza a mutação de um indivíduo com probabilidade pMutation.
        """
        for param, values in self.hyperparam_dict.items():
            if random.random() < pMutation:
                while True:
                    if isinstance(values, tuple):  # Se for intervalo numérico
                        new_value = random.randint(*values)
                    else:  # Se for lista de valores categóricos
                        new_value = random.choice(values)

                    if new_value != self.hyperparam[param]:
                        self.hyperparam[param] = new_value
                        break
