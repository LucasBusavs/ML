import random
from sklearn.neighbors import KNeighborsClassifier


class Individual_KNN():
    fitness = None
    parent1 = None
    parent2 = None

    hyperparam_dict = {
        'n_neighbors': (3, 101),  # Número de vizinhos (mínimo: 3, máximo: 101)
        'weights': ['uniform', 'distance'],  # Tipo de ponderação
        'p': [1, 2]  # Distância de Minkowski: 1 (Manhattan) ou 2 (Euclidiana)
    }

    def __init__(self, n_classes, n_instances):
        """
        Inicializa um indivíduo com hiperparâmetros aleatórios dentro dos intervalos definidos.
        """
        self.n_classes = n_classes
        self.n_instances = n_instances
        self.hyperparam = {
            'n_neighbors': self.get_valid_k(),
            'weights': random.choice(self.hyperparam_dict['weights']),
            'p': random.choice(self.hyperparam_dict['p'])
        }

    def get_valid_k(self):
        """Gera um valor válido de 'n_neighbors' de acordo com as restrições."""
        min_k, max_k = self.hyperparam_dict['n_neighbors']
        # Garante K > n_classes e K não divisível por n_classes
        k = random.randint(max(self.n_classes + 1, min_k), max_k)
        while k % self.n_classes == 0:
            k = random.randint(max(self.n_classes, min_k), max_k)
        return k

    def get_model(self):
        """
        Retorna um modelo KNN com os hiperparâmetros do indivíduo.
        """
        return KNeighborsClassifier(**self.hyperparam)

    # TODO: Implementação de chromossomo binário facilitaria a mutação
    def mutation(self, pMutation):
        """
        Realiza a mutação de um indivíduo com probabilidade pMutation.
        """
        for param, values in self.hyperparam_dict.items():
            if random.random() < pMutation:
                while True:
                    if isinstance(values, tuple):  # Se for intervalo numérico
                        new_value = self.get_valid_k()
                    else:  # Se for lista de valores categóricos
                        new_value = random.choice(values)

                    if new_value != self.hyperparam[param]:
                        self.hyperparam[param] = new_value
                        break

    def show_hyperparam(self):
        """
        Mostra os hiperparâmetros do indivíduo.
        """
        print(self.hyperparam)
