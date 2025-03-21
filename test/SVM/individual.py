import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


class Individual_KNN():
    fitness = None
    parent1 = None
    parent2 = None

    hyperparam_dict = {
        'n_neighbors': (3, 101),  # Número de vizinhos (mínimo: 3, máximo: 101)
        'weights': ['uniform', 'distance'],  # Tipo de ponderação
        'p': [1, 2]  # Distância de Minkowski: 1 (Manhattan) ou 2 (Euclidiana)
    }

    def __init__(self, n_classes, n_instances, n_neighbors=None, weights=None, p=None):
        """
        Inicializa um indivíduo. Se os hiperparâmetros forem passados, utiliza-os. 
        Caso contrário, gera valores aleatórios.
        """
        self.n_classes = n_classes
        self.n_instances = n_instances
        self.hyperparam = {
            'n_neighbors': n_neighbors if n_neighbors is not None else self.get_valid_k(),
            'weights': weights if weights is not None else random.choice(self.hyperparam_dict['weights']),
            'p': p if p is not None else random.choice(self.hyperparam_dict['p'])
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


class Individual_RF():
    fitness = None
    parent1 = None
    parent2 = None

    hyperparam_dict = {
        'n_estimators': list(range(10, 200, 10)),   # Quantidade de árvores
        # Mínimo de amostras para dividir um nó
        'min_samples_split': [2, 5, 10, 15, 20],
        # Mínimo de amostras em uma folha
        'min_samples_leaf': [1, 2, 4, 8, 16],
        'max_features': ['sqrt', 'log2'],  # Número de features por divisão
        'criterion': ['gini', 'entropy'],  # Critério de divisão
    }

    def __init__(self, n_estimators=None, min_samples_split=None, min_samples_leaf=None, max_features=None, criterion=None):
        """
        Inicializa um indivíduo. Se os hiperparâmetros forem passados, utiliza-os. 
        Caso contrário, gera valores aleatórios.
        """
        self.hyperparam = {
            'n_estimators': n_estimators if n_estimators is not None else random.choice(self.hyperparam_dict['n_estimators']),
            'min_samples_split': min_samples_split if min_samples_split is not None else random.choice(self.hyperparam_dict['min_samples_split']),
            'min_samples_leaf': min_samples_leaf if min_samples_leaf is not None else random.choice(self.hyperparam_dict['min_samples_leaf']),
            'max_features': max_features if max_features is not None else random.choice(self.hyperparam_dict['max_features']),
            'criterion': criterion if criterion is not None else random.choice(self.hyperparam_dict['criterion'])
        }

    def mutation(self, pMutation):
        """
        Realiza a mutação de um indivíduo com probabilidade pMutation.
        """
        for param, values in self.hyperparam_dict.items():
            if random.random() < pMutation:
                while True:
                    new_value = random.choice(values)

                    if new_value != self.hyperparam[param]:
                        self.hyperparam[param] = new_value
                        break

    def get_model(self):
        """
        Retorna um modelo KNN com os hiperparâmetros do indivíduo.
        """
        return RandomForestClassifier(**self.hyperparam)

    def show_hyperparam(self):
        """
        Mostra os hiperparâmetros do indivíduo.
        """
        print(self.hyperparam)


class Individual_SVM():
    fitness = None
    parent1 = None
    parent2 = None

    hyperparam_dict = {
        'C': [0.01, 0.1, 1, 10, 100, 1000],
        'kernel': ['rbf', 'poly', 'sigmoid', 'linear'],
        # Só para rbf, poly, sigmoid
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10],
        'degree': [2, 3, 4, 5],  # Apenas para poly
        # Para poly e sigmoid
        'coef0': [-1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0],
        'tol': [1e-4, 1e-3, 1e-2],
        'class_weight': [None, 'balanced'],
        'max_iter': [1000, 5000, 10000]
    }

    def __init__(self, C=None, kernel=None, gamma=None, degree=None, coef0=None, tol=None, class_weight=None, max_iter=None):
        """
        Inicializa um indivíduo. Se os hiperparâmetros forem passados, utiliza-os. 
        Caso contrário, gera valores aleatórios.
        """
        self.hyperparam = {
            'C': C if C is not None else random.choice(self.hyperparam_dict['C']),
            'kernel': kernel if kernel is not None else random.choice(self.hyperparam_dict['kernel']),
            'gamma': gamma if gamma is not None else random.choice(self.hyperparam_dict['gamma']),
            'degree': degree if degree is not None else random.choice(self.hyperparam_dict['degree']),
            'coef0': coef0 if coef0 is not None else random.choice(self.hyperparam_dict['coef0']),
            'tol': tol if tol is not None else random.choice(self.hyperparam_dict['tol']),
            'class_weight': class_weight if class_weight is not None else random.choice(self.hyperparam_dict['class_weight']),
            'max_iter': max_iter if max_iter is not None else random.choice(self.hyperparam_dict['max_iter'])
        }

    def mutation(self, pMutation):
        """
        Realiza a mutação de um indivíduo com probabilidade pMutation.
        """
        for param, values in self.hyperparam_dict.items():
            if random.random() < pMutation:
                while True:
                    new_value = random.choice(values)

                    if new_value != self.hyperparam[param]:
                        self.hyperparam[param] = new_value
                        break

    def get_model(self):
        """
        Retorna um modelo KNN com os hiperparâmetros do indivíduo.
        """
        return SVC(**self.hyperparam)

    def show_hyperparam(self):
        """
        Mostra os hiperparâmetros do indivíduo.
        """
        print(self.hyperparam)
