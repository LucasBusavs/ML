import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class Individual_KNN():
    fitness = None
    parent1 = None
    parent2 = None

    hyperparam_dict = {
        'n_neighbors': (3, 101),
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }

    def __init__(self, n_classes, n_instances, n_neighbors=None, weights=None, p=None):
        """
        Initizalize a KNN individual. If hyperparameters are passed, use them.
        Otherwise, generate random values.

        If any optional hyperparameter is not passed, it will be randomly generated based on the hyperparam_dict.
        The 'n_neighbors' hyperparameter is generated to be greater than the number of classes and not divisible by the number of classes.
        The 'weights' hyperparameter is randomly selected from the list of possible values.
        The 'p' hyperparameter is randomly selected from the list of possible values.

        Parameters
        ----------
        n_classes : int  
            Number of classes in the dataset.
        n_instances : int
            Number of instances in the dataset.
        n_neighbors : int, optional
            Number of neighbors to use (default is random).
        weights : str, optional
            Weight function used in prediction (default is random).
            - 'uniform': uniform weights.
            - 'distance': weight points by the inverse of their distance.
        p : int, optional
            Power parameter for the Minkowski distance (default is random).
            - 1: Manhattan distance.
            - 2: Euclidean distance.

        Returns
        -------
        None
        """
        self.n_classes = n_classes
        self.n_instances = n_instances
        self.hyperparam = {
            'n_neighbors': n_neighbors if n_neighbors is not None else self.get_valid_k(),
            'weights': weights if weights is not None else random.choice(self.hyperparam_dict['weights']),
            'p': p if p is not None else random.choice(self.hyperparam_dict['p'])
        }

    def get_valid_k(self):
        """
        Gera um valor válido de 'n_neighbors' de acordo com as restrições.
        """

        min_k, max_k = self.hyperparam_dict['n_neighbors']

        # Garante K > n_classes e K não divisível por n_classes
        k = random.randint(max(self.n_classes + 1, min_k), max_k)

        while k % self.n_classes == 0:
            k = random.randint(max(self.n_classes + 1, min_k), max_k)

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


class Individual_DT():
    fitness = None
    parent1 = None
    parent2 = None

    hyperparam_dict = {
        'splitter': ['best', 'random'],
        'min_samples_split': [2, 5, 10, 15, 20],
        'min_samples_leaf': [1, 2, 4, 8, 16],
        'max_features': ['sqrt', 'log2'],
        'criterion': ['gini', 'entropy'],
        'class_weight': [None, 'balanced'],
    }

    def __init__(self, splitter=None, min_samples_split=None, min_samples_leaf=None, max_features=None, criterion=None, class_weight=None):
        """
        Inicializa um indivíduo. Se os hiperparâmetros forem passados, utiliza-os. 
        Caso contrário, gera valores aleatórios.
        """
        self.hyperparam = {
            'splitter': splitter if splitter is not None else random.choice(self.hyperparam_dict['splitter']),
            'min_samples_split': min_samples_split if min_samples_split is not None else random.choice(self.hyperparam_dict['min_samples_split']),
            'min_samples_leaf': min_samples_leaf if min_samples_leaf is not None else random.choice(self.hyperparam_dict['min_samples_leaf']),
            'max_features': max_features if max_features is not None else random.choice(self.hyperparam_dict['max_features']),
            'criterion': criterion if criterion is not None else random.choice(self.hyperparam_dict['criterion']),
            'class_weight': class_weight if class_weight is not None else random.choice(self.hyperparam_dict['class_weight'])
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
        return DecisionTreeClassifier(**self.hyperparam)

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
        'n_estimators': list(range(10, 200, 10)),
        'min_samples_split': [2, 5, 10, 15, 20],
        'min_samples_leaf': [1, 2, 4, 8, 16],
        'max_features': ['sqrt', 'log2', None],
        'criterion': ['gini', 'entropy'],
        'class_weight': [None, 'balanced', 'balanced_subsample'],
        'n_jobs': -1,
    }

    def __init__(self, n_estimators=None, min_samples_split=None, min_samples_leaf=None, max_features=None, criterion=None, class_weight=None, n_jobs=None):
        """
        Inicializa um indivíduo. Se os hiperparâmetros forem passados, utiliza-os. 
        Caso contrário, gera valores aleatórios.
        """
        self.hyperparam = {
            'n_estimators': n_estimators if n_estimators is not None else random.choice(self.hyperparam_dict['n_estimators']),
            'min_samples_split': min_samples_split if min_samples_split is not None else random.choice(self.hyperparam_dict['min_samples_split']),
            'min_samples_leaf': min_samples_leaf if min_samples_leaf is not None else random.choice(self.hyperparam_dict['min_samples_leaf']),
            'max_features': max_features if max_features is not None else random.choice(self.hyperparam_dict['max_features']),
            'criterion': criterion if criterion is not None else random.choice(self.hyperparam_dict['criterion']),
            'class_weight': class_weight if class_weight is not None else random.choice(self.hyperparam_dict['class_weight']),
            'n_jobs': n_jobs if n_jobs is not None else self.hyperparam_dict['n_jobs']
        }

    def mutation(self, pMutation):
        """
        Realiza a mutação de um indivíduo com probabilidade pMutation.
        """
        for param, values in self.hyperparam_dict.items():
            if param != 'n_jobs' and random.random() < pMutation:
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
        'class_weight': [None, 'balanced'],
        'max_iter': 4000
    }

    def __init__(self, C=None, kernel=None, gamma=None, degree=None, coef0=None, class_weight=None, max_iter=None):
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
            'class_weight': class_weight if class_weight is not None else random.choice(self.hyperparam_dict['class_weight']),
            'max_iter': max_iter if max_iter is not None else self.hyperparam_dict['max_iter']
        }

    def mutation(self, pMutation):
        """
        Realiza a mutação de um indivíduo com probabilidade pMutation.
        """
        for param, values in self.hyperparam_dict.items():
            if param != 'max_iter' and random.random() < pMutation:
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
