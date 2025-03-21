import random
import numpy as np
from individual import Individual_KNN, Individual_RF, Individual_SVM
from sklearn.preprocessing import StandardScaler
from score import pipeline_score
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
import warnings


class Population_KNN:
    """
    Representa uma população de indivíduos para otimização de hiperparâmetros via algoritmo genético.
    """

    def __init__(self, pSize, n_classes, n_instances, individuals=None):
        """
        Inicializa uma população de indivíduos.

        :param pSize: Número de indivíduos na população.
        :param n_classes: Número de classes do problema.
        :param n_instances: Número de instâncias no dataset.
        :param gen: Número da geração.
        :param individuals: Lista opcional de indivíduos para inicialização via JSON.
        """
        self.n_classes = n_classes
        self.n_instances = n_instances
        self.pSize = pSize
        self.fitness = []

        if individuals is None:
            self.individuals = [Individual_KNN(
                n_classes, n_instances) for _ in range(pSize)]
        else:
            self.individuals = [Individual_KNN(
                n_classes, n_instances, **ind) for ind in individuals]

    def fitness_function(self, X_train, X_test, y_train, y_test):
        """
        Avalia o fitness de todos os indivíduos da população.

        :param X_train, y_train: Dados de treino.
        :param X_test, y_test: Dados de validação.
        """
        for ind in self.individuals:
            model = ind.get_model()
            model.fit(X_train, y_train)  # Treina o modelo
            y_pred = model.predict(X_test)  # Faz previsões
            score = pipeline_score(y_test, y_pred)  # Avalia o fitness
            ind.fitness = score  # Atribui o fitness ao indivíduo
            self.fitness.append(score)  # Adiciona à lista de fitness

    def select(self):
        """
        Seleciona a partir de roulette wheel selection.
        """
        partSum = 0
        sumFitness = sum(self.fitness)
        rand = random.random() * sumFitness
        for i in range(self.pSize):
            partSum += self.fitness[i]
            if partSum >= rand:
                return self.individuals[i]

    # TODO: Melhoraria com a implementação de um cromossomo binário
    def crossover(self, parent1, parent2, pCross, pMutation):
        """
        Realiza o crossover entre pares de indivíduos.
        """

        child1 = Individual_KNN(self.n_classes, self.n_instances)
        child2 = Individual_KNN(self.n_classes, self.n_instances)

        if random.random() < pCross:
            jCross = random.randint(
                1, len(self.individuals[0].hyperparam) - 1)  # Ponto de corte
        else:
            # Sem crossover
            jCross = len(self.individuals[0].hyperparam)

        for i in range(jCross):
            child1.hyperparam[list(self.individuals[0].hyperparam.keys())[
                i]] = parent1.hyperparam[list(parent1.hyperparam.keys())[i]]
            child2.hyperparam[list(self.individuals[0].hyperparam.keys())[
                i]] = parent2.hyperparam[list(parent2.hyperparam.keys())[i]]

        if jCross < len(self.individuals[0].hyperparam):
            for i in range(jCross, len(self.individuals[0].hyperparam)):
                child1.hyperparam[list(self.individuals[0].hyperparam.keys())[
                    i]] = parent2.hyperparam[list(parent2.hyperparam.keys())[i]]
                child2.hyperparam[list(self.individuals[0].hyperparam.keys())[
                    i]] = parent1.hyperparam[list(parent1.hyperparam.keys())[i]]

        child1.mutation(pMutation)
        child2.mutation(pMutation)

        child1.parent1 = parent1
        child1.parent2 = parent2
        child2.parent1 = parent1
        child2.parent2 = parent2

        return child1, child2

    def statistics(self):
        """
        Retorna métricas estatísticas da população.
        """
        return {
            "minFitness": min(self.fitness),
            "maxFitness": max(self.fitness),
            "meanFitness": np.mean(self.fitness),
            "sumFitness": sum(self.fitness)
        }


class Population_RF:
    """
    Representa uma população de indivíduos para otimização de hiperparâmetros via algoritmo genético.
    """

    def __init__(self, pSize, individuals=None):
        """
        Inicializa uma população de indivíduos.

        :param pSize: Número de indivíduos na população.
        :param n_classes: Número de classes do problema.
        :param n_instances: Número de instâncias no dataset.
        :param gen: Número da geração.
        :param individuals: Lista opcional de indivíduos para inicialização via JSON.
        """
        self.pSize = pSize
        self.fitness = []

        if individuals is None:
            self.individuals = [Individual_RF() for _ in range(pSize)]
        else:
            self.individuals = [Individual_RF(**ind) for ind in individuals]

    def fitness_function(self, X_train, X_test, y_train, y_test):
        """
        Avalia o fitness de todos os indivíduos da população.

        :param X_train, y_train: Dados de treino.
        :param X_test, y_test: Dados de validação.
        """
        for ind in self.individuals:
            model = ind.get_model()
            model.fit(X_train, y_train)  # Treina o modelo
            y_pred = model.predict(X_test)  # Faz previsões
            score = pipeline_score(y_test, y_pred)  # Avalia o fitness
            ind.fitness = score  # Atribui o fitness ao indivíduo
            self.fitness.append(score)  # Adiciona à lista de fitness

    def select(self):
        """
        Seleciona a partir de roulette wheel selection.
        """
        partSum = 0
        sumFitness = sum(self.fitness)
        rand = random.random() * sumFitness
        for i in range(self.pSize):
            partSum += self.fitness[i]
            if partSum >= rand:
                return self.individuals[i]

    # TODO: Melhoraria com a implementação de um cromossomo binário
    def crossover(self, parent1, parent2, pCross, pMutation):
        """
        Realiza o crossover entre pares de indivíduos.
        """

        child1 = Individual_RF()
        child2 = Individual_RF()

        if random.random() < pCross:
            jCross = random.randint(
                1, len(self.individuals[0].hyperparam) - 1)  # Ponto de corte
        else:
            # Sem crossover
            jCross = len(self.individuals[0].hyperparam)

        for i in range(jCross):
            child1.hyperparam[list(self.individuals[0].hyperparam.keys())[
                i]] = parent1.hyperparam[list(parent1.hyperparam.keys())[i]]
            child2.hyperparam[list(self.individuals[0].hyperparam.keys())[
                i]] = parent2.hyperparam[list(parent2.hyperparam.keys())[i]]

        if jCross < len(self.individuals[0].hyperparam):
            for i in range(jCross, len(self.individuals[0].hyperparam)):
                child1.hyperparam[list(self.individuals[0].hyperparam.keys())[
                    i]] = parent2.hyperparam[list(parent2.hyperparam.keys())[i]]
                child2.hyperparam[list(self.individuals[0].hyperparam.keys())[
                    i]] = parent1.hyperparam[list(parent1.hyperparam.keys())[i]]

        child1.mutation(pMutation)
        child2.mutation(pMutation)

        child1.parent1 = parent1
        child1.parent2 = parent2
        child2.parent1 = parent1
        child2.parent2 = parent2

        return child1, child2

    def statistics(self):
        """
        Retorna métricas estatísticas da população.
        """
        return {
            "minFitness": min(self.fitness),
            "maxFitness": max(self.fitness),
            "meanFitness": np.mean(self.fitness),
            "sumFitness": sum(self.fitness)
        }


class Population_SVM:
    """
    Representa uma população de indivíduos para otimização de hiperparâmetros via algoritmo genético.
    """

    def __init__(self, pSize, individuals=None):
        """
        Inicializa uma população de indivíduos.

        :param pSize: Número de indivíduos na população.
        :param n_classes: Número de classes do problema.
        :param n_instances: Número de instâncias no dataset.
        :param gen: Número da geração.
        :param individuals: Lista opcional de indivíduos para inicialização via JSON.
        """
        self.pSize = pSize
        self.fitness = []

        if individuals is None:
            self.individuals = [Individual_SVM() for _ in range(pSize)]
        else:
            self.individuals = [Individual_SVM(**ind) for ind in individuals]

    def fitness_function(self, X_train, X_test, y_train, y_test):
        """
        Avalia o fitness de todos os indivíduos da população.

        :param X_train, y_train: Dados de treino.
        :param X_test, y_test: Dados de validação.
        """
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UndefinedMetricWarning)

        for ind in self.individuals:
            model = ind.get_model()
            model.fit(X_train, y_train)  # Treina o modelo
            y_pred = model.predict(X_test)  # Faz previsões
            score = pipeline_score(y_test, y_pred)  # Avalia o fitness
            ind.fitness = score  # Atribui o fitness ao indivíduo
            self.fitness.append(score)  # Adiciona à lista de fitness

    def select(self):
        """
        Seleciona a partir de roulette wheel selection.
        """
        partSum = 0
        sumFitness = sum(self.fitness)
        rand = random.random() * sumFitness
        for i in range(self.pSize):
            partSum += self.fitness[i]
            if partSum >= rand:
                return self.individuals[i]

    # TODO: Melhoraria com a implementação de um cromossomo binário
    def crossover(self, parent1, parent2, pCross, pMutation):
        """
        Realiza o crossover entre pares de indivíduos.
        """

        child1 = Individual_SVM()
        child2 = Individual_SVM()

        if random.random() < pCross:
            jCross = random.randint(
                1, len(self.individuals[0].hyperparam) - 1)  # Ponto de corte
        else:
            # Sem crossover
            jCross = len(self.individuals[0].hyperparam)

        for i in range(jCross):
            child1.hyperparam[list(self.individuals[0].hyperparam.keys())[
                i]] = parent1.hyperparam[list(parent1.hyperparam.keys())[i]]
            child2.hyperparam[list(self.individuals[0].hyperparam.keys())[
                i]] = parent2.hyperparam[list(parent2.hyperparam.keys())[i]]

        if jCross < len(self.individuals[0].hyperparam):
            for i in range(jCross, len(self.individuals[0].hyperparam)):
                child1.hyperparam[list(self.individuals[0].hyperparam.keys())[
                    i]] = parent2.hyperparam[list(parent2.hyperparam.keys())[i]]
                child2.hyperparam[list(self.individuals[0].hyperparam.keys())[
                    i]] = parent1.hyperparam[list(parent1.hyperparam.keys())[i]]

        child1.mutation(pMutation)
        child2.mutation(pMutation)

        child1.parent1 = parent1
        child1.parent2 = parent2
        child2.parent1 = parent1
        child2.parent2 = parent2

        return child1, child2

    def statistics(self):
        """
        Retorna métricas estatísticas da população.
        """
        return {
            "minFitness": min(self.fitness),
            "maxFitness": max(self.fitness),
            "meanFitness": np.mean(self.fitness),
            "sumFitness": sum(self.fitness)
        }
