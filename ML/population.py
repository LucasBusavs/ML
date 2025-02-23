import random
import numpy as np
from individual import Individual_KNN
from score import pipeline_score


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Population:
    """
    Representa uma população de indivíduos para otimização de hiperparâmetros via algoritmo genético.
    """

    def __init__(self, pSize):
        """
        Inicializa uma população de indivíduos.

        :param pSize: Número de indivíduos na população.
        """
        self.pSize = pSize
        self.individuals = [Individual_KNN() for _ in range(pSize)]
        self.fitness = []  # Lista para armazenar os valores de fitness

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

    # TODO: Analisar se é a melhor forma de ser feito
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

        child1 = Individual_KNN()
        child2 = Individual_KNN()

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
