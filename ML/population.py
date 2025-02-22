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

    def crossover(self):
        """
        Realiza o crossover entre pares de indivíduos.
        """
        nova_geracao = []
        while len(nova_geracao) < self.tamanho:
            # Escolhe 2 indivíduos aleatórios
            pai1, pai2 = random.sample(self.individuos, 2)
            filho = self.recombinar(pai1, pai2)
            nova_geracao.append(filho)

        self.individuos = nova_geracao  # Substitui pela nova geração

    def recombinar(self, pai1, pai2):
        """
        Faz o crossover de dois indivíduos, gerando um novo.

        :param pai1, pai2: Indivíduos que serão recombinados.
        :return: Novo indivíduo gerado.
        """
        filho = Individual_KNN()
        for chave in filho.hiperparametros.keys():
            filho.hiperparametros[chave] = random.choice(
                [pai1.hiperparametros[chave], pai2.hiperparametros[chave]])
        return filho

    def mutation(self, pMutation=0.02):
        """
        Aplica mutação em todos os indivíduos da população.

        :param pMutation: Probabilidade de mutação.
        """
        for individual in self.individuals:
            individual.mutation(pMutation)

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

    def __repr__(self):
        return f"Populacao({len(self.individuos)} indivíduos)"


# Carregar dataset
dataset = pd.read_csv('docs/db/dados_preprocessados.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Divisão dos dados
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

# Normalizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

initPop = Population(10)
initPop.fitness_function(X_train, X_test, y_train, y_test)
mate1 = initPop.select()
mate2 = initPop.select()
