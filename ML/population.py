import random
import numpy as np
from individual import Individual_KNN
from score import pipeline_score


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

    def selecionar_melhores(self, num_selecionados):
        """
        Seleciona os melhores indivíduos com base no fitness.

        :param num_selecionados: Quantidade de indivíduos a serem mantidos.
        """
        melhores_indices = np.argsort(
            self.fitness)[-num_selecionados:]  # Ordena e pega os melhores
        self.individuos = [self.individuos[i] for i in melhores_indices]
        self.fitness = [self.fitness[i] for i in melhores_indices]

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
            if random.random() < pMutation:
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
