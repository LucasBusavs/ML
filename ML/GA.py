from sklearn.model_selection import train_test_split
import pandas as pd
from population import Population
import time
import numpy as np
import json

# Carregar dataset
dataset = pd.read_csv('docs/db/dados_preprocessados.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Contar o número de classes únicas
n_classes = len(np.unique(y))  # Número de classes
instances = len(dataset)  # Número de instâncias

# Divisão dos dados
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

# Algoritmo Genético Principal
# TODO: Definir critério de parada


def generation(generations=10, popSize=30, pCross=0.8, pMutation=0.02):
    # Cria uma população inicial
    old_pop = Population(popSize, n_classes, instances)

    # json_data = '''[
    # {"n_neighbors": 21, "weights": "uniform", "p": 1},
    # {"n_neighbors": 25, "weights": "distance", "p": 2},
    # {"n_neighbors": 27, "weights": "uniform", "p": 1},
    # {"n_neighbors": 35, "weights": "distance", "p": 2},
    # {"n_neighbors": 33, "weights": "uniform", "p": 1},
    # {"n_neighbors": 37, "weights": "distance", "p": 2},
    # {"n_neighbors": 45, "weights": "uniform", "p": 1},
    # {"n_neighbors": 43, "weights": "distance", "p": 2},
    # {"n_neighbors": 51, "weights": "uniform", "p": 1},
    # {"n_neighbors": 55, "weights": "distance", "p": 2}
    # ]'''
    # # Convertendo JSON para lista de dicionários
    # dados = json.loads(json_data)
    # old_pop = Population(popSize, n_classes, instances, dados)

    maxIndiv = old_pop.individuals[0]  # Inicializa com um indivíduo válido

    for gen in range(generations):
        old_pop.fitness_function(X_train, X_test, y_train, y_test)
        # Calcula o fitness da população
        print(f"Geração {gen+1}")
        print(old_pop.statistics())

        maxIndivGen = old_pop.individuals[0]
        i = 1
        for i in range(popSize):
            if old_pop.individuals[i].fitness > maxIndivGen.fitness:
                maxIndivGen = old_pop.individuals[i]

        if maxIndivGen.fitness > maxIndiv.fitness:
            maxIndiv = maxIndivGen

        print(f"Melhor indivíduo da geração {gen+1}")
        maxIndivGen.show_hyperparam()
        print("\n")

        # Cria uma nova população vazia
        new_pop = Population(0, n_classes, instances)

        while new_pop.pSize < popSize:
            # Seleciona os pais
            mate1 = old_pop.select()

            # Garante que mate2 é diferente de mate1
            while True:
                mate2 = old_pop.select()
                if mate2 != mate1:
                    break

            # Realiza o crossover
            child1, child2 = old_pop.crossover(mate1, mate2, pCross, pMutation)
            new_pop.individuals.extend([child1, child2])

            new_pop.pSize += 2

        old_pop = new_pop

    return maxIndiv


if __name__ == '__main__':
    start_time = time.perf_counter()

    # Executa o algoritmo genético
    bestIndividual = generation()
    print("Melhor indivíduo encontrado:")
    bestIndividual.show_hyperparam()
    print(f"Fitness: {bestIndividual.fitness}")

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    # Tempo de execução
    print(f"Tempo de execução: {elapsed_time:.6f} segundos")
