from sklearn.model_selection import train_test_split
import pandas as pd
from population import Population
import time
import numpy as np

# Carregar dataset
dataset = pd.read_csv('docs/db/dados_preprocessados.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Contar o número de classes únicas
n_classes = len(np.unique(y))  # Número de classes

# Divisão dos dados
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

# Algoritmo Genético Principal
# TODO: Definir critério de parada


def generation(generations=10, popSize=20, pCross=0.8, pMutation=0.02):
    old_pop = Population(popSize, n_classes)    # Cria uma população inicial
    old_pop.fitness_function(X_train, X_test, y_train, y_test)
    maxIndiv = old_pop.individuals[0]  # Inicializa com um indivíduo válido

    for gen in range(generations):
        # Calcula o fitness da população
        print(f"Geração {gen+1}")
        print(old_pop.statistics())

        for i in range(popSize):
            if old_pop.individuals[i].fitness > maxIndiv.fitness:
                maxIndiv = old_pop.individuals[i]

        # Cria uma nova população vazia
        new_pop = Population(0, n_classes)

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
        old_pop.fitness_function(X_train, X_test, y_train, y_test)

    return maxIndiv


# Executando a otimização
start_time = time.perf_counter()
bestIndividual = generation()
bestIndividual.show_hyperparam()
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Tempo de execução: {elapsed_time:.6f} segundos")
