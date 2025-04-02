from sklearn.model_selection import train_test_split
import pandas as pd
from population import Population_KNN, Population_RF, Population_SVM
import time
import numpy as np
import json
from ucimlrepo import fetch_ucirepo

dataset = fetch_ucirepo(id=891)
X = dataset.data.features
y = dataset.data.targets
y = np.array(y).ravel()

# Carregar dataset
# dataset = pd.read_csv('docs/db/dados_preprocessados.csv')
# X = dataset.iloc[:, :-1].values
# y = dataset.iloc[:, -1].values

# Contar o número de classes únicas
n_classes = len(np.unique(y))  # Número de classes
instances = len(dataset)  # Número de instâncias

# Divisão dos dados
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Algoritmo Genético Principal
# TODO: Definir critério de parada


def generation_KNN(generations=10, popSize=20, pCross=0.8, pMutation=0.02, json_data=None):
    # Cria uma população inicial

    if json_data is None:
        old_pop = Population_KNN(popSize, n_classes, instances)
    else:
        # Convertendo JSON para lista de dicionários
        dados = json.loads(json_data)
        old_pop = Population_KNN(popSize, n_classes, instances, dados)

    # json_data = '''[
    # {"n_neighbors": 14, "weights": "distance", "p": 1},
    # {"n_neighbors": 18, "weights": "distance", "p": 2},
    # {"n_neighbors": 12, "weights": "distance", "p": 2},
    # {"n_neighbors": 31, "weights": "uniform", "p": 1},
    # {"n_neighbors": 10, "weights": "uniform", "p": 2},
    # {"n_neighbors": 37, "weights": "distance", "p": 2},
    # {"n_neighbors": 26, "weights": "distance", "p": 1},
    # {"n_neighbors": 29, "weights": "distance", "p": 1},
    # {"n_neighbors": 24, "weights": "uniform", "p": 1},
    # {"n_neighbors": 38, "weights": "uniform", "p": 1},
    # {"n_neighbors": 41, "weights": "distance", "p": 2},
    # {"n_neighbors": 15, "weights": "uniform", "p": 1},
    # {"n_neighbors": 36, "weights": "distance", "p": 2},
    # {"n_neighbors": 34, "weights": "distance", "p": 1},
    # {"n_neighbors": 15, "weights": "distance", "p": 2},
    # {"n_neighbors": 34, "weights": "uniform", "p": 1},
    # {"n_neighbors": 19, "weights": "uniform", "p": 1},
    # {"n_neighbors": 18, "weights": "distance", "p": 2},
    # {"n_neighbors": 42, "weights": "distance", "p": 2},
    # {"n_neighbors": 24, "weights": "uniform", "p": 2}
    # ]'''

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
        new_pop = Population_KNN(0, n_classes, instances)

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


def generation_RF(generations=10, popSize=30, pCross=0.8, pMutation=0.02, json_data=None):
    # Cria uma população inicial

    if json_data is None:
        old_pop = Population_RF(popSize)
    else:
        # Convertendo JSON para lista de dicionários
        dados = json.loads(json_data)
        old_pop = Population_RF(popSize, dados)

    # json_data = '''[
    # {"n_estimators": 128, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt", "criterion": "gini"},
    # {"n_estimators": 37, "min_samples_split": 15, "min_samples_leaf": 4, "max_features": "log2", "criterion": "entropy"},
    # {"n_estimators": 135, "min_samples_split": 15, "min_samples_leaf": 4, "max_features": "log2", "criterion": "entropy"},
    # {"n_estimators": 49, "min_samples_split": 20, "min_samples_leaf": 8, "max_features": "log2", "criterion": "entropy"},
    # {"n_estimators": 192, "min_samples_split": 2, "min_samples_leaf": 8, "max_features": "sqrt", "criterion": "entropy"},
    # {"n_estimators": 12, "min_samples_split": 10, "min_samples_leaf": 2, "max_features": "sqrt", "criterion": "gini"},
    # {"n_estimators": 136, "min_samples_split": 10, "min_samples_leaf": 1, "max_features": "log2", "criterion": "gini"},
    # {"n_estimators": 141, "min_samples_split": 2, "min_samples_leaf": 2, "max_features": "log2", "criterion": "entropy"},
    # {"n_estimators": 163, "min_samples_split": 15, "min_samples_leaf": 4, "max_features": "log2", "criterion": "entropy"},
    # {"n_estimators": 134, "min_samples_split": 2, "min_samples_leaf": 1, "max_features": "sqrt", "criterion": "entropy"},
    # {"n_estimators": 115, "min_samples_split": 15, "min_samples_leaf": 1, "max_features": "log2", "criterion": "entropy"},
    # {"n_estimators": 64, "min_samples_split": 2, "min_samples_leaf": 4, "max_features": "sqrt", "criterion": "entropy"},
    # {"n_estimators": 134, "min_samples_split": 5, "min_samples_leaf": 2, "max_features": "log2", "criterion": "gini"},
    # {"n_estimators": 57, "min_samples_split": 5, "min_samples_leaf": 16, "max_features": "log2", "criterion": "gini"},
    # {"n_estimators": 26, "min_samples_split": 2, "min_samples_leaf": 16, "max_features": "log2", "criterion": "entropy"},
    # {"n_estimators": 182, "min_samples_split": 10, "min_samples_leaf": 8, "max_features": "log2", "criterion": "entropy"},
    # {"n_estimators": 63, "min_samples_split": 2, "min_samples_leaf": 8, "max_features": "log2", "criterion": "gini"},
    # {"n_estimators": 133, "min_samples_split": 10, "min_samples_leaf": 4, "max_features": "sqrt", "criterion": "entropy"},
    # {"n_estimators": 21, "min_samples_split": 10, "min_samples_leaf": 4, "max_features": "log2", "criterion": "entropy"},
    # {"n_estimators": 147, "min_samples_split": 10, "min_samples_leaf": 16, "max_features": "log2", "criterion": "entropy"},
    # {"n_estimators": 145, "min_samples_split": 20, "min_samples_leaf": 1, "max_features": "log2", "criterion": "gini"},
    # {"n_estimators": 199, "min_samples_split": 5, "min_samples_leaf": 4, "max_features": "log2", "criterion": "gini"},
    # {"n_estimators": 109, "min_samples_split": 10, "min_samples_leaf": 4, "max_features": "sqrt", "criterion": "gini"},
    # {"n_estimators": 94, "min_samples_split": 15, "min_samples_leaf": 4, "max_features": "log2", "criterion": "gini"},
    # {"n_estimators": 77, "min_samples_split": 15, "min_samples_leaf": 16, "max_features": "log2", "criterion": "entropy"},
    # {"n_estimators": 37, "min_samples_split": 10, "min_samples_leaf": 4, "max_features": "log2", "criterion": "gini"},
    # {"n_estimators": 198, "min_samples_split": 15, "min_samples_leaf": 4, "max_features": "log2", "criterion": "gini"},
    # {"n_estimators": 12, "min_samples_split": 2, "min_samples_leaf": 8, "max_features": "sqrt", "criterion": "entropy"},
    # {"n_estimators": 22, "min_samples_split": 5, "min_samples_leaf": 2, "max_features": "log2", "criterion": "entropy"},
    # {"n_estimators": 111, "min_samples_split": 20, "min_samples_leaf": 2, "max_features": "sqrt", "criterion": "gini"}
    # ]'''

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
        new_pop = Population_RF(0)

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


def generation_SVM(generations=10, popSize=50, pCross=0.8, pMutation=0.02, json_data=None):
    # Cria uma população inicial

    if json_data is None:
        old_pop = Population_SVM(popSize)
    else:
        # Convertendo JSON para lista de dicionários
        dados = json.loads(json_data)
        old_pop = Population_SVM(popSize, dados)

    # json_data = '''[
    # {"C": 1000, "kernel": "rbf", "tol": 0.001, "class_weight": null, "max_iter": 5000, "gamma": 0.1},
    # {"C": 0.1, "kernel": "poly", "tol": 0.0001, "class_weight": null, "max_iter": 5000, "gamma": "scale", "degree": 4, "coef0": 1.0},
    # {"C": 1000, "kernel": "poly", "tol": 0.01, "class_weight": null, "max_iter": 1000, "gamma": "scale", "degree": 2, "coef0": -0.5},
    # {"C": 0.1, "kernel": "rbf", "tol": 0.0001, "class_weight": "balanced", "max_iter": 5000, "gamma": 0.01},
    # {"C": 1, "kernel": "poly", "tol": 0.01, "class_weight": "balanced", "max_iter": 1000, "gamma": 1, "degree": 2, "coef0": -0.5},
    # {"C": 0.1, "kernel": "linear", "tol": 0.0001, "class_weight": "balanced", "max_iter": 1000},
    # {"C": 1, "kernel": "sigmoid", "tol": 0.0001, "class_weight": "balanced", "max_iter": 1000, "gamma": 0.1},
    # {"C": 1000, "kernel": "linear", "tol": 0.01, "class_weight": "balanced", "max_iter": 5000},
    # {"C": 0.1, "kernel": "sigmoid", "tol": 0.001, "class_weight": null, "max_iter": 5000, "gamma": 1},
    # {"C": 10, "kernel": "linear", "tol": 0.0001, "class_weight": "balanced", "max_iter": 1000},
    # {"C": 1, "kernel": "rbf", "tol": 0.0001, "class_weight": null, "max_iter": 5000, "gamma": 0.1},
    # {"C": 0.1, "kernel": "sigmoid", "tol": 0.001, "class_weight": null, "max_iter": 5000, "gamma": 0.01},
    # {"C": 1000, "kernel": "sigmoid", "tol": 0.01, "class_weight": null, "max_iter": 1000, "gamma": 1},
    # {"C": 10, "kernel": "sigmoid", "tol": 0.0001, "class_weight": null, "max_iter": 1000, "gamma": 1},
    # {"C": 100, "kernel": "sigmoid", "tol": 0.001, "class_weight": null, "max_iter": 1000, "gamma": 0.1},
    # {"C": 1000, "kernel": "rbf", "tol": 0.01, "class_weight": "balanced", "max_iter": 5000, "gamma": 0.01},
    # {"C": 10, "kernel": "sigmoid", "tol": 0.001, "class_weight": null, "max_iter": 1000, "gamma": 1},
    # {"C": 0.1, "kernel": "poly", "tol": 0.001, "class_weight": "balanced", "max_iter": 1000, "gamma": "auto", "degree": 2, "coef0": 0.5},
    # {"C": 10, "kernel": "poly", "tol": 0.01, "class_weight": null, "max_iter": 1000, "gamma": 0.1, "degree": 4, "coef0": 0.5},
    # {"C": 1000, "kernel": "rbf", "tol": 0.0001, "class_weight": null, "max_iter": 5000, "gamma": 1},
    # {"C": 1, "kernel": "sigmoid", "tol": 0.0001, "class_weight": null, "max_iter": 1000, "gamma": "scale"},
    # {"C": 1000, "kernel": "poly", "tol": 0.01, "class_weight": null, "max_iter": 1000, "gamma": "auto", "degree": 4, "coef0": 0.5},
    # {"C": 0.1, "kernel": "linear", "tol": 0.01, "class_weight": null, "max_iter": 1000},
    # {"C": 10, "kernel": "rbf", "tol": 0.001, "class_weight": null, "max_iter": 5000, "gamma": "scale"},
    # {"C": 0.1, "kernel": "sigmoid", "tol": 0.01, "class_weight": null, "max_iter": 5000, "gamma": 1},
    # {"C": 0.1, "kernel": "sigmoid", "tol": 0.0001, "class_weight": "balanced", "max_iter": 5000, "gamma": 0.01},
    # {"C": 1000, "kernel": "poly", "tol": 0.001, "class_weight": "balanced", "max_iter": 1000, "gamma": "scale", "degree": 3, "coef0": -0.5},
    # {"C": 100, "kernel": "sigmoid", "tol": 0.001, "class_weight": null, "max_iter": 5000, "gamma": 1},
    # {"C": 1000, "kernel": "poly", "tol": 0.0001, "class_weight": "balanced", "max_iter": 5000, "gamma": 0.1, "degree": 4, "coef0": 0.0},
    # {"C": 10, "kernel": "linear", "tol": 0.0001, "class_weight": null, "max_iter": 5000},
    # {"C": 1000, "kernel": "sigmoid", "tol": 0.001, "class_weight": "balanced", "max_iter": 1000, "gamma": "scale"},
    # {"C": 1000, "kernel": "sigmoid", "tol": 0.0001, "class_weight": "balanced", "max_iter": 1000, "gamma": 0.1},
    # {"C": 0.1, "kernel": "sigmoid", "tol": 0.001, "class_weight": null, "max_iter": 1000, "gamma": 0.1},
    # {"C": 0.1, "kernel": "poly", "tol": 0.001, "class_weight": "balanced", "max_iter": 1000, "gamma": 0.01, "degree": 3, "coef0": 0.5},
    # {"C": 0.1, "kernel": "sigmoid", "tol": 0.01, "class_weight": "balanced", "max_iter": 5000, "gamma": 0.01},
    # {"C": 1, "kernel": "rbf", "tol": 0.0001, "class_weight": null, "max_iter": 1000, "gamma": "scale"},
    # {"C": 100, "kernel": "rbf", "tol": 0.0001, "class_weight": null, "max_iter": 1000, "gamma": 0.01},
    # {"C": 1, "kernel": "linear", "tol": 0.0001, "class_weight": "balanced", "max_iter": 1000},
    # {"C": 10, "kernel": "poly", "tol": 0.001, "class_weight": null, "max_iter": 5000, "gamma": "auto", "degree": 2, "coef0": 0.0},
    # {"C": 1000, "kernel": "poly", "tol": 0.0001, "class_weight": null, "max_iter": 1000, "gamma": 1, "degree": 2, "coef0": 0.5},
    # {"C": 10, "kernel": "rbf", "tol": 0.001, "class_weight": null, "max_iter": 1000, "gamma": 1},
    # {"C": 0.1, "kernel": "linear", "tol": 0.001, "class_weight": "balanced", "max_iter": 1000},
    # {"C": 10, "kernel": "rbf", "tol": 0.0001, "class_weight": null, "max_iter": 5000, "gamma": 0.01},
    # {"C": 10, "kernel": "sigmoid", "tol": 0.0001, "class_weight": null, "max_iter": 5000, "gamma": "auto"},
    # {"C": 1, "kernel": "poly", "tol": 0.01, "class_weight": "balanced", "max_iter": 1000, "gamma": 0.01, "degree": 2, "coef0": 1.0},
    # {"C": 1000, "kernel": "poly", "tol": 0.0001, "class_weight": null, "max_iter": 1000, "gamma": 0.01, "degree": 3, "coef0": 0.0},
    # {"C": 0.1, "kernel": "poly", "tol": 0.01, "class_weight": null, "max_iter": 1000, "gamma": "auto", "degree": 3, "coef0": 0.5},
    # {"C": 10, "kernel": "linear", "tol": 0.01, "class_weight": null, "max_iter": 5000},
    # {"C": 10, "kernel": "linear", "tol": 0.01, "class_weight": "balanced", "max_iter": 1000},
    # {"C": 1000, "kernel": "sigmoid", "tol": 0.01, "class_weight": "balanced", "max_iter": 1000, "gamma": 1}
    # ]'''

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
        new_pop = Population_SVM(0)

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


def tunning(json_KNN=None, json_RF=None, json_SVM=None, verbose=False):
    start_time = time.perf_counter()

    # Executa o algoritmo genético para cada algoritmo
    if (verbose):
        print("Executando algoritmo genético para KNN\n")
        bestInd_KNN = generation_KNN(json_data=json_KNN)
        end_time = time.perf_counter()
        elapsed_time_KNN = end_time - start_time
        print("Melhor indivíduo KNN encontrado:")
        bestInd_KNN.show_hyperparam()
        print(f"Fitness: {bestInd_KNN.fitness}")
        print(f"Tempo de execução: {elapsed_time_KNN:.6f} segundos\n")

        print("Executando algoritmo genético para RF\n")
        start_time_RF = time.perf_counter()
        bestInd_RF = generation_RF(json_data=json_RF)
        end_time = time.perf_counter()
        elapsed_time_RF = end_time - start_time_RF
        print("Melhor indivíduo RF encontrado:")
        bestInd_RF.show_hyperparam()
        print(f"Fitness: {bestInd_RF.fitness}")
        print(f"Tempo de execução: {elapsed_time_RF:.6f} segundos\n")

        print("Executando algoritmo genético para SVM\n")
        start_time_SVM = time.perf_counter()
        bestInd_SVM = generation_SVM(json_data=json_SVM)
        end_time = time.perf_counter()
        elapsed_time_SVM = end_time - start_time_SVM
        print("Melhor indivíduo SVM encontrado:")
        bestInd_SVM.show_hyperparam()
        print(f"Fitness: {bestInd_SVM.fitness}")
        print(f"Tempo de execução: {elapsed_time_SVM:.6f} segundos\n")
    else:
        bestInd_KNN = generation_KNN(json_data=json_KNN)
        bestInd_RF = generation_RF(json_data=json_RF)
        bestInd_SVM = generation_SVM(json_data=json_SVM)

    if bestInd_KNN.fitness > bestInd_RF.fitness and bestInd_KNN.fitness > bestInd_SVM.fitness:
        bestIndividual = bestInd_KNN
    elif bestInd_RF.fitness > bestInd_KNN.fitness and bestInd_RF.fitness > bestInd_SVM.fitness:
        bestIndividual = bestInd_RF
    else:
        bestIndividual = bestInd_SVM

    return bestIndividual


if __name__ == '__main__':
    start_time = time.perf_counter()

    bestIndividual = tunning(verbose=True)
    print("Melhor indivíduo geral encontrado: ")
    bestIndividual.show_hyperparam()
    print(f"Fitness: {bestIndividual.fitness}\n")

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    # Tempo de execução
    print(f"Tempo de execução total: {elapsed_time:.6f} segundos")
