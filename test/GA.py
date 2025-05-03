from sklearn.model_selection import train_test_split
import pandas as pd
from population import Population_KNN, Population_RF, Population_SVM, Population_DT
import time
import numpy as np
import json
from score import pipeline_score
import os
import io
import sys
import csv


# Algoritmo Genético Principal
# TODO: Definir critério de parada


def generation_KNN(generations=10, popSize=20, pCross=0.8, pMutation=0.02, json_data=None):
    if json_data is None:
        old_pop = Population_KNN(popSize, n_classes, instances)
    else:
        # Convertendo JSON para lista de dicionários
        dados = json.loads(json_data)
        old_pop = Population_KNN(popSize, n_classes, instances, dados)

    maxIndiv = old_pop.individuals[0]  # Inicializa com um indivíduo válido

    for gen in range(generations):
        old_pop.fitness_function(X_train, y_train)
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


def generation_DT(generations=10, popSize=20, pCross=0.8, pMutation=0.02, json_data=None):
    if json_data is None:
        old_pop = Population_DT(popSize)
    else:
        # Convertendo JSON para lista de dicionários
        dados = json.loads(json_data)
        old_pop = Population_DT(popSize, dados)

    maxIndiv = old_pop.individuals[0]  # Inicializa com um indivíduo válido

    for gen in range(generations):
        old_pop.fitness_function(X_train, y_train)
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
        new_pop = Population_DT(0)

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
    if json_data is None:
        old_pop = Population_RF(popSize)
    else:
        # Convertendo JSON para lista de dicionários
        dados = json.loads(json_data)
        old_pop = Population_RF(popSize, dados)

    maxIndiv = old_pop.individuals[0]  # Inicializa com um indivíduo válido

    for gen in range(generations):
        old_pop.fitness_function(X_train, y_train)
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
    if json_data is None:
        old_pop = Population_SVM(popSize)
    else:
        # Convertendo JSON para lista de dicionários
        dados = json.loads(json_data)
        old_pop = Population_SVM(popSize, dados)

    maxIndiv = old_pop.individuals[0]  # Inicializa com um indivíduo válido

    for gen in range(generations):
        old_pop.fitness_function(X_train, y_train)
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


def tunning(json_KNN=None, json_DT=None, json_RF=None, json_SVM=None, verbose=False):
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

        print("Executando algoritmo genético para DT\n")
        start_time_DT = time.perf_counter()
        bestInd_DT = generation_DT(json_data=json_DT)
        end_time = time.perf_counter()
        elapsed_time_DT = end_time - start_time_DT
        print("Melhor indivíduo DT encontrado:")
        bestInd_DT.show_hyperparam()
        print(f"Fitness: {bestInd_DT.fitness}")
        print(f"Tempo de execução: {elapsed_time_DT:.6f} segundos\n")

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
        bestInd_DT = generation_DT(json_data=json_DT)
        bestInd_RF = generation_RF(json_data=json_RF)
        bestInd_SVM = generation_SVM(json_data=json_SVM)

    if bestInd_KNN.fitness > bestInd_RF.fitness and bestInd_KNN.fitness > bestInd_SVM.fitness and bestInd_KNN.fitness > bestInd_DT.fitness:
        bestIndividual = bestInd_KNN
    elif bestInd_RF.fitness > bestInd_KNN.fitness and bestInd_RF.fitness > bestInd_SVM.fitness and bestInd_RF.fitness > bestInd_DT.fitness:
        bestIndividual = bestInd_RF
    elif bestInd_DT.fitness > bestInd_KNN.fitness and bestInd_DT.fitness > bestInd_SVM.fitness and bestInd_DT.fitness > bestInd_RF.fitness:
        bestIndividual = bestInd_DT
    else:
        bestIndividual = bestInd_SVM

    return bestIndividual


def capturar_e_salvar_prints_em_csv(func, *args, csv_filename="log.csv", **kwargs):
    buffer = io.StringIO()
    stdout_original = sys.stdout  # Guarda o stdout original
    sys.stdout = buffer           # Redireciona para o buffer

    try:
        resultado = func(*args, **kwargs)
    finally:
        sys.stdout = stdout_original  # Restaura o stdout original

    # Pega o conteúdo dos prints
    saida = buffer.getvalue().splitlines()

    # Salva em um CSV
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["mensagem"])
        for linha in saida:
            writer.writerow([linha])

    return resultado


if __name__ == '__main__':
    # Listar todos os arquivos CSV na pasta
    datasets = [f for f in os.listdir(
        'docs/db/dataSets') if f.endswith(".csv")]

    # Caminho do arquivo CSV onde os resultados serão armazenados
    result_csv_path = "results/GAIA_results.csv"

    # Se o arquivo de resultados ainda não existir, cria com cabeçalho
    if not os.path.exists(result_csv_path):
        pd.DataFrame(columns=["dataset", "best_params", "test_score", "execution_time"]).to_csv(
            result_csv_path, index=False)

    for dataset in datasets:
        if dataset != "CDC Diabetes Health Indicators.csv":
            print(f"\nProcessando: {dataset}")

            # Carregar o dataset
            df = pd.read_csv(f'docs/db/dataSets/{dataset}')
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values

            # Contar o número de classes únicas
            n_classes = len(np.unique(y))  # Número de classes
            instances = len(dataset)  # Número de instâncias

            # Divisão dos dados
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=42
            )

            start_time = time.perf_counter()
            bestInd = tunning()
            end_time = time.perf_counter()
            exec_time = end_time - start_time

            model = bestInd.get_model()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            final_score = capturar_e_salvar_prints_em_csv(
                pipeline_score, y_test, y_pred, verbosity=True, csv_filename=f"tabela {dataset}")

            # Criar um dicionário com os resultados
            result = {
                "dataset": dataset,
                "best_params": str(bestInd.hyperparam),
                "test_score": final_score,
                "execution_time": exec_time,
            }

            # Adicionar ao arquivo CSV imediatamente
            pd.DataFrame([result]).to_csv(result_csv_path,
                                          mode="a", header=False, index=False
                                          )
