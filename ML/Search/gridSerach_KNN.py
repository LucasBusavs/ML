import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from score import pipeline_score
import numpy as np
import os
import time

# Caminho da pasta onde os datasets estão salvos
dataset_dir = "docs/db/dataSets"

# Listar todos os arquivos CSV na pasta
datasets = [f for f in os.listdir(dataset_dir) if f.endswith(".csv")]

# Lista para armazenar os resultados
results = []

# Loop para processar cada dataset
for dataset_file in datasets:
    dataset_path = os.path.join(dataset_dir, dataset_file)

    # Carregar o dataset
    df = pd.read_csv(dataset_path)

    # Exibir nome do dataset e primeiras linhas para verificação
    print(f"\nProcessando: {dataset_file}")
    print(df.head())

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

    # Contabilização de labels
    qntLabel = len(np.unique(y))

    # Definir o modelo KNN
    knn = KNeighborsClassifier()

    # Definir os hiperparâmetros para otimização
    param_grid = {
        'n_neighbors':  list(range(qntLabel + 1, 101)),  # Número de vizinhos
        'weights': ['uniform', 'distance'],  # Peso das amostras
        'p': [1, 2]
    }

    # Configurar o GridSearchCV
    grid_search = GridSearchCV(
        estimator=knn,
        param_grid=param_grid,
        scoring=pipeline_score,  # Métrica para avaliação
        cv=5,  # Validação cruzada com 5 divisões
        verbose=1,  # Para exibir logs
        n_jobs=-1  # Usar todos os núcleos disponíveis
    )

    # Medir o tempo de execução do GridSearch
    start_time = time.time()
    # Executar o GridSearchCV
    grid_search.fit(X_train, y_train)
    end_time = time.time()
    execution_time = end_time - start_time

    # Obter os melhores parâmetros e score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Avaliar no conjunto de teste
    best_knn = grid_search.best_estimator_
    y_pred = best_knn.predict(X_test)
    final_score = pipeline_score(y_test, y_pred)

    print("Melhores hiperparâmetros:", best_params)
    print("Score no treino:", best_score)
    print("Score no teste:", final_score)
    print(f"Tempo de execução: {execution_time:.2f} segundos")
    print("\nRelatório de classificação:")
    print(classification_report(y_test, y_pred))

    # Armazenar os resultados
    results.append({
        "dataset": dataset_file,
        "best_params": best_params,
        "train_score": best_score,
        "test_score": final_score,
        "execution_time": execution_time
    })

# Converter resultados para DataFrame e salvar como CSV
results_df = pd.DataFrame(results)
results_df.to_csv("ml/Search/gridSearch_results_KNN.csv", index=False)

print("\nTodos os resultados foram salvos em 'grid_search_results.csv'")
