import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from score import pipeline_score
from sklearn.metrics import classification_report
import os
import time

# Caminho da pasta onde os datasets estão salvos
dataset_dir = "docs/db/dataSets"

# Caminho do arquivo CSV onde os resultados serão armazenados
result_csv_path = "ml/Results/randomSearch_DT_results.csv"

# Listar todos os arquivos CSV na pasta
datasets = [f for f in os.listdir(dataset_dir) if f.endswith(".csv")]

# Se o arquivo de resultados ainda não existir, cria com cabeçalho
if not os.path.exists(result_csv_path):
    pd.DataFrame(columns=["dataset", "best_params", "train_score",
                 "test_score", "execution_time"]).to_csv(result_csv_path, index=False)

# Loop para processar cada dataset
for dataset in datasets:
    dataset_path = os.path.join(dataset_dir, dataset)

    # Carregar o dataset
    df = pd.read_csv(dataset_path)

    # Exibir nome do dataset e primeiras linhas para verificação
    print(f"\nProcessando: {dataset}")

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

    # Definição do grid de hiperparâmetros
    param_grid = {
        'splitter': ['best', 'random'],   # Quantidade de árvores
        # Mínimo de amostras para dividir um nó
        'min_samples_split': [2, 5, 10, 15, 20],
        # Mínimo de amostras em uma folha
        'min_samples_leaf': [1, 2, 4, 8, 16],
        'max_features': ['sqrt', 'log2'],  # Número de features por divisão
        'criterion': ['gini', 'entropy'],  # Critério de divisão
        'class_weight': [None, 'balanced'],
    }

    # Definição do modelo
    dt = DecisionTreeClassifier(random_state=42)

    # Criar RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=dt,
        param_distributions=param_grid,
        n_iter=30,  # Número de amostras aleatórias a serem testadas
        scoring=pipeline_score,  # Métrica de avaliação
        cv=5,  # Validação cruzada com 5 folds
        random_state=42,
        n_jobs=-1,
        verbose=2
    )

    start_time = time.time()
    # Ajustar o modelo
    random_search.fit(X_train, y_train)
    end_time = time.time()
    execution_time = end_time - start_time

    # Obter os melhores parâmetros e score
    best_params = random_search.best_params_
    best_score = random_search.best_score_

    # Avaliar no conjunto de teste
    best_dt = random_search.best_estimator_
    y_pred = best_dt.predict(X_test)
    final_score = pipeline_score(y_test, y_pred)

    print("Melhores hiperparâmetros:", best_params)
    print("Score no treino:", best_score)
    print("Score no teste:", final_score)
    print(f"Tempo de execução: {execution_time:.2f} segundos")
    print("\nRelatório de classificação:")
    print(classification_report(y_test, y_pred))

    # Criar um dicionário com os resultados
    result = {
        "dataset": dataset,
        # Convertendo para string para evitar problemas de formatação
        "best_params": str(best_params),
        "train_score": best_score,
        "test_score": final_score,
        "execution_time": execution_time,
    }

    # Adicionar ao arquivo CSV imediatamente
    pd.DataFrame([result]).to_csv(result_csv_path,
                                  mode="a", header=False, index=False)
