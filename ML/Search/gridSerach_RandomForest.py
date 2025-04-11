from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
from score import pipeline_score
import os
import time

dataset_dir = 'docs/db/dataSets'

# Caminho do arquivo CSV onde os resultados serão armazenados
result_csv_path = "ML/Results/gridSearch_RF3_results.csv"

datasets = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]

# Se o arquivo de resultados ainda não existir, cria com cabeçalho
if not os.path.exists(result_csv_path):
    pd.DataFrame(columns=["dataset", "best_params", "train_score",
                 "test_score", "execution_time"]).to_csv(result_csv_path, index=False)

for dataset in datasets:
    if dataset == "AIDS Clinical Trials Group Study 175.csv" or dataset == "Breast Cancer Wisconsin (Diagnostic).csv":
        dataset_path = os.path.join(dataset_dir, dataset)

        # Carregar o dataset
        df = pd.read_csv(dataset_path)

        print(f"\nProcessando: {dataset}")

        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        # Dividir os dados em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42)

        # Definição do modelo
        rf = RandomForestClassifier(random_state=42)

        # Definição do grid de hiperparâmetros
        param_grid = {
            'n_estimators': list(range(10, 200, 10)),   # Quantidade de árvores
            # Mínimo de amostras para dividir um nó
            'min_samples_split': [2, 5, 10, 15, 20],
            # Mínimo de amostras em uma folha
            'min_samples_leaf': [1, 2, 4, 8, 16],
            # Número de features por divisão
            'max_features': ['sqrt', 'log2', None],
            'criterion': ['gini', 'entropy'],  # Critério de divisão
            'n_jobs': [-1],  # Usar todos os núcleos disponíveis
        }

        # Inicializando o GridSearchCV
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=5,
            n_jobs=-1,
            verbose=2,
            scoring=pipeline_score
        )

        # Início do tempo de execução
        start_time = time.time()
        # Executando o GridSearchCV
        grid_search.fit(X_train, y_train)
        end_time = time.time()
        elapsed_time = end_time - start_time
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        # Treinando o melhor modelo
        best_rf = grid_search.best_estimator_
        # Avaliando no conjunto de teste
        y_pred = best_rf.predict(X_test)
        final_score = pipeline_score(y_test, y_pred)

        print(f"\nMelhores parâmetros: {best_params}")
        print("Score no treino:", best_score)
        print("Score no teste:", final_score)
        print(f"Tempo de execução: {elapsed_time:.2f} segundos")
        print("\nRelatório de classificação:")
        print(classification_report(y_test, y_pred))

        # Criar um dicionário com os resultados
        result = {
            "dataset": dataset,
            # Convertendo para string para evitar problemas de formatação
            "best_params": str(best_params),
            "train_score": best_score,
            "test_score": final_score,
            "execution_time": elapsed_time
        }

        # Adicionar ao arquivo CSV imediatamente
        pd.DataFrame([result]).to_csv(result_csv_path,
                                      mode="a", header=False, index=False)
