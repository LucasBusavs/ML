from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import pandas as pd
from score import pipeline_score
import os
import time

dataset_dir = 'docs/db/dataSets'

# Caminho do arquivo CSV onde os resultados serão armazenados
result_csv_path = "ml/Results/gridSearch_SVM_results.csv"

datasets = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]

# Se o arquivo de resultados ainda não existir, cria com cabeçalho
if not os.path.exists(result_csv_path):
    pd.DataFrame(columns=["dataset", "best_params", "train_score",
                 "test_score", "execution_time"]).to_csv(result_csv_path, index=False)

for dataset in datasets:
    if dataset != "CDC Diabetes Health Indicators.csv":
        dataset_path = os.path.join(dataset_dir, dataset)

        # Carregar o dataset
        df = pd.read_csv(dataset_path)

        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        # Dividir os dados em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42)

        # Criar pipeline com normalização
        pipeline = Pipeline([
            ('scaler', StandardScaler()),  # Normaliza os dados
            ('clf', SVC())  # Modelo SVM com kernel variado
        ])

        # Definir espaço de busca dos hiperparâmetros
        param_grid = {
            'clf__C': [0.01, 0.1, 1, 10, 100, 1000],
            'clf__kernel': ['rbf', 'poly', 'sigmoid', 'linear'],
            # Só para rbf, poly, sigmoid
            'clf__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10],
            'clf__degree': [2, 3, 4, 5],  # Apenas para poly
            # Para poly e sigmoid
            'clf__coef0': [-1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0],
            'clf__tol': [1e-4, 1e-3, 1e-2],
            'clf__class_weight': [None, 'balanced'],
            'clf__max_iter': [1000, 5000, 10000]
        }

        # # Filtrar combinações inválidas
        # valid_param_grid = []

        # for params in ParameterGrid(param_grid):
        #     if params['clf__kernel'] == 'linear' and params['clf__gamma'] not in ['scale', 'auto']:
        #         continue  # gamma não é usado em SVM linear
        #     if params['clf__kernel'] != 'poly' and params['clf__degree'] != 3:
        #         continue  # degree só é válido para poly
        #     if params['clf__kernel'] not in ['poly', 'sigmoid'] and params['clf__coef0'] != 0.0:
        #         continue  # coef0 só afeta poly e sigmoid
        #     valid_param_grid.append(params)

        # Criar e rodar o GridSearch
        grid_search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            scoring=pipeline_score,
            cv=2,
            n_jobs=-1,
            verbose=1
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
