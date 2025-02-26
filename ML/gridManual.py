import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from itertools import product
from score import pipeline_score
import pandas as pd
import time

# Carregar dataset
dataset = pd.read_csv('docs/db/dados_preprocessados.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Contar o número de classes únicas
n_classes = len(np.unique(y))  # Número de classes

# Divisão dos dados
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

# Definir os hiperparâmetros para otimização
param_grid = {
    'n_neighbors':  list(range(8, 101)),  # Número de vizinhos
    'weights': ['uniform', 'distance'],  # Peso das amostras
    'p': [1, 2]
}

param_combinations = list(
    product(param_grid["n_neighbors"], param_grid["weights"], param_grid["p"]))

# Inicializar variáveis para rastrear o melhor modelo
best_score = 0
best_params = None

start_time = time.perf_counter()

# Grid Search manual
for k, weight, p in param_combinations:
    # Criar e treinar o modelo KNN com os hiperparâmetros atuais
    knn = KNeighborsClassifier(n_neighbors=k, weights=weight, p=p)
    knn.fit(X_train, y_train)

    # Fazer previsões e calcular a acurácia
    y_pred = knn.predict(X_test)
    score = pipeline_score(y_test, y_pred)

    # Atualizar se o modelo for melhor
    if score > best_score:
        best_score = score
        best_params = (k, weight, p)

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Tempo de execução: {elapsed_time:.6f} segundos")

# Exibir os melhores hiperparâmetros encontrados
print(
    f"Melhores parâmetros: k={best_params[0]}, weight={best_params[1]}, metric={best_params[2]}")
print(f"Melhor acurácia obtida: {best_score:.4f}")
