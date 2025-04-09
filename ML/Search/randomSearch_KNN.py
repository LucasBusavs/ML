import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('docs/db/dados_preprocessados.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Definir espaço de busca para os hiperparâmetros do KNN
param_dist = {
    'n_neighbors': np.arange(8, 102),  # Número de vizinhos
    'weights': ['uniform', 'distance'],  # Peso das amostras
    'metric': ['euclidean', 'manhattan', 'minkowski'],  # Métrica de distância
    'p': [1, 2]
}

# Criar modelo KNN
knn = KNeighborsClassifier()

# Criar RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=knn,
    param_distributions=param_dist,
    n_iter=20,  # Número de amostras aleatórias a serem testadas
    scoring='accuracy',  # Métrica de avaliação
    cv=5,  # Validação cruzada com 5 folds
    random_state=42,
    n_jobs=-1
)

# Ajustar o modelo
random_search.fit(X_train, y_train)

# Melhor modelo encontrado
best_knn = random_search.best_estimator_
print(f"Melhores hiperparâmetros encontrados: {random_search.best_params_}")

# Avaliação no conjunto de teste
y_pred = best_knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia no conjunto de teste: {accuracy:.4f}")
