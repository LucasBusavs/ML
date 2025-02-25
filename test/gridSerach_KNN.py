import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from score import pipeline_score
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder

# fetch dataset
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

# data (as pandas dataframes)
X = breast_cancer_wisconsin_diagnostic.data.features
y = breast_cancer_wisconsin_diagnostic.data.targets
y = y.values.flatten()

le = LabelEncoder()
y = le.fit_transform(y)

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

# Feature Scaling
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

# Definir o modelo KNN
knn = KNeighborsClassifier()

# Definir os hiperparâmetros para otimização
param_grid = {
    'n_neighbors':  list(range(3, 25)),  # Número de vizinhos
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

# Executar o GridSearchCV
grid_search.fit(X_train, y_train)

# Exibir os melhores hiperparâmetros encontrados
print("Melhores hiperparâmetros:", grid_search.best_params_)

# Avaliar o modelo no conjunto de teste
best_knn = grid_search.best_estimator_
y_pred = best_knn.predict(X_test)
print("\nRelatório de classificação:")
print(classification_report(y_test, y_pred))
print("Score:", pipeline_score(y_test, y_pred))
