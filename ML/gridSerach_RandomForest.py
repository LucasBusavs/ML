from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
from score import pipeline_score


dataset = pd.read_csv('docs/db/dados_preprocessados.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

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
    'max_features': ['sqrt', 'log2'],  # Número de features por divisão
    'criterion': ['gini', 'entropy'],  # Critério de divisão
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

# Executando o GridSearchCV
grid_search.fit(X_train, y_train)

# Exibir os melhores hiperparâmetros encontrados
print("Melhores hiperparâmetros:", grid_search.best_params_)

# Treinando o melhor modelo
best_rf = grid_search.best_estimator_

# Avaliando no conjunto de teste
y_pred = best_rf.predict(X_test)
print("\nRelatório de classificação:")
print(classification_report(y_test, y_pred))
print("Score:", pipeline_score(y_test, y_pred))
