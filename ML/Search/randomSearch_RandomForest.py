import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from score import pipeline_score
from sklearn.metrics import classification_report

dataset = pd.read_csv('docs/db/dados_preprocessados.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

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

# Definição do modelo
rf = RandomForestClassifier(random_state=42)

# Criar RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_grid,
    n_iter=50,  # Número de amostras aleatórias a serem testadas
    scoring=pipeline_score,  # Métrica de avaliação
    cv=5,  # Validação cruzada com 5 folds
    random_state=42,
    n_jobs=-1
)

# Ajustar o modelo
random_search.fit(X_train, y_train)

# Melhor modelo encontrado
best_rf = random_search.best_estimator_
print(f"Melhores hiperparâmetros encontrados: {random_search.best_params_}")

# Avaliação no conjunto de teste
y_pred = best_rf.predict(X_test)
print("\nRelatório de classificação:")
print(classification_report(y_test, y_pred))
print("Score:", pipeline_score(y_test, y_pred))
