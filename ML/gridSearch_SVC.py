from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import pandas as pd
from score import pipeline_score

# Carregar dataset
dataset = pd.read_csv('docs/db/dados_preprocessados.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

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
    cv=5,
    scoring=pipeline_score,
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

# Exibir melhores parâmetros
print("Melhores parâmetros:", grid_search.best_params_)
print("Melhor score:", grid_search.best_score_)

# Avaliação no conjunto de teste
y_pred = grid_search.best_estimator_.predict(X_test)
print(classification_report(y_test, y_pred))
print("Score:", pipeline_score(y_test, y_pred))
