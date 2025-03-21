from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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

# Definindo o pipeline com normalização (SVM é sensível a escala)
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Normaliza os dados
    ('clf', LinearSVC())  # Modelo SVM linear
])

# Definição do grid de hiperparâmetros
param_grid = {
    'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'clf__penalty': ['l1', 'l2'],
    'clf__loss': ['hinge', 'squared_hinge'],
    'clf__dual': [True, False],  # Lembrando da restrição de L1 com dual=False
    'clf__tol': [1e-5, 1e-4, 1e-3, 1e-2],
    'clf__class_weight': [None, 'balanced'],
    'clf__max_iter': [1000, 5000, 10000]
}

# # Gerar combinações válidas removendo L1 com dual=True
# valid_param_grid = []
# for params in ParameterGrid(param_grid):
#     if params['clf__penalty'] == 'l1' and params['clf__dual']:
#         continue  # Remove combinação inválida
#     valid_param_grid.append(params)

# Criando e rodando o GridSearch
grid_search = GridSearchCV(
    pipeline,
    param_grid=param_grid,  # Agora é uma lista de dicionários válidos
    cv=5,
    scoring=pipeline_score,
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

# Exibindo os melhores parâmetros
print("Melhores parâmetros:", grid_search.best_params_)
print("Melhor score:", grid_search.best_score_)

# Avaliação no conjunto de teste
y_pred = grid_search.best_estimator_.predict(X_test)
print(classification_report(y_test, y_pred))
print("Score:", pipeline_score(y_test, y_pred))
