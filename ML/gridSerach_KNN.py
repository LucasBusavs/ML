from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# Carregar o dataset (substitua pelo seu dataset)
data = load_iris()
X = data.data
y = data.target

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Definir o modelo KNN
knn = KNeighborsClassifier()

# Definir os hiperparâmetros para otimização
param_grid = {
    'n_neighbors':  list(range(1, 102)),  # Número de vizinhos
    'weights': ['uniform', 'distance'],  # Peso das amostras
    'metric': ['euclidean', 'manhattan', 'minkowski']  # Métricas de distância
}

# Configurar o GridSearchCV
grid_search = GridSearchCV(
    estimator=knn,
    param_grid=param_grid,
    scoring='accuracy',  # Métrica para avaliação
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

# Exibir a acurácia no conjunto de teste
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia no conjunto de teste: {accuracy:.2f}")
