import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from score import pipeline_score

"""## Importing the dataset"""

dataset = pd.read_csv('docs/db/dados_preprocessados.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# data = load_iris()
# X = data.data
# y = data.target

"""## Splitting the dataset into the Training set and Test set"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=42)

"""## Feature Scaling"""

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# """## PCA"""

# pca = PCA(n_components=2)
# X_train = pca.fit_transform(X_train)
# X_test = pca.fit_transform(X_test)

"""## Training the K-NN model on the Training set"""

classifier = KNeighborsClassifier(n_neighbors = 8, weights = "distance", p = 1, metric="manhattan")
classifier.fit(X_train, y_train)

"""## Predicting the Test set results"""

y_pred = classifier.predict(X_test)
#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

"""## Making the Confusion Matrix & Classification Report"""

cm = confusion_matrix(y_test, y_pred)
print(cm)

print(classification_report(y_test, y_pred))

score = pipeline_score(y_test, y_pred)
print(f"Score: {score:.2f}")

# """## Visualizing"""

# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X)

# # Transformar os dados de treino e teste
# X_train_pca = pca.transform(X_train)
# X_test_pca = pca.transform(X_test)

# # Limites do gráfico baseados em todos os dados para manter escala consistente
# x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
# y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1

# # Criar grid de pontos
# xx, yy = np.meshgrid(
#     np.arange(x_min, x_max, 0.1),
#     np.arange(y_min, y_max, 0.1)
# )

# # Prever as classes para o grid
# Z = classifier.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
# Z = Z.reshape(xx.shape)

# # Configurar a figura com subplots lado a lado
# fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# # Subplot para dados de treino
# axes[0].contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
# scatter_train = axes[0].scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, edgecolor='k', cmap=plt.cm.Paired)
# axes[0].set_title("Regiões de Decisão - Dados de Treino")
# axes[0].set_xlabel("Componente Principal 1")
# axes[0].set_ylabel("Componente Principal 2")

# # Subplot para dados de teste
# axes[1].contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
# scatter_test = axes[1].scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, edgecolor='k', cmap=plt.cm.Paired)
# axes[1].set_title("Regiões de Decisão - Dados de Teste")
# axes[1].set_xlabel("Componente Principal 1")
# axes[1].set_ylabel("Componente Principal 2")

# # Ajustar espaçamento entre os subplots
# fig.tight_layout()

# # Adicionar uma barra de cores compartilhada
# cbar = fig.colorbar(scatter_train, ax=axes, orientation='vertical', fraction=0.02, pad=0.1, label="Classes")

# plt.show()