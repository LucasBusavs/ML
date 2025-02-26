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
import matplotlib.patches as mpatches

"""## Importing the dataset"""

dataset = pd.read_csv('docs/db/dados_preprocessados.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# data = load_iris()
# X = data.data
# y = data.target

"""## Splitting the dataset into the Training set and Test set"""

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

"""## Feature Scaling"""

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# """## PCA"""

# pca = PCA(n_components=2)
# X_train = pca.fit_transform(X_train)
# X_test = pca.fit_transform(X_test)

"""## Training the K-NN model on the Training set"""

classifier = KNeighborsClassifier(
    n_neighbors=9, weights="distance", p=1)
classifier.fit(X_train, y_train)

"""## Predicting the Test set results"""

y_pred = classifier.predict(X_test)
# print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

"""## Making the Confusion Matrix & Classification Report"""

cm = confusion_matrix(y_test, y_pred)
print(cm)

print(classification_report(y_test, y_pred))

score = pipeline_score(y_test, y_pred)
print(f"Score: {score:.2f}")

"""## PCA for visualization"""
pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test)

# Criar a figura com dois subgráficos (1 linha, 2 colunas)
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Lista de classes assumindo que variam de 0 a 6
classes = range(7)
legend_patches = [mpatches.Patch(color=plt.get_cmap(
    "tab10")(i), label=f"Classe {i}") for i in classes]

# 🔹 Gráfico 1: Valores reais
axes[0].scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test,
                cmap="tab10", edgecolors="k", alpha=0.75)
axes[0].set_title("Valores Reais")
axes[0].set_xlabel("Componente Principal 1")
axes[0].set_ylabel("Componente Principal 2")

# 🔹 Gráfico 2: Previsões do modelo
axes[1].scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_pred,
                cmap="tab10", edgecolors="k", alpha=0.75)
axes[1].set_title("Previsões do Modelo")
axes[1].set_xlabel("Componente Principal 1")
axes[1].set_ylabel("Componente Principal 2")

# Criar legenda única para ambos os gráficos
fig.legend(handles=legend_patches, title="Classes", loc="upper right")

plt.tight_layout()
plt.show()
