import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay

dataset = pd.read_csv('docs/db/dados_preprocessados.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
qntLabel = len(np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

# pca = PCA(n_components=2)
# X_train_pca = pca.fit_transform(X_train)
# X_test_pca = pca.fit_transform(X_test)

# qntK = 20
# firstK = qntLabel + 1
# count = 0
# k = firstK
# while qntK > 0:
#     k = firstK + count
#     if(k % qntLabel == 0):
#         k += 1
#         count += 1
#     print(f"K = {k}")
#     classifier = KNeighborsClassifier(n_neighbors = k, metric = 'minkowski', p = 2)
#     classifier.fit(X_train_pca, y_train)
#     y_pred = classifier.predict(X_test_pca)
#     print(classification_report(y_test,y_pred))
#     count += 1
#     qntK -= 1

# classifier = KNeighborsClassifier(n_neighbors = 7, metric = 'minkowski', p = 2)
# classifier.fit(X_train_pca, y_train)

# y_pred = classifier.predict(X_test_pca)
# print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# cm = confusion_matrix(y_test, y_pred)
# print(cm)
# accuracy_score(y_test, y_pred)

# titles_options = [
#     ("Confusion matrix, without normalization", None),
#     ("Normalized confusion matrix", "true"),
# ]
# for title, normalize in titles_options:
#     disp = ConfusionMatrixDisplay.from_estimator(
#         classifier,
#         X_test_pca,
#         y_test,
#         cmap=plt.cm.Blues,
#         normalize=normalize,
#     )
#     disp.ax_.set_title(title)