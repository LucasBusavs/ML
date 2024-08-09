# LINK TO DB: https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition

from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# fetch dataset 
dataset = fetch_ucirepo(id=544) 

# # variable information 
# print(dataset.variables)

# data (as pandas dataframes) 
X = dataset.data.features 
y = dataset.data.targets

# Encoding Gender
categorias_ordenadas = ['Male', 'Female']
ordinal_encoder = OrdinalEncoder(categories=[categorias_ordenadas])
X.loc[:, 'Gender'] = ordinal_encoder.fit_transform(X[['Gender']])


# Mesma ordem das categorias para as seguintes colunas
categorias_ordenadas = ['no', 'yes']

# Enconding family_history_with_overweight
ordinal_encoder = OrdinalEncoder(categories=[categorias_ordenadas])
X.loc[:, 'family_history_with_overweight'] = ordinal_encoder.fit_transform(X[['family_history_with_overweight']])

# Enconding FAVC
ordinal_encoder = OrdinalEncoder(categories=[categorias_ordenadas])
X.loc[:, 'FAVC'] = ordinal_encoder.fit_transform(X[['FAVC']])

# Enconding SMOKE
ordinal_encoder = OrdinalEncoder(categories=[categorias_ordenadas])
X.loc[:, 'SMOKE'] = ordinal_encoder.fit_transform(X[['SMOKE']])

# Enconding SCC
ordinal_encoder = OrdinalEncoder(categories=[categorias_ordenadas])
X.loc[:, 'SCC'] = ordinal_encoder.fit_transform(X[['SCC']])


# Mesma ordem das categorias para as seguintes colunas
categorias_ordenadas = ['no', 'Sometimes', 'Frequently', 'Always']

# Enconding CAEC
ordinal_encoder = OrdinalEncoder(categories=[categorias_ordenadas])
X.loc[:, 'CAEC'] = ordinal_encoder.fit_transform(X[['CAEC']])

# Enconding CALC
ordinal_encoder = OrdinalEncoder(categories=[categorias_ordenadas])
X.loc[:, 'CALC'] = ordinal_encoder.fit_transform(X[['CALC']])

# HotEnconding MTRANS
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [15])], remainder='passthrough')
X = ct.fit_transform(X)

# Encoding target
categorias_ordenadas = ['Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', 'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']
ordinal_encoder = OrdinalEncoder(categories=[categorias_ordenadas])
y = ordinal_encoder.fit_transform(y)

# Renomeando colunas do X
X = pd.DataFrame(X)
novos_nomes_X = ['Trans1', 'Trans2', 'Trans3', 'Trans4', 'Trans5', 'Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
                 'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC']
X.columns = novos_nomes_X

# Renomear a coluna do y se for um DataFrame
y = pd.DataFrame(y, columns=['NObeyesdad'])

# Salvar X
pd.DataFrame(X).to_csv('docs/db/X_preprocessed.csv', index=False)

# Salvar y
pd.DataFrame(y).to_csv('docs/db/y_preprocessed.csv', index=False)