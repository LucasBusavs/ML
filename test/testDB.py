from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# fetch dataset 
db = fetch_ucirepo(id=544) 

# data (as pandas dataframes) 
X = db.data.features 
y = db.data.targets 

#print(X)
categorias_ordenadas = ['Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', 'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']

ordinal_encoder = OrdinalEncoder(categories=[categorias_ordenadas])
y['NObeyesdad'] = ordinal_encoder.fit_transform(y[['NObeyesdad']])

categorias_ordenadas = ['Male', 'Female']
ordinal_encoder = OrdinalEncoder(categories=[categorias_ordenadas])
X['Gender'] = ordinal_encoder.fit_transform(X[['Gender']])

# variable information 
print(db.variables)

categorias_ordenadas = ['no', 'Sometimes', 'Frequently', 'Always']
ordinal_encoder = OrdinalEncoder(categories=[categorias_ordenadas])
X['CAEC'] = ordinal_encoder.fit_transform(X[['CAEC']])

print(X['CAEC'])