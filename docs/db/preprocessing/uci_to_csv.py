from ucimlrepo import fetch_ucirepo
import pandas as pd

# fetch dataset
dataset = fetch_ucirepo(id=329)

# # variable information
# print(dataset.variables)

# data (as pandas dataframes)
X = dataset.data.features
y = dataset.data.targets

# O dataset pode ter múltiplas partes (features e targets separados, por exemplo)
df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)

# df['age_group'] = df['age_group'].map({'Adult': 0, 'Senior': 1})

# Salvar como CSV
df.to_csv(f"docs/db/{dataset.metadata.name}.csv", index=False)

print(f"Dataset salvo como {dataset.metadata.name}.csv")
