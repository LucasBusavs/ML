import tpot
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

dataset_dir = 'docs/db/dataSets'

# Caminho do arquivo CSV onde os resultados serão armazenados
result_csv_path = "ML/Results/TPOT_results.csv"

datasets = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]

# Se o arquivo de resultados ainda não existir, cria com cabeçalho
if not os.path.exists(result_csv_path):
    pd.DataFrame(columns=["dataset", "best_params", "train_score",
                 "test_score", "execution_time"]).to_csv(result_csv_path, index=False)

for dataset in datasets:
    dataset_path = os.path.join(dataset_dir, dataset)
