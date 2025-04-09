import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Dados dos arquivos
max_iters = [1000, 3000, 4000, 5000, 7000, 10000]
csvs = [f"ml/Results/gridSearch_SVM_max{it}Test.csv" for it in max_iters]

# Carregar todos os CSVs
dfs = []
for it, file in zip(max_iters, csvs):
    df = pd.read_csv(file)
    df["max_iter"] = it
    dfs.append(df)

# Concatenar
all_data = pd.concat(dfs, ignore_index=True)

# Gráfico de linhas: Tempo de execução
plt.figure(figsize=(12, 6))
sns.lineplot(data=all_data, x="max_iter",
             y="execution_time", hue="dataset", marker="o")
plt.title("Max_iter vs Tempo de Execução (por Dataset)")
plt.xlabel("max_iter")
plt.ylabel("Tempo de Execução (s)")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.lineplot(data=all_data, x="max_iter",
             y="test_score", hue="dataset", marker="o")
plt.title("Max_iter vs Score no Teste (por Dataset)")
plt.xlabel("max_iter")
plt.ylabel("Score (Função pipeline_score)")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.lineplot(data=all_data, x="max_iter",
             y="warning_count", hue="dataset", marker="o")
plt.title("Max_iter vs warning_count (por Dataset)")
plt.xlabel("max_iter")
plt.ylabel("warning_count")
plt.grid(True)
plt.tight_layout()
plt.show()
