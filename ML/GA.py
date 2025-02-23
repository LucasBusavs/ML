import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
from score import pipeline_score

# Carregar dataset
dataset = pd.read_csv('docs/db/dados_preprocessados.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Divisão dos dados
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

# Normalizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Cruzamento (crossover)
# TODO: Implementar pCross e definir chromossomo onde será feito o crossover


def crossover(parent1, parent2):
    child1 = (parent1[0], parent2[1], parent1[2])
    child2 = (parent2[0], parent1[1], parent2[2])
    return child1, child2

# Algoritmo Genético Principal
# TODO: Melhorar implementação da seleção de pais
# TODO: Definir critério de parada


def genetic_algorithm(generations=10, population_size=10):
    population = create_population(population_size)

    for gen in range(generations):
        scores = [fitness_function(ind) for ind in population]
        best = select_best(population, scores)
        print(f"Geração {gen+1} - Melhor Score: {max(scores):.4f}")

        next_gen = best[:2]  # Mantém os dois melhores
        while len(next_gen) < population_size:
            parent1, parent2 = random.sample(best, 2)
            child1, child2 = crossover(parent1, parent2)
            next_gen.extend([mutate(child1), mutate(child2)])

        population = next_gen[:population_size]

    return best[0]


# Executando a otimização
best_params = genetic_algorithm()
print("Melhores hiperparâmetros encontrados:", best_params)
