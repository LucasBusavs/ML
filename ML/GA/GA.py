import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

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

# Função de avaliação (fitness)


def evaluate_knn(params):
    n_neighbors, weights, p = params
    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

# Criar população inicial


def create_population(size):
    return [
        (random.randint(1, 101), random.choice(
            ["uniform", "distance"]), random.choice([1, 2]))
        for _ in range(size)
    ]

# Seleção dos melhores indivíduos


def select_best(population, scores, num_best=5):
    sorted_pop = sorted(zip(population, scores),
                        key=lambda x: x[1], reverse=True)
    return [x[0] for x in sorted_pop[:num_best]]

# Cruzamento (crossover)


def crossover(parent1, parent2):
    child1 = (parent1[0], parent2[1], parent1[2])
    child2 = (parent2[0], parent1[1], parent2[2])
    return child1, child2

# Mutação


def mutate(individual, mutation_rate=0.02):
    if random.random() < mutation_rate:
        individual = (random.randint(1, 101), individual[1], individual[2])
    if random.random() < mutation_rate:
        individual = (individual[0], random.choice(
            ["uniform", "distance"]), individual[2])
    if random.random() < mutation_rate:
        individual = (individual[0], individual[1], random.choice([1, 2]))
    return individual

# Algoritmo Genético Principal


def genetic_algorithm(generations=10, population_size=10):
    population = create_population(population_size)

    for generation in range(generations):
        scores = [evaluate_knn(ind) for ind in population]
        best = select_best(population, scores)
        print(f"Geração {generation+1} - Melhor Score: {max(scores):.4f}")

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
