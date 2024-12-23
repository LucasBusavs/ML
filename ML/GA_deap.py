import random
from deap import base, creator, tools, algorithms
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
import pandas as pd

"""## Search Space"""

n_neighbors =  list(range(1, 101))
weights_options = ['uniform', 'distance']
p_options = [1, 2]  # 1 para Manhattan, 2 para Euclidiana

"""## Dataset"""

# data = load_iris()
# X = data.data
# y = data.target

dataset = pd.read_csv('docs/db/dados_preprocessados.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

"""## Fitness Function"""

def fitness_function(individual):
    n_neighbors_val = individual[0]
    weight_val = individual[1]
    p_val = individual[2]

    model = KNeighborsClassifier(n_neighbors=n_neighbors_val, weights=weight_val, p=p_val)

    # Validar o modelo utilizando validação cruzada (StratifiedKFold garante a divisão balanceada de classes)
    cv = StratifiedKFold(n_splits=5)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

    return np.mean(scores),  # O retorno deve ser uma tupla

"""## Individual Creation Function"""

def create_individual():
    n_neighbors_val = random.randint(1, 100)

    # De acordo com o search space
    weight_val = random.choice(weights_options)

    # De acordo com o search space
    p_val = random.choice(p_options)

    #print(f"Indivíduo criado com n_neighbors = {n_neighbors_val}, weights = {weight_val}, p = {p_val}")
    return [n_neighbors_val, weight_val, p_val]

"""## Mutation Function"""

def mutate(individual):
    # Garantir que o valor gerado para n_neighbors esteja dentro do intervalo
    n_neighbors_val = random.randint(1, 100)

    # De acordo com o search space
    weight_val = random.choice(weights_options)

    ## De acordo com o search space
    p_val = random.choice(p_options)

    # Substituir o valor de n_neighbors, weights e p pelo novo valor
    individual[0] = n_neighbors_val
    individual[1] = weight_val
    individual[2] = p_val

    print(f"Indivíduo mutado para n_neighbors = {n_neighbors_val}, weights = {weight_val}, p = {p_val}")
    return individual,

"""## Crossover Function"""

def crossover(ind1, ind2):
    print(f"Antes do crossover: {ind1[0]} - {ind2[0]} | {ind1[1]} - {ind2[1]} | {ind1[2]} - {ind2[2]}")

    # trocando os valores de n_neighbors, weights e p
    ind1[0], ind2[0] = ind2[0], ind1[0]
    ind1[1], ind2[1] = ind2[1], ind1[1]
    ind1[2], ind2[2] = ind2[2], ind1[2]

    print(f"Após crossover: {ind1[0]} - {ind2[0]} | {ind1[1]} - {ind2[1]} | {ind1[2]} - {ind2[2]}")
    return ind1, ind2

"""## Problem Definition"""

creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximizar a fitness
creator.create("Individual", list, fitness=creator.FitnessMax)

"""## Initial Population"""

population = [creator.Individual(create_individual()) for _ in range(50)]

"""## Configurations"""

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", crossover)  # Usando a função de crossover personalizada
toolbox.register("mutate", mutate)  # Usando a função de mutação personalizada
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", fitness_function)

"""## Execution"""

algorithms.eaSimple(population, toolbox, cxpb=0.7, mutpb=0.2, ngen=40, stats=None, halloffame=None, verbose=True)

best_individual = tools.selBest(population, 1)[0]
print(f"Melhor valor de K encontrado: {best_individual[0]}, melhor valor de weights: {best_individual[1]}, melhor valor de p: {best_individual[2]}")