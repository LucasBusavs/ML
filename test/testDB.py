from ucimlrepo import fetch_ucirepo
import pandas as pd

# fetch dataset 
estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition = fetch_ucirepo(id=544) 

# data (as pandas dataframes) 
X = estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.data.features 
y = estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.data.targets 

#print(X)
df = pd.DataFrame(y)
print(y.value_counts())

# metadata 
#print(estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.metadata) 

# variable information 
#print(estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.variables) 