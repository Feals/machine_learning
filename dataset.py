from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
cdc_diabetes_health_indicators = fetch_ucirepo(id=891) 
  
# data (as pandas dataframes) 
X = cdc_diabetes_health_indicators.data.features 
y = cdc_diabetes_health_indicators.data.targets 
  
# metadata 
print(cdc_diabetes_health_indicators.metadata) 
  
# variable information 
print(cdc_diabetes_health_indicators.variables) 

from ucimlrepo import fetch_ucirepo
import pandas as pd

# Fetch dataset
cdc_diabetes_health_indicators = fetch_ucirepo(id=891)

# Data (as pandas dataframes)
X = cdc_diabetes_health_indicators.data.features
y = cdc_diabetes_health_indicators.data.targets

# Concaténer les features et les targets
full_data = pd.concat([X, y], axis=1)

# Exporter en CSV
full_data.to_csv("cdc_diabetes_health_indicators.csv", index=False)

print("Dataset exporté avec succès en tant que 'cdc_diabetes_health_indicators.csv'")
