import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

import joblib


# Définition des listes de variables
numeric_features = ['MentHlth', 'PhysHlth', 'BMI']
ordinal_features = ['GenHlth', 'Age', 'Education', 'Income']
categorical_features = ['HighBP', 'HighChol', 'CholCheck', 'Smoker', 'Stroke', 
                        'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies', 
                        'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk', 'Sex']

# Préprocesseur pour l'imputation
imputer = ColumnTransformer(
    transformers=[
        ('bin_imputer', SimpleImputer(strategy='most_frequent'), ['HighBP', 'HighChol']),
        ('cont_imputer', SimpleImputer(strategy='median'), ['BMI', 'MentHlth', 'PhysHlth', 'Age']),
        ('ord_imputer', SimpleImputer(strategy='constant', fill_value=-1), ['Education', 'Income']),
    ],
    remainder='passthrough'
)

# Pipeline pour les variables numériques
num_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Pipeline pour les variables ordinales
ord_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])


# Pipeline pour les variables catégorielles
categorical_pipeline = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combinaison des pipelines dans un ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, numeric_features),
        ('ord', ord_transformer, ordinal_features),
        ('cat', categorical_pipeline, categorical_features)
    ],
    remainder='passthrough'
)
joblib.dump(preprocessor, "preprocessor_pipeline.pkl")

# --- Pipelines pour les différents modèles ---

# 1. Pipeline pour la régression logistique
logistic_pipeline = Pipeline(steps=[
    ('imputer', imputer),
    ('classifier', LogisticRegression(class_weight='balanced'))
])
joblib.dump(logistic_pipeline, "logistic_pipeline.pkl")

# 2. Pipeline pour Random Forest
rf_pipeline = Pipeline(steps=[
    ('imputer', imputer),
    ('classifier', RandomForestClassifier(class_weight='balanced'))
])
joblib.dump(rf_pipeline, "rf_pipeline.pkl")

# 3. Pipeline pour Gradient Boosting
gb_pipeline = Pipeline(steps=[
    ('imputer', imputer),
    ('classifier', GradientBoostingClassifier())
])
joblib.dump(gb_pipeline, "gb_pipeline.pkl")

# 4.Pipeline pour AdaBoost si tu souhaites l'expérimenter également
adaboost_pipeline = Pipeline(steps=[
    ('imputer', imputer),
    ('classifier', AdaBoostClassifier())
])
joblib.dump(adaboost_pipeline, "adaboost_pipeline.pkl")

print("Tous les pipelines ont été créés et sauvegardés avec succès.")
