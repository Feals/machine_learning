import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Import des modèles
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC

# --- Prétraitement ---
# Listes des colonnes à adapter selon ton dataset
numeric_features = ['age', 'bmi', 'physical_health', 'mental_health']
categorical_features = ['gender', 'smoking', 'alcohol_drinking']

# Pipeline pour les variables numériques
numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Imputation par la moyenne
    ('scaler', StandardScaler())                    # Standardisation
])

# Pipeline pour les variables catégorielles
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Remplacer les valeurs manquantes
    ('encoder', OneHotEncoder(handle_unknown='ignore'))                     # Encodage OneHot
])

# Combinaison des pipelines dans un ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ]
)

# Sauvegarde du pipeline de prétraitement (pour pouvoir l'utiliser indépendamment si besoin)
joblib.dump(preprocessor, "preprocessor_pipeline.pkl")

# --- Pipelines pour les différents modèles ---

# 1. Pipeline pour la régression logistique
logistic_pipeline = Pipeline(steps=[
    ('classifier', LogisticRegression(solver='liblinear'))
])
joblib.dump(logistic_pipeline, "logistic_pipeline.pkl")

# 2. Pipeline pour Random Forest
rf_pipeline = Pipeline(steps=[
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])
joblib.dump(rf_pipeline, "rf_pipeline.pkl")

# 3. Pipeline pour Gradient Boosting
gb_pipeline = Pipeline(steps=[
    ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))
])
joblib.dump(gb_pipeline, "gb_pipeline.pkl")

# 4. Pipeline pour Support Vector Machine
svm_pipeline = Pipeline(steps=[
    ('classifier', SVC(probability=True))  # probability=True si besoin de probabilités
])
joblib.dump(svm_pipeline, "svm_pipeline.pkl")

# (Optionnel) Pipeline pour AdaBoost si tu souhaites l'expérimenter également
adaboost_pipeline = Pipeline(steps=[
    ('classifier', AdaBoostClassifier(n_estimators=50, random_state=42))
])
joblib.dump(adaboost_pipeline, "adaboost_pipeline.pkl")

print("Tous les pipelines ont été créés et sauvegardés avec succès.")
