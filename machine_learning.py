import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    auc
)

# Chargement des données
df = pd.read_csv("cdc_diabetes_health_indicators.csv")

# Séparation des features et de la cible
X = df.drop('target', axis=1)  # adapter le nom de la variable cible
y = df['target']

# Conserver l'ordre et les noms des labels (par exemple "non-diabétique" et "diabétique")
classes = sorted(y.unique())

# Division train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chargement du pipeline de prétraitement
preprocessor = joblib.load("preprocessor_pipeline.pkl")

# Chargement des pipelines pour chaque modèle
pipelines = {
    "LogisticRegression": joblib.load("logistic_pipeline.pkl"),
    "RandomForestClassifier": joblib.load("rf_pipeline.pkl"),
    "GradientBoostingClassifier": joblib.load("gb_pipeline.pkl"),
    "Support_Vector_Machines": joblib.load("svm_pipeline.pkl"),
    "AdaBoostClassifier": joblib.load("adaboost_pipeline.pkl")
}

for model_name, pipeline in pipelines.items():
    print(f"\n\n=== {model_name} ===")
    
    model = pipeline.fit(X_train, y_train)
    
    # Prédictions (classes)
    y_pred = model.predict(X_test)
    
    # Rapport de classification et accuracy
    print(classification_report(y_test, y_pred, target_names=[str(c) for c in classes]))
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f"Matrice de Confusion - {model_name}")
    plt.xlabel("Prédictions")
    plt.ylabel("Véritables")
    plt.show()
    
    # Calcul des scores pour la courbe ROC (pour classification binaire)
    # On tente d'utiliser predict_proba ou decision_function
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]  # Probabilité de la classe positive
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        print(f"Le modèle {model_name} ne fournit pas de scores de probabilité/décision.")
        continue

    # Calcul et affichage de la courbe ROC et de l'AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_score, pos_label=classes[-1])
    roc_auc = auc(fpr, tpr)
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Taux de faux positifs")
    plt.ylabel("Taux de vrais positifs")
    plt.title(f"Courbe ROC - {model_name}")
    plt.legend(loc="lower right")
    plt.show()
