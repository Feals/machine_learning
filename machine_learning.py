import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import RandomOverSampler

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    make_scorer,
    recall_score,
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    auc,
    precision_score,
    f1_score,
    balanced_accuracy_score
)

# Chargement des données
df = pd.read_csv("cdc_diabetes_health_indicators.csv")

X = df.drop('Diabetes_binary', axis=1)
y = df['Diabetes_binary']

classes = sorted(y.unique())

OverS = RandomOverSampler(random_state=42, sampling_strategy='not majority')
x_over, y_over = OverS.fit_resample(X, y)

# Division train/test
X_train, X_test, y_train, y_test = train_test_split(x_over, y_over, test_size=0.2, random_state=42)



# Chargement des pipelines pour chaque modèle
pipelines = {
    "LogisticRegression": joblib.load("logistic_pipeline.pkl"),
    "RandomForestClassifier": joblib.load("rf_pipeline.pkl"),
    "GradientBoostingClassifier": joblib.load("gb_pipeline.pkl"),
    "AdaBoostClassifier": joblib.load("adaboost_pipeline.pkl")
}

# Définition des grilles d'hyperparamètres
param_grids = {
    "LogisticRegression": {
        'classifier__C': [1],
        'classifier__solver': ['liblinear']
    },
    "RandomForestClassifier": {
        'classifier__n_estimators': [10, 50, 100, 200],
        'classifier__max_depth': [8, 10 , 12],
        'classifier__min_samples_split': [5 ,10 ,15,20]
    },
    "GradientBoostingClassifier": {
        'classifier__n_estimators': [200],
        'classifier__learning_rate': [2, 3],
        'classifier__max_depth': [5, 6]
    },
    "AdaBoostClassifier": {
        'classifier__n_estimators': [250],
        'classifier__learning_rate': [2, 3]
    }
}

best_models = {}

for model_name, pipeline in pipelines.items():
    print(f"Optimisation du modèle: {model_name}")
    param_grid = param_grids[model_name]
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring=make_scorer(accuracy_score, pos_label=1), n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_models[model_name] = grid_search.best_estimator_
    print(f"Meilleurs hyperparamètres pour {model_name}: {grid_search.best_params_}")
    print(f"Meilleur score sur l'accuracy: {grid_search.best_score_:.4f}\n")

# Évaluation des meilleurs modèles
for model_name, best_model in best_models.items():
    y_pred = best_model.predict(X_test)
    y_score = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    specificity = recall_score(y_test, y_pred, pos_label=0)

    print(f"\n📊 Évaluation du modèle optimisé: {model_name}")
    print(f"✅ Accuracy: {acc:.4f}")
    print(f"✅ Recall: {rec:.4f}")
    print(f"✅ Précision: {prec:.4f}")
    print(f"✅ F1-score: {f1:.4f}")
    print(f"✅ Balanced Accuracy: {bal_acc:.4f}")
    print(f"✅ Spécificité: {specificity:.4f}")
    print("\n", classification_report(y_test, y_pred))
    
    print(f"\nÉvaluation du modèle optimisé: {model_name}")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Matrice de Confusion - {model_name}")
    plt.xlabel("Prédictions")
    plt.ylabel("Véritables")
    plt.show()
    
    # Courbe ROC
    if y_score is not None:
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        print(f"ROC-AUC: {roc_auc:.4f}")
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlabel("Taux de faux positifs")
        plt.ylabel("Taux de vrais positifs")
        plt.title(f"Courbe ROC - {model_name}")
        plt.legend(loc="lower right")
        plt.show()
    
    joblib.dump(best_model, f"best_{model_name.lower().replace(' ', '_')}_model.pkl")

print("Tous les modèles optimisés ont été sauvegardés et évalués.")