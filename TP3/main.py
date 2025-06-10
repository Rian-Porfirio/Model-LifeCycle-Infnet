import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_curve, auc
)

# Carregar os dados
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names).drop(columns=['mean area'])
y = data.target

# Padronização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Holdout
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y)

#  Baseline: KNN 
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

#  Novo Modelo: Random Forest com GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
}
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_

# Re-treinamento e predição
best_rf.fit(X_train, y_train)
y_pred_rf = best_rf.predict(X_test)

# Métricas individuais (holdout) para comparação clara
metrics = {
    'Modelo': ['KNN', 'Random Forest'],
    'Acurácia': [
        accuracy_score(y_test, y_pred_knn),
        accuracy_score(y_test, y_pred_rf)
    ],
    'Precisão': [
        precision_score(y_test, y_pred_knn),
        precision_score(y_test, y_pred_rf)
    ],
    'Recall': [
        recall_score(y_test, y_pred_knn),
        recall_score(y_test, y_pred_rf)
    ],
    'F1-Score': [
        f1_score(y_test, y_pred_knn),
        f1_score(y_test, y_pred_rf)
    ]
}

df_metrics = pd.DataFrame(metrics)
print("\n== Métricas no conjunto de teste (holdout) ==")
print(df_metrics.to_string(index=False))

# Validação Cruzada 
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ['accuracy', 'precision', 'recall', 'f1']

print("\nValidação Cruzada - KNN:")
cv_knn = cross_validate(knn, X_scaled, y, cv=cv, scoring=scoring)
for m in scoring:
    print(f"{m}: {np.mean(cv_knn[f'test_{m}']):.4f} (+/- {np.std(cv_knn[f'test_{m}']):.4f})")

print("\nValidação Cruzada - Random Forest:")
cv_rf = cross_validate(best_rf, X_scaled, y, cv=cv, scoring=scoring)
for m in scoring:
    print(f"{m}: {np.mean(cv_rf[f'test_{m}']):.4f} (+/- {np.std(cv_rf[f'test_{m}']):.4f})")

# Matriz de Confusão 
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix(y_test, y_pred_knn), annot=True, fmt='d', cmap='Blues')
plt.title("Matriz de Confusão - KNN")
plt.xlabel("Previsto")
plt.ylabel("Real")

plt.subplot(1, 2, 2)
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Greens')
plt.title("Matriz de Confusão - Random Forest")
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.tight_layout()
plt.show()

#  Curva ROC
y_probs_rf = best_rf.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_probs_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

y_probs_knn = knn.predict_proba(X_test)[:, 1]
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_probs_knn)
roc_auc_knn = auc(fpr_knn, tpr_knn)

plt.figure(figsize=(8, 6))
plt.plot(fpr_knn, tpr_knn, label=f'KNN (AUC = {roc_auc_knn:.2f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title("Curva ROC")
plt.xlabel("Taxa de Falsos Positivos (FPR)")
plt.ylabel("Taxa de Verdadeiros Positivos (TPR)")
plt.legend()
plt.grid(True)
plt.show()

#  Gráfico Comparativo de Métricas
df_melted = df_metrics.melt(id_vars="Modelo", var_name="Métrica", value_name="Valor")
plt.figure(figsize=(10, 6))
sns.barplot(data=df_melted, x="Métrica", y="Valor", hue="Modelo")
plt.title("Comparação de Desempenho - KNN vs Random Forest (Holdout)")
plt.ylim(0.8, 1.05)
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()
