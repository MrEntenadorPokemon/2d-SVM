import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('dataset_entrenamiento_gestos.csv')

X = df.drop('gesture_label', axis=1)
y = df['gesture_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42, stratify = y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm_model = SVC(kernel = 'rbf', decision_function_shape = 'ovr')

cv_scores = cross_val_score(svm_model, X_train_scaled, y_train, cv=5)
print(f"Scores de cada fold: {cv_scores}")
print(f"Promedio de Accuracy en CV: {cv_scores.mean():.4f}\n")

svm_model.fit(X_train_scaled, y_train)

# Evaluacion y Generacion de Metricas

y_pred = svm_model.predict(X_test_scaled)

acc_global = accuracy_score(y_test, y_pred)
print(f"~~~ Evaluacion Final ~~~")
print(f"Accuracy Global: {acc_global:.4f}\n")

print("Reporte de Clasificacion: ")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize = (8, 6))
sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues')
plt.title('Matriz de Confusion de Gestos')
plt.ylabel('Etiqueta Real (True Label)')
plt.xlabel('Etiqueta Predicha (Predicted label)')
plt.show()