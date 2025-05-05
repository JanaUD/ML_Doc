Near Neighbors with different metrics
Utilizando los datos del link: Upload data
https://www.kaggle.com/datasets/mahmoudelhemaly/students-grading-dataset?resource=download

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Cargar el dataset
df = pd.read_csv('Students_Grading_Dataset.csv')

# --- 1. Evaluar desequilibrio en 'Grade' ---
print("\n📊 Distribución de calificaciones (Grade):")
grade_distribution = df['Grade'].value_counts(normalize=True) * 100
print(grade_distribution)

plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='Grade', order=df['Grade'].value_counts().index)
plt.title("Distribución de calificaciones (Grade)")
plt.show()

# --- 2. Predecir Study_Hours_per_Week_normalized ---
# Normalizar la variable objetivo
scaler = MinMaxScaler()
df['Study_Hours_per_Week_normalized'] = scaler.fit_transform(df[['Study_Hours_per_Week']])

# Eliminar columnas no útiles para el modelo (IDs, texto, etc.)
df_numeric = df.drop([
    'Student_ID', 'First_Name', 'Last_Name', 'Email', 
    'Study_Hours_per_Week'  # Ya tenemos la normalizada
], axis=1)

# Codificar variables categóricas
df_encoded = pd.get_dummies(df_numeric, columns=[
    'Grade', 'Department', 'Gender', 'Extracurricular_Activities',
    'Internet_Access_at_Home', 'Parent_Education_Level', 'Family_Income_Level'
])

# Separar características (X) y variable objetivo (y)
X = df_encoded.drop('Study_Hours_per_Week_normalized', axis=1)
y = df_encoded['Study_Hours_per_Week_normalized']

# Dividir en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Probar diferentes valores de k y calcular MSE
k_values = range(1, 20)
mse_values = []

for k in k_values:
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_values.append(mse)

# Graficar MSE vs. k
plt.figure(figsize=(10, 5))
plt.plot(k_values, mse_values, marker='o')
plt.title("Error (MSE) vs. k en k-NN")
plt.xlabel("k (número de vecinos)")
plt.ylabel("Error (MSE)")
plt.grid()
plt.show()

# Elegir el mejor k
best_k = k_values[np.argmin(mse_values)]
print(f"\n🔍 Mejor valor de k: {best_k} (MSE mínimo: {min(mse_values):.4f})")

# --- 3. Agregar nuevas variables ---
# Crear interacciones y ratios
df_encoded['Attendance_Score_Interaction'] = df['Attendance (%)'] * df['Total_Score']
df_encoded['Sleep_Stress_Ratio'] = df['Sleep_Hours_per_Night'] / (df['Stress_Level (1-10)'] + 1e-6)

# Normalizar nuevas variables
df_encoded['Attendance_Score_Interaction'] = scaler.fit_transform(df_encoded[['Attendance_Score_Interaction']])
df_encoded['Sleep_Stress_Ratio'] = scaler.fit_transform(df_encoded[['Sleep_Stress_Ratio']])

# Actualizar X y volver a entrenar
X = df_encoded.drop('Study_Hours_per_Week_normalized', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = KNeighborsRegressor(n_neighbors=best_k)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"\n📉 MSE después de agregar nuevas variables: {mse:.4f}")

![image](https://github.com/user-attachments/assets/9be69afc-a4e4-4a2e-972e-044eb7e9ae97)

![image](https://github.com/user-attachments/assets/a667cb56-cd7b-45c7-8485-eeedb75b9508)


