Near Neighbors with different metrics
Utilizando los datos del link: Upload data
https://www.kaggle.com/datasets/mahmoudelhemaly/students-grading-dataset?resource=download

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Cargar datos
df = pd.read_csv('Students_Grading_Dataset.csv')

# 2. Variable objetivo (normalizada) y predictoras
df['Study_Hours_per_Week_Normalized'] = (df['Study_Hours_per_Week'] - df['Study_Hours_per_Week'].mean()) / df['Study_Hours_per_Week'].std()
target = 'Study_Hours_per_Week_Normalized'

# 3. Eliminar columnas irrelevantes
X = df.drop(columns=[target, 'Student_ID', 'First_Name', 'Last_Name', 'Email', 'Study_Hours_per_Week'])
y = df[target]

# 4. Identificar columnas numéricas y categóricas
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# 5. Preprocesamiento
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

# 6. Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Evaluar KNN con distintos k y métricas
results = []
metrics = ['euclidean', 'manhattan', 'chebyshev']
k_values = range(1, 21)

for metric in metrics:
    for k in k_values:
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('knn', KNeighborsRegressor(n_neighbors=k, metric=metric))
        ])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        results.append({'k': k, 'metric': metric, 'mse': mse})

# 8. Convertir a DataFrame
results_df = pd.DataFrame(results)

# 9. Gráfica de MSE
plt.figure(figsize=(10, 6))
for metric in metrics:
    subset = results_df[results_df['metric'] == metric]
    plt.plot(subset['k'], subset['mse'], marker='o', label=f'Métrica: {metric}')
    
plt.title('MSE vs Número de Vecinos (k)')
plt.xlabel('Número de Vecinos (k)')
plt.ylabel('Error Cuadrático Medio (MSE)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 10. Mostrar el mejor modelo
best_result = results_df.loc[results_df['mse'].idxmin()]
print("\n🔍 Mejor combinación:")
print(best_result)
