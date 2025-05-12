# -*- coding: utf-8 -*-
"""KNN_Regresion_Analisis_Estadistico_Completo_Corregido.ipynb

Código completo con:
- Regresión KNN funcional
- Análisis estadístico completo
"""

# 1. Instalación de dependencias
!pip install polars scikit-learn matplotlib seaborn imbalanced-learn --quiet
import os

# 2. Importación de librerías
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from google.colab import files
from sklearn.utils import resample
import pandas as pd

# 3. Configuración de estilos
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid", palette="husl")
%matplotlib inline

# 4. Función para cargar datos
def load_data():
    """Carga datos con múltiples opciones."""
    print("\n" + "="*50)
    print(" OPCIONES DE CARGA DE DATOS ".center(50, "="))
    print("="*50)
    print("1. Usar ruta por defecto (/content/Students_Grading_Dataset.csv)")
    print("2. Ingresar ruta manualmente")
    print("3. Subir archivo manualmente")
    
    choice = input("\nSeleccione opción (1-3): ")
    
    try:
        if choice == "1":
            path = "/content/Students_Grading_Dataset.csv"
        elif choice == "2":
            path = input("Ingrese la ruta completa al archivo CSV: ").strip()
        elif choice == "3":
            print("\nPor favor, suba su archivo CSV:")
            uploaded = files.upload()
            path = list(uploaded.keys())[0]
        else:
            raise ValueError("Opción no válida")
        
        df = pl.read_csv(path)
        print(f"\n✅ Datos cargados exitosamente desde: {path}")
        print(f"📊 Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")
        return df
        
    except Exception as e:
        print(f"\n❌ Error al cargar datos: {str(e)}")
        raise

# 5. Carga de datos
try:
    df = load_data()
    print("\n🔍 Muestra de datos (5 primeras filas):")
    print(df.head())
except Exception as e:
    print(f"\n⚠️ No se pudo cargar el archivo. Error: {e}")
    raise

# 6. Análisis Exploratorio Inicial
print("\n" + "="*50)
print(" ANÁLISIS EXPLORATORIO ".center(50, "="))
print("="*50)

# 6.1 Tipos de datos
plt.figure(figsize=(12, 4))
dtypes = [str(dt) for dt in df.schema.values()]
sns.barplot(x=list(df.columns), y=dtypes, palette="Blues_d")
plt.title('Tipos de Datos por Columna', pad=15)
plt.xticks(rotation=45)
plt.ylabel('Tipo de Dato')
plt.tight_layout()
plt.show()

# 6.2 Selección de variable objetivo
def select_target(df):
    """Selección interactiva de variable objetivo."""
    numeric_cols = [col for col in df.columns 
                   if df[col].dtype in (pl.Float64, pl.Float32, pl.Int64, pl.Int32)]
    
    if not numeric_cols:
        raise ValueError("No se encontraron columnas numéricas")
    
    # Visualización de distribuciones
    print("\n📊 Distribuciones de variables numéricas:")
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot((len(numeric_cols)//3)+1, 3, i)
        sns.histplot(df[col].to_numpy(), kde=True, color='skyblue')
        plt.title(f'Distribución de {col}', fontsize=10)
    plt.tight_layout()
    plt.show()
    
    # Selección interactiva
    print("\n🔢 Seleccione la columna objetivo:")
    for i, col in enumerate(numeric_cols, 1):
        print(f"{i}. {col}")
    
    while True:
        try:
            choice = int(input("\nIngrese el número de la columna objetivo: "))
            if 1 <= choice <= len(numeric_cols):
                selected_col = numeric_cols[choice-1]
                
                if df[selected_col].dtype in (pl.Int32, pl.Int64):
                    print(f"⚠️ Convirtiendo '{selected_col}' a float...")
                    df = df.with_columns(pl.col(selected_col).cast(pl.Float64))
                
                return df, selected_col
            print("❌ Número fuera de rango. Intente nuevamente.")
        except ValueError:
            print("❌ Ingrese un número válido.")

try:
    df, TARGET_COL = select_target(df)
    print(f"\n🎯 Columna objetivo seleccionada: '{TARGET_COL}'")
    
    # Visualización de la variable objetivo
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.boxplot(y=df[TARGET_COL].to_numpy(), color='lightblue')
    plt.title('Diagrama de Caja')
    
    plt.subplot(1, 2, 2)
    sns.histplot(df[TARGET_COL].to_numpy(), kde=True, color='lightgreen')
    plt.title('Distribución')
    plt.tight_layout()
    plt.show()
    
except Exception as e:
    print(f"\n❌ Error en selección de variable objetivo: {e}")
    raise

# 7. Ingeniería de Características
print("\n" + "="*50)
print(" INGENIERÍA DE CARACTERÍSTICAS ".center(50, "="))
print("="*50)

# 7.1 Eliminar columnas irrelevantes
irrelevant_cols = ['Student_ID', 'First_Name', 'Last_Name', 'Email']
df = df.drop([col for col in irrelevant_cols if col in df.columns])

# 7.2 Procesamiento de variables categóricas
categorical_cols = [col for col in df.columns if df[col].dtype == pl.Utf8 and col != TARGET_COL]
print(f"🧠 Variables categóricas a transformar: {categorical_cols}")

if categorical_cols:
    # Guardar nombres originales para referencia
    original_categorical_cols = categorical_cols.copy()
    
    # Aplicar one-hot encoding
    df = df.to_dummies(columns=categorical_cols)
    
    # Obtener nuevas columnas dummy creadas
    dummy_cols = [col for col in df.columns 
                 if any(col.startswith(cat_col) for cat_col in original_categorical_cols)]
    
    print(f"\n📐 Nuevas columnas tras one-hot encoding: {df.shape[1]}")
    print("\n🔍 Muestra de columnas transformadas:")
    
    # Mostrar solo algunas columnas dummy como ejemplo (máximo 5)
    sample_dummy_cols = dummy_cols[:min(5, len(dummy_cols))]
    print(df.select(sample_dummy_cols).head())

# 8. Análisis Estadístico Completo
print("\n" + "="*50)
print(" ANÁLISIS ESTADÍSTICO ".center(50, "="))
print("="*50)

# 8.1 Función para identificar columnas numéricas
def get_numeric_cols(df):
    """Identifica columnas numéricas excluyendo la objetivo."""
    numeric_types = (pl.Int8, pl.Int16, pl.Int32, pl.Int64, 
                    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                    pl.Float32, pl.Float64)
    return [col for col in df.columns 
            if df[col].dtype in numeric_types and col != TARGET_COL]

numeric_cols = get_numeric_cols(df)
if not numeric_cols:
    raise ValueError("No se encontraron columnas numéricas para análisis")

print(f"🔢 Columnas numéricas para análisis: {numeric_cols}")

# 8.2 Matriz de Correlación (con tamaño de letra ajustado)
print("\n📊 Matriz de Correlación:")
corr_matrix = df.select(numeric_cols + [TARGET_COL]).to_pandas().corr()

plt.figure(figsize=(12, 10))
heatmap = sns.heatmap(
    corr_matrix,
    annot=True,
    annot_kws={'size': 8},  # Tamaño de fuente reducido
    fmt=".2f",
    cmap='coolwarm',
    center=0,
    vmin=-1,
    vmax=1,
    linewidths=0.5,
    cbar_kws={"label": "Coeficiente de Correlación", "shrink": 0.75}
)

# Ajustar tamaño de letra de los ejes
heatmap.set_xticklabels(heatmap.get_xticklabels(), 
                       rotation=45, 
                       ha='right',
                       fontsize=9)

heatmap.set_yticklabels(heatmap.get_yticklabels(), 
                       rotation=0,
                       fontsize=9)

plt.title('Matriz de Correlación', pad=20, fontsize=12)
plt.tight_layout()
plt.show()

# 8.3 Análisis de Varianza
print("\n📈 Análisis de Varianza:")
variance_df = df.select([pl.col(col).var().alias(col) for col in numeric_cols])

plt.figure(figsize=(12, 6))
ax = sns.barplot(data=variance_df.transpose().to_pandas(), palette='viridis')
plt.title('Varianza por Variable', pad=20, fontsize=14)
plt.xlabel('Variables')
plt.ylabel('Varianza')
plt.xticks(rotation=45)

# Añadir etiquetas de valor
for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}", 
                (p.get_x() + p.get_width()/2., p.get_height()), 
                ha='center', va='center', 
                xytext=(0, 10), 
                textcoords='offset points',
                fontsize=9)
plt.tight_layout()
plt.show()

# 9. Modelado KNN (Versión Corregida con manejo de desbalance)
print("\n" + "="*50)
print(" MODELADO KNN ".center(50, "="))
print("="*50)

# 9.1 Preparación de datos
X = df.select(numeric_cols).to_numpy()
y = df[TARGET_COL].to_numpy()

# Discretización de la variable objetivo para análisis de balance
# Convertir a Pandas para usar pd.cut
y_pd = pd.Series(y)
y_binned = pd.cut(y_pd, bins=3, labels=['Low', 'Medium', 'High'])
class_distribution = y_binned.value_counts(normalize=True)

print("\n📊 Distribución de clases (discretizadas):")
print(class_distribution)

# 9.1.1 Verificar desbalance significativo
if class_distribution.max() > 0.7:  # Si una clase tiene más del 70%
    print("\n⚠️ Advertencia: Distribución desbalanceada detectada")
    print("🔧 Aplicando técnicas de balanceo...")
    
    # Convertir a DataFrame para balanceo
    df_pd = df.select(numeric_cols + [TARGET_COL]).to_pandas()
    df_pd['target_binned'] = pd.cut(df_pd[TARGET_COL], bins=3, labels=['Low', 'Medium', 'High'])
    
    # Separar por clases
    majority_class = class_distribution.idxmax()
    minority_classes = [c for c in class_distribution.index if c != majority_class]
    
    df_majority = df_pd[df_pd.target_binned == majority_class]
    df_minorities = [df_pd[df_pd.target_binned == c] for c in minority_classes]
    
    # Balancear todas las clases
    n_samples = min(len(df_majority), *[len(df) for df in df_minorities])
    
    # Downsample majority class
    df_majority_downsampled = df_majority.sample(n=n_samples, random_state=42)
    
    # Upsample minority classes
    df_minorities_upsampled = [df.sample(n=n_samples, replace=True, random_state=42) 
                              for df in df_minorities]
    
    # Combinar
    df_balanced = pd.concat([df_majority_downsampled] + df_minorities_upsampled)
    
    # Ver nueva distribución
    print("\n📊 Nueva distribución después del balanceo:")
    print(df_balanced.target_binned.value_counts(normalize=True))
    
    # Preparar datos balanceados
    X = df_balanced[numeric_cols].values
    y = df_balanced[TARGET_COL].values
else:
    print("\n✅ Los datos están balanceados, procediendo sin ajustes")

# Resto del código permanece igual...
# Escalado
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# División train-test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# 9.2 Búsqueda del mejor k
print("\n🔍 Buscando el mejor valor de k...")
k_range = range(1, 21)
rmse_scores = []

for k in k_range:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Corrección aplicada aquí
    rmse_scores.append(rmse)
    print(f"k={k:2d}: RMSE = {rmse:.4f}")

# Visualización del RMSE
plt.figure(figsize=(10, 6))
plt.plot(k_range, rmse_scores, marker='o', linestyle='--', color='royalblue')
plt.title('RMSE para diferentes valores de k', pad=15)
plt.xlabel('Número de vecinos (k)')
plt.ylabel('RMSE')
plt.xticks(k_range)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

best_k = k_range[np.argmin(rmse_scores)]
print(f"\n✅ Mejor valor de k: {best_k} con RMSE = {min(rmse_scores):.4f}")

# 9.3 Modelo final (Versión Corregida)
print("\n🏆 Entrenando modelo final con el mejor k...")
final_knn = KNeighborsRegressor(n_neighbors=best_k)
final_knn.fit(X_train, y_train)
final_preds = final_knn.predict(X_test)

# Métricas finales (Corrección aplicada aquí)
final_r2 = r2_score(y_test, final_preds)
final_rmse = np.sqrt(mean_squared_error(y_test, final_preds))  # Calculamos RMSE manualmente

print("\n📊 Resultados del modelo final:")
print(f"- R²: {final_r2:.4f}")
print(f"- RMSE: {final_rmse:.4f}")

# Visualización de predicciones vs reales
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=final_preds, alpha=0.6, color='royalblue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 
         linestyle='--', color='red', linewidth=1)
plt.title('Predicciones vs Valores Reales', pad=15)
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()

print("\n" + "="*50)
print(" ANÁLISIS COMPLETADO ".center(50, "="))
print("="*50)


# 9.4 Visualización de Errores (Alternativa a Matriz de Confusión para Regresión)
print("\n📈 Visualización de Errores de Predicción")

# Crear bins para discretizar las predicciones y valores reales
bins = np.linspace(min(y_test.min(), final_preds.min()), 
                   max(y_test.max(), final_preds.max()), 10)

y_test_binned = np.digitize(y_test, bins)
preds_binned = np.digitize(final_preds, bins)

# Crear matriz de conteo
confusion_matrix = np.zeros((len(bins)+1, len(bins)+1))
for true, pred in zip(y_test_binned, preds_binned):
    confusion_matrix[true-1, pred-1] += 1

# Visualización
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix, annot=True, fmt='.0f', cmap='Blues',
            xticklabels=[f"{bins[i-1]:.1f}-{bins[i]:.1f}" for i in range(len(bins))],
            yticklabels=[f"{bins[i-1]:.1f}-{bins[i]:.1f}" for i in range(len(bins))])
plt.title('Distribución de Predicciones vs Valores Reales', pad=20)
plt.xlabel('Predicciones (binned)')
plt.ylabel('Valores Reales (binned)')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Alternativa: Gráfico de dispersión con residuos
plt.figure(figsize=(12, 6))
residuals = y_test - final_preds
sns.scatterplot(x=final_preds, y=residuals, alpha=0.6, color='royalblue')
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Análisis de Residuos', pad=15)
plt.xlabel('Predicciones')
plt.ylabel('Residuos (Real - Predicción)')
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()

print("\n" + "="*50)
print(" ANÁLISIS COMPLETADO ".center(50, "="))
print(" Hecho por: Jannet Ortiz Aguilar".center(50, "="))
print("="*50)

![image](https://github.com/user-attachments/assets/264fba30-f5d0-4a04-bd8d-e9980d159ecd)

![image](https://github.com/user-attachments/assets/27c41925-6dc1-498d-9a8d-2759699eddfc)

![image](https://github.com/user-attachments/assets/fe91c3ae-4f9e-4476-bd11-fb58ee36a26c)

![image](https://github.com/user-attachments/assets/d1e1e3e4-0f79-4fef-bc2d-0e86b65000d2)




