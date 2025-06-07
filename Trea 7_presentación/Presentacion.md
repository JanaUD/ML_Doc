# Coreeción de la presentación
# Se propone trabajar una ecuación para hallaR el  valor 

!pip install scikit-learn
# Instalación (si usas Google Colab)
# !pip install polars scikit-learn matplotlib seaborn

import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def calcular_tcr_individual(datos_iniciales: dict, datos_finales: dict) -> pl.DataFrame:
    if len(datos_iniciales['id_planta']) != len(datos_finales['id_planta']):
        raise ValueError("El número de plantas no coincide")

    if datos_iniciales['tiempo'] >= datos_finales['tiempo']:
        raise ValueError("El tiempo final debe ser mayor al inicial")

    df_inicial = pl.DataFrame({
        'id_planta': datos_iniciales['id_planta'],
        'masa_inicial': datos_iniciales['masa_seca'],
        'tiempo_inicial': datos_iniciales['tiempo']
    })

    df_final = pl.DataFrame({
        'id_planta': datos_finales['id_planta'],
        'masa_final': datos_finales['masa_seca'],
        'tiempo_final': datos_finales['tiempo']
    })

    df = df_inicial.join(df_final, on='id_planta')

    df = df.with_columns([
        (pl.col('tiempo_final') - pl.col('tiempo_inicial')).alias('delta_tiempo'),
        (pl.col('masa_final') - pl.col('masa_inicial')).alias('incremento_masa'),
        ((pl.col('masa_final').log() - pl.col('masa_inicial').log()) /
         (pl.col('tiempo_final') - pl.col('tiempo_inicial'))).alias('tcr'),
        ((pl.col('masa_final') - pl.col('masa_inicial')) /
         (pl.col('tiempo_final') - pl.col('tiempo_inicial'))).alias('tasa_crecimiento_absoluto')
    ])

    return df

def entrenar_modelo_rf(df_polars: pl.DataFrame):
    df = df_polars.to_pandas()

    X = df[['masa_inicial', 'masa_final', 'tiempo_inicial', 'tiempo_final', 'delta_tiempo', 'incremento_masa', 'tasa_crecimiento_absoluto']]
    y = df['tcr']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    print("\n Evaluación del Modelo Random Forest:")
    print(f"R²: {r2_score(y_test, y_pred):.4f}")
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE: {rmse:.4f}")

    # Importancia de características
    importancias = modelo.feature_importances_
    plt.figure(figsize=(10, 5))
    sns.barplot(x=X.columns, y=importancias)
    plt.title("Importancia de las características en Random Forest")
    plt.ylabel("Importancia")
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# Nuevos datos: 10 muestras
datos_iniciales = {
    'id_planta': list(range(1, 11)),
    'masa_seca': [0.0016, 0.032, 0.045, 0.027, 0.004, 0.012, 0.038, 0.028, 0.022, 0.015],
    'tiempo': 0
}

datos_finales = {
    'id_planta': list(range(1, 11)),
    'masa_seca': [0.0394, 0.0807, 0.0581, 0.0988, 0.0468, 0.062, 0.087, 0.059, 0.044, 0.033],
    'tiempo': 30

}

try:
    resultados = calcular_tcr_individual(datos_iniciales, datos_finales)
    print("📄 Resultados calculados:")
    print(resultados)

    entrenar_modelo_rf(resultados)

except Exception as e:
    print(f"\n Error: {str(e)}")
  print("")
  print ("Jannet Ortiz Aguilar")

  # RESULTADOS

  ![image](https://github.com/user-attachments/assets/7f6b80c1-2efe-48e9-9242-4bf7f0643e91)

![image](https://github.com/user-attachments/assets/3e047634-539b-4991-a32b-2bd2b436d3fe)
