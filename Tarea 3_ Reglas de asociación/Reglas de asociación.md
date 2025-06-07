# Tarea 3: implementación de uso de Polars con los siguiente algoritmos para encontrar reglas de asociación

# 1. APLICACIÓN DEL ALGORITMO APRIORI

# Primero, se instalan las librerías necesarias en Colab
  !pip install polars efficient-apriori great-tables
     # Importar las librerías necesarias
import polars as pl
from efficient_apriori import apriori
from great_tables import GT, loc, style

def main():
    # Datos de las transacciones (mejor estructuración)
    transacciones = [
        ["Milk", "Bread", "Butter"],
        ["Milk", "Bread"],
        ["Bread", "Butter"],
        ["Milk", "Butter"],
        ["Milk", "Bread", "Butter"]
    ]
    
    # Aplicar el algoritmo Apriori con parámetros ajustados
    itemsets, rules = apriori(
        transacciones, 
        min_support=0.3,  # Aumentado para este dataset pequeño
        min_confidence=0.6  # Aumentado para reglas más significativas
    )
    
    # Procesamiento más eficiente de las reglas
    if not rules:
        print("No se encontraron reglas significativas con los parámetros actuales.")
        return
    
    # Extracción de métricas en una sola iteración
    rule_data = {
        "Antecedente": [list(rule.lhs) for rule in rules],
        "Consecuente": [list(rule.rhs) for rule in rules],
        "Confianza": [round(rule.confidence, 3) for rule in rules],
        "Soporte": [round(rule.support, 3) for rule in rules],
        "Lift": [round(rule.lift, 3) for rule in rules]
    }
    
    # Creación del DataFrame con Polars
    df_rules = pl.DataFrame(rule_data).sort("Lift", descending=True)
    
    # Visualización mejorada
    print("\n=== REGLAS DE ASOCIACIÓN ENCONTRADAS ===")
    print(df_rules)
    
    # Creación de tabla visual con Great Tables
    if len(df_rules) > 0:
        gt_table = (
            GT(df_rules.to_pandas())
            .tab_header(title="Análisis de Reglas de Asociación")
            .fmt_number(columns=["Confianza", "Soporte", "Lift"], decimals=3)
            .data_color(
                columns=["Lift"],
                palette=["red", "green"],
                domain=[min(rule_data["Lift"]), max(rule_data["Lift"])]
            )
        )
        print("\nVisualización tabular:")
        display(gt_table)  # Para entornos como Jupyter/Colab
    
    print("\nJannet Ortiz Aguilar")

if __name__ == "__main__":
    main()

RESULTADOS: 

R=== REGLAS DE ASOCIACIÓN ENCONTRADAS ===
shape: (9, 5)
![image](https://github.com/user-attachments/assets/d798bc8f-0b7b-4ae0-8bd5-1803a2c10a3f)


Jannet Ortiz Aguilar


# 2. APLIACACIÓN DEL ALGORITMO FP-GROWTH
# Primero,  se instalan las librerías necesarias
 import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
from tabulate import tabulate

# Datos de las transacciones
transacciones = [
    ["Milk", "Bread", "Butter"],
    ["Milk", "Bread"],
    ["Bread", "Butter"],
    ["Milk", "Butter"],
    ["Milk", "Bread", "Butter"]
]

# Se aplica el algoritmo TransactionEncoder para transformar las transacciones
encoder = TransactionEncoder()
encoded_array = encoder.fit(transacciones).transform(transacciones)

# Se converte los datos transformados en un DataFrame
df = pd.DataFrame(encoded_array, columns=encoder.columns_)

# Se calculan los patrones frecuentes (soporte mínimo de 0.1) y reglas de asociación (soporte mínimo de 0.6 para el lift)
frequent_itemsets = fpgrowth(df, min_support=0.1, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.6)

# Se Convierte los conjuntos de antecedente y consecuente en listas para que sean más legibles
rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x))
rules['consequents'] = rules['consequents'].apply(lambda x: list(x))
print(rules)

# Se Seleccionan las columnas relevantes: antecedentes, consecuentes, confianza, soporte y lift
result = rules[['antecedents', 'consequents', 'confidence', 'support', 'lift']]

# Se da formato a la tabla utilizando tabulate para hacerla más legible
formatted_result = tabulate(result, headers='keys', tablefmt='fancy_grid', showindex=False)

# Imprimir la tabla formateada
print(formatted_result)
    
print("")
print("Jannet Ortiz Aguilar")

if __name__ == "__main__":
    main()

RESULTADOS

=== REGLAS DE ASOCIACIÓN ENCONTRADAS ===
shape: (9, 5)
![image](https://github.com/user-attachments/assets/9c296eb1-0b92-436b-a97e-60080b43d39d)


# 3. Comparación de resultados Apriori Vs FP-GROWTH (Visualización tabular)

![image](https://github.com/user-attachments/assets/4857f34f-4a86-437e-b4d1-6c2b7134ca51)

