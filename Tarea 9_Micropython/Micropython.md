# Tarea Mycropython
A continuación se explica línea por línea el código que se implementa un clasificador Random Forest básico en MicroPython

# 1. Importaciones y compatibilidad
 
      import math
    
    # Compatibilidad entre Python normal y MicroPython
      try:
          import urandom as random
      except ImportError:
          import random

# Teniendo en cuenta que el algoritmo import math importa funciones como raíz cuadrada (sqrt); además, el bloque del código try hasta el except lo que hace es intentar importar urandom, el cuál es usado en Micropython, pero como se está corriendo en colab es necesario adaptarlo para que funcione corectamente.

# 2. Funciones auxiliares
  def contar(lista):
      conteo = {}
      for item in lista:
          if item in conteo:
              conteo[item] += 1
          else:
              conteo[item] = 1
      return conteo

# La función del anterior bloque es para contar las veces que aparece cada elemento en una lista, adicionalmente, devuelve un diccionario: elemento → frecuencia
 
  def mayoritario(lista):
      conteo = contar(lista)
      max_cuenta = -1
      clase = None
      for k in conteo:
          if conteo[k] > max_cuenta:
              max_cuenta = conteo[k]
              clase = k
      return clase

# Encuentra el valor más frecuente en una lista (la "clase mayoritaria").

# 3. Clase Uso del algoritmo DecisionTree

  class DecisionTree:

# 3.1 Constructor

    def __init__(self, max_depth=5, min_samples_split=2)

# donde: se utiliza el algoritmo max_depth para representar la profundidad máxima que puede alcanzar el árbol de decisiones, en otras palabras, el número de niveles desde la raíz hasta la hoja más profunda que sea posible revisar durante el entrenamiento.

# 3.2 Entrenamiento  

  def fit(self, X, y)

# El cuál define un método que ajusta o entrena el modelo usando los datos de X y las etiquetas y

# 3.3 Predicción

  def predict(self, X)

# Cada árbol vota una clase y elige la clase mayoritaria, entras palabras voto por mayoría.

# 3.4 Prueba simple

  if __name__ == '__main__'

  # Código de prueba con una tabla de verdad del oprerador lógico OR

      X = [[0, 0], [0, 1], [1, 0], [1, 1]]
      y = [0, 1, 1, 1]
      
  ![image](https://github.com/user-attachments/assets/56c2f844-6f1d-480a-99f4-6074b96074bc)

# 3.4.1 Entrenamiento con estos ejemplos

    pred = rf.predict(test_X)
    print("Predicciones:", pred)
    print("Reales:", y)

Se obtienen e imprimen las predicciones del Random Forest y la salida esperada

# RESULTADOS

![image](https://github.com/user-attachments/assets/9fb6461b-19da-46f2-9fc6-962b8373b50d)


