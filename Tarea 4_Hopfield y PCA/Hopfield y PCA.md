# Tarea 4_Hopfield y PCA: se busca el recorrido por todas las ciudades que demore menos tiempo, sin repetir ciudad utilizando redes de Hopfield

# Uso de redes de Hopfield para buscar el recorrido por todas las ciudades que demore menos tiempo y sin repetir ciudad. 
# Importar librerías necesarias
  import random
  import math
  from itertools import permutations
  import matplotlib.pyplot as plt

# 1. Datos del problema
  ciudades = ['A', 'B', 'C', 'D', 'E']

# Matriz de distancias entre ciudades
  distancias = [
      [0, 5, 5, 6, 4],
      [5, 0, 3, 7, 8],
      [5, 3, 0, 4, 8],
      [6, 7, 4, 0, 5],
      [4, 8, 8, 5, 0]
  ]
# 2. Solución por Fuerza Bruta
  def fuerza_bruta_tsp():
      num_ciudades = len(ciudades)
      mejor_ruta = None
      menor_distancia = float('inf')
      
      for perm in permutations(range(num_ciudades)):
          distancia_total = 0
          for i in range(num_ciudades):
              distancia_total += distancias[perm[i]][perm[(i+1) % num_ciudades]]
          
          if distancia_total < menor_distancia:
              menor_distancia = distancia_total
              mejor_ruta = perm
      
      return mejor_ruta, menor_distancia
# Ejecutar fuerza bruta
    ruta_optima, dist_optima = fuerza_bruta_tsp()
    print("Ruta óptima:", [ciudades[i] for i in ruta_optima])
    print("Distancia óptima:", dist_optima)
    print()

# 3. Red de Hopfield
    def hopfield_tsp_simple(max_iter=1000):
        num_ciudades = len(ciudades)
        neuronas = [[random.random() for _ in range(num_ciudades)] for _ in range(num_ciudades)]
        
        A, B, C = 10, 10, 0.1
        
        for _ in range(max_iter):
            for ciudad in range(num_ciudades):
                for orden in range(num_ciudades):
                    suma_fila = sum(neuronas[ciudad]) - neuronas[ciudad][orden]
                    suma_columna = sum(neuronas[c][orden] for c in range(num_ciudades)) - neuronas[ciudad][orden]
                    
                    suma_distancias = 0
                    for otra_ciudad in range(num_ciudades):
                        if otra_ciudad != ciudad:
                            siguiente = (orden + 1) % num_ciudades
                            anterior = (orden - 1) % num_ciudades
                            suma_distancias += distancias[ciudad][otra_ciudad] * (
                                neuronas[otra_ciudad][siguiente] + neuronas[otra_ciudad][anterior])
                    
                    delta = -A * suma_fila - B * suma_columna - C * suma_distancias
                    neuronas[ciudad][orden] = 1 / (1 + math.exp(-delta))
        
        ruta = [max(range(num_ciudades), key=lambda x: neuronas[ciudad][x]) for ciudad in range(num_ciudades)]
        distancia = sum(distancias[ruta[i]][ruta[(i+1) % num_ciudades]] for i in range(num_ciudades))
        
        return ruta, distancia
  
# Ejecutar Hopfield
    ruta_hopfield, dist_hopfield = hopfield_tsp_simple()
    print("Ruta Hopfield:", [ciudades[i] for i in ruta_hopfield])
    print("Distancia Hopfield:", dist_hopfield)
    print()

# 4. Visualización con Gráfica de Colores
# Coordenadas ficticias basadas en distancias
    coordenadas = {
          'A': (1, 2),
          'B': (3, 5),
          'C': (4, 3),
          'D': (6, 4),
          'E': (2, 1)
      }

# Crear figura
    plt.figure(figsize=(10, 6))

# Dibujar ciudades
    for ciudad, (x, y) in coordenadas.items():
        plt.scatter(x, y, s=200, zorder=5)
        plt.text(x, y, ciudad, fontsize=12, ha='center', va='center')
  
# Función para dibujar rutas
    def dibujar_ruta(ruta, color, estilo, etiqueta):
        x = [coordenadas[ciudades[i]][0] for i in ruta]
        y = [coordenadas[ciudades[i]][1] for i in ruta]
        x.append(x[0])  # Cerrar el ciclo
        y.append(y[0])
        plt.plot(x, y, color=color, linestyle=estilo, linewidth=2, label=etiqueta)

# Dibujar ambas rutas
    dibujar_ruta(ruta_optima, 'green', '-', 'Ruta Óptima')
    dibujar_ruta(ruta_hopfield, 'blue', '--', 'Ruta Hopfield')

# Configuraciones adicionales
    plt.title('Comparación de Rutas TSP', fontsize=14)
    plt.xlabel('Coordenada X', fontsize=12)
    plt.ylabel('Coordenada Y', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)

# Mostrar gráfica
    plt.tight_layout()
    plt.show()
    print("")
  print ("Jannet Ortiz Aguilar") 

# RESULTADOS

![image](https://github.com/user-attachments/assets/797c038e-764f-4879-a1da-dba6bcbb5da1)


# Código para utilización PCA que permite visualizar gráficas en 2D una base de datos de MNIST

!pip install numpy

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml

# Cargar los datos de MNIST desde OpenML
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

# Extraer las imágenes y etiquetas
X = mnist.data  # Matriz de características (70,000 imágenes de 784 píxeles cada una)
y = mnist.target.astype(int)  # Etiquetas de los dígitos

# Aplicar PCA con 2 componentes
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Definir una paleta de colores para los dígitos
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7, s=15)

# Añadir la barra de color con el nuevo label
plt.colorbar(scatter, label="Dígitos", ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# Etiquetas y título
plt.xlabel("Componente Principal 1 - Dirección de máxima varianza")
plt.ylabel("Componente Principal 2 - Dirección perpendicular a PC1")
plt.title("VISUALIZACIÓN DE MNIST EN 2D USANDO PCA")

# Mostrar la gráfica
plt.show()
print("")
print ("Jannet Ortiz Aguilar") 

# RESULTADOS

![image](https://github.com/user-attachments/assets/5b6aa71e-8991-4049-b017-05374efbfdf0)

