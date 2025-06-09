# Tarea 9_Mycropython_RandonForestClassifieer

# Explicación: 
  RandomForestClassifier (Clasificadodr de Bosque Aleatorio) es un algotirmo de     aprendizaje automático supervisado utilizado para realizar tareas de clasificación;  pero, también es posible utilizarlo para rregresión (teniendo en cuenta pequeños  cambios); entendiendo que un bosque aleatorio es un modelo de aprendizaje automático (se basa en otros modelos para hacer predicciones), que entrena múltiples clasificadores de árboles de decisión sobre diferentes submuestras del conjunto de datos utilizando el promedio de los mismos. Adicionalmente, RandomForestClassifieer es parte de scitik.learn (biblioteca de apendizaje automático), esta utiliza dependencias externas como numpy, scipy, entre otras; realizando cálculos intensivos en memoria y CPU dl pc, las cuales no son compatibles con Micropython, 

Ahora bien, la intención es realizar un código que haga la función del algoritmo RandomForestClassifier para Micropython que simule dicho algoritmo; por tal motivo, a continuac se presenta en MicroPython

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
      def __init__(self, max_depth=5, min_samples_split=2):
          self.max_depth = max_depth
          self.min_samples_split = min_samples_split
          self.tree = None

# 3.1 Constructor

   def fit(self, X, y):
      self.tree = self._build_tree(X, y)

# donde: se utiliza el algoritmo max_depth para representar la profundidad máxima que puede alcanzar el árbol de decisiones, en otras palabras, el número de niveles desde la raíz hasta la hoja más profunda que sea posible revisar durante el entrenamiento.

# 3.2 Entrenamiento  

  def predict(self, X):
    return [self._predict_single(x, self.tree) for x in X]

# El cuál define un método que ajusta o entrena el modelo usando los datos de X y las etiquetas y, predice una lista de entradas usando el árbol construido.

# 3.3 Construcción del árbol
  
  def _build_tree(self, X, y, depth=0):
      n_samples = len(X)
      n_features = len(X[0]) if n_samples > 0 else 0

  # 3.4 Define el número de muestras y características.

  if (depth >= self.max_depth or n_samples < self.min_samples_split or len(set(y)) == 1):
      return {'class': mayoritario(y)}
  
# 3.5 Condición para detener la división: profundidad máxima, pocas muestras o todos los valores y son iguales.

  best_feature, best_threshold = self._best_split(X, y)

# 3.6 Busca la mejor división (feature y umbral).

  if best_feature is None:
      return {'class': mayoritario(y)}

# 3.7 Si no se puede dividir, se devuelve la clase mayoritaria.

  left_X, left_y, right_X, right_y = [], [], [], []
  for i in range(n_samples):
      if X[i][best_feature] <= best_threshold:
          left_X.append(X[i])
          left_y.append(y[i])
      else:
          right_X.append(X[i])
          right_y.append(y[i])

# 3.7 Se separan los datos según la mejor división.

  left_subtree = self._build_tree(left_X, left_y, depth + 1)
  right_subtree = self._build_tree(right_X, right_y, depth + 1)

# 3.8 Construye los subárboles recursivamente

return {
    'feature': best_feature,
    'threshold': best_threshold,
    'left': left_subtree,
    'right': right_subtree
}
# Devuelve el nodo como un diccionario.

# 4. Método para elegir la mejor división

  def _best_split(self, X, y)
    best_gini = 1
    best_feature = None
    best_threshold = None

   m = min(int(math.sqrt(n_features)) + 1, n_features)

# Se selecciona un subconjunto aleatorio de atributos (típico en Random Forest).

     features = []
  while len(features) < m:
      f = random.getrandbits(8) % n_features
      if f not in features:
          features.append(f)

# Elige m características aleatorias sin repetir.


   for feature in features:
    thresholds = []
      for i in range(len(X)):
      val = X[i][feature]
        if val not in thresholds:
        thresholds.append(val)

    for threshold in thresholds:
      left_y, right_y = [], []
        for i in range(len(X)):
            if X[i][feature] <= threshold:
              left_y.append(y[i])
              else:
                right_y.append(y[i])

# Para cada característica, se prueban todos los valores únicos como umbrales.

if not left_y or not right_y: 
 continue
# Se ignoran divisiones vacías.

    gini = self._gini_index(left_y, right_y)
    if gini < best_gini:
      best_gini = gini
      best_feature = feature
      best_threshold = threshold
 
 return best_feature, best_threshold

# Se calcula el índice Gini de la división.

# 5. Gini
  def _gini_index(self, left_y, right_y):
     n_left = len(left_y)
        n_right = len(right_y)
        n_total = n_left + n_right

        def gini(y_list):
            clases = set(y_list)
            score = 1.0
            for c in clases:
                p = y_list.count(c) / len(y_list)
                score -= p * p
            return score

        return (n_left / n_total) * gini(left_y) + (n_right / n_total) * gini(right_y)
  

# 6 Predicción de un solo dato

  def _predict_single(self, x, node):
    if 'class' in node:
            return node['class']
        if x[node['feature']] <= node['threshold']:
            return self._predict_single(x, node['left'])
        else:
            return self._predict_single(x, node['right'])

# Baja recursivamente por el árbol hasta llegar a una hoja ('class'), devolviendo la clase

# 7. Clase RandomForest
  class RandomForest:
    def __init__(self, n_estimators=5, max_depth=5, min_samples_split=2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

# Define un bosque de n_estimators árboles con cierta profundidad máxima

  def fit(self, X, y):
   self.trees = []
        n_samples = len(X)
        
# Entrena cada árbol con una muestra aleatoria del dataset (bootstrap)

  for _ in range(n_samples):
      indices.append(random.getrandbits(8) % n_samples)
      
# Se realiza muestreo con reemplazo para crear un subconjunto de entrenamiento

# 8. Predicción con votación mayoritaria

  def predict(self, X):
    self.trees = []
        n_samples = len(X)
        for _ in range(self.n_estimators):
            indices = []
            for _ in range(n_samples):
                indices.append(random.getrandbits(8) % n_samples)
            X_sample = [X[i] for i in indices]
            y_sample = [y[i] for i in indices]
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        predictions = []
        for x in X:
            votes = []
            for tree in self.trees:
                vote = tree.predict([x])[0]
                votes.append(vote)
            predictions.append(mayoritario(votes))
        return predictions

# Cada árbol da su voto. Se elige la clase más votada (mayoritario(votes))


# 9. Prueba simple

  if __name__ == '__main__':
      X = [[0, 0], [0, 1], [1, 0], [1, 1]]
      y = [0, 1, 1, 1]
  
      rf = RandomForest(n_estimators=5, max_depth=3)
      rf.fit(X, y)
  
      test_X = [[0, 0], [0, 1], [1, 0], [1, 1]]
      pred = rf.predict(test_X)
  
      print("Predicciones:", pred)
      print("Reales:", y)
      print("")
      print("Jannet Ortiz Aguilar")

Se obtienen e imprimen las predicciones del Random Forest y la salida esperada

# RESULTADOS

![image](https://github.com/user-attachments/assets/9fb6461b-19da-46f2-9fc6-962b8373b50d)


