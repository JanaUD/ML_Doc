1. Numpy (biblioteca de anális de datos)
  # los vectores y matrices permiten el análisis de datos de manera organizada, 
  por lo que son eficientes para trabajar los datos númericos y científicos; 
  además, es simple de trabajar ya que consiste en un puntero a la memoria, 
  utilizados para interpretar datos almacenados tales como "Tipo dato" y los "saltos"
  Un ejemplo de ello se evidencia en el artículo:Harris, C. R., Millman, K. J., Van Der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., ... & Oliphant, T. E. (2020). Array programming with NumPy. Nature, 585(7825), 357-362.
  En la Fig. 1: The NumPy array incorporates several fundamental array concepts.


2. Pandas y Polars (juntas son bibliotecas de python que permiten analizar grandes conjuntos de datos estructurados en filas y columnas)
  Permiten filtar datos, agrupar por columnas, realizar operaciones como sumas y conteos entre otros.
  Permite unir, fusionar y combinar DataFrames.
  Pandas es más antiguó pero es reconocido entre la comunudad de las personas que trabajan con python, 
  pero hace más uso de memoria cuando trabaja con grandes volumenes de datos;
  a diferencia de Polars, que es más reciente, esc onocido por ser más eficiente tn el uso de la memoria para procesamiento de datos.
  En terminos generales sus características son:
  Pandas                                                   Polars
  Bueno en conjuntos de datos pequeños                     Es más efciciente en análisis de grandes volumenes de datos
  consume más memoria                                      Más eficiente en el uso de memoria
  Ejecuta todas las operaciones de manera inmediata        No realiza operaciones de no ser de necsario
  
  Por lo anterior Polars aunque es una biblioteca más resiente esta creciendo en popularidad por su efectividad, al contrario 
  Pandas es una opción popular que ya ha sido utilizada y garantiza datos confiables.


3. Matplotlip y hvPlot (bibliotecas de python que proporcionan diseño gráfico)
  Matplotlip permite visualización en 2D aunque requiere más código para los detalles de las magenes.
  hvPlot permite la creación de gráficos en 3D, utilizando hv.Scatter3D para gráficos de dispersión y hv.Surface para gráficos de superficie.
  En terminos generales sus características son:
  Matplotlip                                             hvPlot
  No es tan amigable                                     es fácil de usar especialmente con pandas
  tiene limitada interactividad en su funcionalidad      su interactividad viene por defecto
  Necesita código adicional para gráficos complejos      Variedad de gráficos listos para usar
  
  Las dos soon herramientas poderosas para trabajar con python, su uso depende de las necesidades del usuario; 
  los dos son versatiles y ampliamente utilizadas; Matplotlip es excelente para trabajar gráficas estáticas y 
  hvPlot excelente en el uso de gráficos interactivos en contextos de exploración y análisis de datos.
