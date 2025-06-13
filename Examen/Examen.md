# Corrección de examen para prueba de escritorio

 # Datos de estudiantes = {
      "E1 (Ingeniería)": {"lógica", "álgebra", "pensamiento_analítico"},
      "E2 (Ingeniería)": {"estadística", "álgebra", "lógica", "probabilidad"},
      "E3 (Psicología)": {"empatía", "comunicación", "trabajo_en_equipo"},
      "E4 (Psicología)": {"lectura", "participación", "expresión_oral"},
      "E5 (Matemáticas)": {"resolución_problemas", "abstracción", "cálculo"},
      "E6 (Matemáticas)": {"teoría_números", "lógica", "probabilidad"}

# Cálculo de similitud

   def jaccard_manual(set1, set2):
     interseccion = set1.intersection(set2)  # (1) Cursos que aparecen en los dos conjuntos revisados
     union = set1.union(set2)                # (2) Los cursos de la unión de los conjuntos sin repetir
     if len(union) == 0:                     # (3) Evitar división por cero
         return 0.0
     return len(interseccion) / len(union)   # (4) Fórmula para hallar la similitud de Jaccar

# Comparación de pares de los conjuntos de los estudiantes con todos los demás sin repetir

   nombres = list(estudiantes.keys())
   print("=== Resultados de Similitud de Jaccard ===")
   for i in range(len(nombres)):
       for j in range(i + 1, len(nombres)):
           estudiante1 = nombres[i]
           estudiante2 = nombres[j]
           similitud = jaccard_manual(estudiantes[estudiante1], estudiantes[estudiante2])
           print(f"{estudiante1} vs {estudiante2}: {similitud:.2f}")

# Matriz de similitud (sólo para observar los resultados)
   print("\n=== Matriz de Similitud ===")
   print("Estudiante\t" + "\t".join(nombres))
   for i in range(len(nombres)):
       fila = [nombres[i]]
       for j in range(len(nombres)):
           if i == j:
               fila.append("1.00")  # Autosimilitud (un estudiante consigo mismo)
           else:
               similitud = jaccard_manual(estudiantes[nombres[i]], estudiantes[nombres[j]])
               fila.append(f"{similitud:.2f}")
      print("\t".join(fila))
      print("")
      print("\nJannet Ortiz Aguilar")
