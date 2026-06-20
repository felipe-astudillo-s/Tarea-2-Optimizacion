================================================================
  TAREA 2 - METAHEURÍSTICAS: ALGORITMO GENÉTICO PARA MCDP
================================================================

INTEGRANTES
-----------
  - Felipe Astudillo    RUT: 21.734.001-2
  - Diego Zuñiga        RUT: 21763321-4

DESCRIPCIÓN
-----------
Implementación de un Algoritmo Genético (AG) para resolver el
Manufacturing Cell Design Problem (MCDP). El objetivo es maximizar
la Grouping Efficacy (GE), agrupando máquinas en celdas de forma
que se minimicen los movimientos inter-celda y los elementos vacíos.

ARCHIVOS
--------
  mcdp_ga.py         Módulo Python con todas las clases del AG:
                       - MCDPInstance  : representa una instancia del problema
                       - Individual    : solución (asignación máquina → celda)
                       - GeneticAlgorithm : lógica evolutiva completa
                       - ExperimentRunner : ejecuta N corridas y agrega estadísticas
                       - parse_instance_file() : lee el archivo de instancias

  solver.ipynb       Notebook Jupyter con la ejecución completa:
                       1. Carga de instancias
                       2. Configuración de parámetros
                       3. Ejecución de 10 corridas por instancia
                       4. Tabla de resultados (media, desviación estándar)
                       5. Box plots
                       6. Curvas de convergencia
                       7. Visualización de la mejor solución

  Instancias/
    instanciasMCDP.txt   Tres instancias del problema:
                           - Instancia 1 (pequeña):  5 máquinas × 7 partes
                           - Instancia 2 (Boctor 1): 16 máquinas × 30 partes
                           - Instancia 3 (Boctor 10): 16 máquinas × 30 partes

REQUISITOS
----------
  Python >= 3.9
  numpy
  matplotlib
  pandas
  jupyter  (para ejecutar el notebook)

  Instalar dependencias:
    pip install numpy matplotlib pandas jupyter

CÓMO EJECUTAR
-------------
  Opción 1 — Notebook interactivo (recomendado):
    jupyter notebook solver.ipynb
    Ejecutar todas las celdas en orden (Kernel → Restart & Run All).

  Opción 2 — Módulo directo (prueba rápida):
    python mcdp_ga.py
    Ejecuta una prueba en Instancia 1 con 50 generaciones.

PARÁMETROS CONFIGURABLES (celda 3 del notebook)
-------------------------------------------------
  pop_size          : tamaño de la población
  max_generations   : número máximo de generaciones por corrida
  max_time_seconds  : tiempo máximo por corrida (en segundos)
  tournament_size   : tamaño del torneo en selección
  crossover_prob    : probabilidad de cruce
  mutation_prob     : probabilidad de mutación por individuo
  elite_count       : número de élites preservados por generación
  crossover_type    : "two_point" o "one_point"
  random_seed       : semilla base (None = no determinístico)
  N_RUNS            : número de ejecuciones por instancia (default 10)

FUNCIÓN OBJETIVO
----------------
  GE = e_in / (e_in + e_out + e_voids)

  donde:
    e_in    = número de 1s dentro de bloques diagonales (intra-celda)
    e_out   = número de 1s fuera de bloques (elementos excepcionales)
    e_voids = número de 0s dentro de bloques diagonales (vacíos)

  Se maximiza GE ∈ [0, 1].

================================================================
