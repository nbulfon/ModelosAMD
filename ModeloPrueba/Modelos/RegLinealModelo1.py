# Módulo RegLineal para el modelo 1 de prueba con datos de AM/AMD,etc.
# Actúa como punto de entrada para ejecutar el proyecto.
# Este script importa los otros módulos y orquesta el flujo completo desde la conexión
# a la base de datos hasta la evaluación del modelo y posteriores conclusiones.

###
### El objetivo de este segundo modelo, es predecir la cantidad de infracciones,
### en una  determinada ubicacion (dada por latitud y longitud) y un determinado momento (FechaYHoraInfraccion).
### (Regresion Lineal Multiple).
###
import matplotlib.pyplot as _plt
import pandas as _pandas
import numpy as _numpy
import statsmodels.formula.api as _smf

import Modulos.ManejoDatos as _manejoData
import EntrenamientoTesteoDatos as _trainingTestData
import VisualizacionDatos as _visualizeData

# Paso 1
# llamo a la función para limpiar, procesar y generar datos

#IMPORTANTE. COMO AHORA ESTOY HACIENDO UN GROUP BY, PROBAR PRIMERO BIEN
# SIN AGREGARLE DATOS ALEATORIOS, Y LUEGO PROBARLO DE NUEVO, CON ESOS DATOS EXTRA.
df = _manejoData.PrepararDatos_Modelo_2(num_filas_aleatorias_requeridas=0)

# Mostrar las primeras filas del DataFrame resultante
print(df.head())
print(f"Número total de filas en df: {len(df)}")

# Paso 2: creacion del modelo.
# => Y  a + b*X -- funcion logistica.
# TipoInfraccion = a + b1 * Latitud + b2 * Longitud + b3 * Horainfraccion + b4 * DiaSemanaInfraccion + b5 * MesInfraccion
# separo variable explicada y explicativas ->
Y = df['CantidadInfracciones']
X = df.drop(columns=['CantidadInfracciones'])



#--------------------------------------------------------
# Recordar que:
# scikit-learn.LogisticRegression con configuración multinomial es adecuado para
# tareas de clasificación práctica y optimización en contextos de producción, y
# proporciona herramientas para validar y ajustar el modelo.    
#--------------------------------------------------------

# Validación del modelo logístico con conjuntos de entrenamiento y prueba
















