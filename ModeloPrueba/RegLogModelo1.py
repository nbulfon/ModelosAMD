# Módulo RegLogistica para el modelo 1 de prueba con datos de AM/AMD,etc.
# Actúa como punto de entrada para ejecutar el proyecto.
# Este script importa los otros módulos y orquesta el flujo completo desde la conexión
# a la base de datos hasta la evaluación del modelo y posteriores conclusiones.

###
### El objetivo de este primer modelo, es predecir la probabilidad de ocurrencia de un tipo de infraccion
### en una  determinada ubicacion (dada por latitud y longitud) y un determinado momento (FechaYHoraInfraccion).
### (Regresion logistica multinomial).
###
import matplotlib.pyplot as _plt
import pandas as _pandas
import numpy as _numpy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder

import ManejoDatos as _manejoData
import EntrenamientoTesteoDatos as _trainingTestData
import VisualizacionDatos as _visualizeData


# Paso 1
# llamo a la función para limpiar, procesar y generar datos
df = _manejoData.PrepararDatos_Modelo_1(num_filas_aleatorias_requeridas=5000)

# Mostrar las primeras filas del DataFrame resultante
print(df.head())
print(f"Número total de filas en df: {len(df)}")
# FIN Paso 1

# Paso 2: creacion del modelo.
# => Y = 1/(1+ e^-(a + b*X)) -- funcion logistica.
# TipoInfraccion = 1/ 1 +( e^-( a + b1 * Latitud + b2 * Longitud + b3 * Horainfraccion + b4 * DiaSemanaInfraccion + b5 * MesInfraccion))
# separo variable explicada y explicativas ->
X = df[['LatitudInfraccion', 'LongitudInfraccion', 'HoraDelDia', 'DiaDeLaSemana', 'Mes']]
Y = df['TipoInfraccion']

# Convertir Y a formato numerico con LabelEncoder
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)

# Agregar constante para el termino de intercepcion
X_sm = sm.add_constant(X)  

Y_encoded.shape
X_sm.shape

# Crear el modelo logit multinomial con las versiones alineadas y sin nulos de X e Y
mnlogit_model = sm.MNLogit(Y_encoded, X_sm)
# Ajustar el modelo
result = mnlogit_model.fit()

# Mostrar el resumen del modelo
print(result.summary())

logit_model_sm = sm.MNLogit(Y_encoded, X_sm)
result = logit_model_sm.fit()
# Calcular McFadden's R²
llf = result.llf  # log-likelihood del modelo ajustado
llnull = result.llnull  # log-likelihood del modelo solo con el intercepto

mcfadden_r2 = 1 - (llf / llnull)
print(f"McFadden's R²: {mcfadden_r2:.4f}")

#--------------------------------------------------------
# Recordar que:
# scikit-learn.LogisticRegression con configuración multinomial es adecuado para
# tareas de clasificación práctica y optimización en contextos de producción, y
# proporciona herramientas para validar y ajustar el modelo.    
#--------------------------------------------------------

# Validación del modelo logístico con conjuntos de entrenamiento y prueba
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
# Entrenar el modelo con el conjunto de entrenamiento
lm = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
lm.fit(X_train, Y_train)

# Realizar predicciones
Y_pred = lm.predict(X_test)
Y_pred_proba = lm.predict_proba(X_test)

# veo las primeras 10 predicciones ->
print(Y_pred[:10])
# veo las probabilidades de las primeras 10 observaciones.
# IMPORTANTE PARA ENTENDER:
# Interpretación de Y_pred_proba ->
#Cuando usas lm.predict_proba(X_test), Y_pred_proba será un array de 2 dimensiones:
 #Cada fila del array representa una observación o muestra del conjunto de prueba (X_test).
 #Cada columna representa la probabilidad de que la observación pertenezca a una clase particular.    
print(Y_pred_proba[:10])

# Evaluar el modelo
accuracy = accuracy_score(Y_test, Y_pred) # es como el R2...(masomenos)
print(f'Exactitud en el conjunto de prueba: {accuracy:.2f}')
# Reporte de clasificación
print("\nReporte de Clasificación:")
print(classification_report(Y_test, Y_pred))

# FIN Paso 2.


# Paso 3: Visualizar resultados
_visualizeData.GraficarResultados(Y_test,Y_pred)
_visualizeData.VisualizarDistribucionProbabilidades(Y_pred_proba,4)
#_visualizeData.GraficarDispersion(X_test, Y_test, Y_pred)

# Paso 4: Sacar conclusiones.

