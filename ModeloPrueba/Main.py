# Módulo Main.
# Actúa como punto de entrada para ejecutar el proyecto.
# Este script importa los otros módulos y orquesta el flujo completo desde la conexión
# a la base de datos hasta la evaluación del modelo.

###
### El objetivo de este primer modelo, es predecir la probabilidad de ocurrencia de un tipo de infraccion
### en una  determinada ubicacion (dada por latitud y longitud) y un determinado momento (FechaYHoraInfraccion).
###

import pandas as _pandas
import numpy as _numpy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import statsmodels.api as sm

import ConexionBd as _context
import ManejoDatos as _manejoData
import EntrenamientoTesteoDatos as _trainingTestData
import VisualizacionDatos as _visualizeData

# Paso 1: Conectar a la base de datos y obtener datos
data = _context.EjecutarQuery('SELECT * FROM "Infraccion"')
dataTipoInf = _context.EjecutarQuery('SELECT * FROM "TipoInfraccion"')
# FIN Paso 1

# Paso 2
# llamo a la función para limpiar, procesar y generar datos
df = _manejoData.PrepararDatos_Modelo_1(data, dataTipoInf, num_filas_aleatorias=5000)
# Mostrar las primeras filas del DataFrame resultante
print(df.head())
print(f"Número total de filas en df: {len(df)}")
# FIN Paso 2

# Paso 3: creacion del modelo.
# => Y = 1/(1+ e^-(a + b*X)) -- funcion logistica.
# TipoInfraccion = 1/ 1 +( e^-( a + b1 * Latitud + b2 * Longitud + b3 * Horainfraccion + b4 * DiaSemanaInfraccion + b5 * MesInfraccion))
# separo variable explicada y explicativas ->
X = df[['LatitudInfraccion', 'LongitudInfraccion', 'HoraDelDia', 'DiaDeLaSemana', 'Mes']]
Y = df['TipoInfraccion']


# divido en conjuntos de entrenamiento y prueba. Por defecto seteo el tamanio para probar en 20%.
# Establezco la random seed (semilla aleatoria, para que todos los experimentos sean reproducibles) en 42 (discrecional).

# Implementacion del modelo en Python con statsmodel.api
import statsmodels.api as sm

logit_model = sm.Logit(Y, X)

result = logit_model.fit()

result.summary()


# Implementación del modelo en Python con scikit-learn

from sklearn import linear_model

logit_model = linear_model.LogisticRegression()
logit_model.fit(X,Y)

logit_model.score(X,Y)

1-Y.mean()  

_pandas.DataFrame(list(zip(X.columns, _numpy.transpose(logit_model.coef_))))

# Validación del modelo logístico
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state=0)

lm = linear_model.LogisticRegression()
lm.fit(X_train, Y_train)


#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# verifico la distribucion del modelo ->
#print(Y_train.value_counts())

# Entrenar un modelo de regresión logística
#modelo_logistico = _trainingTestData.EntrenarModeloLogistico(X_train,Y_train)
# FIN Paso 4.

# Paso 5: Testeo del modelo

# hago las predicciones
Y_pred = modelo_logistico.predict(X_test)


# Evaluar el modelo
exactitud = accuracy_score(Y_test, Y_pred)
print(f'Exactitud del modelo: {exactitud:.2f}')

# reporte de clasificación
print(classification_report(Y_test, Y_pred))


# matriz de confusión
print(confusion_matrix(Y_test, Y_pred))

# FIN Paso 5.


# Paso 6: Visualizar resultados


# Paso 7: Sacar conclusiones.

