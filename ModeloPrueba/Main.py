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

# Paso 3: creacion del modelo.
# => Y = 1/(1+ e^-(a + b*X)) -- funcion logistica.
# TipoInfraccion = 1/ 1 +( e^-( a + b1 * Latitud + b2 * Longitud + b3 * Horainfraccion + b4 * DiaSemanaInfraccion + b5 * MesInfraccion))
# separo variable explicada y explicativas ->
X = df[['LatitudInfraccion', 'LongitudInfraccion', 'HoraDelDia', 'DiaDeLaSemana', 'Mes']]
Y = df['TipoInfraccion']





# Implementacion del modelo en Python con statsmodel.api

# Asegurarse de que todos los datos en X sean numericos
X = X.apply(_pandas.to_numeric, errors='coerce')

# Codificar la variable dependiente Y de tipo string a numerico usando LabelEncoder
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)

# Agregar una constante a X para el termino de intercepcion
X = sm.add_constant(X)

# Convertir los datos a arrays de NumPy
X = _numpy.asarray(X)
Y_encoded = _numpy.asarray(Y_encoded)

# Crear el modelo logit
logit_model = sm.Logit(Y_encoded, X)

# Ajustar el modelo
result = logit_model.fit()

# Mostrar el resumen
print(result.summary())

# Mostrar las clases codificadas para referencia
print("Clases codificadas:", label_encoder.classes_)


# Implementación del modelo en Python con scikit-learn


logit_model = linear_model.LogisticRegression()
logit_model.fit(X,Y)

logit_model.score(X,Y)

1-Y.mean()  

_pandas.DataFrame(list(zip(X.columns, _numpy.transpose(logit_model.coef_))))

# Validación del modelo logístico

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state=0)

lm = linear_model.LogisticRegression()
lm.fit(X_train, Y_train)








# Validación del modelo logístico con conjuntos de entrenamiento y prueba
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

# Entrenar el modelo con el conjunto de entrenamiento
lm = linear_model.LogisticRegression()
lm.fit(X_train, Y_train)

# Realizar predicciones
Y_pred = lm.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(Y_test, Y_pred)
print(f'Exactitud en el conjunto de prueba: {accuracy:.2f}')
print(classification_report(Y_test, Y_pred))
print(confusion_matrix(Y_test, Y_pred))



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

