# Módulo RegLogistica para el modelo 2 de prueba con datos de AM/AMD,etc.
# Actúa como punto de entrada para ejecutar el proyecto.
# Este script importa los otros módulos y orquesta el flujo completo desde la conexión
# a la base de datos hasta la evaluación del modelo y posteriores conclusiones.

###
### El objetivo de este primer modelo, es predecir la probabilidad de ocurrencia de que
### los proximos n registros sean con infraccion.
### (Regresion logistica multinomial).
### IMPORTANTE. PARA MODELOS DE VAR. DEPENDIENTE DICOTOMICA, EL R^2 NO SIRVE MUCHO -> VER GUJARATI.
###
import matplotlib.pyplot as _plt
import pandas as _pandas
import numpy as _numpy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

import ManejoDatos as _manejoData
#import EntrenamientoTesteoDatos as _trainingTestData
import VisualizacionDatos as _visualizeData


# Paso 1
# llamo a la función para limpiar, procesar y generar datos ->
X , Y = _manejoData.PrepararDatos_Modelo_3(siGeneraratosAleatorios=True)
# FIN Paso 1

# Paso 2
# entreno un modelo de Random Forest para evaluar la importancia de c/ caracteristica.
forest = RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True)
forest.fit(X, Y)

# obtengo las caracteristicas importantes dados los datos que me proporciona
# el bosque aleatorio que entrene y segun mi umbral de decision ->
caracteristicas_importantes = _manejoData.SeleccionarCaracteristicasImportantes(
    X, forest, umbral=0.05)

# Filtrar X para usar solo características importantes
X_filtrado = X[caracteristicas_importantes]
# FIN Paso 2.

# Paso 3: creacion del modelo de Regresion Logistica.
# => Y = 1/(1+ e^-(a + b*X)) -- funcion logistica.
# TipoInfraccion = 1/ 1 +( e^-( a + b1 * Latitud + b2 * Longitud + b3 * Horainfraccion + b4 * DiaSemanaInfraccion + b5 * MesInfraccion))
# Convertir Y a formato numerico con LabelEncoder
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y) # 0 = False. 1 = True.
# chequeo de que Y_encoded sea un arreglo de numpy de tipo float
Y_encoded = _numpy.array(Y_encoded, dtype=float)
# Agregar constante para el termino de intercepcion
X_sm = sm.add_constant(X_filtrado)  

# Verificar multicolinealidad con VIF.
# Valores de VIF mayores a 5 o 10 indican una alta multicolinealidad que podría ser problemática.
vif_data = _pandas.DataFrame()
vif_data["feature"] = X_sm.columns
vif_data["VIF"] = [variance_inflation_factor(X_sm.values, i) for i in range(X_sm.shape[1])]
print(vif_data)


# Crear el modelo logit multinomial con las versiones alineadas y sin nulos de X e Y
mnlogit_model = sm.MNLogit(Y_encoded, X_sm)
# Ajustar el modelo
result = mnlogit_model.fit()
# Mostrar el resumen del modelo
print(result.summary())


#--------------------------------------------------------
# Recordar que:
# scikit-learn.LogisticRegression con configuración multinomial es adecuado para
# tareas de clasificación práctica y optimización en contextos de producción, y
# proporciona herramientas para validar y ajustar el modelo.    
#--------------------------------------------------------

# Validación del modelo logístico con conjuntos de entrenamiento y prueba
X_train, X_test, Y_train, Y_test = train_test_split(X_filtrado, Y, test_size=0.3, random_state=42)

# escalado de los datos.
scaler = StandardScaler()
# ajustado y transformado de los datos de entrenamiento,
# transformando también los datos de prueba.
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# convierto arrays escalados de vuelta a DataFrames para mantener nombres de columnas.
X_train_scaled = _pandas.DataFrame(X_train_scaled, columns=X_filtrado.columns)
X_test_scaled = _pandas.DataFrame(X_test_scaled, columns=X_filtrado.columns)

# entreno al modelo con el conjunto de entrenamiento ->
lm = LogisticRegression(multi_class='multinomial',
                        solver='lbfgs',
                        max_iter=2000,
                        class_weight='balanced')
lm.fit(X_train_scaled, Y_train)

# realizo predicciones ->
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

# Evaluacion del modelo (METRICAS IMPORTANTES DEBAJO).

import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score,classification_report

# Accuracy (Exactitud).
accuracy = accuracy_score(Y_test, Y_pred)
print(f'Exactitud en el conjunto de prueba: {accuracy:.2f}')

# Precision: mide la proporción de verdaderos positivos
# sobre el total de positivos predichos.
# Es útil cuando el costo de un falso positivo es alto.

# Recall (Sensibilidad o Tasa de Verdaderos Positivos):
# Mide la proporción de verdaderos positivos sobre el total de positivos reales.
# Es útil cuando el costo de un falso negativo es alto.

# F1-Score: Es la media armónica de la precisión y el recall,
# proporcionando una métrica equilibrada cuando hay una relación
# desigual entre precision y recall.

# Reporte de clasificación general ->
print("\nReporte de Clasificación:")
print(classification_report(Y_test, Y_pred))

# Cross validation ->
from sklearn.model_selection import cross_val_score
scores = cross_val_score(lm, X_train_scaled, Y_train, cv=5, scoring='f1_weighted')
print(f"F1-Score promedio con validación cruzada: {scores.mean():.4f}")


# FIN Paso 3.

# Paso 3: Visualizar resultados

# Matriz de Confusión.
from sklearn.metrics import confusion_matrix
matrizConfusion = confusion_matrix(Y_test, Y_pred)
sns.heatmap(matrizConfusion, annot=True, fmt="d")

_visualizeData.GraficarCurvaROC(Y_test, Y_pred_proba)

# graficar la distribucion de los errores
 # Calcular los errores residuales (predicción - valor real)
#errores_residuales = Y - Y_pred
#_visualizeData.GraficarDistribucionErrores(errores_residuales)
_visualizeData.GraficarHistogramaProbInfraccion(Y_pred_proba)
_visualizeData.GraficarInfraccionesPorTiempo(X_test,Y_pred_proba)
_visualizeData.MapaDeCalorPrediccionesUbicacion(X_test,Y_pred_proba)
_visualizeData.GraficarDispersion_Prob(
    X_test,
    Y_pred_proba[:, 1],
    nombreColumna_X='VelocidadRegistrada')

_visualizeData.GraficarDispersion_Prob(
    X_test,
    Y_pred_proba[:, 1],
    nombreColumna_X='LatitudInfraccion')
_visualizeData.GraficarDispersion_Prob(
    X_test,
    Y_pred_proba[:, 1],
    nombreColumna_X='LongitudInfraccion')

_visualizeData.GraficarBarras_Prob(
    X_test,
    Y_pred_proba[:, 1],
    nombreColumna_X='HoraDelDia')





# Paso 4: Sacar conclusiones.



