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

import ManejoDatos as _manejoData
#import EntrenamientoTesteoDatos as _trainingTestData
import VisualizacionDatos as _visualizeData


# Paso 1
# llamo a la función para limpiar, procesar y generar datos ->
df = _manejoData.PrepararDatos_Modelo_3(num_filas_aleatorias_requeridas=15000)

# muestro las primeras filas del DataFrame resultante ->
print(df.head())
print(f"Número total de filas en df: {len(df)}")
# FIN Paso 1


# Codificacion de variables categoricas. (Dummies)
columnas_categoricas = ['NumeroDeSerieEquipo',
                        #'ProvinciaInfraccion',
                        #'PartidoInfraccion',
                        'TipoInfraccion',
                        'TipoVehiculo',
                        'GrupoVehiculo']

for columna in columnas_categoricas:
    df = _manejoData.CodificarColumnasCategoricas(df, columna)
# FIN Codificacion de variables categoricas (Dummies)

# convierto todas las columnas a tipo float64 para luego hacer la regresion.
df = df.astype(float)


# Paso 2: creacion del modelo.
# => Y = 1/(1+ e^-(a + b*X)) -- funcion logistica.
# TipoInfraccion = 1/ 1 +( e^-( a + b1 * Latitud + b2 * Longitud + b3 * Horainfraccion + b4 * DiaSemanaInfraccion + b5 * MesInfraccion))
# separo variable explicada y explicativas ->
Y = df['Si_Infraccion']
X = df.drop(columns=['Si_Infraccion'])

# Convertir Y a formato numerico con LabelEncoder
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y) # 0 = False. 1 = True.
# chequeo de que Y_encoded sea un arreglo de numpy de tipo float
Y_encoded = _numpy.array(Y_encoded, dtype=float)
# Agregar constante para el termino de intercepcion
X_sm = sm.add_constant(X)  

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
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=50)

# escalado de los datos.
scaler = StandardScaler()
# ajustado y transformado de los datos de entrenamiento,
# transformando también los datos de prueba.
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# convierto arrays escalados de vuelta a DataFrames para mantener nombres de columnas.
X_train_scaled = _pandas.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = _pandas.DataFrame(X_test_scaled, columns=X.columns)

# entreno al modelo con el conjunto de entrenamiento ->
lm = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=2000)
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

from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score,classification_report

# Matriz de Confusión.
matrizConfusion = confusion_matrix(Y_test, Y_pred)
sns.heatmap(matrizConfusion, annot=True, fmt="d")

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

precision = precision_score(Y_test, Y_pred, average='weighted')
recall = recall_score(Y_test, Y_pred, average='weighted')
f1 = f1_score(Y_test, Y_pred, average='weighted')

# ROC Curve y AUC (Área Bajo la Curva)
#_visualizeData.GraficarCurvaROC(Y_test, Y_pred_proba)
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(Y_test['Si_Infraccion'], Y_pred_proba)
roc_auc = auc(fpr, tpr)

# Graficar la curva ROC
_plt.figure()
_plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
_plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
_plt.xlim([0.0, 1.0])
_plt.ylim([0.0, 1.05])
_plt.xlabel('False Positive Rate')
_plt.ylabel('True Positive Rate')
_plt.title('Receiver Operating Characteristic')
_plt.legend(loc="lower right")
_plt.show()



# Reporte de clasificación
print("\nReporte de Clasificación:")
print(classification_report(Y_test, Y_pred))

# FIN Paso 2.


# Paso 3: Visualizar resultados
_visualizeData.GraficarResultados(Y_test,Y_pred)
_visualizeData.VisualizarDistribucionProbabilidades(Y_pred_proba,4)
_visualizeData.GraficarDispersion(X_test, Y_test, Y_pred)

# graficar la distribucion de los errores
 # Calcular los errores residuales (predicción - valor real)
errores_residuales = Y - Y_pred
_visualizeData.GraficarDistribucionErrores(errores_residuales)

# Paso 4: Sacar conclusiones.












