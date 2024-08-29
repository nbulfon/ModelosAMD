# Módulo RegLogistica para el modelo 2 de prueba con datos de AM/AMD,etc.
# Actúa como punto de entrada para ejecutar el proyecto.
# Este script importa los otros módulos y orquesta el flujo completo desde la conexión
# a la base de datos hasta la evaluación del modelo y posteriores conclusiones.

###
### El objetivo de este primer modelo, es predecir la probabilidad de ocurrencia de que
### los proximos n registros sean con infraccion.
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
