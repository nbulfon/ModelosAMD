# Modulo de visualización de los datos.

import matplotlib.pyplot as _plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

def GraficarResultados(y_test, y_pred, titulo='Resultados de la Prediccion'):
    """
    Grafica los resultados de la prediccion comparando los valores reales con los predichos.

    Args:
    y_test (pd.Series): Valores reales del conjunto de prueba.
    y_pred (np.ndarray): Valores predichos por el modelo.
    titulo (str, optional): Titulo del grafico. Valor por defecto es 'Resultados de la Prediccion'.
    """
    _plt.figure(figsize=(10, 6))
    _plt.scatter(y_test, y_pred, color='blue')
    _plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, color='red')
    _plt.xlabel('Valores Reales')
    _plt.ylabel('Valores Predichos')
    _plt.title(titulo)
    _plt.show()
    

def VisualizarDistribucionProbabilidades(Y_pred_proba, num_clases):
    """
    Visualiza la distribución de las probabilidades predichas para cada clase.
    
    Args:
    Y_pred_proba (np.array): Array de probabilidades predichas con forma (n_samples, num_clases).
    num_clases (int): Número de clases en el problema de clasificación.
    
    Returns:
    None: Muestra gráficos de las distribuciones de probabilidades.
    """
    
    _plt.figure(figsize=(10, 6))
    
    # Iterar sobre el número de clases para graficar la distribución de probabilidades de cada clase
    for i in range(num_clases):
        sns.kdeplot(Y_pred_proba[:, i], label=f'Clase {i}', shade=True)
    
    _plt.xlabel('Probabilidad predicha')
    _plt.ylabel('Densidad')
    _plt.title('Distribución de probabilidades por clase')
    _plt.legend()
    _plt.show()
    
    
def GraficarDispersion(X, Y_test, Y_pred):
    """
    Genera un gráfico de dispersión para visualizar las predicciones del modelo.

    Args:
    X (pd.DataFrame o np.array): Conjunto de características de prueba (al menos 2 columnas).
    Y_test (pd.Series o np.array): Conjunto de etiquetas de prueba.
    Y_pred (pd.Series o np.array): Predicciones del modelo.

    Returns:
    None: Muestra un gráfico de dispersión.
    """

    # Asumimos que las dos primeras columnas de X son las características a graficar
    _plt.figure(figsize=(10, 6))
    
    # Gráfico de dispersión con las clases reales
    _plt.scatter(X[:, 0], X[:, 1], c=Y_test, marker='o', cmap='coolwarm', label='Real', alpha=0.6)
    
    # Gráfico de dispersión con las predicciones del modelo
    _plt.scatter(X[:, 0], X[:, 1], c=Y_pred, marker='x', cmap='coolwarm', label='Predicción', alpha=0.6)
    
    _plt.xlabel('Característica 1')
    _plt.ylabel('Característica 2')
    _plt.title('Gráfico de Dispersión de Predicciones')
    _plt.legend()
    _plt.show()
    

def GraficarDistribucionErrores(errores_residuales):
    """
    Grafica la distribución de los errores residuales usando un histograma
    con una curva de densidad superpuesta.

    Args:
    errores_residuales (array-like): Array o lista de errores residuales.

    Returns:
    None
    """
    _plt.figure(figsize=(10, 6))
    sns.histplot(errores_residuales, kde=True, bins=30, color='blue', alpha=0.7)
    _plt.title("Distribución de Errores Residuales")
    _plt.xlabel("Error Residual")
    _plt.ylabel("Frecuencia")
    _plt.grid()
    _plt.show()
    
    
def GraficarCurvaROC(Y_test, Y_pred_proba):

    fpr, tpr, thresholds = roc_curve(Y_test, Y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)

    # Graficar la curva ROC
    _plt.figure()
    _plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    _plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    _plt.xlim([0.0, 1.0])
    _plt.ylim([0.0, 1.05])
    _plt.xlabel('Tasa de Falsos Positivos')
    _plt.ylabel('Tasa de Verdaderos Positivos')
    _plt.title('Curva ROC para Predicción de Infracciones')
    _plt.legend(loc="lower right")
    _plt.show()

def GraficarHistogramaProbInfraccion(Y_pred_proba):
    _plt.hist(Y_pred_proba[:, 1], bins=10, color='skyblue', edgecolor='black')
    _plt.axvline(x=0.5, color='red', linestyle='--', label='Umbral de decisión (0.5)')
    _plt.xlabel('Probabilidad Predicha de Infracción')
    _plt.ylabel('Frecuencia')
    _plt.title('Distribución de Probabilidades Predichas de Infracción')
    _plt.legend()
    _plt.show()

def GraficarInfraccionesPorTiempo(X_test,Y_pred_proba):
    X_test['Probabilidad_Infraccion'] = Y_pred_proba[:, 1]
    infraccion_por_hora = X_test.groupby('HoraDelDia')['Probabilidad_Infraccion'].mean()

    _plt.bar(infraccion_por_hora.index, infraccion_por_hora.values, color='coral')
    _plt.xlabel('Hora del Día')
    _plt.ylabel('Probabilidad Media de Infracción')
    _plt.title('Probabilidad de Infracción por Hora del Día')
    _plt.xticks(range(24))  # Horas de 0 a 23
    _plt.show()

def MapaDeCalorPrediccionesUbicacion(X_test,Y_pred_proba):
    _plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_test['LongitudInfraccion'], 
                y=X_test['LatitudInfraccion'], 
                hue=Y_pred_proba[:, 1], 
                palette='coolwarm', 
                size=Y_pred_proba[:, 1], 
                sizes=(10, 200), 
                legend=None)
    _plt.colorbar(label='Probabilidad de Infracción')
    _plt.xlabel('Longitud')
    _plt.ylabel('Latitud')
    _plt.title('Mapa de Calor de Probabilidades de Infracción')
    _plt.show()

def GraficarDispersion_Prob(
        df,
        probabilidades_infraccion,
        nombreColumna_X):
    """
    Crea un grafico de dispersion para mostrar la relacion entre la columna por param.
    y la probabilidad de infraccion.

    Args:
    df (pd.DataFrame): DataFrame que contiene los datos, incluyendo la columna 'VelocidadRegistrada'.
    probabilidades_infraccion (array-like): Array que contiene las probabilidades de infraccion predichas.

    Returns:
    None
    """
    # Agregar las probabilidades de infracción al DataFrame
    df['Probabilidad_Infraccion'] = probabilidades_infraccion

    # Crear un gráfico de dispersión
    _plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=nombreColumna_X, y='Probabilidad_Infraccion', alpha=0.6, color='blue')

    # Agregar una línea de tendencia suavizada usando seaborn
    sns.regplot(data=df, x=nombreColumna_X, y='Probabilidad_Infraccion', scatter=False, color='red', lowess=True)

    # Configuración de etiquetas y título
    _plt.xlabel(nombreColumna_X)
    _plt.ylabel('Probabilidad de Infracción')
    _plt.title(f"Relación entre {nombreColumna_X} y Probabilidad de Infracción")
    _plt.grid(True)

    # Mostrar el gráfico
    _plt.show()

def GraficarBarras_Prob(
        df,
        probabilidades_infraccion,
        nombreColumna_X):
    """
    Crea un grafico de barras para mostrar la relacion entre la columna por param.
    y la probabilidad media de infraccion.

    Args:
    df (pd.DataFrame): DataFrame que contiene los datos, incluyendo la columna 'HoraDelDia'.
    probabilidades_infraccion (array-like): Array que contiene las probabilidades de infraccion predichas.
    
    Returns:
    None
    """
    # Agregar las probabilidades de infracción al DataFrame
    df['Probabilidad_Infraccion'] = probabilidades_infraccion

    # Calcular la probabilidad media de infracción por cada hora del día
    probabilidad_por_hora = df.groupby(nombreColumna_X)['Probabilidad_Infraccion'].mean()

    # Crear un gráfico de barras
    _plt.figure(figsize=(10, 6))
    probabilidad_por_hora.plot(kind='bar', color='skyblue', edgecolor='black')
    
    # Configuración de etiquetas y título
    _plt.xlabel(nombreColumna_X)
    _plt.ylabel('Probabilidad Media de Infracción')
    _plt.title(f"Relación entre {nombreColumna_X} y Probabilidad de Infracción")
    _plt.xticks(rotation=0)
    _plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Mostrar el gráfico
    _plt.show()









