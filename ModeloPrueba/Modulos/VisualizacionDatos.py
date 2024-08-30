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

    fpr, tpr, thresholds = roc_curve(Y_test, Y_pred_proba)
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

    
