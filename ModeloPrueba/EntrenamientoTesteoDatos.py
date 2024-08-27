# Modulo para el entrenamiento y el testeo de los datos.

# model_training.py
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def EntrenarModeloLineal(X_train, Y_train):
    """
    Entrena un modelo de regresion lineal.

    Args:
    X_train (pd.DataFrame): Conjunto de caracteristicas de entrenamiento.
    Y_train (pd.Series): Conjunto de etiquetas de entrenamiento.

    Returns:
    LinearRegression: Modelo de regresion lineal entrenado.
    """
    modelo_lineal = LinearRegression()
    modelo_lineal.fit(X_train, Y_train)
    return modelo_lineal

def EntrenarModeloLogistico(X_train, Y_train):
    """
    Entrena un modelo de regresion logistica.

    Args:
    X_train (pd.DataFrame): Conjunto de caracteristicas de entrenamiento.
    Y_train (pd.Series): Conjunto de etiquetas de entrenamiento.

    Returns:
    LogisticRegression: Modelo de regresion logistica entrenado.
    """
    modelo_logistico = LogisticRegression(solver='lbfgs', max_iter=1000)
    modelo_logistico.fit(X_train, Y_train)
    return modelo_logistico


def Predecir(modelo, X_test):
    """
    Utiliza un modelo entrenado para hacer predicciones sobre un conjunto de datos de prueba.

    Args:
    modelo: Modelo de machine learning entrenado.
    X_test (pd.DataFrame): Conjunto de caracteristicas de prueba.

    Returns:
    np.ndarray: Predicciones realizadas por el modelo.
    """
    return modelo.predict(X_test)


def EvaluarModeloLineal(modelo, X_test, Y_test):
    """
    Evalua un modelo de regresion lineal utilizando R^2 y el error cuadratico medio.

    Args:
    modelo: Modelo de regresion lineal entrenado.
    X_test (pd.DataFrame): Conjunto de caracteristicas de prueba.
    Y_test (pd.Series): Conjunto de etiquetas de prueba.

    Returns:
    dict: Diccionario con R^2 y MSE.
    """
    predicciones = modelo.predict(X_test)
    r2 = r2_score(Y_test, predicciones)
    mse = mean_squared_error(Y_test, predicciones)
    return {'R2': r2, 'MSE': mse}

def EvaluarModeloLogistico(modelo, X_test, y_test):
    """
    Evalua un modelo de regresion logistica utilizando exactitud.

    Args:
    modelo: Modelo de regresion logistica entrenado.
    X_test (pd.DataFrame): Conjunto de caracteristicas de prueba.
    Y_test (pd.Series): Conjunto de etiquetas de prueba.

    Returns:
    float: Exactitud del modelo.
    """
    predicciones = modelo.predict(X_test)
    exactitud = accuracy_score(y_test, predicciones)
    return exactitud
