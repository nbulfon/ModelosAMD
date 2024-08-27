# Modulo de visualización de los datos.

import matplotlib.pyplot as _plt

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