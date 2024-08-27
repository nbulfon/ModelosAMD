# Modulo que se va a encargar del data cleaning.

import pandas as _pandas

import GeneracionDatosAleatorios as _generacionDatos

# Esta funcion, es para el modelo que intentara predecir la probabilidad de ocurrencia 
# de un tipo de infraccion en una determinada ubicacion.
def PrepararDatos_Modelo_1(data, dataTipoInf, num_filas_aleatorias):
    """
    Limpia y procesa los datos, y genera datos aleatorios adicionales.
    Esta funcion, es para el modelo que intentara predecir la probabilidad de ocurrencia 
    de un tipo de infraccion en una determinada ubicacion.

    Args:
    data (pd.DataFrame): DataFrame que contiene los datos originales de infracciones.
    dataTipoInf (pd.DataFrame): DataFrame que contiene la relacion de TipoInfraccionId con Descripcion.
    num_filas_aleatorias (int): Numero de filas de datos aleatorios a generar para pruebas.

    Returns:
    pd.DataFrame: DataFrame con los datos procesados y ampliados con datos aleatorios.
    """
    # Limpiar y procesar los datos

    # Creo un df nuevo con las columnas que necesito
    df = data[['LatitudInfraccion', 'LongitudInfraccion', 'TipoInfraccionId']]

    # Verifico tipos de dato
    print(df.dtypes)
    
    # Intento convertir 'LatitudInfraccion' y 'LongitudInfraccion' a tipo float. Los nulls los pongo en NaN.
    df['LatitudInfraccion'] = _pandas.to_numeric(df['LatitudInfraccion'], errors='coerce')
    df['LongitudInfraccion'] = _pandas.to_numeric(df['LongitudInfraccion'], errors='coerce')

    # Relleno valores nulls
    df = RellenarNullsConLaMedia(df, 'LatitudInfraccion')
    df = RellenarNullsConLaMedia(df, 'LongitudInfraccion')

    # Creo nuevas columnas para tener la fecha separada
    df['HoraDelDia'] = data['FechaYHoraInfraccion'].dt.hour
    df['DiaDeLaSemana'] = data['FechaYHoraInfraccion'].dt.dayofweek  # Lunes = 0, Domingo = 6
    df['Mes'] = data['FechaYHoraInfraccion'].dt.month

    # Creo un diccionario de mapeo desde el DataFrame dataTipoInf
    tipo_infraccion_dict = dataTipoInf.set_index('TipoInfraccionId')['Descripcion'].to_dict()

    # Uso map() para asignar las descripciones de las infracciones a la columna 'TipoInfraccion'
    df['TipoInfraccion'] = df['TipoInfraccionId'].map(tipo_infraccion_dict)

    # Elimino la columna 'TipoInfraccionId' que asigne temporalmente
    df.drop(columns=['TipoInfraccionId'], inplace=True)
    
    # Limpiar y procesar los datos

    # Generacion de datos aleatorios para probar en desarrollo
    df_aleatorio = _generacionDatos.GenerarDatosAleatorios(num_filas_aleatorias)
    
    # Concateno los datos originales con los datos generados aleatorios
    df = _pandas.concat([df, df_aleatorio], ignore_index=True)
    # FIN Generacion de datos aleatorios
    
    return df


def RellenarNullsConLaMedia(df, nombreColumna):
    """
    Rellena los valores nulos en la columna especificada con la media de esa columna.

    Args:
    df (pd.DataFrame): El DataFrame que contiene los datos.
    nombreColumna (str): El nombre de la columna en la que se deben rellenar los valores nulos.

    Returns:
    pd.DataFrame: El DataFrame modificado con los valores nulos rellenados con la media.
    """
    df[nombreColumna] = df[nombreColumna].fillna(df[nombreColumna].mean())
    return df


def EliminarColumna(df, nombreColumna):
    """
    Elimina la columna especificada del DataFrame.

    Args:
    df (pd.DataFrame): El DataFrame del que se eliminara la columna.
    nombreColumna (str): El nombre de la columna que se debe eliminar.

    Returns:
    pd.DataFrame: El DataFrame modificado sin la columna especificada.
    """
    df = df.drop(nombreColumna, axis=1)
    return df


def EliminarFilasConNulls(df, columnas=None):
    """
    Elimina las filas que contienen valores nulos en las columnas especificadas.

    Args:
    df (pd.DataFrame): El DataFrame que contiene los datos.
    columnas (list, optional): Lista de nombres de columnas en las cuales se buscaran valores nulos.
                               Si no se especifica, se eliminaran filas con nulls en cualquier columna.

    Returns:
    pd.DataFrame: El DataFrame modificado sin las filas que contienen valores nulos.
    """
    df = df.dropna(subset=columnas)
    return df


def EliminarDuplicados(df):
    """
    Elimina filas duplicadas en el DataFrame.

    Args:
    df (pd.DataFrame): El DataFrame del cual se eliminaran las filas duplicadas.

    Returns:
    pd.DataFrame: El DataFrame modificado sin filas duplicadas.
    """
    df = df.drop_duplicates()
    return df


def RellenarNullsConValor(df, nombreColumna, valor):
    """
    Rellena los valores nulos en la columna especificada con un valor especifico.

    Args:
    df (pd.DataFrame): El DataFrame que contiene los datos.
    nombreColumna (str): El nombre de la columna en la que se deben rellenar los valores nulos.
    valor: El valor con el que se rellenaran los valores nulos.

    Returns:
    pd.DataFrame: El DataFrame modificado con los valores nulos rellenados con el valor especificado.
    """
    df[nombreColumna] = df[nombreColumna].fillna(valor)
    return df


def ConvertirATipo(df, nombreColumna, tipo):
    """
    Convierte los datos de la columna especificada a un tipo de dato especifico.

    Args:
    df (pd.DataFrame): El DataFrame que contiene los datos.
    nombreColumna (str): El nombre de la columna a convertir.
    tipo: El tipo de dato al cual se quiere convertir la columna (por ejemplo, float, int, str).

    Returns:
    pd.DataFrame: El DataFrame modificado con la columna convertida al tipo de dato especificado.
    """
    df[nombreColumna] = df[nombreColumna].astype(tipo)
    return df


def NormalizarColumna(df, nombreColumna):
    """
    Normaliza los valores de una columna para que esten en un rango de 0 a 1.

    Args:
    df (pd.DataFrame): El DataFrame que contiene los datos.
    nombreColumna (str): El nombre de la columna a normalizar.

    Returns:
    pd.DataFrame: El DataFrame modificado con los valores de la columna normalizados.
    """
    df[nombreColumna] = (df[nombreColumna] - df[nombreColumna].min()) / (df[nombreColumna].max() - df[nombreColumna].min())
    return df


def CodificarColumnasCategoricas(df, nombreColumna):
    """
    Convierte columnas categoricas en variables dummy para su uso en modelos de Machine Learning.

    Args:
    df (pd.DataFrame): El DataFrame que contiene los datos.
    nombreColumna (str): El nombre de la columna categorica a convertir en variables dummy.

    Returns:
    pd.DataFrame: El DataFrame modificado con las columnas categoricas convertidas en variables dummy.
    """
    df = _pandas.get_dummies(df, columns=[nombreColumna])
    return df


# JOINS ->
def UnirTablasPorColumna(df1, df2, columna, tipo_union='inner'):
    """
    Une dos DataFrames basandose en una columna comun.

    Args:
    df1 (pd.DataFrame): El primer DataFrame.
    df2 (pd.DataFrame): El segundo DataFrame.
    columna (str): El nombre de la columna comun en ambos DataFrames para realizar la union.
    tipo_union (str, optional): El tipo de union a realizar: 'inner', 'left', 'right', 'outer'. 
                                Valor por defecto es 'inner'.

    Returns:
    pd.DataFrame: El DataFrame resultante despues de la union.
    """
    return _pandas.merge(df1, df2, on=columna, how=tipo_union)

def UnirTablasPorIndices(df1, df2, tipo_union='inner'):
    """
    Une dos DataFrames basandose en sus indices.

    Args:
    df1 (pd.DataFrame): El primer DataFrame.
    df2 (pd.DataFrame): El segundo DataFrame.
    tipo_union (str, optional): El tipo de union a realizar: 'inner', 'left', 'right', 'outer'. 
                                Valor por defecto es 'inner'.

    Returns:
    pd.DataFrame: El DataFrame resultante despues de la union.
    """
    return df1.join(df2, how=tipo_union)
    """
    Une dos DataFrames basandose en sus indices.

    Args:
    df1 (pd.DataFrame): El primer DataFrame.
    df2 (pd.DataFrame): El segundo DataFrame.
    tipo_union (str, optional): El tipo de union a realizar: 'inner', 'left', 'right', 'outer'. 
                                Valor por defecto es 'inner'.

    Returns:
    pd.DataFrame: El DataFrame resultante despues de la union.
    """
    return df1.join(df2, how=tipo_union)

def ConcatenarTablas(lista_dfs, axis=0):
    """
    Concatena una lista de DataFrames.

    Args:
    lista_dfs (list of pd.DataFrame): Lista de DataFrames a concatenar.
    axis (int, optional): El eje a lo largo del cual concatenar: 0 para filas, 1 para columnas. 
                          Valor por defecto es 0.

    Returns:
    pd.DataFrame: El DataFrame resultante despues de la concatenacion.
    """
    return _pandas.concat(lista_dfs, axis=axis)

def UnirTablasConClavesDiferentes(df1, df2, columna_df1, columna_df2, tipo_union='inner'):
    """
    Une dos DataFrames basandose en diferentes nombres de columnas clave.

    Args:
    df1 (pd.DataFrame): El primer DataFrame.
    df2 (pd.DataFrame): El segundo DataFrame.
    columna_df1 (str): El nombre de la columna en df1 para la union.
    columna_df2 (str): El nombre de la columna en df2 para la union.
    tipo_union (str, optional): El tipo de union a realizar: 'inner', 'left', 'right', 'outer'. 
                                Valor por defecto es 'inner'.

    Returns:
    pd.DataFrame: El DataFrame resultante despues de la union.
    """
    return _pandas.merge(df1, df2, left_on=columna_df1, right_on=columna_df2, how=tipo_union)

def UnirTablasConMultiplesColumnas(df1, df2, columnas_comunes, tipo_union='inner'):
    """
    Une dos DataFrames basandose en multiples columnas comunes.

    Args:
    df1 (pd.DataFrame): El primer DataFrame.
    df2 (pd.DataFrame): El segundo DataFrame.
    columnas_comunes (list of str): Lista de nombres de columnas comunes en ambos DataFrames para la union.
    tipo_union (str, optional): El tipo de union a realizar: 'inner', 'left', 'right', 'outer'. 
                                Valor por defecto es 'inner'.

    Returns:
    pd.DataFrame: El DataFrame resultante despues de la union.
    """
    return _pandas.merge(df1, df2, on=columnas_comunes, how=tipo_union)
