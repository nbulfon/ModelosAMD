# Modulo que se va a encargar del data cleaning.

import pandas as _pandas

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
    df (pd.DataFrame): El DataFrame del que se eliminar� la columna.
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
    columnas (list, optional): Lista de nombres de columnas en las cuales se buscar�n valores nulos.
                               Si no se especifica, se eliminar�n filas con nulls en cualquier columna.

    Returns:
    pd.DataFrame: El DataFrame modificado sin las filas que contienen valores nulos.
    """
    df = df.dropna(subset=columnas)
    return df


def EliminarDuplicados(df):
    """
    Elimina filas duplicadas en el DataFrame.

    Args:
    df (pd.DataFrame): El DataFrame del cual se eliminar�n las filas duplicadas.

    Returns:
    pd.DataFrame: El DataFrame modificado sin filas duplicadas.
    """
    df = df.drop_duplicates()
    return df


def RellenarNullsConValor(df, nombreColumna, valor):
    """
    Rellena los valores nulos en la columna especificada con un valor espec�fico.

    Args:
    df (pd.DataFrame): El DataFrame que contiene los datos.
    nombreColumna (str): El nombre de la columna en la que se deben rellenar los valores nulos.
    valor: El valor con el que se rellenar�n los valores nulos.

    Returns:
    pd.DataFrame: El DataFrame modificado con los valores nulos rellenados con el valor especificado.
    """
    df[nombreColumna] = df[nombreColumna].fillna(valor)
    return df


def ConvertirATipo(df, nombreColumna, tipo):
    """
    Convierte los datos de la columna especificada a un tipo de dato espec�fico.

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
    Normaliza los valores de una columna para que est�n en un rango de 0 a 1.

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
    Convierte columnas categ�ricas en variables dummy para su uso en modelos de Machine Learning.

    Args:
    df (pd.DataFrame): El DataFrame que contiene los datos.
    nombreColumna (str): El nombre de la columna categ�rica a convertir en variables dummy.

    Returns:
    pd.DataFrame: El DataFrame modificado con las columnas categ�ricas convertidas en variables dummy.
    """
    df = _pandas.get_dummies(df, columns=[nombreColumna])
    return df


# JOINS ->
def UnirTablasPorColumna(df1, df2, columna, tipo_union='inner'):
    """
    Une dos DataFrames bas�ndose en una columna com�n.

    Args:
    df1 (pd.DataFrame): El primer DataFrame.
    df2 (pd.DataFrame): El segundo DataFrame.
    columna (str): El nombre de la columna com�n en ambos DataFrames para realizar la uni�n.
    tipo_union (str, optional): El tipo de uni�n a realizar: 'inner', 'left', 'right', 'outer'. 
                                Valor por defecto es 'inner'.

    Returns:
    pd.DataFrame: El DataFrame resultante despu�s de la uni�n.
    """
    return _pandas.merge(df1, df2, on=columna, how=tipo_union)

def UnirTablasPorIndices(df1, df2, tipo_union='inner'):
    """
    Une dos DataFrames bas�ndose en sus �ndices.

    Args:
    df1 (pd.DataFrame): El primer DataFrame.
    df2 (pd.DataFrame): El segundo DataFrame.
    tipo_union (str, optional): El tipo de uni�n a realizar: 'inner', 'left', 'right', 'outer'. 
                                Valor por defecto es 'inner'.

    Returns:
    pd.DataFrame: El DataFrame resultante despu�s de la uni�n.
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
    pd.DataFrame: El DataFrame resultante despu�s de la concatenaci�n.
    """
    return _pandas.concat(lista_dfs, axis=axis)

def UnirTablasConClavesDiferentes(df1, df2, columna_df1, columna_df2, tipo_union='inner'):
    """
    Une dos DataFrames bas�ndose en diferentes nombres de columnas clave.

    Args:
    df1 (pd.DataFrame): El primer DataFrame.
    df2 (pd.DataFrame): El segundo DataFrame.
    columna_df1 (str): El nombre de la columna en df1 para la uni�n.
    columna_df2 (str): El nombre de la columna en df2 para la uni�n.
    tipo_union (str, optional): El tipo de uni�n a realizar: 'inner', 'left', 'right', 'outer'. 
                                Valor por defecto es 'inner'.

    Returns:
    pd.DataFrame: El DataFrame resultante despu�s de la uni�n.
    """
    return _pandas.merge(df1, df2, left_on=columna_df1, right_on=columna_df2, how=tipo_union)

def UnirTablasConMultiplesColumnas(df1, df2, columnas_comunes, tipo_union='inner'):
    """
    Une dos DataFrames bas�ndose en m�ltiples columnas comunes.

    Args:
    df1 (pd.DataFrame): El primer DataFrame.
    df2 (pd.DataFrame): El segundo DataFrame.
    columnas_comunes (list of str): Lista de nombres de columnas comunes en ambos DataFrames para la uni�n.
    tipo_union (str, optional): El tipo de uni�n a realizar: 'inner', 'left', 'right', 'outer'. 
                                Valor por defecto es 'inner'.

    Returns:
    pd.DataFrame: El DataFrame resultante despu�s de la uni�n.
    """
    return _pandas.merge(df1, df2, on=columnas_comunes, how=tipo_union)





