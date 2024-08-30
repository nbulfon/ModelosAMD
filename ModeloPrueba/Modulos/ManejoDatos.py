# Modulo que se va a encargar del data cleaning.

import pandas as _pandas
import ConexionBd as _context
import GeneracionDatosAleatorios as _generacionDatos

# Esta funcion, es para preparar los datos del modelo que intentara predecir la probabilidad de ocurrencia 
# de un tipo de infraccion en una determinada ubicacion.
def PrepararDatos_Modelo_1(num_filas_aleatorias_requeridas):
    """
    Limpia y procesa los datos, y genera datos aleatorios adicionales.
    Esta funcion, es para el modelo que intentara predecir la probabilidad de ocurrencia 
    de un tipo de infraccion en una determinada ubicacion.

    Args:
    num_filas_aleatorias_requeridas (int): Numero de filas de datos aleatorios a generar para pruebas.

    Returns:
    pd.DataFrame: DataFrame con los datos procesados y ampliados con datos aleatorios.
    """
    
    # Conectar a la base de datos y obtener datos
    data = _context.EjecutarQuery('SELECT * FROM "Infraccion"')
    dataTipoInf = _context.EjecutarQuery('SELECT * FROM "TipoInfraccion"')
    # FIN Conectar a la base de datos y obtener datos
    
    
    # Limpiar y procesar los datos

    # Creo un df nuevo con las columnas que necesito
    df = data[['LatitudInfraccion', 'LongitudInfraccion', 'TipoInfraccionId']]

    # Verifico tipos de dato
    print(df.dtypes)
    
    # Intento convertir 'LatitudInfraccion' y 'LongitudInfraccion' a tipo float. Los nulls los pongo en NaN.
    df['LatitudInfraccion'] = _pandas.to_numeric(df['LatitudInfraccion'], errors='coerce')
    df['LongitudInfraccion'] = _pandas.to_numeric(df['LongitudInfraccion'], errors='coerce')

    # Relleno valores NaN con la media. -----> ESTO ES UN CRITERIO DISCRECIONAL.HAY QUE CHARLARLO.
    df = RellenarNullsConLaMedia(df, 'LatitudInfraccion')
    df = RellenarNullsConLaMedia(df, 'LongitudInfraccion')

    # Creo nuevas columnas para tener la fecha separada
    df['HoraDelDia'] = data['FechaYHoraInfraccion'].dt.hour
    df['DiaDeLaSemana'] = data['FechaYHoraInfraccion'].dt.dayofweek  # Lunes = 0, Domingo = 6
    df['Mes'] = data['FechaYHoraInfraccion'].dt.month

    # mapeo de columnmas para tener la descripcion y no el id en mi df a utilizar.
    df = MapearColumna(df, dataTipoInf, 'TipoInfraccionId', 'Descripcion', 'TipoInfraccion')
    
    # Generacion aleatoria de datos. Sirve para ambiente de DESARROLLO.
    if (num_filas_aleatorias_requeridas > 0):
        # Generacion de datos aleatorios para probar en desarrollo
        df_aleatorio = _generacionDatos.GenerarDatosAleatorios_Modelo_1(num_filas_aleatorias_requeridas)
        
        # Concateno los datos originales con los datos generados aleatorios
        df = _pandas.concat([df, df_aleatorio], ignore_index=True)
    # FIN Generacion de datos aleatorios
    
    return df

# Esta funcion, es para preparar los datos del modelo que intentara predecir
# la cantidad de infracciones, en una determinada ubicacion, en un determinado momento.
def PrepararDatos_Modelo_2(num_filas_aleatorias_requeridas=5000):
    """
    Limpia y procesa los datos, y genera datos aleatorios adicionales.
    Esta funcion es para el modelo que intentara predecir la cantidad de infracciones
    en una determinada ubicacion, en una determinada fecha (dia, mes o anio).

    Args:
    num_filas_aleatorias_requeridas (int): Numero de filas de datos aleatorios a generar para pruebas.

    Returns:
    pd.DataFrame: DataFrame con los datos procesados y agregados.
    """
    
    # Conectar a la base de datos y obtener datos
    data = _context.EjecutarQuery('SELECT * FROM "Infraccion"')
    dataTipoInf = _context.EjecutarQuery('SELECT * FROM "TipoInfraccion"')
    dataTipoVehiculo = _context.EjecutarQuery('SELECT * FROM "TipoVehiculo"')
    dataGrupoVehiculo = _context.EjecutarQuery('SELECT * FROM "GrupoVehiculo"')
    
    # Crear un df nuevo con las columnas que necesito
    df = data[['NumeroDeSerieEquipo', 'FechaYHoraInfraccion', 
               'TipoInfraccionId', 'GrupoVehiculoId', 'TipoVehiculoId',
               'ProvinciaInfraccion', 'PartidoInfraccion']]
    
    # Rellenar valores NaN con la media o la moda s/corresponda.
    df = RellenarNullsConLaMedia(df, 'TipoVehiculoId')
    df = RellenarNullsConLaMedia(df, 'GrupoVehiculoId')
    df = RellenarNullsConLaModa(df, 'ProvinciaInfraccion')
    df = RellenarNullsConLaModa(df, 'PartidoInfraccion')
    
    # Crear nuevas columnas para tener la fecha separada
    df['HoraDelDia'] = data['FechaYHoraInfraccion'].dt.hour
    df['DiaDeLaSemana'] = data['FechaYHoraInfraccion'].dt.dayofweek  # Lunes = 0, Domingo = 6
    df['Mes'] = data['FechaYHoraInfraccion'].dt.month

    # Mapear descripciones
    df = MapearColumna(df, dataTipoInf, 'TipoInfraccionId', 'Descripcion', 'TipoInfraccion')
    df = MapearColumna(df, dataTipoVehiculo, 'TipoVehiculoId', 'Descripcion', 'TipoVehiculo')
    df = MapearColumna(df, dataGrupoVehiculo, 'GrupoVehiculoId', 'Descripcion', 'GrupoVehiculo')
    
    # Generacion aleatoria de datos
    if (num_filas_aleatorias_requeridas > 0):
        df_aleatorio = _generacionDatos.GenerarDatosAleatorios_Modelo_2(
            dataTipoInf.columns,
            dataTipoVehiculo.columns,
            dataGrupoVehiculo.columns,
            data['NumeroDeSerieEquipo'].values,
            numeroFilasRequeridas=5000)
        df = _pandas.concat([df, df_aleatorio], ignore_index=True)


    df_agrupado = df.groupby('NumeroDeSerieEquipo').agg(
       CantidadInfracciones=('NumeroDeSerieEquipo', 'size'),
       TipoInfraccion=('TipoInfraccion', 'first'),              # o 'mode' para el valor más frecuente
       TipoVehiculo=('TipoVehiculo', 'first'),                  # o 'mode'
       GrupoVehiculo=('GrupoVehiculo', 'first'),                # o 'mode'
       HoraDelDia=('HoraDelDia', 'first'),                      # o 'mean' si necesitas el promedio
       DiaDeLaSemana=('DiaDeLaSemana', 'first'),                # o 'mean'
       Mes=('Mes', 'first'),
       Provinci=('ProvinciaInfraccion','first'),
       Partido=('PartidoInfraccion','first')
   ).reset_index()
    
    
    # Hacer el groupBy por NumeroDeSerieEquipo, para luego predecir CantidadInfracciones,
    # me deja con muy pocos grados de libertad...por lo cual, tendria, o ver que en produccion, hayan muchos 
    # equipos diferentes (algo raro), o, usar otra variable para agrupar, o, no agrupar y usar otra variable...
    
    return df_agrupado

# Esta funcion, es para preparar los datos del modelo que intentara predecir la probabilidad de ocurrencia
# de que los proximos n registros sean con infraccion.
def PrepararDatos_Modelo_3(siGeneraratosAleatorios=False):
    """
    Limpia y procesa los datos, y genera datos aleatorios adicionales.
    Esta funcion, es para el modelo que intentara predecir la probabilidad de ocurrencia 
    de que los proximos n registros sean con infraccion.

    Args:
    num_filas_aleatorias_requeridas (int): Numero de filas de datos aleatorios a generar para pruebas.

    Returns:
    pd.DataFrame: DataFrame con los datos procesados y ampliados con datos aleatorios.
    """
    
    # Conectar a la base de datos y obtener datos
    data = _context.EjecutarQuery('SELECT * FROM "Infraccion"')
    data['Si_Infraccion'] = True
    dataTipoInf = _context.EjecutarQuery('SELECT * FROM "TipoInfraccion"')
    dataGrupoVehiculo = _context.EjecutarQuery('SELECT * FROM "GrupoVehiculo"')
    dataTipoVehiculo = _context.EjecutarQuery('SELECT * FROM "TipoVehiculo"')
    def asignar_grupo_a_tipoVehiculo(descripcion):
        if descripcion.startswith('L'):
            return 'L'
        elif descripcion.startswith('N'):
            return 'N'
        elif descripcion.startswith('M'):
            return 'M'
        elif descripcion.startswith('O'):
            return 'O'
    
    dataTipoVehiculo['GrupoTipoVehiculo'] = dataTipoVehiculo['Descripcion'].apply(asignar_grupo_a_tipoVehiculo) 

    
    ## -----ESTA PARTE LUEGO SE VA, FALTA AGREGAR LA TABLA EN LA BASE DE DATOS-----
    filepath = r"C:\\Nicolas\\ModelosAMD\\ModeloPrueba"
    file = filepath + "\\" + "URBAN_000372.csv"
    dataEstadisticas = _pandas.read_csv(file, sep=";")
    dataEstadisticas['Si_Infraccion'] = False
    estadisticas_seleccionadas = dataEstadisticas[['Si_Infraccion','VELOCIDAD','Carril']]
    
    dataEstadisticas['FECHA'] = _pandas.to_datetime(dataEstadisticas['FECHA'], format='%d/%m/%Y')
    dataEstadisticas['HORA'] = _pandas.to_datetime(dataEstadisticas['HORA'], format='%H:%M:%S')
    estadisticas_seleccionadas['Mes'] = dataEstadisticas['FECHA'].dt.month
    estadisticas_seleccionadas['DiaDeLaSemana'] = dataEstadisticas['FECHA'].dt.dayofweek
    estadisticas_seleccionadas['HoraDelDia'] = dataEstadisticas['HORA'].dt.hour
    
    # renombro columnas para luego hacerlas coincidir los datos en el df ->
    estadisticas_seleccionadas.rename(columns={'VELOCIDAD': 'VelocidadRegistrada'}, inplace=True)
    ## -----ESTA PARTE LUEGO SE VA, FALTA AGREGAR LA TABLA EN LA BASE DE DATOS-----
    
    # Crear un df nuevo con las columnas que necesito
    df = data[['Si_Infraccion','NumeroDeSerieEquipo', 'FechaYHoraInfraccion', 
               'TipoInfraccionId', 'GrupoVehiculoId', 'TipoVehiculoId',
               'VelocidadRegistrada','VelocidadPermitida',
               #'ProvinciaInfraccion', 'PartidoInfraccion'
               'LatitudInfraccion','LongitudInfraccion']]
    
    
    # Rellenar valores NaN con la media o la moda s/corresponda.
    df = RellenarNullsConLaMedia(df, 'TipoVehiculoId')
    df = RellenarNullsConLaMedia(df, 'GrupoVehiculoId')
    #df = RellenarNullsConLaModa(df, 'ProvinciaInfraccion')
    #df = RellenarNullsConLaModa(df, 'PartidoInfraccion')
    
    # convertir valores no convertibles a NaN
    df['LatitudInfraccion'] = _pandas.to_numeric(df['LatitudInfraccion'], errors='coerce')
    df['LongitudInfraccion'] = _pandas.to_numeric(df['LongitudInfraccion'], errors='coerce')
    df = RellenarNullsConLaMedia(df, 'LatitudInfraccion')
    df = RellenarNullsConLaMedia(df, 'LongitudInfraccion')
    
    # Crear nuevas columnas para tener la fecha separada
    df['HoraDelDia'] = data['FechaYHoraInfraccion'].dt.hour
    df['DiaDeLaSemana'] = data['FechaYHoraInfraccion'].dt.dayofweek  # Lunes = 0, Domingo = 6
    df['Mes'] = data['FechaYHoraInfraccion'].dt.month

    # Mapear descripciones
    df = MapearColumna(df, dataTipoInf, 'TipoInfraccionId', 'Descripcion', 'TipoInfraccion')
    df = MapearColumna(df, dataTipoVehiculo, 'TipoVehiculoId', 'Descripcion', 'TipoVehiculo')
    df = MapearColumna(df, dataGrupoVehiculo, 'GrupoVehiculoId', 'Descripcion', 'GrupoVehiculo')
    
    # datos compartidos entre los df:
    # Si_Infraccion, VelocidadRegistrada, Mes, DiaDeLaSemana, HoraDelDia
    # no compartido ---> Carril
    columnas_comunes = ['Si_Infraccion', 'VelocidadRegistrada', 'Mes', 'DiaDeLaSemana', 'HoraDelDia']
    
    df['VelocidadRegistrada'] = df['VelocidadRegistrada'].str.replace(',', '.').astype('float64')
    df['VelocidadPermitida'] = df['VelocidadPermitida'].str.replace(',', '.').astype('float64')
    df['LatitudInfraccion'] = df['LatitudInfraccion'].astype('float64')
    
    #print(df.LatitudInfraccion.dtypes)
    
    df = _pandas.concat([df, estadisticas_seleccionadas], ignore_index=True)
    df = EliminarColumna(df, 'FechaYHoraInfraccion')
    df = RellenarNullsConLaMedia(df, 'VelocidadPermitida')
    df = RellenarNullsConLaMedia(df, 'VelocidadRegistrada')
    df = RellenarNullsConLaMedia(df, 'Carril')
    df = RellenarNullsConLaModa(df, 'TipoVehiculo')
    df = RellenarNullsConLaModa(df, 'GrupoVehiculo')
    df = RellenarNullsConLaModa(df, 'TipoInfraccion')
    df = RellenarNullsConLaModa(df, 'NumeroDeSerieEquipo')
    #df = RellenarNullsConLaModa(df, 'ProvinciaInfraccion')
    #df = RellenarNullsConLaModa(df, 'PartidoInfraccion')
    df = RellenarNullsConLaModa(df, 'NumeroDeSerieEquipo')
    df = RellenarNullsConLaMedia(df, 'LatitudInfraccion')
    df = RellenarNullsConLaMedia(df, 'LongitudInfraccion')
    
    # Generacion aleatoria de datos
    if (siGeneraratosAleatorios):
        df_aleatorio = _generacionDatos.GenerarDatosAleatorios_Modelo_3(
            dataTipoInf['Descripcion'].values,
            dataTipoVehiculo['GrupoTipoVehiculo'].values,
            dataGrupoVehiculo['Descripcion'].values,
            data['NumeroDeSerieEquipo'].values,
            numeroFilasRequeridas=25000,
            proporción_clase_positiva=0.5)
        df = _pandas.concat([df, df_aleatorio], ignore_index=True)
    
    
    # Codificacion de variables categoricas. (Dummies)
    columnas_categoricas = ['NumeroDeSerieEquipo',
                            #'ProvinciaInfraccion',
                            #'PartidoInfraccion',
                            'TipoInfraccion',
                            'TipoVehiculo',
                            'GrupoVehiculo']

    for columna in columnas_categoricas:
        df = CodificarColumnasCategoricas(df, columna)
    # FIN Codificacion de variables categoricas (Dummies)

    # convierto todas las columnas a tipo float64 para luego hacer la regresion.
    df = df.astype(float)
    
    # muestro las primeras filas del DataFrame resultante ->
    print(df.head())
    print(f"Número total de filas en df: {len(df)}")

    # separo variable explicada y explicativas ->
    Y = df['Si_Infraccion']
    X = df.drop(columns=['Si_Infraccion'])
    
    return X , Y

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

def RellenarNullsConLaModa(df, nombreColumna):
    """
    Rellena los valores NaN en una columna de tipo string con el valor mas frecuente.

    Args:
    df (pd.DataFrame): DataFrame en el que se realizara la operacion.
    nombreColumna (str): Nombre de la columna a procesar.

    Returns:
    pd.DataFrame: DataFrame con los NaN rellenados.
    """
    moda = df[nombreColumna].mode()[0]
    df[nombreColumna] = df[nombreColumna].fillna(moda)
    return df

def MapearColumna(df_a_mapear, df_para_ser_mapeado, column_id, column_description, new_column_name):
    """
    Mapea una columna de un DataFrame utilizando un diccionario de mapeo creado
    a partir de otro DataFrame.

    Parametros:
    df (DataFrame): DataFrame principal al que se le aplicara el mapeo.
    df_para_ser_mapeado (DataFrame): DataFrame de mapeo que contiene las columnas para crear el diccionario.
    column_id (str): Nombre de la columna en df que sera utilizada para el mapeo.
    column_description (str): Nombre de la columna en mapping_df que contiene las descripciones para el mapeo.
    new_column_name (str): Nombre de la nueva columna que se creará en df después de aplicar el mapeo.

    Retorna:
    DataFrame: DataFrame actualizado con la nueva columna mapeada y sin la columna original.
    """
    # Creo un diccionario de mapeo desde el DataFrame mapping_df
    mapping_dict = df_para_ser_mapeado.set_index(column_id)[column_description].to_dict()

    # Uso map() para asignar las descripciones a la nueva columna
    df_a_mapear[new_column_name] = df_a_mapear[column_id].map(mapping_dict)

    # Elimino la columna original que se utilizó para el mapeo
    df_a_mapear.drop(columns=[column_id], inplace=True)

    return df_a_mapear

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

def SeleccionarCaracteristicasImportantes(X, forest, umbral):
    """
    Selecciona caracterIsticas importantes basadas en su importancia
    en el modelo de Random Forest.
    
    Args:
    forest (RandomForestClassifier): Modelo de Random Forest entrenado.
    umbral (float): Umbral de importancia para seleccionar características.
    
    Returns:
    list: Lista de nombres de caracterIsticas seleccionadas.
    """
    importancias = _pandas.Series(forest.feature_importances_, index=X.columns)
    caracteristicas_importantes = importancias[importancias > umbral].index.tolist()
    print(f"Importancias: {importancias}")
    print(f"CaracterIsticas seleccionadas (umbral={umbral}): {caracteristicas_importantes}")
    return caracteristicas_importantes

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
    
    # Agrego drop_first=True para evitar problemas de colinealidad
    # (donde una variable se puede predecir perfectamente con otras variables).
    # porque para una variable Dicotomica, el resultante debe ser n-1. Es decir, la
    # ultima se deduce por el negativo del resto.
    df = _pandas.get_dummies(df, columns=[nombreColumna], drop_first=True)
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
