# Modulo generar datos aleatorios y demas.

import pandas as _pandas
import numpy as _numpy

def GenerarDatosAleatorios_Modelo_1(numeroFilasRequeridas):
    """
    Genera un DataFrame con datos estocasticos para pruebas.
    Este metodo se usa para el modelo de prediccion de escenario de probabilidades
    (usando algoritmo Regresion Logistica Multinomial)

    Args:
    num_filas_aleatorias (int): Numero de filas de datos aleatorios a generar.

    Returns:
    pd.DataFrame: DataFrame con datos aleatorios.
    """
    # Generar datos aleatorios
    latitudes_aleatorias = _numpy.random.uniform(low=34.0, high=35.0, size=numeroFilasRequeridas)
    longitudes_aleatorias = _numpy.random.uniform(low=-119.0, high=-117.0, size=numeroFilasRequeridas)
    horas_aleatorias = _numpy.random.randint(0, 24, size=numeroFilasRequeridas)
    dias_semana_aleatorios = _numpy.random.randint(0, 7, size=numeroFilasRequeridas)
    meses_aleatorios = _numpy.random.randint(1, 13, size=numeroFilasRequeridas)

    
    tipos_infraccion = ['Velocidad', 'Luces', 'Semáforo','Estacionamiento']
    tipo_infraccion_aleatorios = _numpy.random.choice(tipos_infraccion, size=numeroFilasRequeridas)

    # Crear DataFrame de los datos aleatorios
    df_aleatorio = _pandas.DataFrame({
        'FechaYHoraInfraccion': 
            _pandas.to_datetime('2024-08-01')+ 
            _pandas.to_timedelta(_numpy.random.randint(
                0,
                86400,
                size=numeroFilasRequeridas),
                unit='s'),
        'LatitudInfraccion': latitudes_aleatorias,
        'LongitudInfraccion': longitudes_aleatorias,
        'HoraDelDia': horas_aleatorias,
        'DiaDeLaSemana': dias_semana_aleatorios,
        'Mes': meses_aleatorios,
        'TipoInfraccion': tipo_infraccion_aleatorios
    })
    
    return df_aleatorio

def GenerarDatosAleatorios_Modelo_2(columnas_tiposInfraccion,
                                    columnas_tiposVehiculo,
                                    columnas_gruposVehiculo,
                                    valores_numeroSerieEquipo,
                                    numeroFilasRequeridas):
    """
    Genera un DataFrame con datos estocasticos para pruebas.
    Este metodo se usa para el modelo de prediccion de cantidad de infracciones
    en base a parametros.
    (algoritmo Regresion Lineal Multiple)
    Args:
    num_filas_aleatorias (int): Numero de filas de datos aleatorios a generar.

    Returns:
    pd.DataFrame: DataFrame con datos aleatorios.
    """
    # Generar datos aleatorios
    horas_aleatorias = _numpy.random.randint(0, 24, size=numeroFilasRequeridas)
    dias_semana_aleatorios = _numpy.random.randint(0, 7, size=numeroFilasRequeridas)
    meses_aleatorios = _numpy.random.randint(1, 13, size=numeroFilasRequeridas)

    # Provincia
    provincias = ['Buenos Aires']
    provincias_aleatorios = _numpy.random.choice(provincias, size=numeroFilasRequeridas)
    
    # Partido
    partidos = ['La Plata', 'ALMIRANTE BROWN', 'San Vicente', 'San Nicolas', 'Ensenada', 'Berisso']
    partidos_aleatorios = _numpy.random.choice(partidos, size=numeroFilasRequeridas)
    
    # numero de serie equipo ->
    equipos = valores_numeroSerieEquipo
    equipos_aleatorios = _numpy.random.choice(equipos, size=numeroFilasRequeridas)
    
    # tipos infraccion ->
    tipos_infraccion = columnas_tiposInfraccion
    tipo_infraccion_aleatorios = _numpy.random.choice(tipos_infraccion, size=numeroFilasRequeridas)
    
    # tipos vehiculo ->
    tipos_vehiculo = columnas_tiposVehiculo
    tipo_vehiculo_aleatorios = _numpy.random.choice(tipos_vehiculo, size=numeroFilasRequeridas)
    
    # grupos vehiculo ->
    grupos_vehiculo = columnas_gruposVehiculo
    grupos_vehiculo_aleatorios = _numpy.random.choice(grupos_vehiculo, size=numeroFilasRequeridas)
    
    
    # Crear DataFrame de los datos aleatorios
    df_aleatorio = _pandas.DataFrame({
        'HoraDelDia': horas_aleatorias,
        'DiaDeLaSemana': dias_semana_aleatorios,
        'Mes': meses_aleatorios,
        'TipoInfraccion': tipo_infraccion_aleatorios,
        'TipoVehiculo': tipo_vehiculo_aleatorios,
        'GrupoVehiculo': grupos_vehiculo_aleatorios,
        'NumeroDeSerieEquipo': equipos_aleatorios,
        'ProvinciaInfraccion': provincias_aleatorios,
        'PartidoInfraccion':partidos_aleatorios
    })
    
    return df_aleatorio


def GenerarDatosAleatorios_Modelo_3(columnas_tiposInfraccion,
                                    columnas_tiposVehiculo,
                                    columnas_gruposVehiculo,
                                    valores_numeroSerieEquipo,
                                    numeroFilasRequeridas,
                                    proporción_clase_positiva):
    """
    Genera un DataFrame con datos estocasticos para pruebas.
    Este metodo se usa para el modelo de prediccion de la probabilidad de ocurrencia
    de que los proximos n registros sean con infraccion.
    (algoritmo Regresion Logistica Multinomial)
    Args:
    numeroFilasRequeridas (int): Numero de filas de datos aleatorios a generar.
    proporción_clase_positiva (float): Proporción de casos donde Si_Infraccion es True (entre 0 y 1).
    
    Returns:
    pd.DataFrame: DataFrame con datos aleatorios.
    """
    # Generar datos aleatorios para columnas comunes
    horas_aleatorias = _numpy.random.randint(0, 24, size=numeroFilasRequeridas)
    dias_semana_aleatorios = _numpy.random.randint(0, 7, size=numeroFilasRequeridas)
    meses_aleatorios = _numpy.random.randint(1, 13, size=numeroFilasRequeridas)

    # Generar valores booleanos aleatorios para Si_Infraccion,
    #con una proporcion determinada (recibida por param).
    si_infraccion_aleatorios = _numpy.random.choice(
       [True, False], 
       size=numeroFilasRequeridas, 
       p=[proporción_clase_positiva, 1 - proporción_clase_positiva]
    )

    # Generar velocidades aleatorias para VelocidadRegistrada y VelocidadPermitida
    velocidad_registrada_aleatoria = _numpy.random.uniform(20, 120, size=numeroFilasRequeridas)  # Ajustar el rango según sea necesario
    velocidad_permitida_aleatoria = _numpy.random.uniform(30, 100, size=numeroFilasRequeridas)  # Ajustar el rango según sea necesario
    
    # lat long ->
    latitudes_aleatorias = _numpy.random.uniform(low=34.0, high=35.0, size=numeroFilasRequeridas)
    longitudes_aleatorias = _numpy.random.uniform(low=-119.0, high=-117.0, size=numeroFilasRequeridas)
    
    # Provincia
    #provincias = ['Buenos Aires']
    #provincias_aleatorios = _numpy.random.choice(provincias, size=numeroFilasRequeridas)
    
    # Partido
    #partidos = ['La Plata', 'ALMIRANTE BROWN', 'San Vicente', 'San Nicolas', 'Ensenada', 'Berisso']
    #partidos_aleatorios = _numpy.random.choice(partidos, size=numeroFilasRequeridas)
    
    # Numero de serie equipo ->
    equipos_validos = [equipo for equipo in valores_numeroSerieEquipo if equipo is not None]
    if not equipos_validos:
       equipos_validos = 'NEO_0366'
    equipos_aleatorios = _numpy.random.choice(equipos_validos, size=numeroFilasRequeridas)
    
    # Tipos infraccion ->
    tipos_infraccion = columnas_tiposInfraccion
    tipo_infraccion_aleatorios = _numpy.random.choice(tipos_infraccion, size=numeroFilasRequeridas)
    
    # Tipos vehiculo ->
    tipos_vehiculo = columnas_tiposVehiculo
    tipo_vehiculo_aleatorios = _numpy.random.choice(tipos_vehiculo, size=numeroFilasRequeridas)
    
    # Grupos vehiculo ->
    grupos_vehiculo = columnas_gruposVehiculo
    grupos_vehiculo_aleatorios = _numpy.random.choice(grupos_vehiculo, size=numeroFilasRequeridas)
    
    # Generar datos aleatorios para la columna Carril
    carril_aleatorio = _numpy.random.choice([1, 2, 3], size=numeroFilasRequeridas)
    
    # Combinar fechas y horas en una sola columna de tipo datetime
    años_aleatorios = _numpy.random.randint(2020, 2025, size=numeroFilasRequeridas)  # rango de años puede ajustarse.
    fechas_aleatorias = _pandas.to_datetime({
        'year': años_aleatorios,
        'month': meses_aleatorios,
        'day': _numpy.random.randint(1, 29, size=numeroFilasRequeridas),  # para simplificar, usar hasta 28 días.
        'hour': horas_aleatorias,
        'minute': _numpy.random.randint(0, 60, size=numeroFilasRequeridas),
        'second': _numpy.random.randint(0, 60, size=numeroFilasRequeridas)
    })

    # Crear DataFrame de los datos aleatorios
    df_aleatorio = _pandas.DataFrame({
        'Si_Infraccion': si_infraccion_aleatorios,
        'NumeroDeSerieEquipo': equipos_aleatorios,
        'TipoInfraccion': tipo_infraccion_aleatorios,
        'GrupoVehiculo': grupos_vehiculo_aleatorios,
        'TipoVehiculo': tipo_vehiculo_aleatorios,
        'VelocidadRegistrada': velocidad_registrada_aleatoria,
        'VelocidadPermitida': velocidad_permitida_aleatoria,
        #'ProvinciaInfraccion': provincias_aleatorios,
        #'PartidoInfraccion': partidos_aleatorios,
        'HoraDelDia': horas_aleatorias,
        'DiaDeLaSemana': dias_semana_aleatorios,
        'Mes': meses_aleatorios,
        'Carril': carril_aleatorio,
        'LatitudInfraccion': latitudes_aleatorias,
        'LongitudInfraccion':longitudes_aleatorias
    })
    
    return df_aleatorio

