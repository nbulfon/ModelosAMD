# Modulo generar datos aleatorios y demas.

import pandas as _pandas
import numpy as _numpy

def GenerarDatosAleatorios_Modelo_1(numeroFilasRequeridas):
    """
    Genera un DataFrame con datos estocasticos para pruebas.
    Este metodo se usa para el modelo de prediccion de escenario de probabilidades
    (usando algoritmo Regresion Logistica Multinomial)

    Args:
    num_filas_aleatorias (int): Número de filas de datos aleatorios a generar.

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
    num_filas_aleatorias (int): Número de filas de datos aleatorios a generar.

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

