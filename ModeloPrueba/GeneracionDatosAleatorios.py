# Modulo generar datos aleatorios y demas.

import pandas as _pandas
import numpy as _numpy

def GenerarDatosAleatorios(numeroFilasRequeridas):
    """
    Genera un DataFrame con datos estocasticos para pruebas.

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
