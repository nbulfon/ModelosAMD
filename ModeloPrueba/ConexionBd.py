# Módulo para la conexión a la base de datos de MercurIA.


import psycopg2
import pandas as _pandas

def ObtenerConexionBD():
    """
    Obtiene la conexion a la base de datos.
    """
    conn = psycopg2.connect(
        host='qa9.dotech.io',
        database='MercurIA',
        user='postgres',
        password='SQL01*',
        port='5432'
    )
    return conn

def EjecutarQuery(query):
    """
    Ejecuta una query.
    
    Args:
    query: query que se va a ejecutar.
    """
    conn = ObtenerConexionBD()
    df = _pandas.read_sql(query, conn)
    conn.close()
    return df




#------Esto es para probar que ande------
# Escribir una consulta SQL
#query = 'SELECT * FROM "Infraccion"'

# Ejecutar la consulta y traer los datos a un DataFrame de Pandas
#df = execute_query(query)

# Mostrar las primeras filas del DataFrame
#print(df)
#------------