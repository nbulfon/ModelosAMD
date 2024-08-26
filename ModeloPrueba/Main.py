# Módulo Main.
# Actúa como punto de entrada para ejecutar el proyecto.
# Este script importa los otros módulos y orquesta el flujo completo desde la conexión
# a la base de datos hasta la evaluación del modelo.

import ConexionBd
import ManejoDatos
import EntrenamientoTesteoDatos
import VisualizacionDatos

# Paso 1: Conectar a la base de datos y obtener datos
query = 'SELECT * FROM nombre_de_la_tabla'
df = ConexionBd.execute_query(query)

# Paso 2: Limpiar y procesar los datos.
df = ManejoDatos.clean_data(df)
df = ManejoDatos.feature_engineering(df)

# Paso 3: Entrenar el modelo y Testearlo.
X = df.drop('target', axis=1)  # 'target' es la columna que quieres predecir
y = df['target']
model, accuracy = EntrenamientoTesteoDatos.train_model(X, y)
print(f'Accuracy: {accuracy}')

# Paso 4: Visualizar resultados


# Paso 5: Sacar conclusiones.

