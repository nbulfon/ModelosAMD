# Módulo RegLineal para el modelo 1 de prueba con datos de AM/AMD,etc.
# Actúa como punto de entrada para ejecutar el proyecto.
# Este script importa los otros módulos y orquesta el flujo completo desde la conexión
# a la base de datos hasta la evaluación del modelo y posteriores conclusiones.

###
### El objetivo de este primer modelo, es predecir la probabilidad de ocurrencia de un tipo de infraccion
### en una  determinada ubicacion (dada por latitud y longitud) y un determinado momento (FechaYHoraInfraccion).
### (Regresion logistica multinomial).
###