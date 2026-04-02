module DivisionNeuronal

# Tipos base
include("tipos.jl")

# Excepciones
include("errores.jl")

# Validación
include("validacion.jl")

# Generador de subconfiguraciones
include("generador.jl")

# Evaluador
include("evaluador.jl")

# Comparador (función es_mejor)
include("comparador.jl")

# Mapa de soluciones
include("mapa_soluciones.jl")

# Serialización y deserialización
include("serializacion.jl")

# Progreso y cancelación
include("progreso.jl")

# Entrenamiento
include("entrenamiento.jl")

# Motor de división (orquestador principal)
include("motor.jl")

# Selección de mejor solución
include("seleccion.jl")

# Exportar tipos
export RedBase, Subconfiguracion, ResultadoEvaluacion
export EntradaMapaSoluciones, MapaDeSoluciones
export ConfiguracionDivision, ProgresoExploracion

# Exportar excepciones
export RedBaseNoInicializadaError, NeuronasInvalidasError, UmbralFueraDeRangoError

# Exportar funciones de validación
export validar_red_base, validar_neuronas, validar_umbral

# Exportar generador
export GeneradorDeSubconfiguraciones, bitmask_a_indices, extraer_subconfiguracion

# Exportar evaluador
export evaluar, es_mejor

# Exportar mapa de soluciones
export inicializar_mapa, actualizar_si_mejor!

# Exportar serialización
export serializar, deserializar, formatear

# Exportar progreso y cancelación
export reportar_progreso, debe_parar

# Exportar entrenamiento
export entrenar_y_evaluar!, entrenar_mapa!, EstadoAdam

# Exportar motor
export ejecutar_division

# Exportar selección
export ResultadoSeleccion, calcular_score, seleccionar_mejor

end # module DivisionNeuronal
