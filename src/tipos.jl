"""
Tipos paramétricos para el Método de la División Neuronal.

Todos los tipos numéricos son paramétricos en `T <: AbstractFloat`
para soportar Float32, Float64, etc.
"""

"""
    RedBase{T <: AbstractFloat}

Red neuronal inicializada con pesos aleatorios que sirve como punto de partida
para el proceso de división neuronal.
"""
struct RedBase{T <: AbstractFloat}
    pesos::Vector{Matrix{T}}      # Pesos por capa [entrada→oculta1, ..., ocultaN→salida]
    biases::Vector{Vector{T}}     # Biases por capa
    n_entradas::Int               # Número máximo de neuronas de entrada
    n_salidas::Int                # Número máximo de neuronas de salida
end

"""
    Subconfiguracion{T <: AbstractFloat}

Subred extraída de la RedBase, representando un subconjunto de neuronas
y conexiones que forma una subred funcional.
"""
struct Subconfiguracion{T <: AbstractFloat}
    indices_entrada::Vector{Int}   # Índices de neuronas de entrada seleccionadas
    indices_salida::Vector{Int}    # Índices de neuronas de salida seleccionadas
    pesos::Vector{Matrix{T}}       # Pesos recortados de la Red_Base
    biases::Vector{Vector{T}}      # Biases recortados
    n_neuronas_activas::Int        # Total de neuronas activas (entrada + ocultas + salida)
end

"""
    ResultadoEvaluacion{T <: AbstractFloat}

Resultado de evaluar una subconfiguración contra datos de validación.
"""
struct ResultadoEvaluacion{T <: AbstractFloat}
    precision_global::T                          # Precisión sobre todas las salidas
    precisiones_parciales::Dict{Vector{Int}, T}  # Precisión por subconjunto de salidas
end

"""
    EntradaMapaSoluciones{T <: AbstractFloat}

Entrada individual del mapa de soluciones. Es mutable para permitir
actualización de métricas pre/post entrenamiento.
"""
mutable struct EntradaMapaSoluciones{T <: AbstractFloat}
    subconfiguracion::Union{Nothing, Subconfiguracion{T}}
    precision::T
    precision_pre_entrenamiento::T
    precision_post_entrenamiento::T
end

"""
    MapaDeSoluciones{T <: AbstractFloat}

Diccionario que almacena las mejores subconfiguraciones encontradas
para la solución global y cada subconjunto de salidas.
Incluye una referencia completa (todas las entradas y salidas) como baseline.
"""
struct MapaDeSoluciones{T <: AbstractFloat}
    global_::EntradaMapaSoluciones{T}                        # Solución global (mejor subred)
    parciales::Dict{Vector{Int}, EntradaMapaSoluciones{T}}   # Soluciones parciales
    referencia_completa::EntradaMapaSoluciones{T}             # Red completa como baseline
end

"""
    ConfiguracionDivision{T <: AbstractFloat}

Parámetros de configuración del proceso de división neuronal.
"""
struct ConfiguracionDivision{T <: AbstractFloat}
    umbral_de_acierto::T    # Valor entre 0.0 y 1.0, por defecto 0.4
end

"""
    ProgresoExploracion

Estado de progreso reportado durante la exploración de subconfiguraciones.
No es paramétrico ya que solo contiene contadores enteros.
"""
struct ProgresoExploracion
    evaluadas::Int                # Subconfiguraciones evaluadas
    total::Int                    # Total de subconfiguraciones
    soluciones_globales::Int      # Número de soluciones globales encontradas
    soluciones_parciales::Int     # Número de soluciones parciales encontradas
end
