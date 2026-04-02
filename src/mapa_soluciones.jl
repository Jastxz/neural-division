"""
Mapa de Soluciones para el Método de la División Neuronal.

Gestiona la inicialización y actualización del mapa que almacena las mejores
subconfiguraciones encontradas para la solución global y cada subconjunto de salidas.
"""

"""
    inicializar_mapa(n_salidas::Int, ::Type{T}; referencia::Union{Nothing, Subconfiguracion{T}}=nothing)::MapaDeSoluciones{T} where T

Inicializa un `MapaDeSoluciones{T}` con:
- Una entrada global con `subconfiguracion = nothing`
- Una entrada parcial por cada subconjunto no vacío de `1:n_salidas`, todas con `subconfiguracion = nothing`
- Una referencia completa (red con todas las entradas y salidas) si se proporciona

Las claves parciales son vectores de índices ordenados generados por bitmask.
Total de claves parciales: `2^n_salidas - 1`.
"""
function inicializar_mapa(n_salidas::Int, ::Type{T}; referencia::Union{Nothing, Subconfiguracion{T}}=nothing)::MapaDeSoluciones{T} where T
    # Crear entrada global vacía
    global_ = EntradaMapaSoluciones{T}(nothing, T(0), T(0), T(0))

    # Crear entradas parciales para cada subconjunto no vacío de 1:n_salidas
    parciales = Dict{Vector{Int}, EntradaMapaSoluciones{T}}()
    for mask in 1:((1 << n_salidas) - 1)
        indices = bitmask_a_indices(mask, n_salidas)
        parciales[indices] = EntradaMapaSoluciones{T}(nothing, T(0), T(0), T(0))
    end

    # Crear entrada de referencia completa
    ref_entrada = EntradaMapaSoluciones{T}(referencia, T(0), T(0), T(0))

    return MapaDeSoluciones{T}(global_, parciales, ref_entrada)
end

"""
    actualizar_si_mejor!(mapa::MapaDeSoluciones{T}, subconfig::Subconfiguracion{T}, resultado::ResultadoEvaluacion{T}, umbral::T) where T

Actualiza el mapa de soluciones si la subconfiguración es mejor que la almacenada.

Para la solución global:
- Si `resultado.precision_global > umbral` y `es_mejor(subconfig, precision_global, mapa.global_)`,
  reemplaza la entrada global.

Para cada solución parcial:
- Si la precisión parcial para ese subconjunto `> umbral` y `es_mejor(subconfig, prec, entrada)`,
  reemplaza la entrada parcial correspondiente.
"""
function actualizar_si_mejor!(mapa::MapaDeSoluciones{T}, subconfig::Subconfiguracion{T}, resultado::ResultadoEvaluacion{T}, umbral::T) where T
    # Comprobar solución global
    if resultado.precision_global > umbral && es_mejor(subconfig, resultado.precision_global, mapa.global_)
        mapa.global_.subconfiguracion = subconfig
        mapa.global_.precision = resultado.precision_global
    end

    # Comprobar cada solución parcial
    for (subconj, prec) in resultado.precisiones_parciales
        if haskey(mapa.parciales, subconj) && prec > umbral
            entrada = mapa.parciales[subconj]
            if es_mejor(subconfig, prec, entrada)
                entrada.subconfiguracion = subconfig
                entrada.precision = prec
            end
        end
    end
end
