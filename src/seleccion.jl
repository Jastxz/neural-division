"""
Selección de la mejor solución del MapaDeSoluciones.

Compara la referencia completa contra las subredes encontradas
usando un score que pesa precisión > coste.
"""

"""
    ResultadoSeleccion{T}

Resultado de la selección de la mejor solución.
"""
struct ResultadoSeleccion{T <: AbstractFloat}
    tipo::Symbol                    # :referencia, :global, :parcial
    entrada::EntradaMapaSoluciones{T}
    clave::Vector{Int}              # [] para referencia/global, [i...] para parcial
    score::T
    precision::T
    neuronas::Int
    neuronas_referencia::Int        # neuronas de la red completa para calcular ahorro
end

"""
    calcular_score(precision::T, neuronas::Int, max_neuronas::Int; peso_precision::T=T(0.8)) where T

Calcula un score combinado de precisión y eficiencia.
- `peso_precision`: peso relativo de la precisión (0.8 por defecto = 80% precisión, 20% ahorro)
- El ahorro se normaliza como `1 - neuronas/max_neuronas`
"""
function calcular_score(precision::T, neuronas::Int, max_neuronas::Int; peso_precision::T=T(0.8)) where T
    peso_eficiencia = one(T) - peso_precision
    eficiencia = one(T) - T(neuronas) / T(max(max_neuronas, 1))
    return peso_precision * precision + peso_eficiencia * eficiencia
end

"""
    seleccionar_mejor(mapa::MapaDeSoluciones{T}; peso_precision::T=T(0.8))::ResultadoSeleccion{T} where T

Selecciona la mejor solución del mapa comparando:
1. La referencia completa (red con todas las entradas/salidas)
2. La solución global (mejor subred encontrada)
3. Las soluciones parciales

Usa un score que combina precisión post-entrenamiento y eficiencia (ahorro de neuronas),
pesando más la precisión (80% por defecto).

Devuelve un `ResultadoSeleccion` con la mejor opción.
"""
function seleccionar_mejor(mapa::MapaDeSoluciones{T}; peso_precision::T=T(0.8))::ResultadoSeleccion{T} where T
    ref = mapa.referencia_completa
    max_neuronas = ref.subconfiguracion !== nothing ? ref.subconfiguracion.n_neuronas_activas : 1

    mejor = nothing

    # Evaluar referencia completa
    if ref.subconfiguracion !== nothing
        prec = ref.precision_post_entrenamiento
        neur = ref.subconfiguracion.n_neuronas_activas
        s = calcular_score(prec, neur, max_neuronas; peso_precision=peso_precision)
        mejor = ResultadoSeleccion{T}(:referencia, ref, Int[], s, prec, neur, max_neuronas)
    end

    # Evaluar solución global
    g = mapa.global_
    if g.subconfiguracion !== nothing
        prec = g.precision_post_entrenamiento
        neur = g.subconfiguracion.n_neuronas_activas
        s = calcular_score(prec, neur, max_neuronas; peso_precision=peso_precision)
        if mejor === nothing || s > mejor.score
            mejor = ResultadoSeleccion{T}(:global, g, Int[], s, prec, neur, max_neuronas)
        end
    end

    # Evaluar soluciones parciales
    for (clave, entrada) in mapa.parciales
        if entrada.subconfiguracion !== nothing
            prec = entrada.precision_post_entrenamiento
            neur = entrada.subconfiguracion.n_neuronas_activas
            s = calcular_score(prec, neur, max_neuronas; peso_precision=peso_precision)
            if mejor === nothing || s > mejor.score
                mejor = ResultadoSeleccion{T}(:parcial, entrada, clave, s, prec, neur, max_neuronas)
            end
        end
    end

    return mejor
end
