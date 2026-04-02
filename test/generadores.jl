"""
Generadores de datos aleatorios para tests de propiedades usando PropCheck.jl.

Cada generador produce instancias válidas o inválidas de los tipos del módulo
DivisionNeuronal para verificar propiedades universales.
"""

using PropCheck
using DivisionNeuronal

"""
    gen_red_base(T, max_entradas, max_salidas, max_capas_ocultas)

Genera `RedBase{T}` aleatorias válidas con:
- Entre 1 y `max_entradas` neuronas de entrada
- Entre 1 y `max_salidas` neuronas de salida
- Entre 0 y `max_capas_ocultas` capas ocultas (cada una con 1-8 neuronas)
- Pesos y biases con dimensiones consistentes
"""
function gen_red_base(::Type{T}, max_entradas::Int, max_salidas::Int, max_capas_ocultas::Int) where T <: AbstractFloat
    gen_n_ent = isample(1:max_entradas)
    gen_n_sal = isample(1:max_salidas)
    gen_n_ocultas = isample(0:max_capas_ocultas)

    gen_params = interleave(gen_n_ent, gen_n_sal, gen_n_ocultas)

    map(gen_params) do (n_ent, n_sal, n_ocultas)
        # Construir la arquitectura: dimensiones de cada capa
        dims = Int[n_ent]
        for _ in 1:n_ocultas
            push!(dims, rand(1:8))
        end
        push!(dims, n_sal)

        # Generar pesos y biases consistentes
        n_capas = length(dims) - 1
        pesos = Matrix{T}[]
        biases = Vector{T}[]
        for i in 1:n_capas
            push!(pesos, randn(T, dims[i], dims[i + 1]))
            push!(biases, randn(T, dims[i + 1]))
        end

        RedBase{T}(pesos, biases, n_ent, n_sal)
    end
end


"""
    gen_datos_validacion(n_entradas, n_salidas, n_muestras)

Genera conjuntos de datos de validación sintéticos como NamedTuple
`(entradas=Matrix{T}, salidas=Matrix{T})` con datos aleatorios.

- `n_entradas`: número de columnas de la matriz de entradas
- `n_salidas`: número de columnas de la matriz de salidas
- `n_muestras`: número de filas (muestras)

Las entradas son valores aleatorios en [0, 1] y las salidas son valores binarios (0.0 o 1.0).
"""
function gen_datos_validacion(::Type{T}, n_entradas::Int, n_salidas::Int, n_muestras::Int) where T <: AbstractFloat
    return map(iconst((n_entradas, n_salidas, n_muestras))) do (n_ent, n_sal, n_m)
        entradas = rand(T, n_m, n_ent)
        salidas = T.(rand(Bool, n_m, n_sal))
        (entradas=entradas, salidas=salidas)
    end
end


"""
    gen_mapa_soluciones(n_salidas)

Genera `MapaDeSoluciones{Float64}` aleatorios válidos para testing de serialización.
Produce mapas con algunas entradas pobladas (con subconfiguraciones aleatorias) y
algunas vacías (subconfiguracion = nothing), simulando un estado realista del mapa
tras una exploración parcial.
"""
function gen_mapa_soluciones(n_salidas::Int)
    gen_n = iconst(n_salidas)
    map(gen_n) do n_sal
        mapa = inicializar_mapa(n_sal, Float64)

        # Decidir aleatoriamente si poblar la entrada global
        if rand(Bool)
            n_ent = rand(1:4)
            idx_ent = sort(unique(rand(1:max(n_sal, 2), n_ent)))
            idx_sal = sort(unique(rand(1:n_sal, rand(1:n_sal))))
            isempty(idx_ent) && (idx_ent = [1])
            isempty(idx_sal) && (idx_sal = [1])

            # Crear una subconfiguracion simple (1 capa)
            n_e = length(idx_ent)
            n_s = length(idx_sal)
            pesos = [randn(Float64, n_e, n_s)]
            biases = [randn(Float64, n_s)]
            n_activas = n_e + n_s
            sc = Subconfiguracion{Float64}(idx_ent, idx_sal, pesos, biases, n_activas)

            mapa.global_.subconfiguracion = sc
            mapa.global_.precision = rand(Float64)
            mapa.global_.precision_pre_entrenamiento = rand(Float64)
            mapa.global_.precision_post_entrenamiento = rand(Float64)
        end

        # Poblar aleatoriamente algunas entradas parciales
        for (clave, entrada) in mapa.parciales
            if rand() < 0.4  # 40% de probabilidad de poblar cada entrada
                n_ent = rand(1:4)
                idx_ent = sort(unique(rand(1:max(n_sal, 2), n_ent)))
                isempty(idx_ent) && (idx_ent = [1])

                n_e = length(idx_ent)
                n_s = length(clave)
                pesos = [randn(Float64, n_e, n_s)]
                biases = [randn(Float64, n_s)]
                n_activas = n_e + n_s
                sc = Subconfiguracion{Float64}(idx_ent, copy(clave), pesos, biases, n_activas)

                entrada.subconfiguracion = sc
                entrada.precision = rand(Float64)
                entrada.precision_pre_entrenamiento = rand(Float64)
                entrada.precision_post_entrenamiento = rand(Float64)
            end
        end

        mapa
    end
end
