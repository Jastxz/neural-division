"""
Embeddings y representaciones modernas para el Método de la División Neuronal.

Soporta:
- Embeddings aprendibles para features categóricas
- Representaciones densas combinando features continuas y embeddings
- Evaluación batch de subconfiguraciones con operaciones matriciales
"""

using LinearAlgebra

"""
    TablaEmbedding{T}

Tabla de embeddings para una feature categórica.
Mapea cada categoría (entero 1..n_categorias) a un vector denso de dimensión `dim_embedding`.
"""
mutable struct TablaEmbedding{T <: AbstractFloat}
    pesos::Matrix{T}        # n_categorias × dim_embedding
    n_categorias::Int
    dim_embedding::Int
end

function TablaEmbedding(::Type{T}, n_categorias::Int, dim_embedding::Int) where T
    pesos = randn(T, n_categorias, dim_embedding) .* T(0.1)
    TablaEmbedding{T}(pesos, n_categorias, dim_embedding)
end

"""
    lookup(tabla::TablaEmbedding{T}, indices::Vector{Int}) -> Matrix{T}

Busca los embeddings para los índices dados.
Devuelve una matriz (n_muestras × dim_embedding).
"""
function lookup(tabla::TablaEmbedding{T}, indices::AbstractVector{<:Integer}) where T
    return tabla.pesos[indices, :]
end

"""
    EsquemaEmbedding{T}

Define cómo transformar features crudas en representaciones densas.
Combina features continuas (pasadas directamente) con embeddings de features categóricas.
"""
struct EsquemaEmbedding{T <: AbstractFloat}
    tablas::Vector{TablaEmbedding{T}}           # una tabla por feature categórica
    indices_continuos::Vector{Int}               # índices de features continuas en los datos crudos
    indices_categoricos::Vector{Int}             # índices de features categóricas en los datos crudos
    dim_total::Int                               # dimensión total de la representación
end

"""
    crear_esquema(T, n_features, categoricos; dim_embedding=4)

Crea un esquema de embedding.
- `categoricos`: Dict(indice_feature => n_categorias) para cada feature categórica
- `dim_embedding`: dimensión de cada embedding (default 4)
"""
function crear_esquema(::Type{T}, n_features::Int, categoricos::Dict{Int,Int};
                        dim_embedding::Int=4) where T
    indices_cat = sort(collect(keys(categoricos)))
    indices_cont = [i for i in 1:n_features if !(i in indices_cat)]

    tablas = [TablaEmbedding(T, categoricos[i], dim_embedding) for i in indices_cat]

    dim_total = length(indices_cont) + length(indices_cat) * dim_embedding

    EsquemaEmbedding{T}(tablas, indices_cont, indices_cat, dim_total)
end

"""
    transformar(esquema, datos_crudos) -> Matrix{T}

Transforma datos crudos en representaciones densas usando el esquema de embedding.
- Features continuas: se copian directamente
- Features categóricas: se reemplazan por sus embeddings

Entrada: datos_crudos (n_muestras × n_features_crudas)
Salida: representacion (n_muestras × dim_total)
"""
function transformar(esquema::EsquemaEmbedding{T}, datos_crudos::AbstractMatrix) where T
    n_muestras = size(datos_crudos, 1)
    resultado = zeros(T, n_muestras, esquema.dim_total)

    col = 1

    # Copiar features continuas
    for idx in esquema.indices_continuos
        resultado[:, col] .= T.(datos_crudos[:, idx])
        col += 1
    end

    # Embeddings de features categóricas
    for (i, idx) in enumerate(esquema.indices_categoricos)
        indices = Int.(datos_crudos[:, idx])
        emb = lookup(esquema.tablas[i], indices)  # n_muestras × dim_embedding
        dim_e = esquema.tablas[i].dim_embedding
        resultado[:, col:col+dim_e-1] .= emb
        col += dim_e
    end

    return resultado
end

"""
    RedBaseConEmbedding{T}

Red base que incluye un esquema de embedding para transformar datos crudos
en representaciones densas antes de pasar por las capas de la red.
"""
struct RedBaseConEmbedding{T <: AbstractFloat}
    esquema::EsquemaEmbedding{T}
    red::RedBase{T}
end

"""
    crear_red_con_embedding(T, n_features_crudas, categoricos, capas_ocultas, n_salidas;
                             dim_embedding=4, seed=42)

Crea una red con embedding integrado.
- `capas_ocultas`: vector con neuronas por capa oculta, e.g. [32, 16]
- La primera capa de la red tiene dim_total entradas (tras embedding)
"""
function crear_red_con_embedding(::Type{T}, n_features_crudas::Int,
                                  categoricos::Dict{Int,Int},
                                  capas_ocultas::Vector{Int}, n_salidas::Int;
                                  dim_embedding::Int=4, seed::Int=42) where T
    Random.seed!(seed)

    esquema = crear_esquema(T, n_features_crudas, categoricos; dim_embedding=dim_embedding)

    # Construir capas: [dim_total, ocultas..., n_salidas]
    dims = vcat([esquema.dim_total], capas_ocultas, [n_salidas])
    n_capas = length(dims) - 1
    pesos = [randn(T, dims[i], dims[i+1]) for i in 1:n_capas]
    biases = [randn(T, dims[i+1]) for i in 1:n_capas]

    red = RedBase{T}(pesos, biases, esquema.dim_total, n_salidas)

    RedBaseConEmbedding{T}(esquema, red)
end

"""
    evaluar_batch_subconfiguraciones(red, datos, indices_salida_totales, subconfiguraciones)

Evalúa múltiples subconfiguraciones en batch usando operaciones matriciales.
Más eficiente que evaluar una por una cuando hay muchas subconfiguraciones.
"""
function evaluar_batch_subconfiguraciones(red::RedBase{T}, datos,
        indices_salida_totales::Vector{Int},
        subconfiguraciones::Vector{Tuple{Vector{Int}, Vector{Int}}};
        umbral::T=T(0.4)) where T

    resultados = Vector{Tuple{Union{Nothing, Subconfiguracion{T}}, ResultadoEvaluacion{T}}}()

    # Pre-computar datos de entrada una vez
    entradas = datos.entradas
    salidas = datos.salidas

    for (idx_ent, idx_sal) in subconfiguraciones
        subconfig = extraer_subconfiguracion(red, idx_ent, idx_sal)
        subconfig === nothing && continue

        # Forward pass vectorizado
        entrada_sub = entradas[:, subconfig.indices_entrada]
        activacion = entrada_sub
        for i in eachindex(subconfig.pesos)
            # Operación matricial batch: (n_muestras × dim_in) * (dim_in × dim_out)
            activacion = sigmoid.(activacion * subconfig.pesos[i] .+ subconfig.biases[i]')
        end

        # Calcular precisión global
        predicciones_bin = activacion .> T(0.5)
        targets_bin = round.(Int, salidas[:, subconfig.indices_salida])
        n_muestras = size(entradas, 1)

        correctas = sum(all(predicciones_bin[i, :] .== targets_bin[i, :]) for i in 1:n_muestras)
        precision_global = T(correctas) / T(n_muestras)

        # Precisiones parciales (simplificado para batch)
        todos_subconjuntos = subconjuntos_no_vacios(indices_salida_totales)
        precisiones_parciales = Dict{Vector{Int}, T}()
        for subconj in todos_subconjuntos
            indices_en_sub = Int[]
            for idx in subconj
                pos = findfirst(==(idx), subconfig.indices_salida)
                pos !== nothing && push!(indices_en_sub, pos)
            end
            if isempty(indices_en_sub)
                precisiones_parciales[subconj] = T(0)
            else
                corr = sum(all(predicciones_bin[i, indices_en_sub] .== targets_bin[i, indices_en_sub]) for i in 1:n_muestras)
                precisiones_parciales[subconj] = T(corr) / T(n_muestras)
            end
        end

        resultado = ResultadoEvaluacion{T}(precision_global, precisiones_parciales)
        push!(resultados, (subconfig, resultado))
    end

    return resultados
end
