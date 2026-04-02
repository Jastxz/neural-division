"""
Generador lazy de subconfiguraciones para el Método de la División Neuronal.

Itera sobre el producto cartesiano de todos los subconjuntos no vacíos de
neuronas de entrada × todos los subconjuntos no vacíos de neuronas de salida,
usando enumeración por bitmask para evitar materializar todas las combinaciones.
"""

"""
    GeneradorDeSubconfiguraciones{T <: AbstractFloat}

Iterador lazy que enumera todas las combinaciones posibles de subconjuntos
no vacíos de neuronas de entrada y salida de una `RedBase`.

Cada iteración produce un par `(indices_entrada::Vector{Int}, indices_salida::Vector{Int})`.
"""
struct GeneradorDeSubconfiguraciones{T <: AbstractFloat}
    red_base::RedBase{T}
end

"""
    bitmask_a_indices(mask::Int, n::Int) -> Vector{Int}

Convierte un bitmask entero en un vector de índices.
El bit `i` (0-indexed) encendido indica que el índice `i+1` está incluido.
"""
function bitmask_a_indices(mask::Integer, n::Int)
    indices = Int[]
    for i in 0:(n - 1)
        if (mask >> i) & 1 == 1
            push!(indices, i + 1)
        end
    end
    return indices
end

"""
    Base.length(gen::GeneradorDeSubconfiguraciones)

Devuelve el número total de subconfiguraciones: `(2^n_entradas - 1) × (2^n_salidas - 1)`.
"""
function Base.length(gen::GeneradorDeSubconfiguraciones)
    return (2^gen.red_base.n_entradas - 1) * (2^gen.red_base.n_salidas - 1)
end

"""
    Base.eltype(::Type{GeneradorDeSubconfiguraciones{T}}) where T

Tipo de elemento producido por el iterador.
"""
function Base.eltype(::Type{GeneradorDeSubconfiguraciones{T}}) where T
    return Tuple{Vector{Int}, Vector{Int}}
end

"""
    Base.iterate(gen::GeneradorDeSubconfiguraciones{T}, state=nothing) where T

Implementa el protocolo de iteración de Julia para el generador.

El estado es un par `(mask_entrada, mask_salida)` donde cada mask va de 1 a 2^n - 1.
- `mask_entrada` itera de 1 a `2^n_entradas - 1`
- `mask_salida` itera de 1 a `2^n_salidas - 1`

El orden de iteración es: para cada mask_entrada, recorrer todos los mask_salida.
"""
function Base.iterate(gen::GeneradorDeSubconfiguraciones{T}, state=nothing) where T
    n_ent = gen.red_base.n_entradas
    n_sal = gen.red_base.n_salidas
    max_mask_ent = (1 << n_ent) - 1  # 2^n_entradas - 1
    max_mask_sal = (1 << n_sal) - 1  # 2^n_salidas - 1

    # Estado inicial
    if state === nothing
        mask_ent = 1
        mask_sal = 1
    else
        (mask_ent, mask_sal) = state
    end

    # Verificar si hemos terminado
    if mask_ent > max_mask_ent
        return nothing
    end

    # Convertir bitmasks a vectores de índices
    indices_entrada = bitmask_a_indices(mask_ent, n_ent)
    indices_salida = bitmask_a_indices(mask_sal, n_sal)

    # Calcular siguiente estado
    next_mask_sal = mask_sal + 1
    next_mask_ent = mask_ent
    if next_mask_sal > max_mask_sal
        next_mask_sal = 1
        next_mask_ent = mask_ent + 1
    end

    return ((indices_entrada, indices_salida), (next_mask_ent, next_mask_sal))
end


"""
    extraer_subconfiguracion(red_base::RedBase{T}, indices_entrada, indices_salida) where T

Extrae una subconfiguración de la red base recortando las matrices de pesos
según los índices de entrada y salida seleccionados.

- La primera capa de pesos se recorta por filas (neuronas de entrada seleccionadas)
- La última capa de pesos se recorta por columnas (neuronas de salida seleccionadas)
- Las capas ocultas intermedias se mantienen completas
- Los biases de capas ocultas se mantienen completos; el último bias se recorta por `indices_salida`

Devuelve `nothing` si alguna matriz de pesos resultante está vacía.
"""
function extraer_subconfiguracion(red_base::RedBase{T}, indices_entrada, indices_salida) where T
    idx_ent = collect(Int, indices_entrada)
    idx_sal = collect(Int, indices_salida)
    n_capas = length(red_base.pesos)

    if n_capas == 0
        return nothing
    end

    nuevos_pesos = Matrix{T}[]
    nuevos_biases = Vector{T}[]

    if n_capas == 1
        # Caso especial: una sola capa (entrada → salida directa)
        W = red_base.pesos[1][idx_ent, idx_sal]
        if isempty(W)
            return nothing
        end
        push!(nuevos_pesos, W)
        push!(nuevos_biases, red_base.biases[1][idx_sal])
    else
        # Primera capa: recortar filas por índices de entrada
        W_primera = red_base.pesos[1][idx_ent, :]
        if isempty(W_primera)
            return nothing
        end
        push!(nuevos_pesos, W_primera)
        push!(nuevos_biases, copy(red_base.biases[1]))

        # Capas ocultas intermedias: se mantienen completas
        for i in 2:(n_capas - 1)
            W = red_base.pesos[i]
            if isempty(W)
                return nothing
            end
            push!(nuevos_pesos, copy(W))
            push!(nuevos_biases, copy(red_base.biases[i]))
        end

        # Última capa: recortar columnas por índices de salida
        W_ultima = red_base.pesos[end][:, idx_sal]
        if isempty(W_ultima)
            return nothing
        end
        push!(nuevos_pesos, W_ultima)
        push!(nuevos_biases, red_base.biases[end][idx_sal])
    end

    # Calcular neuronas activas: entrada + ocultas + salida
    n_ocultas = 0
    if n_capas > 1
        for i in 1:(length(red_base.biases) - 1)
            n_ocultas += length(red_base.biases[i])
        end
    end
    n_neuronas_activas = length(idx_ent) + n_ocultas + length(idx_sal)

    return Subconfiguracion{T}(
        idx_ent,
        idx_sal,
        nuevos_pesos,
        nuevos_biases,
        n_neuronas_activas
    )
end
