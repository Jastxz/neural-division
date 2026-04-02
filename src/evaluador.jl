"""
Evaluador de subconfiguraciones para el Método de la División Neuronal.

Realiza forward pass con activación sigmoide y calcula precisión global
y parcial sobre datos de validación.
"""

"""
    sigmoid(x)

Función de activación sigmoide: σ(x) = 1 / (1 + exp(-x))
"""
sigmoid(x) = one(x) / (one(x) + exp(-x))

"""
    forward_pass(subconfig::Subconfiguracion{T}, entrada::AbstractMatrix{T}) where T

Realiza el forward pass a través de las capas de la subconfiguración.
Cada capa aplica: output = sigmoid(input * weights .+ biases')

`entrada` tiene dimensiones (n_muestras × n_features_entrada).
Devuelve una matriz (n_muestras × n_salidas).
"""
function forward_pass(subconfig::Subconfiguracion{T}, entrada::AbstractMatrix{T}) where T
    activacion = entrada
    for i in eachindex(subconfig.pesos)
        # activacion: (muestras × dim_in), pesos[i]: (dim_in × dim_out)
        # biases[i]: (dim_out,) → broadcast como fila
        activacion = sigmoid.(activacion * subconfig.pesos[i] .+ subconfig.biases[i]')
    end
    return activacion
end

"""
    subconjuntos_no_vacios(indices::Vector{Int})

Genera todos los subconjuntos no vacíos de `indices` usando enumeración por bitmask.
"""
function subconjuntos_no_vacios(indices::Vector{Int})
    n = length(indices)
    resultado = Vector{Int}[]
    for mask in 1:((1 << n) - 1)
        subconjunto = Int[]
        for i in 0:(n - 1)
            if (mask >> i) & 1 == 1
                push!(subconjunto, indices[i + 1])
            end
        end
        push!(resultado, subconjunto)
    end
    return resultado
end

"""
    evaluar(subconfig::Subconfiguracion{T}, datos_validacion, indices_salida_totales::Vector{Int})::ResultadoEvaluacion{T} where T

Evalúa una subconfiguración contra datos de validación.

Parámetros:
- `subconfig`: Subconfiguración a evaluar
- `datos_validacion`: NamedTuple o struct con campos `entradas::Matrix{T}` y `salidas::Matrix{T}`
  - `entradas`: (n_muestras × n_features_totales) — se seleccionan las columnas de `subconfig.indices_entrada`
  - `salidas`: (n_muestras × n_salidas_totales) — se seleccionan las columnas de `subconfig.indices_salida`
- `indices_salida_totales`: Todos los índices de salida del problema (para calcular subconjuntos parciales)

Devuelve `ResultadoEvaluacion{T}` con:
- `precision_global`: Fracción de muestras donde TODAS las salidas de la subconfiguración son correctas
- `precisiones_parciales`: Dict con precisión para cada subconjunto no vacío de `indices_salida_totales`
"""
function evaluar(subconfig::Subconfiguracion{T}, datos_validacion, indices_salida_totales::Vector{Int})::ResultadoEvaluacion{T} where T
    entradas = datos_validacion.entradas
    salidas = datos_validacion.salidas
    n_muestras = size(entradas, 1)

    # Seleccionar columnas de entrada según la subconfiguración
    entrada_sub = entradas[:, subconfig.indices_entrada]

    # Forward pass
    predicciones_raw = forward_pass(subconfig, entrada_sub)

    # Binarizar predicciones (umbral 0.5) y targets
    predicciones_bin = predicciones_raw .> T(0.5)
    targets_bin = round.(Int, salidas[:, subconfig.indices_salida])

    # Precisión global: fracción de muestras donde TODAS las salidas coinciden
    correctas_global = 0
    for i in 1:n_muestras
        if all(predicciones_bin[i, :] .== targets_bin[i, :])
            correctas_global += 1
        end
    end
    precision_global = T(correctas_global) / T(n_muestras)

    # Precisiones parciales para cada subconjunto no vacío de indices_salida_totales
    todos_subconjuntos = subconjuntos_no_vacios(indices_salida_totales)
    precisiones_parciales = Dict{Vector{Int}, T}()

    for subconj in todos_subconjuntos
        # Encontrar cuáles de estos índices están en la subconfiguración
        indices_en_subconfig = Int[]
        for idx in subconj
            pos = findfirst(==(idx), subconfig.indices_salida)
            if pos !== nothing
                push!(indices_en_subconfig, pos)
            end
        end

        if isempty(indices_en_subconfig)
            # La subconfiguración no cubre ninguna salida de este subconjunto
            precisiones_parciales[subconj] = T(0)
        else
            # Calcular precisión para este subconjunto
            correctas = 0
            for i in 1:n_muestras
                if all(predicciones_bin[i, indices_en_subconfig] .== targets_bin[i, indices_en_subconfig])
                    correctas += 1
                end
            end
            precisiones_parciales[subconj] = T(correctas) / T(n_muestras)
        end
    end

    return ResultadoEvaluacion{T}(precision_global, precisiones_parciales)
end


