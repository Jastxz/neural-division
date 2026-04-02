"""
Entrenamiento de subconfiguraciones para el Método de la División Neuronal.

Implementa Adam optimizer con mini-batches, early stopping, y backpropagation
con activación sigmoide y binary cross-entropy loss.
"""

using Random: randperm
using Statistics: mean

"""
    EstadoAdam{T}

Estado del optimizador Adam para una capa (pesos + biases).
"""
mutable struct EstadoAdam{T <: AbstractFloat}
    m_w::Matrix{T}   # Primer momento (media) de gradientes de pesos
    v_w::Matrix{T}   # Segundo momento (varianza) de gradientes de pesos
    m_b::Vector{T}   # Primer momento de gradientes de biases
    v_b::Vector{T}   # Segundo momento de gradientes de biases
end

function crear_estado_adam(::Type{T}, pesos::Vector{Matrix{T}}, biases::Vector{Vector{T}}) where T
    [EstadoAdam{T}(
        zeros(T, size(pesos[i])),
        zeros(T, size(pesos[i])),
        zeros(T, length(biases[i])),
        zeros(T, length(biases[i]))
    ) for i in eachindex(pesos)]
end

"""
Aplica un paso de Adam para una capa.
"""
function adam_paso!(pesos::Matrix{T}, biases::Vector{T}, estado::EstadoAdam{T},
                    grad_w::Matrix{T}, grad_b::Vector{T}, t::Int;
                    lr::T=T(0.001), β1::T=T(0.9), β2::T=T(0.999), ε::T=T(1e-8)) where T
    # Actualizar momentos
    estado.m_w .= β1 .* estado.m_w .+ (one(T) - β1) .* grad_w
    estado.v_w .= β2 .* estado.v_w .+ (one(T) - β2) .* grad_w .^ 2
    estado.m_b .= β1 .* estado.m_b .+ (one(T) - β1) .* grad_b
    estado.v_b .= β2 .* estado.v_b .+ (one(T) - β2) .* grad_b .^ 2

    # Corrección de sesgo
    corr1 = one(T) - β1^t
    corr2 = one(T) - β2^t

    # Actualizar parámetros
    pesos .-= lr .* (estado.m_w ./ corr1) ./ (sqrt.(estado.v_w ./ corr2) .+ ε)
    biases .-= lr .* (estado.m_b ./ corr1) ./ (sqrt.(estado.v_b ./ corr2) .+ ε)
end

"""
    forward_con_cache(subconfig, entrada)

Forward pass almacenando activaciones para backprop.
Devuelve (activaciones, salida).
"""
function forward_con_cache(subconfig::Subconfiguracion{T}, entrada::AbstractMatrix{T}) where T
    activaciones = Vector{Matrix{T}}(undef, length(subconfig.pesos) + 1)
    activaciones[1] = entrada
    for i in eachindex(subconfig.pesos)
        z = activaciones[i] * subconfig.pesos[i] .+ subconfig.biases[i]'
        activaciones[i + 1] = sigmoid.(z)
    end
    return activaciones
end

"""
    backward!(subconfig, activaciones, target, estados_adam, t; lr, batch_size)

Backpropagation con actualización Adam.
"""
function backward!(subconfig::Subconfiguracion{T}, activaciones::Vector{Matrix{T}},
                   target::AbstractMatrix{T}, estados_adam::Vector{EstadoAdam{T}}, t::Int;
                   lr::T=T(0.001)) where T
    n_muestras = T(size(target, 1))
    delta = activaciones[end] .- target

    for i in length(subconfig.pesos):-1:1
        grad_w = activaciones[i]' * delta / n_muestras
        grad_b = vec(sum(delta, dims=1)) / n_muestras

        # Propagar delta ANTES de actualizar pesos
        if i > 1
            delta = delta * subconfig.pesos[i]' .* activaciones[i] .* (one(T) .- activaciones[i])
        end

        # Actualizar con Adam
        adam_paso!(subconfig.pesos[i], subconfig.biases[i], estados_adam[i],
                   grad_w, grad_b, t; lr=lr)
    end
end

"""
    calcular_loss(predicciones::Matrix{T}, targets::Matrix{T}) where T

Binary cross-entropy loss.
"""
function calcular_loss(predicciones::Matrix{T}, targets::Matrix{T}) where T
    ε = T(1e-7)
    pred_clamp = clamp.(predicciones, ε, one(T) - ε)
    -mean(targets .* log.(pred_clamp) .+ (one(T) .- targets) .* log.(one(T) .- pred_clamp))
end

"""
    entrenar_y_evaluar!(subconfig, datos_entrenamiento, datos_validacion; kwargs...)

Entrena una subconfiguración con Adam optimizer, mini-batches y early stopping.

Parámetros con nombre:
- `epochs`: Máximo de épocas (default 1000)
- `lr`: Learning rate para Adam (default 0.001)
- `batch_size`: Tamaño de mini-batch (default 32, o total si hay menos muestras)
- `paciencia`: Épocas sin mejora antes de parar (default 50)

Devuelve `(precision_pre, precision_post)`.
"""
function entrenar_y_evaluar!(subconfig::Subconfiguracion{T}, datos_entrenamiento, datos_validacion;
                              epochs::Int=1000, lr::T=T(0.001),
                              batch_size::Int=32, paciencia::Int=50)::Tuple{T, T} where T
    indices_salida_totales = collect(1:size(datos_validacion.salidas, 2))

    # Medir precisión pre-entrenamiento
    resultado_pre = evaluar(subconfig, datos_validacion, indices_salida_totales)
    precision_pre = resultado_pre.precision_global

    # Preparar datos
    entrada_train = datos_entrenamiento.entradas[:, subconfig.indices_entrada]
    target_train = datos_entrenamiento.salidas[:, subconfig.indices_salida]
    n_muestras = size(entrada_train, 1)
    bs = min(batch_size, n_muestras)

    # Inicializar Adam
    estados = crear_estado_adam(T, subconfig.pesos, subconfig.biases)
    paso_global = 0

    # Early stopping
    mejor_loss = T(Inf)
    epocas_sin_mejora = 0

    for epoch in 1:epochs
        # Shuffle de índices
        perm = randperm(n_muestras)

        # Mini-batches
        for inicio in 1:bs:n_muestras
            fin = min(inicio + bs - 1, n_muestras)
            idx = perm[inicio:fin]

            batch_entrada = entrada_train[idx, :]
            batch_target = target_train[idx, :]

            paso_global += 1

            # Forward + backward
            activaciones = forward_con_cache(subconfig, batch_entrada)
            backward!(subconfig, activaciones, batch_target, estados, paso_global; lr=lr)
        end

        # Calcular loss para early stopping (cada 5 épocas para no ralentizar)
        if epoch % 5 == 0
            entrada_val = datos_validacion.entradas[:, subconfig.indices_entrada]
            target_val = datos_validacion.salidas[:, subconfig.indices_salida]
            act_val = forward_con_cache(subconfig, entrada_val)
            loss = calcular_loss(act_val[end], target_val)

            if loss < mejor_loss - T(1e-6)
                mejor_loss = loss
                epocas_sin_mejora = 0
            else
                epocas_sin_mejora += 5
            end

            if epocas_sin_mejora >= paciencia
                break
            end
        end
    end

    # Medir precisión post-entrenamiento
    resultado_post = evaluar(subconfig, datos_validacion, indices_salida_totales)
    precision_post = resultado_post.precision_global

    return (precision_pre, precision_post)
end

"""
    entrenar_mapa!(mapa, datos_entrenamiento, datos_validacion; kwargs...)

Entrena todas las subconfiguraciones no vacías del mapa (referencia completa,
solución global, y soluciones parciales).
"""
function entrenar_mapa!(mapa::MapaDeSoluciones{T}, datos_entrenamiento, datos_validacion;
                         epochs::Int=1000, lr::T=T(0.001),
                         batch_size::Int=32, paciencia::Int=50) where T
    kwargs = (epochs=epochs, lr=lr, batch_size=batch_size, paciencia=paciencia)

    # Entrenar referencia completa
    if mapa.referencia_completa.subconfiguracion !== nothing
        (pre, post) = entrenar_y_evaluar!(mapa.referencia_completa.subconfiguracion,
            datos_entrenamiento, datos_validacion; kwargs...)
        mapa.referencia_completa.precision_pre_entrenamiento = pre
        mapa.referencia_completa.precision_post_entrenamiento = post
        mapa.referencia_completa.precision = post
    end

    # Entrenar solución global
    if mapa.global_.subconfiguracion !== nothing
        (pre, post) = entrenar_y_evaluar!(mapa.global_.subconfiguracion,
            datos_entrenamiento, datos_validacion; kwargs...)
        mapa.global_.precision_pre_entrenamiento = pre
        mapa.global_.precision_post_entrenamiento = post
        if post < pre
            @warn "El entrenamiento no produjo mejora para la solución global" precision_pre=pre precision_post=post diferencia=post-pre
        end
    end

    # Entrenar soluciones parciales
    for (subconj, entrada) in mapa.parciales
        if entrada.subconfiguracion !== nothing
            (pre, post) = entrenar_y_evaluar!(entrada.subconfiguracion,
                datos_entrenamiento, datos_validacion; kwargs...)
            entrada.precision_pre_entrenamiento = pre
            entrada.precision_post_entrenamiento = post
            if post < pre
                @warn "El entrenamiento no produjo mejora para la solución parcial" subconjunto=subconj precision_pre=pre precision_post=post diferencia=post-pre
            end
        end
    end

    return mapa
end
