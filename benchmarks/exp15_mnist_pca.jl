"""
Experimento 15: MNIST reducido con PCA.

Variante A: 10 componentes PCA, 2 clases (0 vs 1), exploración completa.
  Subconfiguraciones: (2^10-1)×(2^2-1) = 3069
Variante B: 10 componentes PCA, 3 clases (0 vs 1 vs 7), exploración acotada 2-6.
  Subconfiguraciones: ~1474 × 7 = 10318
Variante C: 15 componentes PCA, 2 clases (0 vs 1), exploración acotada 3-10.
  Subconfiguraciones: ~31K

Ejecución:
    julia --project=. benchmarks/exp15_mnist_pca.jl
"""

include("utils.jl")
using CodecZlib, LinearAlgebra

function leer_imagenes(ruta)
    data = transcode(GzipDecompressor, read(ruta))
    n = Int(ntoh(reinterpret(UInt32, data[5:8])[1]))
    rows = Int(ntoh(reinterpret(UInt32, data[9:12])[1]))
    cols = Int(ntoh(reinterpret(UInt32, data[13:16])[1]))
    imgs = reshape(Float64.(data[17:end]), rows * cols, n)
    return imgs ./ 255.0  # 784 × n
end

function leer_labels(ruta)
    data = transcode(GzipDecompressor, read(ruta))
    return Int.(data[9:end])
end

function aplicar_pca(X_train, X_test, n_componentes; n_pca_muestras=10000)
    # PCA sobre submuestra del train
    Random.seed!(42)
    idx = randperm(size(X_train, 2))[1:min(n_pca_muestras, size(X_train, 2))]
    X_sub = X_train[:, idx]
    mu = mean(X_sub, dims=2)
    X_centered = X_sub .- mu
    U, S, V = svd(X_centered)

    # Proyectar
    W = U[:, 1:n_componentes]  # 784 × k
    Z_train = W' * (X_train .- mu)  # k × n_train
    Z_test = W' * (X_test .- mu)    # k × n_test

    # Normalizar a [0,1]
    for j in 1:n_componentes
        row = Z_train[j, :]
        mn, mx = minimum(row), maximum(row)
        rango = mx - mn
        if rango > 0
            Z_train[j, :] .= (Z_train[j, :] .- mn) ./ rango
            Z_test[j, :] .= clamp.((Z_test[j, :] .- mn) ./ rango, 0.0, 1.0)
        end
    end

    var_total = sum(S.^2)
    var_k = sum(S[1:n_componentes].^2) / var_total * 100

    return Z_train', Z_test', var_k  # n × k (filas = muestras)
end

function filtrar_clases(X, y, clases)
    mask = [l in clases for l in y]
    X_filt = X[:, mask]
    y_filt = y[mask]
    # Remapear a 1-based
    mapa = Dict(c => i for (i, c) in enumerate(sort(collect(clases))))
    y_mapped = [mapa[l] for l in y_filt]
    n_clases = length(clases)
    salidas = zeros(Float64, length(y_mapped), n_clases)
    for (i, c) in enumerate(y_mapped)
        salidas[i, c] = 1.0
    end
    return X_filt, salidas
end

# Iterador de combinaciones
struct CombIter; n::Int; k::Int; end
combinations_iter(n, k) = CombIter(n, k)
function Base.iterate(c::CombIter, state=nothing)
    if state === nothing
        c.k > c.n && return nothing
        combo = collect(1:c.k)
        return (copy(combo), combo)
    end
    combo = state
    i = c.k
    while i > 0 && combo[i] == c.n - c.k + i; i -= 1; end
    i == 0 && return nothing
    combo[i] += 1
    for j in (i+1):c.k; combo[j] = combo[j-1] + 1; end
    return (copy(combo), combo)
end
Base.length(c::CombIter) = binomial(c.n, c.k)

function ejecutar_mnist_pca(nombre, ent_train, sal_train, ent_test, sal_test,
                             capas, seeds; umbral=0.4, epochs=500, lr=0.01,
                             rango_entradas=nothing)
    n_ent = size(ent_train, 2)
    n_sal = size(sal_train, 2)
    n_total = sum(capas)

    acotado = rango_entradas !== nothing
    if acotado
        min_e, max_e = rango_entradas
        n_sub = sum(binomial(n_ent, k) for k in min_e:max_e) * (2^n_sal - 1)
        @printf("\n─── %s | Red %s (%d neur) | entradas %d-%d | %d subconfigs ───\n\n",
            nombre, join(capas,"→"), n_total, min_e, max_e, n_sub)
    else
        n_sub = (2^n_ent - 1) * (2^n_sal - 1)
        @printf("\n─── %s | Red %s (%d neur) | %d subconfigs ───\n\n",
            nombre, join(capas,"→"), n_total, n_sub)
    end

    println("┌────────┬──────────────────────┬──────────────────────┬──────────┐")
    println("│  Seed  │  Ref. Completa       │  Mejor Seleccionada  │ Tipo     │")
    println("│        │ Neur │ Post (test)    │ Neur │ Post (test)   │          │")
    println("├────────┼──────┼────────────────┼──────┼───────────────┼──────────┤")

    ref_posts = Float64[]
    sel_posts = Float64[]
    sel_neurs = Int[]
    sel_tipos = Symbol[]
    sel_n_ents = Int[]
    tiempos = Float64[]

    datos_train = (entradas=ent_train, salidas=sal_train)
    datos_test = (entradas=ent_test, salidas=sal_test)

    for seed in seeds
        red = crear_red(capas; seed=seed)
        config = ConfiguracionDivision{Float64}(umbral)

        if acotado
            min_e, max_e = rango_entradas
            mapa = inicializar_mapa(n_sal, Float64;
                referencia=extraer_subconfiguracion(red, collect(1:n_ent), collect(1:n_sal)))
            indices_salida_totales = collect(1:n_sal)

            t = @elapsed begin
                for ne in min_e:max_e
                    for combo in combinations_iter(n_ent, ne)
                        for mask_sal in 1:(2^n_sal - 1)
                            idx_sal = bitmask_a_indices(mask_sal, n_sal)
                            subconfig = extraer_subconfiguracion(red, combo, idx_sal)
                            subconfig === nothing && continue
                            resultado = evaluar(subconfig, datos_test, indices_salida_totales)
                            actualizar_si_mejor!(mapa, subconfig, resultado, umbral)
                        end
                    end
                end
                entrenar_mapa!(mapa, datos_train, datos_test;
                    epochs=epochs, lr=Float64(lr), paciencia=50, batch_size=128)
            end
        else
            t = @elapsed mapa = ejecutar_division(red, datos_test, config;
                datos_entrenamiento=datos_train,
                epochs=epochs, lr=Float64(lr), paciencia=50, batch_size=128)
        end

        sel = seleccionar_mejor(mapa)
        ref = mapa.referencia_completa

        push!(ref_posts, ref.precision_post_entrenamiento)
        push!(sel_posts, sel.precision)
        push!(sel_neurs, sel.neuronas)
        push!(sel_tipos, sel.tipo)
        push!(tiempos, t)
        n_e = sel.entrada.subconfiguracion !== nothing ? length(sel.entrada.subconfiguracion.indices_entrada) : n_ent
        push!(sel_n_ents, n_e)

        @printf("│ %6d │ %4d │    %5.1f%%       │ %4d │    %5.1f%%      │ %-8s │\n",
            seed, n_total, ref.precision_post_entrenamiento * 100,
            sel.neuronas, sel.precision * 100, sel.tipo)
    end

    println("├────────┼──────┼────────────────┼──────┼───────────────┼──────────┤")
    n_ref = count(t -> t == :referencia, sel_tipos)
    @printf("│  Media │ %4d │    %5.1f%%       │ %4.0f │    %5.1f%%      │ ref:%-4d │\n",
        n_total, mean(ref_posts)*100, mean(sel_neurs), mean(sel_posts)*100, n_ref)
    @printf("│  Std   │      │    %5.1f%%       │ %4.1f │    %5.1f%%      │ sub:%-4d │\n",
        std(ref_posts)*100, std(sel_neurs), std(sel_posts)*100, length(seeds)-n_ref)
    println("└────────┴──────┴────────────────┴──────┴───────────────┴──────────┘")
    @printf("  Tiempo medio: %.1fs | Entradas medias: %.1f / %d\n\n", mean(tiempos), mean(sel_n_ents), n_ent)
end

function exp15()
    println("╔══════════════════════════════════════════════════════════════╗")
    println("║  Experimento 15: MNIST con PCA                             ║")
    println("╚══════════════════════════════════════════════════════════════╝")

    # Cargar MNIST
    println("\nCargando MNIST...")
    X_train_raw = leer_imagenes(joinpath(@__DIR__, "..", "csv", "train-images-idx3-ubyte.gz"))
    y_train = leer_labels(joinpath(@__DIR__, "..", "csv", "train-labels-idx1-ubyte.gz"))
    X_test_raw = leer_imagenes(joinpath(@__DIR__, "..", "csv", "t10k-images-idx3-ubyte.gz"))
    y_test = leer_labels(joinpath(@__DIR__, "..", "csv", "t10k-labels-idx1-ubyte.gz"))
    println("  Train: $(size(X_train_raw, 2)), Test: $(size(X_test_raw, 2))")

    seeds = SEEDS_20[1:10]  # 10 semillas para velocidad

    # === Variante A: 10 PCA, 0 vs 1 ===
    println("\n═══ Variante A: 10 PCA, dígitos 0 vs 1 ═══")
    X_tr_01, sal_tr_01 = filtrar_clases(X_train_raw, y_train, Set([0, 1]))
    X_te_01, sal_te_01 = filtrar_clases(X_test_raw, y_test, Set([0, 1]))
    ent_tr_a, ent_te_a, var_a = aplicar_pca(X_tr_01, X_te_01, 10)
    @printf("  %d train, %d test, %.1f%% varianza explicada\n",
        size(ent_tr_a, 1), size(ent_te_a, 1), var_a)

    ejecutar_mnist_pca("MNIST 0vs1 10PCA", ent_tr_a, sal_tr_01, ent_te_a, sal_te_01,
        [10, 16, 2], seeds; umbral=0.5)

    # === Variante B: 10 PCA, 0 vs 1 vs 7 ===
    println("\n═══ Variante B: 10 PCA, dígitos 0 vs 1 vs 7 ═══")
    X_tr_017, sal_tr_017 = filtrar_clases(X_train_raw, y_train, Set([0, 1, 7]))
    X_te_017, sal_te_017 = filtrar_clases(X_test_raw, y_test, Set([0, 1, 7]))
    ent_tr_b, ent_te_b, var_b = aplicar_pca(X_tr_017, X_te_017, 10)
    @printf("  %d train, %d test, %.1f%% varianza explicada\n",
        size(ent_tr_b, 1), size(ent_te_b, 1), var_b)

    ejecutar_mnist_pca("MNIST 0vs1vs7 10PCA", ent_tr_b, sal_tr_017, ent_te_b, sal_te_017,
        [10, 16, 3], seeds; umbral=0.4, rango_entradas=(2, 7))

    # === Variante C: 15 PCA, 0 vs 1 ===
    println("\n═══ Variante C: 15 PCA, dígitos 0 vs 1 ═══")
    ent_tr_c, ent_te_c, var_c = aplicar_pca(X_tr_01, X_te_01, 15)
    @printf("  %d train, %d test, %.1f%% varianza explicada\n",
        size(ent_tr_c, 1), size(ent_te_c, 1), var_c)

    ejecutar_mnist_pca("MNIST 0vs1 15PCA", ent_tr_c, sal_tr_01, ent_te_c, sal_te_01,
        [15, 16, 2], seeds; umbral=0.5, rango_entradas=(3, 10))
end

exp15()
