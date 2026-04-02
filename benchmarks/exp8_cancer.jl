"""
Experimento 8: Breast Cancer Wisconsin (Diagnostic).

Variante A: 10 features (mean), 2 clases, 569 muestras.
  Subconfiguraciones: (2^10-1)×(2^2-1) = 3069
Variante B: 30 features (todas), 2 clases, con cancelación por tiempo.

Ejecución:
    julia --project=. benchmarks/exp8_cancer.jl
"""

include("utils.jl")
using Base.Threads: Atomic

function cargar_cancer(; solo_mean::Bool=false)
    ruta = joinpath(@__DIR__, "wdbc.csv")
    if !isfile(ruta)
        println("Descargando Breast Cancer dataset...")
        run(`curl -sL -o $ruta https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data`)
    end

    lines = readlines(ruta)
    n = length(lines)
    n_features = solo_mean ? 10 : 30
    entradas = zeros(Float64, n, n_features)
    salidas = zeros(Float64, n, 2)  # [benigno, maligno]

    for (i, line) in enumerate(lines)
        vals = split(line, ',')
        # Columna 2: M/B
        if vals[2] == "M"
            salidas[i, 2] = 1.0  # maligno
        else
            salidas[i, 1] = 1.0  # benigno
        end
        # Columnas 3-32: features (10 mean, 10 SE, 10 worst)
        for j in 1:n_features
            entradas[i, j] = parse(Float64, vals[j + 2])
        end
    end

    # Normalizar a [0, 1]
    for j in 1:n_features
        col = entradas[:, j]
        mn, mx = minimum(col), maximum(col)
        rango = mx - mn
        if rango > 0
            entradas[:, j] .= (col .- mn) ./ rango
        end
    end

    return entradas, salidas
end

function split_cancer(entradas, salidas, ratio_train=0.8; seed=42)
    Random.seed!(seed)
    n = size(entradas, 1)
    idx_train = Int[]
    idx_test = Int[]
    for c in 1:2
        idx_clase = findall(salidas[:, c] .== 1.0)
        perm = randperm(length(idx_clase))
        n_train = round(Int, length(idx_clase) * ratio_train)
        append!(idx_train, idx_clase[perm[1:n_train]])
        append!(idx_test, idx_clase[perm[n_train+1:end]])
    end
    shuffle!(idx_train)
    shuffle!(idx_test)
    return (entradas=entradas[idx_train, :], salidas=salidas[idx_train, :]),
           (entradas=entradas[idx_test, :], salidas=salidas[idx_test, :])
end

function exp8_cancer()
    println("╔══════════════════════════════════════════════════════════════╗")
    println("║  Experimento 8: Breast Cancer Wisconsin                    ║")
    println("╚══════════════════════════════════════════════════════════════╝")

    seeds = SEEDS_20
    epochs = 1000
    lr = 0.01
    paciencia = 100

    # ── Variante A: 10 features (mean) ──
    entradas_10, salidas_10 = cargar_cancer(solo_mean=true)
    total_sub_10 = (2^10 - 1) * (2^2 - 1)

    println("\n═══ Variante A: 10 features (mean) ═══")
    println("Subconfiguraciones: $total_sub_10")

    nombres_10 = ["Radius", "Texture", "Perimeter", "Area", "Smoothness",
                   "Compactness", "Concavity", "ConcavePts", "Symmetry", "FractalDim"]

    for n_ocultas in [16, 32]
        capas = [10, n_ocultas, 2]
        n_total = sum(capas)

        println("\n─── Red 10→$(n_ocultas)→2 ($n_total neuronas) | umbral=0.5 ───\n")

        println("┌────────┬──────────────────────┬──────────────────────┬──────────┐")
        println("│  Seed  │  Ref. Completa       │  Mejor Seleccionada  │ Tipo     │")
        println("│        │ Neur │ Post (test)    │ Neur │ Post (test)   │          │")
        println("├────────┼──────┼────────────────┼──────┼───────────────┼──────────┤")

        ref_posts = Float64[]
        sel_posts = Float64[]
        sel_neurs = Int[]
        sel_tipos = Symbol[]
        sel_ents_list = Vector{Int}[]
        tiempos = Float64[]

        for seed in seeds
            datos_train, datos_test = split_cancer(entradas_10, salidas_10; seed=seed)
            red = crear_red(capas; seed=seed)
            config = ConfiguracionDivision{Float64}(0.5)

            t = @elapsed mapa = ejecutar_division(red, datos_test, config;
                datos_entrenamiento=datos_train,
                epochs=epochs, lr=Float64(lr), paciencia=paciencia)

            sel = seleccionar_mejor(mapa)
            ref = mapa.referencia_completa

            push!(ref_posts, ref.precision_post_entrenamiento)
            push!(sel_posts, sel.precision)
            push!(sel_neurs, sel.neuronas)
            push!(sel_tipos, sel.tipo)
            push!(tiempos, t)

            if sel.entrada.subconfiguracion !== nothing
                push!(sel_ents_list, sel.entrada.subconfiguracion.indices_entrada)
            else
                push!(sel_ents_list, Int[])
            end

            @printf("│ %6d │ %4d │    %5.1f%%       │ %4d │    %5.1f%%      │ %-8s │\n",
                seed, ref.subconfiguracion.n_neuronas_activas,
                ref.precision_post_entrenamiento * 100,
                sel.neuronas, sel.precision * 100, sel.tipo)
        end

        println("├────────┼──────┼────────────────┼──────┼───────────────┼──────────┤")
        n_ref = count(t -> t == :referencia, sel_tipos)
        @printf("│  Media │ %4d │    %5.1f%%       │ %4.0f │    %5.1f%%      │ ref:%-4d │\n",
            n_total, mean(ref_posts)*100, mean(sel_neurs), mean(sel_posts)*100, n_ref)
        println("└────────┴──────┴────────────────┴──────┴───────────────┴──────────┘")
        @printf("  Tiempo medio: %.1fs\n", mean(tiempos))

        # Feature importance
        conteo = zeros(Int, 10)
        n_sub = 0
        for ents in sel_ents_list
            length(ents) == 10 && continue  # skip referencia
            isempty(ents) && continue
            for idx in ents
                conteo[idx] += 1
            end
            n_sub += 1
        end
        if n_sub > 0
            println("  Features en subredes seleccionadas ($n_sub subredes):")
            orden = sortperm(conteo, rev=true)
            for idx in orden
                conteo[idx] == 0 && continue
                @printf("    %2d. %-14s: %d/%d (%.0f%%)\n",
                    idx, nombres_10[idx], conteo[idx], n_sub, conteo[idx]/n_sub*100)
            end
        end
        println()
    end

    # ── Variante B: 30 features con cancelación ──
    entradas_30, salidas_30 = cargar_cancer(solo_mean=false)

    println("\n═══ Variante B: 30 features (todas) con cancelación por tiempo ═══")
    println("Subconfiguraciones totales: $((2^30-1)*(2^2-1)) (≈3.2 mil millones)")
    println("Exploración limitada a 10000 subconfiguraciones por ejecución\n")

    for n_ocultas in [32]
        capas = [30, n_ocultas, 2]
        n_total = sum(capas)

        println("─── Red 30→$(n_ocultas)→2 ($n_total neuronas) | umbral=0.5 ───\n")

        println("┌────────┬──────────────────────┬──────────────────────┬──────────┐")
        println("│  Seed  │  Ref. Completa       │  Mejor Seleccionada  │ Tipo     │")
        println("│        │ Neur │ Post (test)    │ Neur │ Post (test)   │          │")
        println("├────────┼──────┼────────────────┼──────┼───────────────┼──────────┤")

        ref_posts = Float64[]
        sel_posts = Float64[]
        sel_neurs = Int[]
        sel_tipos = Symbol[]
        tiempos = Float64[]

        for seed in seeds
            datos_train, datos_test = split_cancer(entradas_30, salidas_30; seed=seed)
            red = crear_red(capas; seed=seed)
            config = ConfiguracionDivision{Float64}(0.5)

            # Cancelar tras 10000 evaluaciones
            señal = Atomic{Bool}(false)
            evaluadas_count = Ref(0)
            callback = function(p::ProgresoExploracion)
                evaluadas_count[] = p.evaluadas
                if p.evaluadas >= 10000
                    señal[] = true
                end
            end

            t = @elapsed mapa = ejecutar_division(red, datos_test, config;
                datos_entrenamiento=datos_train,
                epochs=epochs, lr=Float64(lr), paciencia=paciencia,
                callback_progreso=callback, señal_parada=señal)

            sel = seleccionar_mejor(mapa)
            ref = mapa.referencia_completa

            push!(ref_posts, ref.precision_post_entrenamiento)
            push!(sel_posts, sel.precision)
            push!(sel_neurs, sel.neuronas)
            push!(sel_tipos, sel.tipo)
            push!(tiempos, t)

            n_ent = sel.entrada.subconfiguracion !== nothing ? length(sel.entrada.subconfiguracion.indices_entrada) : 30
            @printf("│ %6d │ %4d │    %5.1f%%       │ %4d │    %5.1f%%      │ %-8s │\n",
                seed, ref.subconfiguracion.n_neuronas_activas,
                ref.precision_post_entrenamiento * 100,
                sel.neuronas, sel.precision * 100, sel.tipo)
        end

        println("├────────┼──────┼────────────────┼──────┼───────────────┼──────────┤")
        n_ref = count(t -> t == :referencia, sel_tipos)
        @printf("│  Media │ %4d │    %5.1f%%       │ %4.0f │    %5.1f%%      │ ref:%-4d │\n",
            n_total, mean(ref_posts)*100, mean(sel_neurs), mean(sel_posts)*100, n_ref)
        println("└────────┴──────┴────────────────┴──────┴───────────────┴──────────┘")
        @printf("  Tiempo medio: %.1fs (10000 subconfiguraciones exploradas)\n\n", mean(tiempos))
    end
end

exp8_cancer()
