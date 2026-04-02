"""
Experimento 7: Dataset Wine.

13 entradas, 3 clases, 178 muestras.
Subconfiguraciones: (2^13-1)×(2^3-1) = 57337
Split 80/20 train/test estratificado.

Ejecución:
    julia --project=. benchmarks/exp7_wine.jl
"""

include("utils.jl")

function cargar_wine()
    ruta = joinpath(@__DIR__, "wine.csv")
    if !isfile(ruta)
        println("Descargando Wine dataset...")
        run(`curl -sL -o $ruta https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data`)
    end

    lines = readlines(ruta)
    n = length(lines)
    entradas = zeros(Float64, n, 13)
    clases = zeros(Int, n)

    for (i, line) in enumerate(lines)
        vals = split(line, ',')
        clases[i] = parse(Int, vals[1])
        for j in 1:13
            entradas[i, j] = parse(Float64, vals[j + 1])
        end
    end

    # Normalizar entradas a [0, 1]
    for j in 1:13
        col = entradas[:, j]
        mn, mx = minimum(col), maximum(col)
        rango = mx - mn
        if rango > 0
            entradas[:, j] .= (col .- mn) ./ rango
        end
    end

    # One-hot encoding (clases 1,2,3 → columnas 1,2,3)
    salidas = zeros(Float64, n, 3)
    for i in 1:n
        salidas[i, clases[i]] = 1.0
    end

    return entradas, salidas
end

function split_datos(entradas, salidas, ratio_train=0.8; seed=42)
    Random.seed!(seed)
    n = size(entradas, 1)
    n_clases = size(salidas, 2)
    idx_train = Int[]
    idx_test = Int[]
    for c in 1:n_clases
        idx_clase = findall(salidas[:, c] .== 1.0)
        perm = randperm(length(idx_clase))
        n_train = round(Int, length(idx_clase) * ratio_train)
        append!(idx_train, idx_clase[perm[1:n_train]])
        append!(idx_test, idx_clase[perm[n_train+1:end]])
    end
    shuffle!(idx_train)
    shuffle!(idx_test)
    datos_train = (entradas=entradas[idx_train, :], salidas=salidas[idx_train, :])
    datos_test = (entradas=entradas[idx_test, :], salidas=salidas[idx_test, :])
    return datos_train, datos_test
end

function exp7_wine()
    println("╔══════════════════════════════════════════════════════════════╗")
    println("║  Experimento 7: Dataset Wine                               ║")
    println("║  13 entradas, 3 salidas (one-hot), 178 muestras            ║")
    println("╚══════════════════════════════════════════════════════════════╝")

    entradas, salidas = cargar_wine()
    seeds = SEEDS_20
    epochs = 1000
    lr = 0.01
    paciencia = 100

    total_subconfigs = (2^13 - 1) * (2^3 - 1)
    println("\nSubconfiguraciones posibles: $total_subconfigs")
    println("Seeds: $(length(seeds)) | Epochs: $epochs | LR: $lr\n")

    for n_ocultas in [16, 32]
        capas = [13, n_ocultas, 3]
        n_total = sum(capas)

        println("─"^60)
        @printf("  Red 13→%d→3 (%d neuronas totales) | umbral=0.4\n", n_ocultas, n_total)
        println("─"^60)

        println("\n┌────────┬────────────────────────────┬────────────────────────────┬──────────┐")
        println("│  Seed  │   Referencia Completa      │   Mejor Seleccionada       │ Tipo     │")
        println("│        │ Neur │ Pre    │ Post (test) │ Neur │ Pre    │ Post (test) │          │")
        println("├────────┼──────┼────────┼─────────────┼──────┼────────┼─────────────┼──────────┤")

        ref_posts = Float64[]
        sel_posts = Float64[]
        sel_neurs = Int[]
        sel_tipos = Symbol[]
        sel_ents = Int[]
        tiempos = Float64[]

        for seed in seeds
            datos_train, datos_test = split_datos(entradas, salidas; seed=seed)

            red = crear_red(capas; seed=seed)
            config = ConfiguracionDivision{Float64}(0.4)

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

            n_ent_sel = sel.tipo == :referencia ? size(entradas, 2) :
                        sel.entrada.subconfiguracion !== nothing ? length(sel.entrada.subconfiguracion.indices_entrada) : 0
            push!(sel_ents, n_ent_sel)

            @printf("│ %6d │ %4d │ %5.1f%% │    %5.1f%%    │ %4d │ %5.1f%% │    %5.1f%%    │ %-8s │\n",
                seed,
                ref.subconfiguracion.n_neuronas_activas,
                ref.precision_pre_entrenamiento * 100,
                ref.precision_post_entrenamiento * 100,
                sel.neuronas,
                sel.entrada.precision_pre_entrenamiento * 100,
                sel.precision * 100,
                sel.tipo)
        end

        println("├────────┼──────┼────────┼─────────────┼──────┼────────┼─────────────┼──────────┤")

        n_ref = count(t -> t == :referencia, sel_tipos)
        n_sub = length(sel_tipos) - n_ref

        @printf("│  Media │ %4d │        │    %5.1f%%    │ %4.0f │        │    %5.1f%%    │ ref:%d    │\n",
            n_total, mean(ref_posts) * 100, mean(sel_neurs), mean(sel_posts) * 100, n_ref)
        @printf("│  Std   │      │        │    %5.1f%%    │ %4.1f │        │    %5.1f%%    │ sub:%d    │\n",
            std(ref_posts) * 100, std(sel_neurs), std(sel_posts) * 100, n_sub)

        println("└────────┴──────┴────────┴─────────────┴──────┴────────┴─────────────┴──────────┘")

        @printf("\n  Tiempo medio: %.1fs | Entradas medias seleccionadas: %.1f / 13\n", mean(tiempos), mean(sel_ents))

        # Análisis de qué entradas se usan más
        println("\n  Frecuencia de entradas en subredes seleccionadas:")
        nombres_feat = ["Alcohol", "MalicAcid", "Ash", "AlcAsh", "Mg",
                        "Phenols", "Flavanoids", "NonFlav", "Proanth",
                        "ColorInt", "Hue", "OD280", "Proline"]
        conteo_ent = zeros(Int, 13)
        n_subredes = 0
        for (i, seed) in enumerate(seeds)
            sel_tipos[i] == :referencia && continue
            datos_train, datos_test = split_datos(entradas, salidas; seed=seed)
            red = crear_red(capas; seed=seed)
            config = ConfiguracionDivision{Float64}(0.4)
            mapa = ejecutar_division(red, datos_test, config;
                datos_entrenamiento=datos_train,
                epochs=epochs, lr=Float64(lr), paciencia=paciencia)
            sel = seleccionar_mejor(mapa)
            if sel.entrada.subconfiguracion !== nothing
                for idx in sel.entrada.subconfiguracion.indices_entrada
                    conteo_ent[idx] += 1
                end
                n_subredes += 1
            end
        end

        if n_subredes > 0
            orden = sortperm(conteo_ent, rev=true)
            for idx in orden
                conteo_ent[idx] == 0 && continue
                @printf("    %2d. %-12s: %d/%d (%.0f%%)\n",
                    idx, nombres_feat[idx], conteo_ent[idx], n_subredes,
                    conteo_ent[idx] / n_subredes * 100)
            end
        end
        println()
    end
end

exp7_wine()
