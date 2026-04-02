"""
Experimento 13: Pima Indians Diabetes y Banknote Authentication.

Pima: 8 features, 2 clases, 768 muestras.
  Subconfiguraciones: (2^8-1)×(2^2-1) = 765
Banknote: 4 features, 2 clases, 1372 muestras.
  Subconfiguraciones: (2^4-1)×(2^2-1) = 45

Ejecución:
    julia --project=. benchmarks/exp13_pima_banknote.jl
"""

include("utils.jl")

function cargar_csv_simple(ruta, n_features; clase_col_last=true)
    lines = filter(!isempty, readlines(ruta))
    n = length(lines)
    entradas = zeros(Float64, n, n_features)
    salidas = zeros(Float64, n, 2)

    for (i, line) in enumerate(lines)
        vals = split(line, ',')
        for j in 1:n_features
            entradas[i, j] = parse(Float64, vals[j])
        end
        clase = parse(Int, vals[end])
        salidas[i, clase + 1] = 1.0  # 0→col1, 1→col2
    end

    for j in 1:n_features
        col = entradas[:, j]
        mn, mx = minimum(col), maximum(col)
        rango = mx - mn
        rango > 0 && (entradas[:, j] .= (col .- mn) ./ rango)
    end

    return entradas, salidas
end

function split_strat(entradas, salidas; ratio=0.8, seed=42)
    Random.seed!(seed)
    n = size(entradas, 1)
    idx_train = Int[]
    idx_test = Int[]
    for c in 1:size(salidas, 2)
        idx_c = findall(salidas[:, c] .== 1.0)
        isempty(idx_c) && continue
        perm = randperm(length(idx_c))
        nt = max(1, round(Int, length(idx_c) * ratio))
        append!(idx_train, idx_c[perm[1:nt]])
        nt < length(idx_c) && append!(idx_test, idx_c[perm[nt+1:end]])
    end
    shuffle!(idx_train)
    shuffle!(idx_test)
    return (entradas=entradas[idx_train, :], salidas=salidas[idx_train, :]),
           (entradas=entradas[idx_test, :], salidas=salidas[idx_test, :])
end

function ejecutar_dataset(nombre, entradas, salidas, capas_list, seeds;
                          umbral=0.4, epochs=1000, lr=0.01, nombres_feat=nothing)
    n_ent = size(entradas, 2)
    n_sal = size(salidas, 2)
    total_sub = (2^n_ent - 1) * (2^n_sal - 1)

    println("\n═══ $nombre ═══")
    println("$n_ent features, $n_sal clases, $(size(entradas,1)) muestras")
    println("Subconfiguraciones: $total_sub\n")

    for capas in capas_list
        n_total = sum(capas)
        println("─── Red $(join(capas, "→")) ($n_total neur) | umbral=$umbral ───\n")

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
            datos_train, datos_test = split_strat(entradas, salidas; seed=seed)
            red = crear_red(capas; seed=seed)
            config = ConfiguracionDivision{Float64}(umbral)

            t = @elapsed mapa = ejecutar_division(red, datos_test, config;
                datos_entrenamiento=datos_train,
                epochs=epochs, lr=Float64(lr), paciencia=100)

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
        @printf("  Tiempo medio: %.2fs\n", mean(tiempos))

        if nombres_feat !== nothing
            conteo = zeros(Int, n_ent)
            n_sub = 0
            for ents in sel_ents_list
                length(ents) == n_ent && continue
                isempty(ents) && continue
                for idx in ents; conteo[idx] += 1; end
                n_sub += 1
            end
            if n_sub > 0
                println("  Features en subredes ($n_sub):")
                orden = sortperm(conteo, rev=true)
                for idx in orden
                    conteo[idx] == 0 && continue
                    @printf("    %2d. %-16s: %d/%d (%.0f%%)\n",
                        idx, nombres_feat[idx], conteo[idx], n_sub, conteo[idx]/n_sub*100)
                end
            end
        end
        println()
    end
end

function exp13()
    println("╔══════════════════════════════════════════════════════════════╗")
    println("║  Experimento 13: Pima Diabetes y Banknote Authentication   ║")
    println("╚══════════════════════════════════════════════════════════════╝")

    seeds = SEEDS_20

    # Pima
    ent_p, sal_p = cargar_csv_simple(joinpath(@__DIR__, "pima.csv"), 8)
    nombres_pima = ["Pregnancies", "Glucose", "BloodPres", "SkinThick",
                     "Insulin", "BMI", "DiabPedigree", "Age"]
    ejecutar_dataset("Pima Indians Diabetes", ent_p, sal_p,
        [[8, 16, 2], [8, 32, 2]], seeds;
        nombres_feat=nombres_pima)

    # Banknote
    ent_b, sal_b = cargar_csv_simple(joinpath(@__DIR__, "banknote.csv"), 4)
    nombres_bank = ["Variance", "Skewness", "Kurtosis", "Entropy"]
    ejecutar_dataset("Banknote Authentication", ent_b, sal_b,
        [[4, 8, 2], [4, 16, 2]], seeds;
        nombres_feat=nombres_bank)
end

exp13()
