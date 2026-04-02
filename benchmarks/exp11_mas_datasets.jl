"""
Experimento 11: Ionosphere, Sonar, Haberman.

Ionosphere: 34 features, 2 clases, 351 muestras (exploración parcial)
Sonar: 60 features, 2 clases, 208 muestras (exploración parcial)
Haberman: 3 features, 2 clases, 306 muestras (exploración completa)

Ejecución:
    julia --project=. benchmarks/exp11_mas_datasets.jl
"""

include("utils.jl")
using Base.Threads: Atomic

function cargar_csv_generico(ruta; n_features, clase_col, clase_map, sep=',', skip_col=nothing)
    lines = filter(!isempty, readlines(ruta))
    n = length(lines)
    entradas = zeros(Float64, n, n_features)
    n_clases = length(unique(values(clase_map)))
    salidas = zeros(Float64, n, n_clases)

    for (i, line) in enumerate(lines)
        vals = split(line, sep)
        feat_idx = 0
        for j in 1:length(vals)
            j == clase_col && continue
            skip_col !== nothing && j == skip_col && continue
            feat_idx += 1
            feat_idx > n_features && break
            entradas[i, feat_idx] = parse(Float64, vals[j])
        end
        clase_str = strip(vals[clase_col])
        salidas[i, clase_map[clase_str]] = 1.0
    end

    # Normalizar
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
    n_clases = size(salidas, 2)
    idx_train = Int[]
    idx_test = Int[]
    for c in 1:n_clases
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

function ejecutar_con_presupuesto(nombre, entradas, salidas, capas, seeds;
        umbral=0.4, epochs=1000, lr=0.01, max_eval=nothing)
    n_ent = size(entradas, 2)
    n_sal = size(salidas, 2)
    n_total = sum(capas)
    total_sub = (2^n_ent - 1) * (2^n_sal - 1)
    parcial = max_eval !== nothing && max_eval < total_sub

    @printf("\n─── %s | Red %s (%d neur) | umbral=%.1f", nombre, join(capas,"→"), n_total, umbral)
    parcial && @printf(" | exploración: %d/%d", max_eval, total_sub)
    println(" ───\n")

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

        kwargs = Dict{Symbol,Any}(
            :datos_entrenamiento => datos_train,
            :epochs => epochs, :lr => Float64(lr), :paciencia => 100)

        if parcial
            señal = Atomic{Bool}(false)
            callback = function(p)
                p.evaluadas >= max_eval && (señal[] = true)
            end
            kwargs[:callback_progreso] = callback
            kwargs[:señal_parada] = señal
        end

        t = @elapsed mapa = ejecutar_division(red, datos_test, config; kwargs...)

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

    # Entradas medias
    n_sub = 0
    total_ent = 0
    for ents in sel_ents_list
        length(ents) == n_ent && continue
        isempty(ents) && continue
        total_ent += length(ents)
        n_sub += 1
    end
    if n_sub > 0
        @printf("  Entradas medias en subredes: %.1f / %d (%d subredes)\n",
            total_ent / n_sub, n_ent, n_sub)
    end
    println()
end

function exp11()
    println("╔══════════════════════════════════════════════════════════════╗")
    println("║  Experimento 11: Ionosphere, Sonar, Haberman               ║")
    println("╚══════════════════════════════════════════════════════════════╝")

    seeds = SEEDS_20

    # --- Haberman: 3 features, exploración completa ---
    println("\n═══ Haberman Survival ═══")
    println("3 features, 2 clases, 306 muestras")
    println("Subconfiguraciones: $((2^3-1)*(2^2-1))")

    ent_h, sal_h = cargar_csv_generico(joinpath(@__DIR__, "..", "csv", "haberman.csv");
        n_features=3, clase_col=4, clase_map=Dict("1"=>1, "2"=>2))

    ejecutar_con_presupuesto("Haberman 3→8→2", ent_h, sal_h, [3, 8, 2], seeds; umbral=0.5)
    ejecutar_con_presupuesto("Haberman 3→16→2", ent_h, sal_h, [3, 16, 2], seeds; umbral=0.5)

    # --- Ionosphere: 34 features, exploración parcial ---
    println("\n═══ Ionosphere ═══")
    println("34 features, 2 clases, 351 muestras")

    ent_i, sal_i = cargar_csv_generico(joinpath(@__DIR__, "..", "csv", "ionosphere.csv");
        n_features=34, clase_col=35, clase_map=Dict("g"=>1, "b"=>2))

    ejecutar_con_presupuesto("Ionosphere 34→16→2", ent_i, sal_i, [34, 16, 2], seeds;
        umbral=0.5, max_eval=10000)
    ejecutar_con_presupuesto("Ionosphere 34→32→2", ent_i, sal_i, [34, 32, 2], seeds;
        umbral=0.5, max_eval=10000)

    # --- Sonar: 60 features, exploración parcial ---
    println("\n═══ Sonar ═══")
    println("60 features, 2 clases, 208 muestras")

    ent_s, sal_s = cargar_csv_generico(joinpath(@__DIR__, "..", "csv", "sonar.csv");
        n_features=60, clase_col=61, clase_map=Dict("R"=>1, "M"=>2))

    ejecutar_con_presupuesto("Sonar 60→32→2", ent_s, sal_s, [60, 32, 2], seeds;
        umbral=0.5, max_eval=10000)
end

exp11()
