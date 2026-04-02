"""
Experimento 12: Ecoli y Balance Scale.

Ecoli: 7 features, 5 clases principales (cp, im, imU, pp, om), 327 muestras.
  Subconfiguraciones: (2^7-1)×(2^5-1) = 3937
Balance Scale: 4 features, 3 clases, 625 muestras.
  Subconfiguraciones: (2^4-1)×(2^3-1) = 105

Ejecución:
    julia --project=. benchmarks/exp12_ecoli_balance.jl
"""

include("utils.jl")

function cargar_ecoli()
    ruta = joinpath(@__DIR__, "ecoli.csv")
    lines = filter(!isempty, readlines(ruta))

    # Filtrar clases con pocas muestras (imL=2, imS=2, omL=5)
    clases_validas = Dict("cp"=>1, "im"=>2, "imU"=>3, "pp"=>4, "om"=>5)
    datos_filtrados = []
    for line in lines
        vals = split(line)
        clase = vals[end]
        haskey(clases_validas, clase) && push!(datos_filtrados, (vals, clases_validas[clase]))
    end

    n = length(datos_filtrados)
    entradas = zeros(Float64, n, 7)
    salidas = zeros(Float64, n, 5)

    for (i, (vals, clase)) in enumerate(datos_filtrados)
        # Columnas 2-8: features (columna 1 es nombre)
        for j in 1:7
            entradas[i, j] = parse(Float64, vals[j + 1])
        end
        salidas[i, clase] = 1.0
    end

    # Normalizar
    for j in 1:7
        col = entradas[:, j]
        mn, mx = minimum(col), maximum(col)
        rango = mx - mn
        rango > 0 && (entradas[:, j] .= (col .- mn) ./ rango)
    end

    return entradas, salidas
end

function cargar_balance()
    ruta = joinpath(@__DIR__, "balance.csv")
    lines = filter(!isempty, readlines(ruta))
    clase_map = Dict("L"=>1, "B"=>2, "R"=>3)

    n = length(lines)
    entradas = zeros(Float64, n, 4)
    salidas = zeros(Float64, n, 3)

    for (i, line) in enumerate(lines)
        vals = split(line, ',')
        salidas[i, clase_map[vals[1]]] = 1.0
        for j in 1:4
            entradas[i, j] = parse(Float64, vals[j + 1])
        end
    end

    # Normalizar
    for j in 1:4
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

function ejecutar_dataset(nombre, entradas, salidas, capas_list, seeds;
                          umbral=0.4, epochs=1000, lr=0.01, nombres_feat=nothing)
    n_ent = size(entradas, 2)
    n_sal = size(salidas, 2)
    total_sub = (2^n_ent - 1) * (2^n_sal - 1)

    println("\n═══ $nombre ═══")
    println("$n_ent features, $n_sal clases, $(size(entradas,1)) muestras")
    println("Subconfiguraciones: $total_sub | Seeds: $(length(seeds))\n")

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
                    @printf("    %2d. %-14s: %d/%d (%.0f%%)\n",
                        idx, nombres_feat[idx], conteo[idx], n_sub, conteo[idx]/n_sub*100)
                end
            end
        end
        println()
    end
end

function exp12()
    println("╔══════════════════════════════════════════════════════════════╗")
    println("║  Experimento 12: Ecoli y Balance Scale                     ║")
    println("╚══════════════════════════════════════════════════════════════╝")

    seeds = SEEDS_20

    # Ecoli
    ent_e, sal_e = cargar_ecoli()
    nombres_ecoli = ["mcg", "gvh", "lip", "chg", "aac", "alm1", "alm2"]
    ejecutar_dataset("Ecoli", ent_e, sal_e,
        [[7, 16, 5], [7, 32, 5]], seeds;
        nombres_feat=nombres_ecoli)

    # Balance Scale
    ent_b, sal_b = cargar_balance()
    nombres_balance = ["LeftWeight", "LeftDist", "RightWeight", "RightDist"]
    ejecutar_dataset("Balance Scale", ent_b, sal_b,
        [[4, 8, 3], [4, 16, 3]], seeds;
        nombres_feat=nombres_balance)
end

exp12()
