"""
Experimento 14B: Adult Census Income — exploración acotada.

11 features, 2 clases, ~30K train.
Exploración limitada a subgrupos de 2-6 entradas.
Subconfiguraciones: 1474 × 3 = 4422

Ejecución:
    julia --project=. benchmarks/exp14b_adult_acotado.jl
"""

include("utils.jl")
using Base.Threads: Atomic

function cargar_adult_11()
    ruta_train = joinpath(@__DIR__, "..", "csv", "adult.csv")
    ruta_test = joinpath(@__DIR__, "..", "csv", "adult_test.csv")

    workclass_map = Dict("Private"=>1, "Self-emp-not-inc"=>2, "Self-emp-inc"=>3,
        "Federal-gov"=>4, "Local-gov"=>5, "State-gov"=>6, "Without-pay"=>7, "Never-worked"=>8)
    marital_map = Dict("Married-civ-spouse"=>1, "Divorced"=>2, "Never-married"=>3,
        "Separated"=>4, "Widowed"=>5, "Married-spouse-absent"=>6, "Married-AF-spouse"=>7)
    relationship_map = Dict("Wife"=>1, "Own-child"=>2, "Husband"=>3,
        "Not-in-family"=>4, "Other-relative"=>5, "Unmarried"=>6)
    race_map = Dict("White"=>1, "Asian-Pac-Islander"=>2, "Amer-Indian-Eskimo"=>3,
        "Other"=>4, "Black"=>5)
    sex_map = Dict("Female"=>1, "Male"=>2)

    function parsear(lines)
        datos = []
        for line in lines
            vals = strip.(split(line, ','))
            length(vals) < 15 && continue
            any(v -> v == "?", vals) && continue
            clase = occursin(">50K", vals[15]) ? 2 : 1
            feats = Float64[
                parse(Float64, vals[1]),                    # 1. Age
                get(workclass_map, vals[2], 0),             # 2. Workclass
                parse(Float64, vals[3]),                    # 3. FnlWgt
                parse(Float64, vals[5]),                    # 4. EduNum
                get(marital_map, vals[6], 0),               # 5. Marital
                get(relationship_map, vals[8], 0),          # 6. Relationship
                get(race_map, vals[9], 0),                  # 7. Race
                get(sex_map, vals[10], 0),                  # 8. Sex
                parse(Float64, vals[11]),                   # 9. CapGain
                parse(Float64, vals[12]),                   # 10. CapLoss
                parse(Float64, vals[13]),                   # 11. HoursWk
            ]
            push!(datos, (feats, clase))
        end
        return datos
    end

    lines_train = filter(l -> !isempty(strip(l)) && !startswith(l, "|"), readlines(ruta_train))
    lines_test = filter(l -> !isempty(strip(l)) && !startswith(l, "|"), readlines(ruta_test))

    dt = parsear(lines_train)
    dtest = parsear(lines_test)

    n_train = length(dt)
    n_test = length(dtest)

    ent_train = zeros(Float64, n_train, 11)
    sal_train = zeros(Float64, n_train, 2)
    for (i, (f, c)) in enumerate(dt)
        ent_train[i, :] .= f
        sal_train[i, c] = 1.0
    end

    ent_test = zeros(Float64, n_test, 11)
    sal_test = zeros(Float64, n_test, 2)
    for (i, (f, c)) in enumerate(dtest)
        ent_test[i, :] .= f
        sal_test[i, c] = 1.0
    end

    for j in 1:11
        mn = minimum(ent_train[:, j])
        mx = maximum(ent_train[:, j])
        rango = mx - mn
        if rango > 0
            ent_train[:, j] .= (ent_train[:, j] .- mn) ./ rango
            ent_test[:, j] .= clamp.((ent_test[:, j] .- mn) ./ rango, 0.0, 1.0)
        end
    end

    return (entradas=ent_train, salidas=sal_train), (entradas=ent_test, salidas=sal_test)
end

function exp14b()
    println("╔══════════════════════════════════════════════════════════════╗")
    println("║  Experimento 14B: Adult — exploración acotada (2-6 ent.)   ║")
    println("╚══════════════════════════════════════════════════════════════╝")

    datos_train, datos_test = cargar_adult_11()
    println("\n$(size(datos_train.entradas, 1)) train, $(size(datos_test.entradas, 1)) test")

    # Contar subconfiguraciones acotadas
    n_sub = sum(binomial(11, k) for k in 2:6) * 3
    println("Subconfiguraciones (entradas 2-6): $n_sub\n")

    seeds = SEEDS_20
    nombres = ["Age", "Workclass", "FnlWgt", "EduNum", "Marital",
               "Relationship", "Race", "Sex", "CapGain", "CapLoss", "HoursWk"]

    for n_ocultas in [32]
        capas = [11, n_ocultas, 2]
        n_total = sum(capas)

        println("─── Red 11→$(n_ocultas)→2 ($n_total neur) | umbral=0.5 | entradas 2-6 ───\n")

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
            red = crear_red(capas; seed=seed)
            config = ConfiguracionDivision{Float64}(0.5)

            # Usar cancelación para filtrar: el generador por bitmask empieza con
            # subconjuntos pequeños. Usamos el motor normal pero con callback que
            # filtra por tamaño de entrada.
            # Alternativa: ejecutar manualmente la exploración acotada.

            # Exploración manual acotada
            mapa = inicializar_mapa(2, Float64;
                referencia=extraer_subconfiguracion(red, collect(1:11), collect(1:2)))
            indices_salida_totales = collect(1:2)

            t = @elapsed begin
                # Evaluar todas las subconfiguraciones con 2-8 entradas
                for n_ent in 2:6
                    for combo in combinations_iter(11, n_ent)
                        for mask_sal in 1:3
                            idx_sal = bitmask_a_indices(mask_sal, 2)
                            subconfig = extraer_subconfiguracion(red, combo, idx_sal)
                            subconfig === nothing && continue
                            resultado = evaluar(subconfig, datos_test, indices_salida_totales)
                            actualizar_si_mejor!(mapa, subconfig, resultado, 0.5)
                        end
                    end
                end

                # Entrenar
                entrenar_mapa!(mapa, datos_train, datos_test;
                    epochs=500, lr=Float64(0.01), paciencia=50, batch_size=256)
            end

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

            n_ent_sel = sel.entrada.subconfiguracion !== nothing ? length(sel.entrada.subconfiguracion.indices_entrada) : 11
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
        @printf("  Tiempo medio: %.1fs\n", mean(tiempos))

        conteo = zeros(Int, 11)
        n_sub_found = 0
        for ents in sel_ents_list
            length(ents) == 11 && continue
            isempty(ents) && continue
            for idx in ents; conteo[idx] += 1; end
            n_sub_found += 1
        end
        if n_sub_found > 0
            println("  Features en subredes ($n_sub_found):")
            orden = sortperm(conteo, rev=true)
            for idx in orden
                conteo[idx] == 0 && continue
                @printf("    %2d. %-14s: %d/%d (%.0f%%)\n",
                    idx, nombres[idx], conteo[idx], n_sub_found, conteo[idx]/n_sub_found*100)
            end
        end
        println()
    end
end

# Generador de combinaciones sin materializar todo en memoria
struct CombIter
    n::Int
    k::Int
end
combinations_iter(n, k) = CombIter(n, k)

function Base.iterate(c::CombIter, state=nothing)
    if state === nothing
        c.k > c.n && return nothing
        combo = collect(1:c.k)
        return (copy(combo), combo)
    end
    combo = state
    # Encontrar la posición más a la derecha que se puede incrementar
    i = c.k
    while i > 0 && combo[i] == c.n - c.k + i
        i -= 1
    end
    i == 0 && return nothing
    combo[i] += 1
    for j in (i+1):c.k
        combo[j] = combo[j-1] + 1
    end
    return (copy(combo), combo)
end

Base.length(c::CombIter) = binomial(c.n, c.k)
Base.eltype(::Type{CombIter}) = Vector{Int}

exp14b()
