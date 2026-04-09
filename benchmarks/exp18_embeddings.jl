"""
Experimento 18: División Neuronal con Embeddings.

Prueba el método con representaciones modernas (embeddings aprendibles)
para features categóricas, comparando contra codificación ordinal.

Dataset: Adult Census Income
- 6 features continuas + 5 categóricas
- Con embeddings: cada categórica se representa como vector denso de dim 4
- Dimensión total con embeddings: 6 continuas + 5×4 embeddings = 26

Ejecución:
    julia --project=. benchmarks/exp18_embeddings.jl
"""

include("utils.jl")

# Iterador de combinaciones
struct CombIter; n::Int; k::Int; end
combinations_iter(n, k) = CombIter(n, k)
function Base.iterate(c::CombIter, state=nothing)
    if state === nothing; c.k > c.n && return nothing; combo = collect(1:c.k); return (copy(combo), combo); end
    combo = state; i = c.k
    while i > 0 && combo[i] == c.n - c.k + i; i -= 1; end
    i == 0 && return nothing; combo[i] += 1
    for j in (i+1):c.k; combo[j] = combo[j-1] + 1; end
    return (copy(combo), combo)
end
Base.length(c::CombIter) = binomial(c.n, c.k)

function cargar_adult_para_embeddings()
    ruta_train = joinpath(@__DIR__, "..", "csv", "adult.csv")
    ruta_test = joinpath(@__DIR__, "..", "csv", "adult_test.csv")

    # Mapeos de categorías a enteros (1-based)
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

            # 11 features: 6 continuas + 5 categóricas (como enteros)
            feats = Float64[
                parse(Float64, vals[1]),                    # 1. Age (continua)
                get(workclass_map, vals[2], 1),             # 2. Workclass (cat, 8 cats)
                parse(Float64, vals[3]),                    # 3. FnlWgt (continua)
                parse(Float64, vals[5]),                    # 4. EduNum (continua)
                get(marital_map, vals[6], 1),               # 5. Marital (cat, 7 cats)
                get(relationship_map, vals[8], 1),          # 6. Relationship (cat, 6 cats)
                get(race_map, vals[9], 1),                  # 7. Race (cat, 5 cats)
                get(sex_map, vals[10], 1),                  # 8. Sex (cat, 2 cats)
                parse(Float64, vals[11]),                   # 9. CapGain (continua)
                parse(Float64, vals[12]),                   # 10. CapLoss (continua)
                parse(Float64, vals[13]),                   # 11. HoursWk (continua)
            ]
            push!(datos, (feats, clase))
        end
        return datos
    end

    lines_train = filter(l -> !isempty(strip(l)) && !startswith(l, "|"), readlines(ruta_train))
    lines_test = filter(l -> !isempty(strip(l)) && !startswith(l, "|"), readlines(ruta_test))

    dt = parsear(lines_train)
    dte = parsear(lines_test)

    # Construir matrices
    n_tr = length(dt); n_te = length(dte)
    ent_tr = zeros(Float64, n_tr, 11)
    sal_tr = zeros(Float64, n_tr, 2)
    for (i, (f, c)) in enumerate(dt)
        ent_tr[i, :] .= f; sal_tr[i, c] = 1.0
    end
    ent_te = zeros(Float64, n_te, 11)
    sal_te = zeros(Float64, n_te, 2)
    for (i, (f, c)) in enumerate(dte)
        ent_te[i, :] .= f; sal_te[i, c] = 1.0
    end

    # Normalizar solo las continuas
    indices_cont = [1, 3, 4, 9, 10, 11]
    for j in indices_cont
        mn = minimum(ent_tr[:, j]); mx = maximum(ent_tr[:, j])
        rango = mx - mn
        if rango > 0
            ent_tr[:, j] .= (ent_tr[:, j] .- mn) ./ rango
            ent_te[:, j] .= clamp.((ent_te[:, j] .- mn) ./ rango, 0.0, 1.0)
        end
    end

    # Definir features categóricas: indice => n_categorias
    categoricos = Dict(2=>8, 5=>7, 6=>6, 7=>5, 8=>2)

    return (entradas=ent_tr, salidas=sal_tr), (entradas=ent_te, salidas=sal_te), categoricos
end

function exp18()
    println("╔══════════════════════════════════════════════════════════════╗")
    println("║  Experimento 18: División Neuronal con Embeddings          ║")
    println("╚══════════════════════════════════════════════════════════════╝")

    datos_train, datos_test, categoricos = cargar_adult_para_embeddings()
    println("\n$(size(datos_train.entradas, 1)) train, $(size(datos_test.entradas, 1)) test")
    println("Features: 6 continuas + 5 categóricas")

    seeds = SEEDS_20[1:10]  # 10 semillas
    dim_emb = 4

    # --- Variante A: Sin embeddings (ordinal, como antes) ---
    println("\n═══ Variante A: Codificación ordinal (11 features → 11 entradas) ═══")
    capas_a = [11, 32, 2]
    println("Red: $(join(capas_a, "→"))\n")

    println("┌────────┬──────────────────────┬──────────────────────┬──────────┐")
    println("│  Seed  │  Ref. Completa       │  Mejor Seleccionada  │ Tipo     │")
    println("│        │ Neur │ Post (test)    │ Neur │ Post (test)   │          │")
    println("├────────┼──────┼────────────────┼──────┼───────────────┼──────────┤")

    ref_a = Float64[]; sel_a = Float64[]
    for seed in seeds
        red = crear_red(capas_a; seed=seed)
        config = ConfiguracionDivision{Float64}(0.5)
        mapa = ejecutar_division(red, datos_test, config;
            datos_entrenamiento=datos_train, epochs=300, lr=Float64(0.01), paciencia=50, batch_size=256)
        sel = seleccionar_mejor(mapa); ref = mapa.referencia_completa
        push!(ref_a, ref.precision_post_entrenamiento); push!(sel_a, sel.precision)
        @printf("│ %6d │ %4d │    %5.1f%%       │ %4d │    %5.1f%%      │ %-8s │\n",
            seed, sum(capas_a), ref.precision_post_entrenamiento*100, sel.neuronas, sel.precision*100, sel.tipo)
    end
    println("├────────┼──────┼────────────────┼──────┼───────────────┼──────────┤")
    @printf("│  Media │      │    %5.1f%%       │      │    %5.1f%%      │          │\n", mean(ref_a)*100, mean(sel_a)*100)
    println("└────────┴──────┴────────────────┴──────┴───────────────┴──────────┘")

    # --- Variante B: Con embeddings ---
    println("\n═══ Variante B: Con embeddings (dim=$dim_emb por categórica) ═══")

    esquema = crear_esquema(Float64, 11, categoricos; dim_embedding=dim_emb)
    println("Dimensión tras embedding: $(esquema.dim_total) (6 cont + 5×$dim_emb emb)")

    # Transformar datos
    ent_tr_emb = transformar(esquema, datos_train.entradas)
    ent_te_emb = transformar(esquema, datos_test.entradas)

    datos_train_emb = (entradas=ent_tr_emb, salidas=datos_train.salidas)
    datos_test_emb = (entradas=ent_te_emb, salidas=datos_test.salidas)

    # Subconjunto pequeño para exploración rápida (2000 muestras)
    Random.seed!(0)
    n_eval = min(2000, size(ent_te_emb, 1))
    idx_eval = randperm(size(ent_te_emb, 1))[1:n_eval]
    datos_eval_emb = (entradas=ent_te_emb[idx_eval, :], salidas=datos_test.salidas[idx_eval, :])

    capas_b = [esquema.dim_total, 32, 2]
    println("Red: $(join(capas_b, "→"))")
    println("Exploración: 10000 subconfiguraciones (entradas 3-6), evaluación sobre $n_eval muestras\n")

    println("┌────────┬──────────────────────┬──────────────────────┬──────────┐")
    println("│  Seed  │  Ref. Completa       │  Mejor Seleccionada  │ Tipo     │")
    println("│        │ Neur │ Post (test)    │ Neur │ Post (test)   │          │")
    println("├────────┼──────┼────────────────┼──────┼───────────────┼──────────┤")

    ref_b = Float64[]; sel_b = Float64[]
    for seed in seeds
        red = crear_red(capas_b; seed=seed)

        # Exploración acotada con subconjunto de evaluación
        mapa = inicializar_mapa(2, Float64;
            referencia=extraer_subconfiguracion(red, collect(1:esquema.dim_total), collect(1:2)))
        indices_sal = collect(1:2)

        evaluadas = 0
        for ne in 3:min(6, esquema.dim_total)
            evaluadas >= 10000 && break
            for combo in combinations_iter(esquema.dim_total, ne)
                evaluadas >= 10000 && break
                for mask_sal in 1:3
                    idx_sal = bitmask_a_indices(mask_sal, 2)
                    subconfig = extraer_subconfiguracion(red, combo, idx_sal)
                    subconfig === nothing && continue
                    resultado = evaluar(subconfig, datos_eval_emb, indices_sal)
                    actualizar_si_mejor!(mapa, subconfig, resultado, 0.5)
                    evaluadas += 1
                end
            end
        end

        # Entrenar con datos completos
        entrenar_mapa!(mapa, datos_train_emb, datos_test_emb;
            epochs=300, lr=Float64(0.01), paciencia=50, batch_size=256)

        sel = seleccionar_mejor(mapa); ref = mapa.referencia_completa
        push!(ref_b, ref.precision_post_entrenamiento); push!(sel_b, sel.precision)
        @printf("│ %6d │ %4d │    %5.1f%%       │ %4d │    %5.1f%%      │ %-8s │\n",
            seed, sum(capas_b), ref.precision_post_entrenamiento*100, sel.neuronas, sel.precision*100, sel.tipo)
    end
    println("├────────┼──────┼────────────────┼──────┼───────────────┼──────────┤")
    @printf("│  Media │      │    %5.1f%%       │      │    %5.1f%%      │          │\n", mean(ref_b)*100, mean(sel_b)*100)
    println("└────────┴──────┴────────────────┴──────┴───────────────┴──────────┘")

    println("\n═══ Comparación ═══")
    @printf("  Ordinal:    ref=%.1f%%, div=%.1f%%\n", mean(ref_a)*100, mean(sel_a)*100)
    @printf("  Embeddings: ref=%.1f%%, div=%.1f%%\n", mean(ref_b)*100, mean(sel_b)*100)
end

exp18()
