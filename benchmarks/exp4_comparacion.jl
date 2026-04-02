"""
Experimento 4: Comparación con red completa entrenada.

Para cada problema, entrena la red completa (sin división) y compara:
¿la subred encontrada alcanza la misma precisión con menos neuronas?

Ejecución:
    julia --project=. benchmarks/exp4_comparacion.jl
"""

include("utils.jl")

"""
Entrena la red completa (todas las entradas, todas las salidas) y devuelve precisión.
"""
function entrenar_red_completa(datos, capas::Vector{Int}, seed::Int; epochs=3000)
    red = crear_red(capas; seed=seed)
    n_ent = capas[1]
    n_sal = capas[end]

    # Crear subconfiguración con TODAS las entradas y salidas
    subconfig = extraer_subconfiguracion(red, collect(1:n_ent), collect(1:n_sal))
    subconfig === nothing && return (prec_pre=0.0, prec_post=0.0, neuronas=sum(capas))

    indices_salida = collect(1:n_sal)
    resultado_pre = evaluar(subconfig, datos, indices_salida)
    prec_pre = resultado_pre.precision_global

    (pre, post) = entrenar_y_evaluar!(subconfig, datos, datos; epochs=epochs)

    return (prec_pre=pre, prec_post=post, neuronas=subconfig.n_neuronas_activas)
end

function exp4_comparacion()
    println("╔══════════════════════════════════════════════════════════════╗")
    println("║  Experimento 4: Comparación con red completa entrenada     ║")
    println("╚══════════════════════════════════════════════════════════════╝")

    seeds = SEEDS_100
    umbral = 0.5
    epochs = 3000

    problemas = [
        ("XOR",            problema_xor(),           [2, 8, 1]),
        ("AND",            problema_and(),           [2, 8, 1]),
        ("Paridad 3b",     problema_paridad_3bits(), [3, 8, 1]),
        ("Encoder 4→2",    problema_encoder_4a2(),   [4, 8, 2]),
        ("Multi AND+OR+XOR", problema_multi_logica(), [2, 8, 3]),
    ]

    println("\nSeeds: $(length(seeds)) | Epochs: $epochs | Umbral división: $umbral\n")
    println("┌──────────────────┬────────────────────────────┬────────────────────────────┬──────────┐")
    println("│ Problema         │   Red Completa             │   División Neuronal        │ Ahorro   │")
    println("│                  │ Neur. │ Pre   │ Post       │ Neur. │ Pre   │ Post       │ neuronas │")
    println("├──────────────────┼───────┼───────┼────────────┼───────┼───────┼────────────┼──────────┤")

    for (nombre, datos, capas) in problemas
        # Red completa: entrenar todas las semillas
        completa_posts = Float64[]
        completa_neur = 0
        for seed in seeds
            r = entrenar_red_completa(datos, capas, seed; epochs=epochs)
            push!(completa_posts, r.prec_post)
            completa_neur = r.neuronas  # siempre igual para misma arquitectura
        end
        completa_pre_med = 0.0  # no relevante, varía mucho
        completa_post_med = mean(completa_posts) * 100
        completa_100 = count(p -> p >= 0.99, completa_posts)

        # División neuronal: ejecutar todas las semillas
        division_posts = Float64[]
        division_neurs = Int[]
        for seed in seeds
            r = ejecutar_una(datos, capas, seed; umbral=umbral, epochs=epochs)
            if r.encontrada
                push!(division_posts, r.prec_post)
                push!(division_neurs, r.neuronas)
            end
        end

        n_div = length(division_posts)
        if n_div > 0
            div_post_med = mean(division_posts) * 100
            div_neur_med = mean(division_neurs)
            div_100 = count(p -> p >= 0.99, division_posts)
            ahorro = (1.0 - div_neur_med / completa_neur) * 100

            @printf("│ %-16s │  %4d │       │ %5.1f%% %2d✓ │  %4.0f │       │ %5.1f%% %2d✓ │  %+5.1f%%  │\n",
                nombre, completa_neur, completa_post_med, completa_100,
                div_neur_med, div_post_med, div_100, ahorro)
        else
            @printf("│ %-16s │  %4d │       │ %5.1f%% %2d✓ │   -   │       │   -        │    -     │\n",
                nombre, completa_neur, completa_post_med, completa_100)
        end
    end

    println("└──────────────────┴───────┴───────┴────────────┴───────┴───────┴────────────┴──────────┘")
    println("\n✓ = semillas que alcanzaron ≥99% precisión")
    println("Ahorro = reducción de neuronas de la subred vs red completa")
end

exp4_comparacion()
