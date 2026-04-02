"""
Experimento 2: Escalado de la red base.

Mismo problema (Paridad 3 bits y XOR) variando neuronas ocultas: 4, 8, 16, 32.
Mide cómo la probabilidad de encontrar buenas subredes escala con el tamaño de la red.

Ejecución:
    julia --project=. benchmarks/exp2_escalado.jl
"""

include("utils.jl")

function exp2_escalado()
    println("╔══════════════════════════════════════════════════════════════╗")
    println("║  Experimento 2: Escalado de la red base                    ║")
    println("╚══════════════════════════════════════════════════════════════╝")

    seeds = SEEDS_100
    ocultas_list = [4, 8, 16, 32]
    umbral = 0.5
    epochs = 3000

    for (nombre, datos, n_ent, n_sal) in [
        ("Paridad 3 bits", problema_paridad_3bits(), 3, 1),
        ("XOR",            problema_xor(),           2, 1),
        ("AND",            problema_and(),           2, 1),
        ("Encoder 4→2",    problema_encoder_4a2(),   4, 2),
    ]
        println("\n--- $nombre (umbral=$umbral, seeds=$(length(seeds)), epochs=$epochs) ---\n")
        println("┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐")
        println("│ Ocultas  │ Tasa éx. │ Neur.med │ Ent.med  │ Pre med  │ Post med │ Δ medio  │ T. med   │")
        println("├──────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┤")

        for n_ocultas in ocultas_list
            capas = [n_ent, n_ocultas, n_sal]
            resultados = []
            for seed in seeds
                r = ejecutar_una(datos, capas, seed; umbral=umbral, epochs=epochs)
                push!(resultados, r)
            end

            exitos = filter(r -> r.encontrada, resultados)
            n_exitos = length(exitos)
            tasa = n_exitos / length(seeds) * 100
            t_med = mean(r.tiempo for r in resultados)

            if n_exitos > 0
                neur_med = mean(r.neuronas for r in exitos)
                ent_med = mean(r.n_entradas for r in exitos)
                pre_med = mean(r.prec_pre for r in exitos) * 100
                post_med = mean(r.prec_post for r in exitos) * 100
                delta_med = post_med - pre_med
                @printf("│   %4d   │  %5.1f%%  │  %6.1f  │  %6.1f  │  %5.1f%%  │  %5.1f%%  │ %+5.1f pp │ %6.3fs  │\n",
                    n_ocultas, tasa, neur_med, ent_med, pre_med, post_med, delta_med, t_med)
            else
                @printf("│   %4d   │  %5.1f%%  │    -     │    -     │    -     │    -     │    -     │ %6.3fs  │\n",
                    n_ocultas, tasa, t_med)
            end
        end

        println("└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘")
    end
end

exp2_escalado()
