"""
Experimento 1: Sensibilidad al umbral de acierto.

Ejecuta Paridad 3 bits con umbrales 0.3, 0.4, 0.5, 0.6, 0.7
y mide tasa de éxito, neuronas activas promedio, y precisión post-entrenamiento.

Ejecución:
    julia --project=. benchmarks/exp1_umbral.jl
"""

include("utils.jl")

function exp1_umbral()
    println("╔══════════════════════════════════════════════════════════════╗")
    println("║  Experimento 1: Sensibilidad al umbral de acierto          ║")
    println("╚══════════════════════════════════════════════════════════════╝")

    datos = problema_paridad_3bits()
    capas = [3, 6, 1]
    seeds = SEEDS_100
    umbrales = [0.3, 0.4, 0.5, 0.6, 0.7]

    println("\nProblema: Paridad 3 bits | Red: 3→6→1 | Seeds: $(length(seeds))")
    println("Epochs: 3000\n")

    # Cabecera
    println("┌─────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐")
    println("│ Umbral  │ Tasa éx. │ Neur.med │ Ent.med  │ Pre med  │ Post med │ Δ medio  │")
    println("├─────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┤")

    for umbral in umbrales
        resultados = []
        for seed in seeds
            r = ejecutar_una(datos, capas, seed; umbral=umbral, epochs=3000)
            push!(resultados, r)
        end

        exitos = filter(r -> r.encontrada, resultados)
        n_exitos = length(exitos)
        tasa = n_exitos / length(seeds) * 100

        if n_exitos > 0
            neur_med = mean(r.neuronas for r in exitos)
            ent_med = mean(r.n_entradas for r in exitos)
            pre_med = mean(r.prec_pre for r in exitos) * 100
            post_med = mean(r.prec_post for r in exitos) * 100
            delta_med = post_med - pre_med
            @printf("│  %.1f    │  %5.1f%%  │  %6.1f  │  %6.1f  │  %5.1f%%  │  %5.1f%%  │ %+5.1f pp │\n",
                umbral, tasa, neur_med, ent_med, pre_med, post_med, delta_med)
        else
            @printf("│  %.1f    │  %5.1f%%  │    -     │    -     │    -     │    -     │    -     │\n",
                umbral, tasa)
        end
    end

    println("└─────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘")

    # Repetir con AND para contraste (problema con sesgo de clase)
    println("\n--- Contraste: AND (2→4→1) ---\n")
    datos_and = problema_and()
    capas_and = [2, 4, 1]

    println("┌─────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐")
    println("│ Umbral  │ Tasa éx. │ Neur.med │ Ent.med  │ Pre med  │ Post med │ Δ medio  │")
    println("├─────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┤")

    for umbral in umbrales
        resultados = []
        for seed in seeds
            r = ejecutar_una(datos_and, capas_and, seed; umbral=umbral, epochs=2000)
            push!(resultados, r)
        end

        exitos = filter(r -> r.encontrada, resultados)
        n_exitos = length(exitos)
        tasa = n_exitos / length(seeds) * 100

        if n_exitos > 0
            neur_med = mean(r.neuronas for r in exitos)
            ent_med = mean(r.n_entradas for r in exitos)
            pre_med = mean(r.prec_pre for r in exitos) * 100
            post_med = mean(r.prec_post for r in exitos) * 100
            delta_med = post_med - pre_med
            @printf("│  %.1f    │  %5.1f%%  │  %6.1f  │  %6.1f  │  %5.1f%%  │  %5.1f%%  │ %+5.1f pp │\n",
                umbral, tasa, neur_med, ent_med, pre_med, post_med, delta_med)
        else
            @printf("│  %.1f    │  %5.1f%%  │    -     │    -     │    -     │    -     │    -     │\n",
                umbral, tasa)
        end
    end

    println("└─────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘")
end

exp1_umbral()
