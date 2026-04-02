"""
Experimento 5: Estadísticas detalladas con 100 semillas.

Reporta media, desviación estándar, mínimo, máximo y distribución
de neuronas activas para los problemas clave.

Ejecución:
    julia --project=. benchmarks/exp5_estadisticas.jl
"""

include("utils.jl")

function distribucion_neuronas(neurs::Vector{Int})
    isempty(neurs) && return ""
    conteo = Dict{Int, Int}()
    for n in neurs
        conteo[n] = get(conteo, n, 0) + 1
    end
    partes = sort(collect(conteo), by=first)
    join(["$(n)n:$(c)" for (n, c) in partes], " | ")
end

function exp5_estadisticas()
    println("╔══════════════════════════════════════════════════════════════╗")
    println("║  Experimento 5: Estadísticas detalladas (100 semillas)     ║")
    println("╚══════════════════════════════════════════════════════════════╝")

    seeds = SEEDS_100
    epochs = 3000

    configs = [
        ("XOR 2→8→1",       problema_xor(),           [2, 8, 1],  0.5),
        ("XOR 2→16→1",      problema_xor(),           [2, 16, 1], 0.5),
        ("AND 2→8→1",       problema_and(),           [2, 8, 1],  0.5),
        ("Paridad 3→8→1",   problema_paridad_3bits(), [3, 8, 1],  0.5),
        ("Paridad 3→16→1",  problema_paridad_3bits(), [3, 16, 1], 0.5),
        ("Encoder 4→8→2",   problema_encoder_4a2(),   [4, 8, 2],  0.5),
        ("Multi 2→8→3",     problema_multi_logica(),  [2, 8, 3],  0.5),
    ]

    for (nombre, datos, capas, umbral) in configs
        println("\n", "─"^60)
        println("  $nombre | umbral=$umbral | epochs=$epochs")
        println("─"^60)

        resultados = []
        for seed in seeds
            r = ejecutar_una(datos, capas, seed; umbral=umbral, epochs=epochs)
            push!(resultados, r)
        end

        exitos = filter(r -> r.encontrada, resultados)
        n_exitos = length(exitos)
        tasa = n_exitos / length(seeds) * 100

        @printf("  Tasa de éxito: %d / %d (%.1f%%)\n", n_exitos, length(seeds), tasa)

        if n_exitos == 0
            println("  Sin soluciones encontradas.\n")
            continue
        end

        # Neuronas activas
        neurs = [r.neuronas for r in exitos]
        @printf("  Neuronas activas: media=%.1f  std=%.1f  min=%d  max=%d\n",
            mean(neurs), std(neurs), minimum(neurs), maximum(neurs))
        println("  Distribución: ", distribucion_neuronas(neurs))

        # Entradas usadas
        ents = [r.n_entradas for r in exitos]
        @printf("  Entradas usadas: media=%.1f  std=%.1f  min=%d  max=%d\n",
            mean(ents), std(ents), minimum(ents), maximum(ents))

        # Precisión exploración
        prec_exp = [r.prec_exp for r in exitos]
        @printf("  Precisión exploración: media=%.1f%%  std=%.1f%%\n",
            mean(prec_exp) * 100, std(prec_exp) * 100)

        # Precisión pre-entrenamiento
        prec_pre = [r.prec_pre for r in exitos]
        @printf("  Precisión pre-entren.: media=%.1f%%  std=%.1f%%\n",
            mean(prec_pre) * 100, std(prec_pre) * 100)

        # Precisión post-entrenamiento
        prec_post = [r.prec_post for r in exitos]
        @printf("  Precisión post-entren: media=%.1f%%  std=%.1f%%\n",
            mean(prec_post) * 100, std(prec_post) * 100)

        # Mejora
        deltas = [r.prec_post - r.prec_pre for r in exitos]
        @printf("  Mejora entrenamiento:  media=%+.1f pp  std=%.1f pp\n",
            mean(deltas) * 100, std(deltas) * 100)

        # Tasa de 100%
        perfectas = count(r -> r.prec_post >= 0.99, exitos)
        @printf("  Soluciones perfectas (≥99%%): %d / %d (%.1f%% de éxitos)\n",
            perfectas, n_exitos, perfectas / n_exitos * 100)

        # Tiempo
        tiempos = [r.tiempo for r in resultados]
        @printf("  Tiempo medio: %.3fs  (total: %.1fs)\n", mean(tiempos), sum(tiempos))
    end

    println("\n", "═"^60)
end

exp5_estadisticas()
