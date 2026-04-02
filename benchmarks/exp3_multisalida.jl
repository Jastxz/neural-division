"""
Experimento 3: Descomposición multi-salida.

Problema con 4 salidas de distinta dificultad:
  Salida 1: AND (lineal)
  Salida 2: OR  (lineal)
  Salida 3: XOR (no lineal)
  Salida 4: NAND (lineal)

¿El mapa de soluciones parciales refleja la estructura del problema?

Ejecución:
    julia --project=. benchmarks/exp3_multisalida.jl
"""

include("utils.jl")

function problema_4salidas()
    e = tabla_verdad(2)
    # AND, OR, XOR, NAND
    s = Float64[
        0 0 0 1;   # 0,0
        0 1 1 1;   # 0,1
        0 1 1 1;   # 1,0
        1 1 0 0    # 1,1
    ]
    (entradas=e, salidas=s)
end

function exp3_multisalida()
    println("╔══════════════════════════════════════════════════════════════╗")
    println("║  Experimento 3: Descomposición multi-salida                ║")
    println("╚══════════════════════════════════════════════════════════════╝")

    datos = problema_4salidas()
    seeds = SEEDS_100
    umbral = 0.5
    epochs = 2000
    nombres_salida = Dict(1=>"AND", 2=>"OR", 3=>"XOR", 4=>"NAND")

    for n_ocultas in [8, 16]
        capas = [2, n_ocultas, 4]
        println("\n--- Red 2→$(n_ocultas)→4 | umbral=$umbral | seeds=$(length(seeds)) ---\n")

        # Recopilar resultados de todas las semillas
        todos_mapas = []
        for seed in seeds
            red = crear_red(capas; seed=seed)
            config = ConfiguracionDivision{Float64}(umbral)
            mapa = ejecutar_division(red, datos, config;
                datos_entrenamiento=datos, epochs=epochs)
            push!(todos_mapas, mapa)
        end

        # Analizar solución global
        n_global = count(m -> m.global_.subconfiguracion !== nothing, todos_mapas)
        println("Solución global encontrada: $n_global / $(length(seeds))")
        if n_global > 0
            globales = filter(m -> m.global_.subconfiguracion !== nothing, todos_mapas)
            neur_med = mean(m.global_.subconfiguracion.n_neuronas_activas for m in globales)
            post_med = mean(m.global_.precision_post_entrenamiento for m in globales) * 100
            @printf("  Neuronas medias: %.1f | Precisión post media: %.1f%%\n", neur_med, post_med)
        end

        # Analizar cada subconjunto de salida individual
        println("\nSoluciones parciales por salida individual:")
        println("┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐")
        println("│ Salida   │ Tasa éx. │ Neur.med │ Ent.med  │ Post med │ Δ medio  │")
        println("├──────────┼──────────┼──────────┼──────────┼──────────┼──────────┤")

        for sal_idx in 1:4
            clave = [sal_idx]
            exitos = []
            for m in todos_mapas
                if haskey(m.parciales, clave) && m.parciales[clave].subconfiguracion !== nothing
                    e = m.parciales[clave]
                    push!(exitos, (neuronas=e.subconfiguracion.n_neuronas_activas,
                                   n_entradas=length(e.subconfiguracion.indices_entrada),
                                   pre=e.precision_pre_entrenamiento,
                                   post=e.precision_post_entrenamiento))
                end
            end

            tasa = length(exitos) / length(seeds) * 100
            nombre = nombres_salida[sal_idx]

            if !isempty(exitos)
                neur_med = mean(e.neuronas for e in exitos)
                ent_med = mean(e.n_entradas for e in exitos)
                post_med = mean(e.post for e in exitos) * 100
                pre_med = mean(e.pre for e in exitos) * 100
                delta = post_med - pre_med
                @printf("│ %4s(%d)  │  %5.1f%%  │  %6.1f  │  %6.1f  │  %5.1f%%  │ %+5.1f pp │\n",
                    nombre, sal_idx, tasa, neur_med, ent_med, post_med, delta)
            else
                @printf("│ %4s(%d)  │  %5.1f%%  │    -     │    -     │    -     │    -     │\n",
                    nombre, sal_idx, tasa)
            end
        end

        println("└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘")

        # Pares de salidas interesantes
        println("\nSoluciones parciales por pares:")
        println("┌──────────────┬──────────┬──────────┬──────────┐")
        println("│ Par salidas  │ Tasa éx. │ Neur.med │ Post med │")
        println("├──────────────┼──────────┼──────────┼──────────┤")

        for par in [[1,2], [1,3], [2,3], [1,4], [2,4], [3,4]]
            exitos = []
            for m in todos_mapas
                if haskey(m.parciales, par) && m.parciales[par].subconfiguracion !== nothing
                    e = m.parciales[par]
                    push!(exitos, (neuronas=e.subconfiguracion.n_neuronas_activas,
                                   post=e.precision_post_entrenamiento))
                end
            end

            tasa = length(exitos) / length(seeds) * 100
            label = join([nombres_salida[i] for i in par], "+")

            if !isempty(exitos)
                neur_med = mean(e.neuronas for e in exitos)
                post_med = mean(e.post for e in exitos) * 100
                @printf("│ %-12s │  %5.1f%%  │  %6.1f  │  %5.1f%%  │\n",
                    label, tasa, neur_med, post_med)
            else
                @printf("│ %-12s │  %5.1f%%  │    -     │    -     │\n",
                    label, tasa)
            end
        end

        println("└──────────────┴──────────┴──────────┴──────────┘")
    end
end

exp3_multisalida()
