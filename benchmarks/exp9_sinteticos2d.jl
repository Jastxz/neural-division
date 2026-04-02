"""
Experimento 9: Clasificación de regiones 2D sintéticas.

Problemas:
  A) Lunas (moons): 2 clases, no linealmente separable, fácil
  B) Círculos concéntricos: 2 clases, no linealmente separable, medio
  C) Espirales: 2 clases, no linealmente separable, difícil
  D) 4 clusters: 4 clases, linealmente separable, fácil

2 entradas (x, y), N muestras sintéticas.
Interesante para ver si el método necesita ambas entradas (debería).

Ejecución:
    julia --project=. benchmarks/exp9_sinteticos2d.jl
"""

include("utils.jl")

# ============================================================================
# Generadores de datos sintéticos
# ============================================================================

function generar_lunas(n::Int; ruido=0.1, seed=0)
    Random.seed!(seed)
    n_mitad = n ÷ 2
    # Luna superior
    θ1 = range(0, π, length=n_mitad)
    x1 = cos.(θ1) .+ randn(n_mitad) .* ruido
    y1 = sin.(θ1) .+ randn(n_mitad) .* ruido
    # Luna inferior (desplazada)
    θ2 = range(0, π, length=n - n_mitad)
    x2 = 1.0 .- cos.(θ2) .+ randn(n - n_mitad) .* ruido
    y2 = -sin.(θ2) .+ 0.5 .+ randn(n - n_mitad) .* ruido

    entradas = vcat(hcat(x1, y1), hcat(x2, y2))
    salidas = vcat(hcat(ones(n_mitad), zeros(n_mitad)),
                   hcat(zeros(n - n_mitad), ones(n - n_mitad)))
    return normalizar_entradas(entradas), salidas
end

function generar_circulos(n::Int; ruido=0.05, seed=0)
    Random.seed!(seed)
    n_mitad = n ÷ 2
    # Círculo interior
    θ1 = rand(n_mitad) .* 2π
    r1 = 0.3 .+ randn(n_mitad) .* ruido
    x1 = r1 .* cos.(θ1)
    y1 = r1 .* sin.(θ1)
    # Círculo exterior
    θ2 = rand(n - n_mitad) .* 2π
    r2 = 1.0 .+ randn(n - n_mitad) .* ruido
    x2 = r2 .* cos.(θ2)
    y2 = r2 .* sin.(θ2)

    entradas = vcat(hcat(x1, y1), hcat(x2, y2))
    salidas = vcat(hcat(ones(n_mitad), zeros(n_mitad)),
                   hcat(zeros(n - n_mitad), ones(n - n_mitad)))
    return normalizar_entradas(entradas), salidas
end

function generar_espirales(n::Int; ruido=0.15, seed=0)
    Random.seed!(seed)
    n_mitad = n ÷ 2
    # Espiral 1
    θ1 = range(0, 3π, length=n_mitad)
    r1 = θ1 ./ (3π)
    x1 = r1 .* cos.(θ1) .+ randn(n_mitad) .* ruido .* 0.1
    y1 = r1 .* sin.(θ1) .+ randn(n_mitad) .* ruido .* 0.1
    # Espiral 2 (rotada π)
    θ2 = range(0, 3π, length=n - n_mitad)
    r2 = θ2 ./ (3π)
    x2 = r2 .* cos.(θ2 .+ π) .+ randn(n - n_mitad) .* ruido .* 0.1
    y2 = r2 .* sin.(θ2 .+ π) .+ randn(n - n_mitad) .* ruido .* 0.1

    entradas = vcat(hcat(x1, y1), hcat(x2, y2))
    salidas = vcat(hcat(ones(n_mitad), zeros(n_mitad)),
                   hcat(zeros(n - n_mitad), ones(n - n_mitad)))
    return normalizar_entradas(entradas), salidas
end

function generar_4clusters(n::Int; ruido=0.15, seed=0)
    Random.seed!(seed)
    n_por_clase = n ÷ 4
    centros = [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)]
    entradas_list = Matrix{Float64}[]
    salidas_list = Matrix{Float64}[]
    for (c, (cx, cy)) in enumerate(centros)
        nc = c == 4 ? n - 3 * n_por_clase : n_por_clase
        x = cx .+ randn(nc) .* ruido
        y = cy .+ randn(nc) .* ruido
        push!(entradas_list, hcat(x, y))
        s = zeros(nc, 4)
        s[:, c] .= 1.0
        push!(salidas_list, s)
    end
    entradas = vcat(entradas_list...)
    salidas = vcat(salidas_list...)
    return normalizar_entradas(entradas), salidas
end

function normalizar_entradas(entradas)
    for j in 1:size(entradas, 2)
        col = entradas[:, j]
        mn, mx = minimum(col), maximum(col)
        rango = mx - mn
        rango > 0 && (entradas[:, j] .= (col .- mn) ./ rango)
    end
    entradas
end

function split_sintetico(entradas, salidas, ratio=0.8; seed=42)
    Random.seed!(seed)
    n = size(entradas, 1)
    perm = randperm(n)
    n_train = round(Int, n * ratio)
    idx_train = perm[1:n_train]
    idx_test = perm[n_train+1:end]
    return (entradas=entradas[idx_train, :], salidas=salidas[idx_train, :]),
           (entradas=entradas[idx_test, :], salidas=salidas[idx_test, :])
end

# ============================================================================
# Experimento
# ============================================================================

function ejecutar_sintetico(nombre, entradas, salidas, capas, seeds; umbral=0.5, epochs=1000, lr=0.01)
    n_total = sum(capas)
    n_sal = capas[end]
    n_ent = capas[1]

    @printf("\n─── %s | Red %s | %d muestras ───\n\n", nombre,
        join(capas, "→"), size(entradas, 1))

    println("┌────────┬──────────────────────┬──────────────────────┬──────────┐")
    println("│  Seed  │  Ref. Completa       │  Mejor Seleccionada  │ Tipo     │")
    println("│        │ Neur │ Post (test)    │ Neur │ Post (test)   │          │")
    println("├────────┼──────┼────────────────┼──────┼───────────────┼──────────┤")

    ref_posts = Float64[]
    sel_posts = Float64[]
    sel_neurs = Int[]
    sel_tipos = Symbol[]
    tiempos = Float64[]

    for seed in seeds
        datos_train, datos_test = split_sintetico(entradas, salidas; seed=seed)
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
end

function exp9_sinteticos()
    println("╔══════════════════════════════════════════════════════════════╗")
    println("║  Experimento 9: Clasificación 2D sintética                 ║")
    println("╚══════════════════════════════════════════════════════════════╝")

    seeds = SEEDS_20
    n_muestras = 300

    # A) Lunas
    ent_l, sal_l = generar_lunas(n_muestras; ruido=0.15, seed=0)
    for ocultas in [8, 16]
        ejecutar_sintetico("Lunas", ent_l, sal_l, [2, ocultas, 2], seeds)
    end

    # B) Círculos
    ent_c, sal_c = generar_circulos(n_muestras; ruido=0.08, seed=0)
    for ocultas in [8, 16]
        ejecutar_sintetico("Círculos", ent_c, sal_c, [2, ocultas, 2], seeds)
    end

    # C) Espirales
    ent_e, sal_e = generar_espirales(n_muestras; ruido=0.2, seed=0)
    for ocultas in [16, 32]
        ejecutar_sintetico("Espirales", ent_e, sal_e, [2, ocultas, 2], seeds)
    end

    # D) 4 clusters
    ent_4, sal_4 = generar_4clusters(n_muestras; ruido=0.2, seed=0)
    for ocultas in [8, 16]
        ejecutar_sintetico("4 Clusters", ent_4, sal_4, [2, ocultas, 4], seeds)
    end
end

exp9_sinteticos()
