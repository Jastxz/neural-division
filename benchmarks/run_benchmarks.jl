"""
Benchmarks del Método de la División Neuronal.

Batería de problemas de complejidad creciente para validar el método
y recopilar métricas de cara a un artículo.

Ejecución:
    julia --project=. benchmarks/run_benchmarks.jl
"""

using DivisionNeuronal
using Random
using Printf
using Statistics

# ============================================================================
# Utilidades
# ============================================================================

"""
Crea una RedBase{Float64} con pesos aleatorios.
`capas` es un vector con las dimensiones de cada capa, e.g. [2, 4, 1].
"""
function crear_red(capas::Vector{Int}; seed::Int=42)
    Random.seed!(seed)
    n_entradas = capas[1]
    n_salidas = capas[end]
    pesos = Matrix{Float64}[]
    biases = Vector{Float64}[]
    for i in 1:(length(capas) - 1)
        push!(pesos, randn(Float64, capas[i], capas[i + 1]))
        push!(biases, randn(Float64, capas[i + 1]))
    end
    RedBase{Float64}(pesos, biases, n_entradas, n_salidas)
end

"""
Genera todas las combinaciones binarias de `n` bits como matriz (2^n × n).
"""
function tabla_verdad(n::Int)
    filas = 2^n
    m = zeros(Float64, filas, n)
    for i in 0:(filas - 1)
        for j in 0:(n - 1)
            m[i + 1, n - j] = Float64((i >> j) & 1)
        end
    end
    m
end

"""
Imprime un resumen compacto de los resultados de un benchmark.
"""
function imprimir_resultado(nombre::String, mapa::MapaDeSoluciones{Float64}, tiempo::Float64, n_salidas::Int)
    println("\n", "="^60)
    println("  $nombre")
    println("="^60)
    @printf("  Tiempo de ejecución: %.3f s\n", tiempo)

    # Solución global
    g = mapa.global_
    if g.subconfiguracion !== nothing
        sc = g.subconfiguracion
        @printf("  Solución Global:\n")
        @printf("    Entradas: %s → Salidas: %s\n", sc.indices_entrada, sc.indices_salida)
        @printf("    Neuronas activas: %d\n", sc.n_neuronas_activas)
        @printf("    Precisión (exploración): %.2f%%\n", g.precision * 100)
        @printf("    Precisión pre-entrenamiento:  %.2f%%\n", g.precision_pre_entrenamiento * 100)
        @printf("    Precisión post-entrenamiento: %.2f%%\n", g.precision_post_entrenamiento * 100)
        dif = g.precision_post_entrenamiento - g.precision_pre_entrenamiento
        @printf("    Mejora por entrenamiento: %+.2f pp\n", dif * 100)
    else
        println("  Solución Global: NO ENCONTRADA")
    end

    # Resumen de soluciones parciales
    n_parciales = count(e -> e.subconfiguracion !== nothing, values(mapa.parciales))
    total_parciales = length(mapa.parciales)
    @printf("  Soluciones parciales: %d / %d\n", n_parciales, total_parciales)

    # Detalle de parciales con solución
    if n_parciales > 0 && n_salidas <= 4
        claves = sort(collect(keys(mapa.parciales)), by=k -> (length(k), k))
        for clave in claves
            entrada = mapa.parciales[clave]
            entrada.subconfiguracion === nothing && continue
            sc = entrada.subconfiguracion
            @printf("    Salidas %s: %d neuronas, %.0f%% → %.0f%% (Δ%+.0f%%)\n",
                clave, sc.n_neuronas_activas,
                entrada.precision_pre_entrenamiento * 100,
                entrada.precision_post_entrenamiento * 100,
                (entrada.precision_post_entrenamiento - entrada.precision_pre_entrenamiento) * 100)
        end
    end
    println()
end

# ============================================================================
# Definición de problemas
# ============================================================================

"""
Problema XOR: 2 entradas, 1 salida. No linealmente separable.
"""
function problema_xor()
    entradas = tabla_verdad(2)
    salidas = reshape(Float64[0, 1, 1, 0], 4, 1)
    return (entradas=entradas, salidas=salidas)
end

"""
Problema AND: 2 entradas, 1 salida. Linealmente separable.
"""
function problema_and()
    entradas = tabla_verdad(2)
    salidas = reshape(Float64[0, 0, 0, 1], 4, 1)
    return (entradas=entradas, salidas=salidas)
end

"""
Problema OR: 2 entradas, 1 salida. Linealmente separable.
"""
function problema_or()
    entradas = tabla_verdad(2)
    salidas = reshape(Float64[0, 1, 1, 1], 4, 1)
    return (entradas=entradas, salidas=salidas)
end

"""
Problema NAND: 2 entradas, 1 salida. Linealmente separable.
"""
function problema_nand()
    entradas = tabla_verdad(2)
    salidas = reshape(Float64[1, 1, 1, 0], 4, 1)
    return (entradas=entradas, salidas=salidas)
end

"""
Problema multi-salida: 2 entradas, 3 salidas (AND, OR, XOR simultáneamente).
Aquí el mapa de soluciones parciales cobra sentido.
"""
function problema_multi_logica()
    entradas = tabla_verdad(2)
    # Columna 1: AND, Columna 2: OR, Columna 3: XOR
    salidas = Float64[
        0 0 0;  # 0,0
        0 1 1;  # 0,1
        0 1 1;  # 1,0
        1 1 0   # 1,1
    ]
    return (entradas=entradas, salidas=salidas)
end

"""
Paridad de 3 bits: 3 entradas, 1 salida.
Salida = 1 si número impar de bits encendidos.
Generalización de XOR a 3 dimensiones.
"""
function problema_paridad_3bits()
    entradas = tabla_verdad(3)
    salidas = reshape(Float64[0, 1, 1, 0, 1, 0, 0, 1], 8, 1)
    return (entradas=entradas, salidas=salidas)
end

"""
Codificador 4→2: 4 entradas (one-hot), 2 salidas (binario).
Aprende la representación comprimida.
"""
function problema_encoder_4a2()
    entradas = Float64[
        1 0 0 0;
        0 1 0 0;
        0 0 1 0;
        0 0 0 1
    ]
    salidas = Float64[
        0 0;
        0 1;
        1 0;
        1 1
    ]
    return (entradas=entradas, salidas=salidas)
end

# ============================================================================
# Ejecución de benchmarks
# ============================================================================

function ejecutar_benchmark(nombre::String, datos, capas::Vector{Int};
                            umbral::Float64=0.4, epochs::Int=500,
                            seeds::Vector{Int}=[42, 123, 7, 2024, 31415, 1, 99, 256, 777, 1337,
                                                 5555, 8080, 9999, 12345, 54321, 65536, 100000,
                                                 271828, 314159, 999999])
    println("\n🔬 Ejecutando: $nombre ($(length(seeds)) semillas, umbral=$(umbral))")

    resultados = []

    for (i, seed) in enumerate(seeds)
        red = crear_red(capas; seed=seed)
        config = ConfiguracionDivision{Float64}(umbral)

        t = @elapsed mapa = ejecutar_division(red, datos, config;
            datos_entrenamiento=datos,
            epochs=epochs)

        push!(resultados, (mapa=mapa, tiempo=t, seed=seed))
        @printf("  Semilla %d: %.2fs", seed, t)

        g = mapa.global_
        if g.subconfiguracion !== nothing
            @printf(" | Global: %d neuronas, %.0f%% → %.0f%%",
                g.subconfiguracion.n_neuronas_activas,
                g.precision_pre_entrenamiento * 100,
                g.precision_post_entrenamiento * 100)
        else
            print(" | Global: -")
        end
        println()
    end

    # Imprimir el mejor resultado (mayor precisión post-entrenamiento global)
    mejor = nothing
    mejor_prec = -1.0
    for r in resultados
        g = r.mapa.global_
        if g.subconfiguracion !== nothing && g.precision_post_entrenamiento > mejor_prec
            mejor_prec = g.precision_post_entrenamiento
            mejor = r
        end
    end

    if mejor !== nothing
        imprimir_resultado("$nombre (mejor resultado, seed=$(mejor.seed))",
            mejor.mapa, mejor.tiempo, capas[end])
    else
        # Mostrar el primer resultado aunque no tenga solución global
        imprimir_resultado("$nombre (seed=$(resultados[1].seed))",
            resultados[1].mapa, resultados[1].tiempo, capas[end])
    end

    # Tabla resumen
    println("  Resumen por semilla:")
    println("  ┌────────┬──────────┬──────────┬──────────┬──────────┐")
    println("  │  Seed  │ Neuronas │ Prec.Exp │ Pre-Ent. │ Post-Ent │")
    println("  ├────────┼──────────┼──────────┼──────────┼──────────┤")
    for r in resultados
        g = r.mapa.global_
        if g.subconfiguracion !== nothing
            @printf("  │ %6d │ %8d │ %7.1f%% │ %7.1f%% │ %7.1f%% │\n",
                r.seed, g.subconfiguracion.n_neuronas_activas,
                g.precision * 100,
                g.precision_pre_entrenamiento * 100,
                g.precision_post_entrenamiento * 100)
        else
            @printf("  │ %6d │    -     │    -     │    -     │    -     │\n", r.seed)
        end
    end
    println("  └────────┴──────────┴──────────┴──────────┴──────────┘")

    return resultados
end

# ============================================================================
# Main
# ============================================================================

function main()
    println("╔══════════════════════════════════════════════════════════════╗")
    println("║   Benchmarks: Método de la División Neuronal               ║")
    println("║   Batería de problemas lógicos                             ║")
    println("╚══════════════════════════════════════════════════════════════╝")

    # --- Nivel 1: Funciones lógicas básicas (2 entradas, 1 salida) ---
    # Red: 2 → 4 ocultas → 1
    capas_basicas = [2, 4, 1]

    println("\n" * "─"^60)
    println("  NIVEL 1: Funciones lógicas básicas (2→4→1)")
    println("─"^60)

    res_xor  = ejecutar_benchmark("XOR (control)", problema_xor(), capas_basicas; umbral=0.5, epochs=2000)
    res_and  = ejecutar_benchmark("AND", problema_and(), capas_basicas; umbral=0.5, epochs=2000)
    res_or   = ejecutar_benchmark("OR", problema_or(), capas_basicas; umbral=0.5, epochs=2000)
    res_nand = ejecutar_benchmark("NAND", problema_nand(), capas_basicas; umbral=0.5, epochs=2000)

    # --- Nivel 2: Multi-salida (2 entradas, 3 salidas) ---
    # Red: 2 → 6 ocultas → 3
    capas_multi = [2, 6, 3]

    println("\n" * "─"^60)
    println("  NIVEL 2: Multi-salida (2→6→3)")
    println("─"^60)

    res_multi = ejecutar_benchmark("AND+OR+XOR (multi-salida)", problema_multi_logica(), capas_multi; umbral=0.5, epochs=2000)

    # --- Nivel 3: Paridad 3 bits ---
    # Red: 3 → 6 ocultas → 1
    capas_paridad = [3, 6, 1]

    println("\n" * "─"^60)
    println("  NIVEL 3: Paridad de 3 bits (3→6→1)")
    println("─"^60)

    res_paridad = ejecutar_benchmark("Paridad 3 bits", problema_paridad_3bits(), capas_paridad; umbral=0.5, epochs=3000)

    # --- Nivel 4: Encoder 4→2 ---
    # Red: 4 → 6 ocultas → 2
    capas_encoder = [4, 6, 2]

    println("\n" * "─"^60)
    println("  NIVEL 4: Encoder 4→2 (4→6→2)")
    println("─"^60)

    res_encoder = ejecutar_benchmark("Encoder 4→2", problema_encoder_4a2(), capas_encoder; umbral=0.5, epochs=2000)

    println("\n" * "═"^60)
    println("  Benchmarks completados.")
    println("═"^60)
end

main()
