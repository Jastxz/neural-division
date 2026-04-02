"""
Experimento 6: Dataset Iris.

4 entradas (sépalos/pétalos), 3 clases (setosa, versicolor, virginica).
150 muestras, codificación one-hot para las salidas.
Split 80/20 train/test.

Ejecución:
    julia --project=. benchmarks/exp6_iris.jl
"""

include("utils.jl")

# ============================================================================
# Dataset Iris (hardcoded para no depender de paquetes externos)
# ============================================================================

function cargar_iris()
    # 150 muestras: sepal_length, sepal_width, petal_length, petal_width
    # Clase: 0=setosa, 1=versicolor, 2=virginica (50 de cada una)
    raw = Float64[
        5.1 3.5 1.4 0.2 0; 4.9 3.0 1.4 0.2 0; 4.7 3.2 1.3 0.2 0; 4.6 3.1 1.5 0.2 0;
        5.0 3.6 1.4 0.2 0; 5.4 3.9 1.7 0.4 0; 4.6 3.4 1.4 0.3 0; 5.0 3.4 1.5 0.2 0;
        4.4 2.9 1.4 0.2 0; 4.9 3.1 1.5 0.1 0; 5.4 3.7 1.5 0.2 0; 4.8 3.4 1.6 0.2 0;
        4.8 3.0 1.4 0.1 0; 4.3 3.0 1.1 0.1 0; 5.8 4.0 1.2 0.2 0; 5.7 4.4 1.5 0.4 0;
        5.4 3.9 1.3 0.4 0; 5.1 3.5 1.4 0.3 0; 5.7 3.8 1.7 0.3 0; 5.1 3.8 1.5 0.3 0;
        5.4 3.4 1.7 0.2 0; 5.1 3.7 1.5 0.4 0; 4.6 3.6 1.0 0.2 0; 5.1 3.3 1.7 0.5 0;
        4.8 3.4 1.9 0.2 0; 5.0 3.0 1.6 0.2 0; 5.0 3.4 1.6 0.4 0; 5.2 3.5 1.5 0.2 0;
        5.2 3.4 1.4 0.2 0; 4.7 3.2 1.6 0.2 0; 4.8 3.1 1.6 0.2 0; 5.4 3.4 1.5 0.4 0;
        5.2 4.1 1.5 0.1 0; 5.5 4.2 1.4 0.2 0; 4.9 3.1 1.5 0.2 0; 5.0 3.2 1.2 0.2 0;
        5.5 3.5 1.3 0.2 0; 4.9 3.6 1.4 0.1 0; 4.4 3.0 1.3 0.2 0; 5.1 3.4 1.5 0.2 0;
        5.0 3.5 1.3 0.3 0; 4.5 2.3 1.3 0.3 0; 4.4 3.2 1.3 0.2 0; 5.0 3.5 1.6 0.6 0;
        5.1 3.8 1.9 0.4 0; 4.8 3.0 1.4 0.3 0; 5.1 3.8 1.6 0.2 0; 4.6 3.2 1.4 0.2 0;
        5.3 3.7 1.5 0.2 0; 5.0 3.3 1.4 0.2 0;
        7.0 3.2 4.7 1.4 1; 6.4 3.2 4.5 1.5 1; 6.9 3.1 4.9 1.5 1; 5.5 2.3 4.0 1.3 1;
        6.5 2.8 4.6 1.5 1; 5.7 2.8 4.5 1.3 1; 6.3 3.3 4.7 1.6 1; 4.9 2.4 3.3 1.0 1;
        6.6 2.9 4.6 1.3 1; 5.2 2.7 3.9 1.4 1; 5.0 2.0 3.5 1.0 1; 5.9 3.0 4.2 1.5 1;
        6.0 2.2 4.0 1.0 1; 6.1 2.9 4.7 1.4 1; 5.6 2.9 3.6 1.3 1; 6.7 3.1 4.4 1.4 1;
        5.6 3.0 4.5 1.5 1; 5.8 2.7 4.1 1.0 1; 6.2 2.2 4.5 1.5 1; 5.6 2.5 3.9 1.1 1;
        5.9 3.2 4.8 1.8 1; 6.1 2.8 4.0 1.3 1; 6.3 2.5 4.9 1.5 1; 6.1 2.8 4.7 1.2 1;
        6.4 2.9 4.3 1.3 1; 6.6 3.0 4.4 1.4 1; 6.8 2.8 4.8 1.4 1; 6.7 3.0 5.0 1.7 1;
        6.0 2.9 4.5 1.5 1; 5.7 2.6 3.5 1.0 1; 5.5 2.4 3.8 1.1 1; 5.5 2.4 3.7 1.0 1;
        5.8 2.7 3.9 1.2 1; 6.0 2.7 5.1 1.6 1; 5.4 3.0 4.5 1.5 1; 6.0 3.4 4.5 1.6 1;
        6.7 3.1 4.7 1.5 1; 6.3 2.3 4.4 1.3 1; 5.6 3.0 4.1 1.3 1; 5.5 2.5 4.0 1.3 1;
        5.5 2.6 4.4 1.2 1; 6.1 3.0 4.6 1.4 1; 5.8 2.6 4.0 1.2 1; 5.0 2.3 3.3 1.0 1;
        5.6 2.7 4.2 1.3 1; 5.7 3.0 4.2 1.2 1; 5.7 2.9 4.2 1.3 1; 6.2 2.9 4.3 1.3 1;
        5.1 2.5 3.0 1.1 1; 5.7 2.8 4.1 1.3 1;
        6.3 3.3 6.0 2.5 2; 5.8 2.7 5.1 1.9 2; 7.1 3.0 5.9 2.1 2; 6.3 2.9 5.6 1.8 2;
        6.5 3.0 5.8 2.2 2; 7.6 3.0 6.6 2.1 2; 4.9 2.5 4.5 1.7 2; 7.3 2.9 6.3 1.8 2;
        6.7 2.5 5.8 1.8 2; 7.2 3.6 6.1 2.5 2; 6.5 3.2 5.1 2.0 2; 6.4 2.7 5.3 1.9 2;
        6.8 3.0 5.5 2.1 2; 5.7 2.5 5.0 2.0 2; 5.8 2.8 5.1 2.4 2; 6.4 3.2 5.3 2.3 2;
        6.5 3.0 5.5 1.8 2; 7.7 3.8 6.7 2.2 2; 7.7 2.6 6.9 2.3 2; 6.0 2.2 5.0 1.5 2;
        6.9 3.2 5.7 2.3 2; 5.6 2.8 4.9 2.0 2; 7.7 2.8 6.7 2.0 2; 6.3 2.7 4.9 1.8 2;
        6.7 3.3 5.7 2.1 2; 7.2 3.2 6.0 1.8 2; 6.2 2.8 4.8 1.8 2; 6.1 3.0 4.9 1.8 2;
        6.4 2.8 5.6 2.1 2; 7.2 3.0 5.8 1.6 2; 7.4 2.8 6.1 1.9 2; 7.9 3.8 6.4 2.0 2;
        6.4 2.8 5.6 2.2 2; 6.3 2.8 5.1 1.5 2; 6.1 2.6 5.6 1.4 2; 7.7 3.0 6.1 2.3 2;
        6.3 3.4 5.6 2.4 2; 6.4 3.1 5.5 1.8 2; 6.0 3.0 4.8 1.8 2; 6.9 3.1 5.4 2.1 2;
        6.7 3.1 5.6 2.4 2; 6.9 3.1 5.1 2.3 2; 5.8 2.7 5.1 1.9 2; 6.8 3.2 5.9 2.3 2;
        6.7 3.3 5.7 2.5 2; 6.7 3.0 5.2 2.3 2; 6.3 2.5 5.0 1.9 2; 6.5 3.0 5.2 2.0 2;
        6.2 3.4 5.4 2.3 2; 5.9 3.0 5.1 1.8 2
    ]

    entradas = raw[:, 1:4]
    clases = Int.(raw[:, 5])

    # Normalizar entradas a [0, 1]
    for j in 1:4
        col = entradas[:, j]
        mn, mx = minimum(col), maximum(col)
        entradas[:, j] .= (col .- mn) ./ (mx - mn)
    end

    # One-hot encoding de salidas
    salidas = zeros(Float64, 150, 3)
    for i in 1:150
        salidas[i, clases[i] + 1] = 1.0
    end

    return entradas, salidas
end

"""
Split estratificado train/test.
"""
function split_datos(entradas, salidas, ratio_train=0.8; seed=42)
    Random.seed!(seed)
    n = size(entradas, 1)
    n_clases = size(salidas, 2)

    idx_train = Int[]
    idx_test = Int[]

    for c in 1:n_clases
        idx_clase = findall(salidas[:, c] .== 1.0)
        perm = randperm(length(idx_clase))
        n_train = round(Int, length(idx_clase) * ratio_train)
        append!(idx_train, idx_clase[perm[1:n_train]])
        append!(idx_test, idx_clase[perm[n_train+1:end]])
    end

    shuffle!(idx_train)
    shuffle!(idx_test)

    datos_train = (entradas=entradas[idx_train, :], salidas=salidas[idx_train, :])
    datos_test = (entradas=entradas[idx_test, :], salidas=salidas[idx_test, :])

    return datos_train, datos_test
end

# ============================================================================
# Experimento
# ============================================================================

function exp6_iris()
    println("╔══════════════════════════════════════════════════════════════╗")
    println("║  Experimento 6: Dataset Iris                               ║")
    println("║  4 entradas, 3 salidas (one-hot), 150 muestras             ║")
    println("╚══════════════════════════════════════════════════════════════╝")

    entradas, salidas = cargar_iris()
    seeds = SEEDS_20
    epochs = 1000
    lr = 0.01
    paciencia = 100

    println("\nSubconfiguraciones posibles: (2^4-1)×(2^3-1) = $(15*7) = 105")
    println("Seeds: $(length(seeds)) | Epochs: $epochs | LR: $lr | Paciencia: $paciencia\n")

    # Probar con distintas arquitecturas
    for n_ocultas in [8, 16, 32]
        capas = [4, n_ocultas, 3]
        n_total = sum(capas)

        println("─"^60)
        @printf("  Red 4→%d→3 (%d neuronas totales) | umbral=0.4\n", n_ocultas, n_total)
        println("─"^60)

        println("\n┌────────┬────────────────────────────┬────────────────────────────┬──────────┐")
        println("│  Seed  │   Referencia Completa      │   Mejor Seleccionada       │ Tipo     │")
        println("│        │ Neur │ Pre    │ Post (test) │ Neur │ Pre    │ Post (test) │          │")
        println("├────────┼──────┼────────┼─────────────┼──────┼────────┼─────────────┼──────────┤")

        ref_posts = Float64[]
        sel_posts = Float64[]
        sel_neurs = Int[]
        sel_tipos = Symbol[]

        for seed in seeds
            datos_train, datos_test = split_datos(entradas, salidas; seed=seed)

            red = crear_red(capas; seed=seed)
            config = ConfiguracionDivision{Float64}(0.4)

            mapa = ejecutar_division(red, datos_test, config;
                datos_entrenamiento=datos_train,
                epochs=epochs, lr=Float64(lr), paciencia=paciencia)

            sel = seleccionar_mejor(mapa)
            ref = mapa.referencia_completa

            ref_post = ref.precision_post_entrenamiento
            push!(ref_posts, ref_post)
            push!(sel_posts, sel.precision)
            push!(sel_neurs, sel.neuronas)
            push!(sel_tipos, sel.tipo)

            @printf("│ %6d │ %4d │ %5.1f%% │    %5.1f%%    │ %4d │ %5.1f%% │    %5.1f%%    │ %-8s │\n",
                seed,
                ref.subconfiguracion.n_neuronas_activas,
                ref.precision_pre_entrenamiento * 100,
                ref_post * 100,
                sel.neuronas,
                sel.entrada.precision_pre_entrenamiento * 100,
                sel.precision * 100,
                sel.tipo)
        end

        println("├────────┼──────┼────────┼─────────────┼──────┼────────┼─────────────┼──────────┤")

        n_ref = count(t -> t == :referencia, sel_tipos)
        n_sub = count(t -> t != :referencia, sel_tipos)

        @printf("│  Media │ %4d │        │    %5.1f%%    │ %4.0f │        │    %5.1f%%    │ ref:%d    │\n",
            n_total,
            mean(ref_posts) * 100,
            mean(sel_neurs),
            mean(sel_posts) * 100,
            n_ref)
        @printf("│  Std   │      │        │    %5.1f%%    │ %4.1f │        │    %5.1f%%    │ sub:%d    │\n",
            std(ref_posts) * 100,
            std(sel_neurs),
            std(sel_posts) * 100,
            n_sub)

        println("└────────┴──────┴────────┴─────────────┴──────┴────────┴─────────────┴──────────┘")

        # Análisis de soluciones parciales
        println("\n  Soluciones parciales más frecuentes (por subconjunto de salidas):")
        # Tomar el último mapa como ejemplo
        datos_train, datos_test = split_datos(entradas, salidas; seed=42)
        red = crear_red(capas; seed=42)
        config = ConfiguracionDivision{Float64}(0.4)
        mapa_ejemplo = ejecutar_division(red, datos_test, config;
            datos_entrenamiento=datos_train,
            epochs=epochs, lr=Float64(lr), paciencia=paciencia)

        nombres_clase = ["Setosa", "Versicolor", "Virginica"]
        claves = sort(collect(keys(mapa_ejemplo.parciales)), by=k -> (length(k), k))
        for clave in claves
            length(clave) > 2 && continue  # solo individuales y pares
            entrada = mapa_ejemplo.parciales[clave]
            nombre_clases = join([nombres_clase[i] for i in clave], "+")
            if entrada.subconfiguracion !== nothing
                sc = entrada.subconfiguracion
                @printf("    Salidas %s (%s): %d neur, entradas %s, %.1f%% → %.1f%%\n",
                    clave, nombre_clases,
                    sc.n_neuronas_activas, sc.indices_entrada,
                    entrada.precision_pre_entrenamiento * 100,
                    entrada.precision_post_entrenamiento * 100)
            else
                @printf("    Salidas %s (%s): sin solución\n", clave, nombre_clases)
            end
        end
        println()
    end
end

exp6_iris()
