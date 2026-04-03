"""
Experimento 16: MONKS Problems.

Tres problemas diseñados para ser difíciles para algoritmos de aprendizaje:
  Monks-1: (a1 == a2) OR (a5 == 1) — regla lógica simple
  Monks-2: exactamente 2 de los 6 atributos tienen su primer valor — combinatoria
  Monks-3: (a5 == 3 AND a4 == 1) OR (a5 != 4 AND a2 != 3) — con 5% ruido

6 atributos, 2 clases, ~124-169 train, 432 test.
Subconfiguraciones: (2^6-1)×(2^2-1) = 189

Ejecución:
    julia --project=. benchmarks/exp16_monks.jl
"""

include("utils.jl")

function cargar_monks(num)
    ruta_train = joinpath(@__DIR__, "..", "csv", "monks$(num)_train.csv")
    ruta_test = joinpath(@__DIR__, "..", "csv", "monks$(num)_test.csv")

    function parsear(ruta)
        lines = filter(!isempty, readlines(ruta))
        n = length(lines)
        entradas = zeros(Float64, n, 6)
        salidas = zeros(Float64, n, 2)
        for (i, line) in enumerate(lines)
            vals = split(strip(line))
            clase = parse(Int, vals[1])  # 0 o 1
            salidas[i, clase + 1] = 1.0
            for j in 1:6
                entradas[i, j] = parse(Float64, vals[j + 1])
            end
        end
        # Normalizar a [0,1]
        for j in 1:6
            col = entradas[:, j]
            mn, mx = minimum(col), maximum(col)
            rango = mx - mn
            rango > 0 && (entradas[:, j] .= (col .- mn) ./ rango)
        end
        return (entradas=entradas, salidas=salidas)
    end

    return parsear(ruta_train), parsear(ruta_test)
end

function ejecutar_monks(nombre, datos_train, datos_test, capas, seeds;
                        umbral=0.4, epochs=1000, lr=0.01)
    n_total = sum(capas)
    n_ent = capas[1]
    n_sal = capas[end]
    total_sub = (2^n_ent - 1) * (2^n_sal - 1)

    @printf("\n─── %s | Red %s (%d neur) | %d subconfigs ───\n",
        nombre, join(capas,"→"), n_total, total_sub)
    @printf("    %d train, %d test\n\n", size(datos_train.entradas, 1), size(datos_test.entradas, 1))

    println("┌────────┬──────────────────────┬──────────────────────┬──────────┐")
    println("│  Seed  │  Ref. Completa       │  Mejor Seleccionada  │ Tipo     │")
    println("│        │ Neur │ Post (test)    │ Neur │ Post (test)   │          │")
    println("├────────┼──────┼────────────────┼──────┼───────────────┼──────────┤")

    ref_posts = Float64[]
    sel_posts = Float64[]
    sel_neurs = Int[]
    sel_tipos = Symbol[]
    sel_ents_list = Vector{Int}[]

    for seed in seeds
        red = crear_red(capas; seed=seed)
        config = ConfiguracionDivision{Float64}(umbral)

        mapa = ejecutar_division(red, datos_test, config;
            datos_entrenamiento=datos_train,
            epochs=epochs, lr=Float64(lr), paciencia=100)

        sel = seleccionar_mejor(mapa)
        ref = mapa.referencia_completa

        push!(ref_posts, ref.precision_post_entrenamiento)
        push!(sel_posts, sel.precision)
        push!(sel_neurs, sel.neuronas)
        push!(sel_tipos, sel.tipo)
        if sel.entrada.subconfiguracion !== nothing
            push!(sel_ents_list, sel.entrada.subconfiguracion.indices_entrada)
        else
            push!(sel_ents_list, Int[])
        end

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

    nombres_attr = ["a1", "a2", "a3", "a4", "a5", "a6"]
    conteo = zeros(Int, 6)
    n_sub = 0
    for ents in sel_ents_list
        length(ents) == 6 && continue
        isempty(ents) && continue
        for idx in ents; conteo[idx] += 1; end
        n_sub += 1
    end
    if n_sub > 0
        println("  Atributos en subredes ($n_sub):")
        orden = sortperm(conteo, rev=true)
        for idx in orden
            conteo[idx] == 0 && continue
            @printf("    %s: %d/%d (%.0f%%)\n", nombres_attr[idx], conteo[idx], n_sub, conteo[idx]/n_sub*100)
        end
    end
    println()
end

function exp16()
    println("╔══════════════════════════════════════════════════════════════╗")
    println("║  Experimento 16: MONKS Problems                            ║")
    println("║  Problemas históricamente difíciles para redes neuronales   ║")
    println("╚══════════════════════════════════════════════════════════════╝")

    seeds = SEEDS_20

    reglas = [
        "Monks-1: (a1==a2) OR (a5==1)",
        "Monks-2: exactamente 2 atributos tienen su primer valor",
        "Monks-3: (a5==3 AND a4==1) OR (a5!=4 AND a2!=3) + 5% ruido"
    ]

    for i in 1:3
        println("\n" * "═"^60)
        println("  $(reglas[i])")
        println("═"^60)

        datos_train, datos_test = cargar_monks(i)

        for n_ocultas in [8, 16, 32]
            ejecutar_monks("Monks-$i", datos_train, datos_test,
                [6, n_ocultas, 2], seeds; epochs=1000, lr=0.01)
        end
    end
end

exp16()
