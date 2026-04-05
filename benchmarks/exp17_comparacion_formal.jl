"""
Experimento 17: Comparación formal y tests estadísticos.

Compara la División Neuronal contra:
  1. Red completa (baseline)
  2. Feature selection por correlación (top-K features más correlacionadas con la salida)
  3. Feature selection aleatoria (K features al azar)

Incluye test de Wilcoxon signed-rank para significancia estadística.

Ejecución:
    julia --project=. benchmarks/exp17_comparacion_formal.jl
"""

include("utils.jl")

# ============================================================================
# Carga de datasets
# ============================================================================

function cargar_wine_datos()
    ruta = joinpath(@__DIR__, "..", "csv", "wine.csv")
    lines = readlines(ruta)
    n = length(lines)
    entradas = zeros(Float64, n, 13)
    salidas = zeros(Float64, n, 3)
    for (i, line) in enumerate(lines)
        vals = split(line, ',')
        clase = parse(Int, vals[1])
        salidas[i, clase] = 1.0
        for j in 1:13; entradas[i, j] = parse(Float64, vals[j+1]); end
    end
    normalizar!(entradas)
    return entradas, salidas
end

function cargar_glass_datos()
    ruta = joinpath(@__DIR__, "..", "csv", "glass.csv")
    lines = filter(!isempty, readlines(ruta))
    n = length(lines)
    entradas = zeros(Float64, n, 9)
    clases_raw = zeros(Int, n)
    for (i, line) in enumerate(lines)
        vals = split(line, ',')
        for j in 1:9; entradas[i, j] = parse(Float64, vals[j+1]); end
        clases_raw[i] = parse(Int, vals[11])
    end
    normalizar!(entradas)
    mapa_c = Dict(1=>1, 2=>2, 3=>3, 5=>4, 6=>5, 7=>6)
    salidas = zeros(Float64, n, 6)
    for i in 1:n; salidas[i, mapa_c[clases_raw[i]]] = 1.0; end
    return entradas, salidas
end

function cargar_seeds_datos()
    ruta = joinpath(@__DIR__, "..", "csv", "seeds.csv")
    lines = filter(!isempty, readlines(ruta))
    n = length(lines)
    entradas = zeros(Float64, n, 7)
    salidas = zeros(Float64, n, 3)
    for (i, line) in enumerate(lines)
        vals = split(line)
        for j in 1:7; entradas[i, j] = parse(Float64, vals[j]); end
        salidas[i, parse(Int, vals[8])] = 1.0
    end
    normalizar!(entradas)
    return entradas, salidas
end

function cargar_ecoli_datos()
    ruta = joinpath(@__DIR__, "..", "csv", "ecoli.csv")
    lines = filter(!isempty, readlines(ruta))
    clases_validas = Dict("cp"=>1, "im"=>2, "imU"=>3, "pp"=>4, "om"=>5)
    datos = []
    for line in lines
        vals = split(line)
        haskey(clases_validas, vals[end]) && push!(datos, (vals, clases_validas[vals[end]]))
    end
    n = length(datos)
    entradas = zeros(Float64, n, 7)
    salidas = zeros(Float64, n, 5)
    for (i, (vals, c)) in enumerate(datos)
        for j in 1:7; entradas[i, j] = parse(Float64, vals[j+1]); end
        salidas[i, c] = 1.0
    end
    normalizar!(entradas)
    return entradas, salidas
end

function normalizar!(entradas)
    for j in 1:size(entradas, 2)
        col = entradas[:, j]; mn, mx = minimum(col), maximum(col)
        rango = mx - mn
        rango > 0 && (entradas[:, j] .= (col .- mn) ./ rango)
    end
end

function split_strat(entradas, salidas; seed=42)
    Random.seed!(seed)
    n = size(entradas, 1); nc = size(salidas, 2)
    idx_tr = Int[]; idx_te = Int[]
    for c in 1:nc
        ic = findall(salidas[:, c] .== 1.0)
        isempty(ic) && continue
        p = randperm(length(ic))
        nt = max(1, round(Int, length(ic) * 0.8))
        append!(idx_tr, ic[p[1:nt]])
        nt < length(ic) && append!(idx_te, ic[p[nt+1:end]])
    end
    shuffle!(idx_tr); shuffle!(idx_te)
    return (entradas=entradas[idx_tr, :], salidas=salidas[idx_tr, :]),
           (entradas=entradas[idx_te, :], salidas=salidas[idx_te, :])
end

# ============================================================================
# Feature selection por correlación
# ============================================================================

function top_k_features_correlacion(entradas_train, salidas_train, k)
    n_feat = size(entradas_train, 2)
    # Correlación media absoluta de cada feature con todas las salidas
    scores = zeros(n_feat)
    for j in 1:n_feat
        for c in 1:size(salidas_train, 2)
            x = entradas_train[:, j]
            y = salidas_train[:, c]
            mx, my = mean(x), mean(y)
            cov_xy = mean((x .- mx) .* (y .- my))
            sx = std(x); sy = std(y)
            if sx > 0 && sy > 0
                scores[j] += abs(cov_xy / (sx * sy))
            end
        end
    end
    return sortperm(scores, rev=true)[1:min(k, n_feat)]
end

# ============================================================================
# Entrenar red con subconjunto de features
# ============================================================================

function entrenar_con_features(entradas_train, salidas_train, entradas_test, salidas_test,
                                indices_feat, n_ocultas; seed=42, epochs=1000, lr=0.01)
    n_feat = length(indices_feat)
    n_sal = size(salidas_train, 2)
    capas = [n_feat, n_ocultas, n_sal]
    red = crear_red(capas; seed=seed)
    subconfig = extraer_subconfiguracion(red, collect(1:n_feat), collect(1:n_sal))

    datos_tr = (entradas=entradas_train[:, indices_feat], salidas=salidas_train)
    datos_te = (entradas=entradas_test[:, indices_feat], salidas=salidas_test)

    (pre, post) = entrenar_y_evaluar!(subconfig, datos_tr, datos_te;
        epochs=epochs, lr=Float64(lr), paciencia=100)
    return post
end

# ============================================================================
# Test de Wilcoxon signed-rank (implementación simple)
# ============================================================================

function wilcoxon_signed_rank(x, y)
    diffs = x .- y
    diffs = filter(d -> d != 0.0, diffs)
    n = length(diffs)
    n == 0 && return (W=0.0, p=1.0, n=0)

    ranks = sortperm(abs.(diffs))
    rank_vals = zeros(n)
    for (r, i) in enumerate(ranks)
        rank_vals[i] = Float64(r)
    end

    W_plus = sum(rank_vals[i] for i in 1:n if diffs[i] > 0; init=0.0)
    W_minus = sum(rank_vals[i] for i in 1:n if diffs[i] < 0; init=0.0)
    W = min(W_plus, W_minus)

    # Aproximación normal para n >= 10
    if n >= 10
        mu_W = n * (n + 1) / 4
        sigma_W = sqrt(n * (n + 1) * (2n + 1) / 24)
        z = (W - mu_W) / sigma_W
        # Aproximación p-valor (two-tailed) usando función error
        p = 2 * 0.5 * erfc(abs(z) / sqrt(2))
    else
        p = NaN  # No calculable con pocos datos
    end

    return (W=W, p=p, n=n, W_plus=W_plus, W_minus=W_minus)
end

# Función error complementaria simple
function erfc(x)
    # Aproximación de Abramowitz and Stegun
    t = 1.0 / (1.0 + 0.3275911 * abs(x))
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    result = poly * exp(-x^2)
    return x >= 0 ? result : 2.0 - result
end

# ============================================================================
# Experimento principal
# ============================================================================

function exp17()
    println("╔══════════════════════════════════════════════════════════════╗")
    println("║  Experimento 17: Comparación formal y tests estadísticos   ║")
    println("╚══════════════════════════════════════════════════════════════╝")

    datasets = [
        ("Wine",  cargar_wine_datos,  13, 3, 16),
        ("Glass", cargar_glass_datos,  9, 6, 16),
        ("Seeds", cargar_seeds_datos,  7, 3, 16),
        ("Ecoli", cargar_ecoli_datos,  7, 5, 16),
    ]

    seeds = SEEDS_20

    for (nombre, cargar_fn, n_feat, n_sal, n_ocultas) in datasets
        entradas, salidas = cargar_fn()
        capas = [n_feat, n_ocultas, n_sal]

        println("\n" * "═"^60)
        @printf("  %s (%d feat, %d clases, %d muestras)\n", nombre, n_feat, n_sal, size(entradas, 1))
        println("═"^60)

        prec_ref = Float64[]       # Red completa
        prec_div = Float64[]       # División neuronal (mejor seleccionada)
        prec_corr = Float64[]      # Feature selection por correlación
        prec_rand = Float64[]      # Feature selection aleatoria
        neur_div = Int[]

        for seed in seeds
            datos_train, datos_test = split_strat(entradas, salidas; seed=seed)

            # 1. Red completa
            p_ref = entrenar_con_features(datos_train.entradas, datos_train.salidas,
                datos_test.entradas, datos_test.salidas, collect(1:n_feat), n_ocultas;
                seed=seed)
            push!(prec_ref, p_ref)

            # 2. División neuronal
            red = crear_red(capas; seed=seed)
            config = ConfiguracionDivision{Float64}(0.4)
            mapa = ejecutar_division(red, datos_test, config;
                datos_entrenamiento=datos_train,
                epochs=1000, lr=Float64(0.01), paciencia=100)
            sel = seleccionar_mejor(mapa)
            push!(prec_div, sel.precision)
            push!(neur_div, sel.neuronas)

            # Determinar K features usadas por la división
            k_div = sel.entrada.subconfiguracion !== nothing ? length(sel.entrada.subconfiguracion.indices_entrada) : n_feat

            # 3. Feature selection por correlación (mismo K que la división)
            k = max(1, min(k_div, n_feat))
            top_feats = top_k_features_correlacion(datos_train.entradas, datos_train.salidas, k)
            p_corr = entrenar_con_features(datos_train.entradas, datos_train.salidas,
                datos_test.entradas, datos_test.salidas, top_feats, n_ocultas; seed=seed)
            push!(prec_corr, p_corr)

            # 4. Feature selection aleatoria (mismo K)
            Random.seed!(seed + 1000)
            rand_feats = randperm(n_feat)[1:k]
            p_rand = entrenar_con_features(datos_train.entradas, datos_train.salidas,
                datos_test.entradas, datos_test.salidas, rand_feats, n_ocultas; seed=seed)
            push!(prec_rand, p_rand)
        end

        # Tabla de resultados
        println("\n┌──────────────────────┬──────────┬──────────┬──────────┐")
        println("│ Método               │ Media    │ Std      │ Neuronas │")
        println("├──────────────────────┼──────────┼──────────┼──────────┤")
        @printf("│ Red completa         │  %5.1f%%  │  %5.1f%%  │ %8d │\n",
            mean(prec_ref)*100, std(prec_ref)*100, sum(capas))
        @printf("│ División Neuronal    │  %5.1f%%  │  %5.1f%%  │ %8.0f │\n",
            mean(prec_div)*100, std(prec_div)*100, mean(neur_div))
        @printf("│ FS Correlación       │  %5.1f%%  │  %5.1f%%  │      -   │\n",
            mean(prec_corr)*100, std(prec_corr)*100)
        @printf("│ FS Aleatoria         │  %5.1f%%  │  %5.1f%%  │      -   │\n",
            mean(prec_rand)*100, std(prec_rand)*100)
        println("└──────────────────────┴──────────┴──────────┴──────────┘")

        # Tests de Wilcoxon
        println("\n  Tests de Wilcoxon signed-rank (p-valor, two-tailed):")

        test_div_ref = wilcoxon_signed_rank(prec_div, prec_ref)
        test_div_corr = wilcoxon_signed_rank(prec_div, prec_corr)
        test_div_rand = wilcoxon_signed_rank(prec_div, prec_rand)
        test_ref_corr = wilcoxon_signed_rank(prec_ref, prec_corr)

        sig(p) = p < 0.01 ? "***" : p < 0.05 ? "**" : p < 0.1 ? "*" : "ns"

        @printf("    División vs Red completa:    p=%.4f %s\n", test_div_ref.p, sig(test_div_ref.p))
        @printf("    División vs FS Correlación:  p=%.4f %s\n", test_div_corr.p, sig(test_div_corr.p))
        @printf("    División vs FS Aleatoria:    p=%.4f %s\n", test_div_rand.p, sig(test_div_rand.p))
        @printf("    Red completa vs FS Correl.:  p=%.4f %s\n", test_ref_corr.p, sig(test_ref_corr.p))
        println("    (*** p<0.01, ** p<0.05, * p<0.1, ns no significativo)")
    end
end

exp17()
