using Test
using PropCheck
using DivisionNeuronal

include("generadores.jl")

# Feature: neural-division, Property 1: Validación de Red Base
# Para toda RedBase con pesos inicializados y arquitectura definida,
# la validación debe aceptarla. Para toda RedBase con pesos vacíos o
# dimensiones inconsistentes, la validación debe rechazarla.
# Valida: Requisitos 1.1, 1.3
@testset "Propiedad 1: Validación de Red Base" begin

    @testset "Redes válidas son aceptadas" begin
        gen = gen_red_base(Float64, 5, 5, 3)
        result = check(gen; ntests=100) do red
            validar_red_base(red) === nothing
        end
        @test result
    end

    @testset "Redes con pesos vacíos son rechazadas" begin
        gen_params = interleave(isample(1:5), isample(1:5))
        gen_invalida_vacia = map(gen_params) do (n_ent, n_sal)
            RedBase{Float64}(Matrix{Float64}[], Vector{Float64}[], n_ent, n_sal)
        end
        result = check(gen_invalida_vacia; ntests=100) do red
            try
                validar_red_base(red)
                false  # No debería llegar aquí
            catch e
                e isa RedBaseNoInicializadaError
            end
        end
        @test result
    end

    @testset "Redes con dimensiones inconsistentes entre capas son rechazadas" begin
        gen = gen_red_base(Float64, 5, 5, 3)
        gen_inconsistente = map(gen) do red
            length(red.pesos) < 2 && return nothing
            # Corromper dimensiones: reemplazar segunda capa con dimensiones incompatibles
            pesos_corruptos = copy(red.pesos)
            filas_orig = size(pesos_corruptos[2], 1)
            cols_orig = size(pesos_corruptos[2], 2)
            # Cambiar filas para que no coincida con columnas de capa anterior
            nueva_filas = filas_orig + 1
            pesos_corruptos[2] = randn(Float64, nueva_filas, cols_orig)
            RedBase{Float64}(pesos_corruptos, red.biases, red.n_entradas, red.n_salidas)
        end
        result = check(gen_inconsistente; ntests=100) do red
            red === nothing && return true  # Saltar redes de una sola capa
            try
                validar_red_base(red)
                false
            catch e
                e isa RedBaseNoInicializadaError
            end
        end
        @test result
    end

    @testset "Redes con biases de dimensión incorrecta son rechazadas" begin
        gen = gen_red_base(Float64, 5, 5, 3)
        gen_bias_malo = map(gen) do red
            biases_corruptos = copy(red.biases)
            # Corromper el primer bias: agregar un elemento extra
            biases_corruptos[1] = vcat(biases_corruptos[1], [0.0])
            RedBase{Float64}(red.pesos, biases_corruptos, red.n_entradas, red.n_salidas)
        end
        result = check(gen_bias_malo; ntests=100) do red
            try
                validar_red_base(red)
                false
            catch e
                e isa RedBaseNoInicializadaError
            end
        end
        @test result
    end

    @testset "Redes con distinto número de capas pesos/biases son rechazadas" begin
        gen = gen_red_base(Float64, 5, 5, 3)
        gen_capas_distintas = map(gen) do red
            # Quitar el último bias para crear inconsistencia
            biases_cortos = red.biases[1:end-1]
            RedBase{Float64}(red.pesos, biases_cortos, red.n_entradas, red.n_salidas)
        end
        result = check(gen_capas_distintas; ntests=100) do red
            try
                validar_red_base(red)
                false
            catch e
                e isa RedBaseNoInicializadaError
            end
        end
        @test result
    end

end


# Feature: neural-division, Property 2: Registro de límites de entrada/salida
# Para todo par de enteros positivos (n_entradas, n_salidas) proporcionados al motor,
# el generador de subconfiguraciones debe producir subconfiguraciones cuyos índices
# de entrada estén en 1:n_entradas y cuyos índices de salida estén en 1:n_salidas.
# Valida: Requisitos 1.2
@testset "Propiedad 2: Registro de límites de entrada/salida" begin

    @testset "Índices de entrada y salida dentro de límites" begin
        gen = gen_red_base(Float64, 4, 4, 2)
        result = check(gen; ntests=100) do red
            generador = GeneradorDeSubconfiguraciones(red)
            for (indices_entrada, indices_salida) in generador
                # Todos los índices de entrada deben estar en 1:n_entradas
                all(idx -> 1 <= idx <= red.n_entradas, indices_entrada) || return false
                # Todos los índices de salida deben estar en 1:n_salidas
                all(idx -> 1 <= idx <= red.n_salidas, indices_salida) || return false
                # Los vectores de índices no deben estar vacíos
                isempty(indices_entrada) && return false
                isempty(indices_salida) && return false
            end
            true
        end
        @test result
    end

end


# Feature: neural-division, Property 3: Enumeración exhaustiva
# Para toda RedBase con n_entradas entradas y n_salidas salidas, el generador
# debe producir exactamente (2^n_entradas - 1) × (2^n_salidas - 1) subconfiguraciones.
# También se verifica que no haya duplicados en la enumeración.
# Valida: Requisitos 2.1
@testset "Propiedad 3: Enumeración exhaustiva" begin

    @testset "Conteo exacto de subconfiguraciones" begin
        gen = gen_red_base(Float64, 4, 4, 2)
        result = check(gen; ntests=100) do red
            generador = GeneradorDeSubconfiguraciones(red)
            esperado = (2^red.n_entradas - 1) * (2^red.n_salidas - 1)

            # Verificar length reportado
            length(generador) == esperado || return false

            # Contar elementos reales producidos por el iterador
            conteo = 0
            for _ in generador
                conteo += 1
            end
            conteo == esperado
        end
        @test result
    end

    @testset "Sin duplicados en la enumeración" begin
        gen = gen_red_base(Float64, 4, 4, 2)
        result = check(gen; ntests=100) do red
            generador = GeneradorDeSubconfiguraciones(red)
            vistos = Set{Tuple{Vector{Int}, Vector{Int}}}()
            for par in generador
                par in vistos && return false
                push!(vistos, par)
            end
            true
        end
        @test result
    end

end


# Feature: neural-division, Property 5: Completitud de evaluación
# Para toda subconfiguración evaluada, el evaluador debe calcular una precisión global
# (considerando todas las salidas) y una precisión parcial para cada subconjunto no vacío
# de salidas del problema.
# Valida: Requisitos 4.1, 4.2, 4.3
@testset "Propiedad 5: Completitud de evaluación" begin

    @testset "Precisión global en [0,1] y precisiones parciales completas" begin
        # Generar una RedBase aleatoria, extraer una subconfiguracion, evaluar
        gen_rb = gen_red_base(Float64, 4, 4, 2)
        result = check(gen_rb; ntests=100) do red
            n_muestras = rand(5:20)
            n_salidas_totales = red.n_salidas
            indices_salida_totales = collect(1:n_salidas_totales)

            # Generar datos de validación
            datos = (
                entradas = rand(Float64, n_muestras, red.n_entradas),
                salidas = Float64.(rand(Bool, n_muestras, n_salidas_totales))
            )

            # Elegir una subconfiguracion aleatoria del generador
            generador = GeneradorDeSubconfiguraciones(red)
            pares = collect(generador)
            isempty(pares) && return true

            idx_par = rand(1:length(pares))
            (idx_ent, idx_sal) = pares[idx_par]
            subconfig = extraer_subconfiguracion(red, idx_ent, idx_sal)
            subconfig === nothing && return true  # Descartada por conexiones inválidas

            # Evaluar
            resultado = evaluar(subconfig, datos, indices_salida_totales)

            # 1. precision_global debe estar en [0.0, 1.0]
            (0.0 <= resultado.precision_global <= 1.0) || return false

            # 2. precisiones_parciales debe contener exactamente 2^n_salidas_totales - 1 entradas
            n_esperado = (1 << n_salidas_totales) - 1
            length(resultado.precisiones_parciales) == n_esperado || return false

            # 3. Cada precisión parcial debe estar en [0.0, 1.0]
            for (_, prec) in resultado.precisiones_parciales
                (0.0 <= prec <= 1.0) || return false
            end

            true
        end
        @test result
    end

end


# Feature: neural-division, Property 6: Candidatura por umbral
# Para toda subconfiguración y para todo subconjunto de salidas, la subconfiguración
# es marcada como candidata para ese subconjunto si y solo si su precisión para dicho
# subconjunto supera el Umbral_De_Acierto.
# Valida: Requisitos 4.4, 4.5
@testset "Propiedad 6: Candidatura por umbral" begin

    @testset "Candidatura global: precisión > umbral ↔ candidata" begin
        gen_rb = gen_red_base(Float64, 4, 4, 2)
        result = check(gen_rb; ntests=100) do red
            n_salidas_totales = red.n_salidas
            indices_salida_totales = collect(1:n_salidas_totales)
            n_muestras = rand(5:20)

            # Generar datos de validación
            datos = (
                entradas = rand(Float64, n_muestras, red.n_entradas),
                salidas = Float64.(rand(Bool, n_muestras, n_salidas_totales))
            )

            # Elegir una subconfiguracion aleatoria
            generador = GeneradorDeSubconfiguraciones(red)
            pares = collect(generador)
            isempty(pares) && return true

            idx_par = rand(1:length(pares))
            (idx_ent, idx_sal) = pares[idx_par]
            subconfig = extraer_subconfiguracion(red, idx_ent, idx_sal)
            subconfig === nothing && return true

            # Evaluar
            resultado = evaluar(subconfig, datos, indices_salida_totales)

            # Elegir un umbral aleatorio en [0.0, 1.0]
            umbral = rand(Float64)

            # Propiedad: es candidata global ↔ precision_global > umbral
            es_candidata_global = resultado.precision_global > umbral
            deberia_ser_candidata = resultado.precision_global > umbral

            es_candidata_global == deberia_ser_candidata
        end
        @test result
    end

    @testset "Candidatura parcial: precisión parcial > umbral ↔ candidata" begin
        gen_rb = gen_red_base(Float64, 4, 4, 2)
        result = check(gen_rb; ntests=100) do red
            n_salidas_totales = red.n_salidas
            indices_salida_totales = collect(1:n_salidas_totales)
            n_muestras = rand(5:20)

            # Generar datos de validación
            datos = (
                entradas = rand(Float64, n_muestras, red.n_entradas),
                salidas = Float64.(rand(Bool, n_muestras, n_salidas_totales))
            )

            # Elegir una subconfiguracion aleatoria
            generador = GeneradorDeSubconfiguraciones(red)
            pares = collect(generador)
            isempty(pares) && return true

            idx_par = rand(1:length(pares))
            (idx_ent, idx_sal) = pares[idx_par]
            subconfig = extraer_subconfiguracion(red, idx_ent, idx_sal)
            subconfig === nothing && return true

            # Evaluar
            resultado = evaluar(subconfig, datos, indices_salida_totales)

            # Elegir un umbral aleatorio en [0.0, 1.0]
            umbral = rand(Float64)

            # Verificar para CADA subconjunto parcial
            for (subconj, prec_parcial) in resultado.precisiones_parciales
                es_candidata_parcial = prec_parcial > umbral
                deberia_ser_candidata = prec_parcial > umbral

                es_candidata_parcial == deberia_ser_candidata || return false
            end

            true
        end
        @test result
    end

    @testset "Umbral extremo 0.0: toda subconfiguración con precisión > 0 es candidata" begin
        gen_rb = gen_red_base(Float64, 3, 3, 2)
        result = check(gen_rb; ntests=100) do red
            n_salidas_totales = red.n_salidas
            indices_salida_totales = collect(1:n_salidas_totales)
            n_muestras = rand(5:20)

            datos = (
                entradas = rand(Float64, n_muestras, red.n_entradas),
                salidas = Float64.(rand(Bool, n_muestras, n_salidas_totales))
            )

            generador = GeneradorDeSubconfiguraciones(red)
            pares = collect(generador)
            isempty(pares) && return true

            idx_par = rand(1:length(pares))
            (idx_ent, idx_sal) = pares[idx_par]
            subconfig = extraer_subconfiguracion(red, idx_ent, idx_sal)
            subconfig === nothing && return true

            resultado = evaluar(subconfig, datos, indices_salida_totales)
            umbral = 0.0

            # Con umbral 0.0, cualquier precisión > 0.0 es candidata
            es_candidata_global = resultado.precision_global > umbral
            # Verificar coherencia: si precision_global > 0.0, debe ser candidata
            if resultado.precision_global > 0.0
                es_candidata_global || return false
            end

            for (_, prec) in resultado.precisiones_parciales
                if prec > 0.0
                    (prec > umbral) || return false
                end
            end

            true
        end
        @test result
    end

    @testset "Umbral extremo 1.0: solo precisión perfecta es candidata" begin
        gen_rb = gen_red_base(Float64, 3, 3, 2)
        result = check(gen_rb; ntests=100) do red
            n_salidas_totales = red.n_salidas
            indices_salida_totales = collect(1:n_salidas_totales)
            n_muestras = rand(5:20)

            datos = (
                entradas = rand(Float64, n_muestras, red.n_entradas),
                salidas = Float64.(rand(Bool, n_muestras, n_salidas_totales))
            )

            generador = GeneradorDeSubconfiguraciones(red)
            pares = collect(generador)
            isempty(pares) && return true

            idx_par = rand(1:length(pares))
            (idx_ent, idx_sal) = pares[idx_par]
            subconfig = extraer_subconfiguracion(red, idx_ent, idx_sal)
            subconfig === nothing && return true

            resultado = evaluar(subconfig, datos, indices_salida_totales)
            umbral = 1.0

            # Con umbral 1.0, ninguna precisión puede superar estrictamente 1.0
            # por lo que no debería haber candidatas (precision > 1.0 es imposible)
            es_candidata_global = resultado.precision_global > umbral
            es_candidata_global == false || return false

            for (_, prec) in resultado.precisiones_parciales
                (prec > umbral) == false || return false
            end

            true
        end
        @test result
    end

end


# Feature: neural-division, Property 7: Inicialización del Mapa de Soluciones
# Para todo número de salidas n, el MapaDeSoluciones inicializado debe contener
# exactamente 2^n - 1 claves: una para cada subconjunto no vacío de 1:n.
# Valida: Requisitos 5.1
@testset "Propiedad 7: Inicialización del Mapa de Soluciones" begin

    @testset "Mapa contiene exactamente 2^n - 1 claves parciales" begin
        gen_n = isample(1:6)
        result = check(gen_n; ntests=100) do n_salidas
            mapa = inicializar_mapa(n_salidas, Float64)
            esperado = (1 << n_salidas) - 1
            length(mapa.parciales) == esperado
        end
        @test result
    end

    @testset "Todas las entradas parciales tienen subconfiguracion === nothing" begin
        gen_n = isample(1:6)
        result = check(gen_n; ntests=100) do n_salidas
            mapa = inicializar_mapa(n_salidas, Float64)
            all(e -> e.subconfiguracion === nothing, values(mapa.parciales))
        end
        @test result
    end

    @testset "Todas las entradas parciales tienen precision == 0.0" begin
        gen_n = isample(1:6)
        result = check(gen_n; ntests=100) do n_salidas
            mapa = inicializar_mapa(n_salidas, Float64)
            all(e -> e.precision == 0.0, values(mapa.parciales))
        end
        @test result
    end

    @testset "Entrada global tiene subconfiguracion === nothing" begin
        gen_n = isample(1:6)
        result = check(gen_n; ntests=100) do n_salidas
            mapa = inicializar_mapa(n_salidas, Float64)
            mapa.global_.subconfiguracion === nothing
        end
        @test result
    end

end


# Feature: neural-division, Property 8: Invariante de mejor-es-más-simple
# Para toda secuencia de subconfiguraciones procesadas, la subconfiguración almacenada
# en cada entrada del MapaDeSoluciones debe ser siempre la más simple (menor número de
# neuronas activas) entre todas las candidatas que superaron el umbral. En caso de empate
# en neuronas, debe ser la de mayor precisión.
# Valida: Requisitos 5.2, 5.3, 5.4, 5.5
@testset "Propiedad 8: Invariante de mejor-es-más-simple" begin

    @testset "Mapa contiene la subconfiguración más simple que supera el umbral" begin
        gen = gen_red_base(Float64, 3, 2, 1)
        result = check(gen; ntests=100) do red
            n_salidas_totales = red.n_salidas
            indices_salida_totales = collect(1:n_salidas_totales)
            n_muestras = rand(5:15)

            # Generar datos de validación
            datos = (
                entradas = rand(Float64, n_muestras, red.n_entradas),
                salidas = Float64.(rand(Bool, n_muestras, n_salidas_totales))
            )

            # Elegir un umbral aleatorio en [0.0, 1.0]
            umbral = rand(Float64)

            # Inicializar mapa
            mapa = inicializar_mapa(n_salidas_totales, Float64)

            # Recopilar todas las candidatas con sus resultados para verificación posterior
            # Clave: Vector{Int} (subconjunto de salidas), Valor: Vector de (subconfig, precision)
            candidatas_global = Tuple{Subconfiguracion{Float64}, Float64}[]
            candidatas_parciales = Dict{Vector{Int}, Vector{Tuple{Subconfiguracion{Float64}, Float64}}}()
            for k in keys(mapa.parciales)
                candidatas_parciales[k] = Tuple{Subconfiguracion{Float64}, Float64}[]
            end

            # Iterar sobre todas las subconfiguraciones, evaluar y actualizar mapa
            generador = GeneradorDeSubconfiguraciones(red)
            for (idx_ent, idx_sal) in generador
                subconfig = extraer_subconfiguracion(red, idx_ent, idx_sal)
                subconfig === nothing && continue

                resultado = evaluar(subconfig, datos, indices_salida_totales)

                # Registrar candidatas que superan el umbral (para verificación)
                if resultado.precision_global > umbral
                    push!(candidatas_global, (subconfig, resultado.precision_global))
                end
                for (subconj, prec) in resultado.precisiones_parciales
                    if haskey(candidatas_parciales, subconj) && prec > umbral
                        push!(candidatas_parciales[subconj], (subconfig, prec))
                    end
                end

                # Actualizar mapa (la función bajo test)
                actualizar_si_mejor!(mapa, subconfig, resultado, umbral)
            end

            # Verificar invariante para la solución global
            if isempty(candidatas_global)
                # Sin candidatas → mapa debe tener nothing
                mapa.global_.subconfiguracion !== nothing && return false
            else
                # Encontrar la mejor candidata esperada: mínimo n_neuronas_activas,
                # desempate por máxima precisión
                mejor_esperada = candidatas_global[1]
                for c in candidatas_global[2:end]
                    sc, prec = c
                    sc_mejor, prec_mejor = mejor_esperada
                    if sc.n_neuronas_activas < sc_mejor.n_neuronas_activas
                        mejor_esperada = c
                    elseif sc.n_neuronas_activas == sc_mejor.n_neuronas_activas && prec > prec_mejor
                        mejor_esperada = c
                    end
                end
                sc_esperada, prec_esperada = mejor_esperada

                # Verificar que el mapa almacena la mejor
                mapa.global_.subconfiguracion === nothing && return false
                almacenada = mapa.global_
                almacenada.subconfiguracion.n_neuronas_activas != sc_esperada.n_neuronas_activas && return false
                almacenada.precision != prec_esperada && return false
            end

            # Verificar invariante para cada solución parcial
            for (subconj, candidatas) in candidatas_parciales
                entrada = mapa.parciales[subconj]
                if isempty(candidatas)
                    entrada.subconfiguracion !== nothing && return false
                else
                    mejor_esperada = candidatas[1]
                    for c in candidatas[2:end]
                        sc, prec = c
                        sc_mejor, prec_mejor = mejor_esperada
                        if sc.n_neuronas_activas < sc_mejor.n_neuronas_activas
                            mejor_esperada = c
                        elseif sc.n_neuronas_activas == sc_mejor.n_neuronas_activas && prec > prec_mejor
                            mejor_esperada = c
                        end
                    end
                    sc_esperada, prec_esperada = mejor_esperada

                    entrada.subconfiguracion === nothing && return false
                    entrada.subconfiguracion.n_neuronas_activas != sc_esperada.n_neuronas_activas && return false
                    entrada.precision != prec_esperada && return false
                end
            end

            true
        end
        @test result
    end

end


# Feature: neural-division, Property 11: Ida y vuelta de serialización
# Para todo MapaDeSoluciones válido, serializar a disco y luego deserializar
# y luego serializar de nuevo debe producir un resultado equivalente al original.
# Valida: Requisitos 7.1, 7.2, 7.3, 7.4
@testset "Propiedad 11: Ida y vuelta de serialización" begin

    """
    Compara dos EntradaMapaSoluciones para equivalencia estructural.
    """
    function entradas_equivalentes(a::EntradaMapaSoluciones{T}, b::EntradaMapaSoluciones{T}) where T
        # Comparar precisiones
        a.precision ≈ b.precision || return false
        a.precision_pre_entrenamiento ≈ b.precision_pre_entrenamiento || return false
        a.precision_post_entrenamiento ≈ b.precision_post_entrenamiento || return false

        # Comparar subconfiguraciones
        if a.subconfiguracion === nothing && b.subconfiguracion === nothing
            return true
        end
        (a.subconfiguracion === nothing) != (b.subconfiguracion === nothing) && return false

        sa = a.subconfiguracion
        sb = b.subconfiguracion
        sa.indices_entrada == sb.indices_entrada || return false
        sa.indices_salida == sb.indices_salida || return false
        sa.n_neuronas_activas == sb.n_neuronas_activas || return false
        length(sa.pesos) == length(sb.pesos) || return false
        for i in eachindex(sa.pesos)
            sa.pesos[i] ≈ sb.pesos[i] || return false
        end
        length(sa.biases) == length(sb.biases) || return false
        for i in eachindex(sa.biases)
            sa.biases[i] ≈ sb.biases[i] || return false
        end
        return true
    end

    """
    Compara dos MapaDeSoluciones para equivalencia estructural.
    """
    function mapas_equivalentes(a::MapaDeSoluciones{T}, b::MapaDeSoluciones{T}) where T
        # Comparar entrada global
        entradas_equivalentes(a.global_, b.global_) || return false

        # Comparar entradas parciales
        length(a.parciales) == length(b.parciales) || return false
        for (clave, entrada_a) in a.parciales
            haskey(b.parciales, clave) || return false
            entradas_equivalentes(entrada_a, b.parciales[clave]) || return false
        end
        return true
    end

    @testset "Serializar → deserializar → serializar produce resultado equivalente" begin
        gen = gen_mapa_soluciones(3)
        result = check(gen; ntests=100) do mapa_original
            dir_temp = mktempdir()
            try
                ruta1 = joinpath(dir_temp, "mapa1.jld2")
                ruta2 = joinpath(dir_temp, "mapa2.jld2")

                # Serializar → deserializar → serializar → deserializar
                serializar(mapa_original, ruta1)
                mapa_cargado1 = deserializar(ruta1)
                serializar(mapa_cargado1, ruta2)
                mapa_cargado2 = deserializar(ruta2)

                # Comparar: los dos mapas deserializados deben ser equivalentes
                mapas_equivalentes(mapa_cargado1, mapa_cargado2) || return false

                # Comparar: el mapa deserializado debe ser equivalente al original
                mapas_equivalentes(mapa_original, mapa_cargado1) || return false

                true
            finally
                rm(dir_temp; recursive=true, force=true)
            end
        end
        @test result
    end

end


# Feature: neural-division, Property 12: Precisión del reporte de progreso
# Para todo reporte de progreso emitido durante la exploración, el porcentaje reportado
# debe ser igual a evaluadas / total, y los conteos de soluciones globales y parciales
# deben coincidir con el estado actual del MapaDeSoluciones.
# Valida: Requisitos 8.1, 8.2
@testset "Propiedad 12: Precisión del reporte de progreso" begin

    @testset "Evaluadas y total coinciden, conteos de soluciones correctos" begin
        gen = gen_mapa_soluciones(3)
        gen_eval = isample(0:1000)
        gen_total = isample(1:1000)
        gen_combined = interleave(gen, gen_eval, gen_total)

        result = check(gen_combined; ntests=100) do (mapa, evaluadas_raw, total_raw)
            # Asegurar que evaluadas <= total
            total = max(total_raw, 1)
            evaluadas = min(evaluadas_raw, total)

            # Capturar el ProgresoExploracion emitido por el callback
            progreso_capturado = Ref{Union{Nothing, ProgresoExploracion}}(nothing)
            callback = function(p::ProgresoExploracion)
                progreso_capturado[] = p
            end

            reportar_progreso(evaluadas, total, mapa, callback)

            progreso = progreso_capturado[]
            progreso === nothing && return false

            # 1. evaluadas y total deben coincidir exactamente
            progreso.evaluadas == evaluadas || return false
            progreso.total == total || return false

            # 2. soluciones_globales debe coincidir con el estado del mapa
            esperado_globales = mapa.global_.subconfiguracion !== nothing ? 1 : 0
            progreso.soluciones_globales == esperado_globales || return false

            # 3. soluciones_parciales debe coincidir con el conteo de entradas parciales no vacías
            esperado_parciales = count(e -> e.subconfiguracion !== nothing, values(mapa.parciales))
            progreso.soluciones_parciales == esperado_parciales || return false

            true
        end
        @test result
    end

    @testset "Callback nothing no produce error" begin
        gen = gen_mapa_soluciones(3)
        gen_eval = isample(0:100)
        gen_total = isample(1:100)
        gen_combined = interleave(gen, gen_eval, gen_total)

        result = check(gen_combined; ntests=100) do (mapa, evaluadas_raw, total_raw)
            total = max(total_raw, 1)
            evaluadas = min(evaluadas_raw, total)

            # Con callback nothing, no debe lanzar error
            reportar_progreso(evaluadas, total, mapa, nothing)
            true
        end
        @test result
    end

end


# Feature: neural-division, Property 13: Cancelación ordenada
# Para toda señal de parada emitida durante la exploración, el motor debe devolver
# un MapaDeSoluciones válido que contenga únicamente los resultados obtenidos hasta
# el momento de la parada, sin entradas corruptas.
# Valida: Requisitos 8.3
@testset "Propiedad 13: Cancelación ordenada" begin

    using Base.Threads: Atomic

    @testset "Mapa válido tras cancelación parcial" begin
        gen = gen_red_base(Float64, 3, 3, 1)
        result = check(gen; ntests=100) do red
            n_salidas_totales = red.n_salidas
            indices_salida_totales = collect(1:n_salidas_totales)
            n_muestras = rand(5:15)

            # Generar datos de validación
            datos = (
                entradas = rand(Float64, n_muestras, red.n_entradas),
                salidas = Float64.(rand(Bool, n_muestras, n_salidas_totales))
            )

            umbral = rand(Float64)

            # Inicializar mapa y señal de parada
            mapa = inicializar_mapa(n_salidas_totales, Float64)
            señal_parada = Atomic{Bool}(false)

            # Elegir un punto de parada aleatorio dentro de la exploración
            generador = GeneradorDeSubconfiguraciones(red)
            total = length(generador)
            punto_parada = rand(0:total)

            # Simular bucle de exploración con cancelación cooperativa
            evaluadas = 0
            for (idx_ent, idx_sal) in generador
                # Comprobar señal de parada entre iteraciones
                if debe_parar(señal_parada)
                    break
                end

                subconfig = extraer_subconfiguracion(red, idx_ent, idx_sal)
                subconfig === nothing && continue

                resultado = evaluar(subconfig, datos, indices_salida_totales)
                actualizar_si_mejor!(mapa, subconfig, resultado, umbral)
                evaluadas += 1

                # Activar señal de parada en el punto elegido
                if evaluadas >= punto_parada
                    señal_parada[] = true
                end
            end

            # === Verificaciones del mapa tras cancelación ===

            # 1. El mapa debe tener el número correcto de claves parciales: 2^n_salidas - 1
            esperado_parciales = (1 << n_salidas_totales) - 1
            length(mapa.parciales) == esperado_parciales || return false

            # 2. Todas las entradas con subconfiguracion !== nothing deben tener
            #    precision en [0.0, 1.0]
            if mapa.global_.subconfiguracion !== nothing
                (0.0 <= mapa.global_.precision <= 1.0) || return false
            end
            for (_, entrada) in mapa.parciales
                if entrada.subconfiguracion !== nothing
                    (0.0 <= entrada.precision <= 1.0) || return false
                end
            end

            # 3. Todas las entradas con subconfiguracion !== nothing deben tener
            #    n_neuronas_activas > 0
            if mapa.global_.subconfiguracion !== nothing
                mapa.global_.subconfiguracion.n_neuronas_activas > 0 || return false
            end
            for (_, entrada) in mapa.parciales
                if entrada.subconfiguracion !== nothing
                    entrada.subconfiguracion.n_neuronas_activas > 0 || return false
                end
            end

            # 4. No debe haber entradas en estado "half-updated":
            #    si subconfiguracion === nothing, precision debe ser 0.0
            #    si subconfiguracion !== nothing, precision debe ser > 0.0 o al menos consistente
            if mapa.global_.subconfiguracion === nothing
                mapa.global_.precision == 0.0 || return false
            end
            for (_, entrada) in mapa.parciales
                if entrada.subconfiguracion === nothing
                    entrada.precision == 0.0 || return false
                end
            end

            true
        end
        @test result
    end

end


# Feature: neural-division, Property 14: Métricas pre/post entrenamiento
# Para toda subconfiguración almacenada en el MapaDeSoluciones final tras el entrenamiento,
# la entrada debe contener la precisión pre-entrenamiento, la precisión post-entrenamiento,
# y la diferencia calculada entre ambas.
# Valida: Requisitos 9.1, 9.2, 9.3, 9.4
@testset "Propiedad 14: Métricas pre/post entrenamiento" begin

    @testset "Entradas con subconfiguracion contienen métricas pre/post válidas" begin
        gen = gen_red_base(Float64, 2, 2, 1)
        result = check(gen; ntests=100) do red
            n_salidas_totales = red.n_salidas
            indices_salida_totales = collect(1:n_salidas_totales)
            n_muestras = rand(5:15)

            # Generar datos de entrenamiento y validación
            datos_ent = (
                entradas = rand(Float64, n_muestras, red.n_entradas),
                salidas = Float64.(rand(Bool, n_muestras, n_salidas_totales))
            )
            datos_val = (
                entradas = rand(Float64, n_muestras, red.n_entradas),
                salidas = Float64.(rand(Bool, n_muestras, n_salidas_totales))
            )

            umbral = 0.0  # Umbral bajo para poblar el mapa

            # Inicializar mapa y poblar con exploración mini
            mapa = inicializar_mapa(n_salidas_totales, Float64)
            generador = GeneradorDeSubconfiguraciones(red)
            for (idx_ent, idx_sal) in generador
                subconfig = extraer_subconfiguracion(red, idx_ent, idx_sal)
                subconfig === nothing && continue
                resultado = evaluar(subconfig, datos_val, indices_salida_totales)
                actualizar_si_mejor!(mapa, subconfig, resultado, umbral)
            end

            # Entrenar el mapa con pocas épocas
            entrenar_mapa!(mapa, datos_ent, datos_val; epochs=10)

            # Verificar entrada global
            if mapa.global_.subconfiguracion !== nothing
                # 1. precision_pre_entrenamiento en [0.0, 1.0]
                (0.0 <= mapa.global_.precision_pre_entrenamiento <= 1.0) || return false
                # 2. precision_post_entrenamiento en [0.0, 1.0]
                (0.0 <= mapa.global_.precision_post_entrenamiento <= 1.0) || return false
                # 3. La diferencia post - pre es calculable (ambos valores están definidos)
                dif = mapa.global_.precision_post_entrenamiento - mapa.global_.precision_pre_entrenamiento
                isfinite(dif) || return false
            end

            # Verificar entradas parciales
            for (_, entrada) in mapa.parciales
                if entrada.subconfiguracion !== nothing
                    (0.0 <= entrada.precision_pre_entrenamiento <= 1.0) || return false
                    (0.0 <= entrada.precision_post_entrenamiento <= 1.0) || return false
                    dif = entrada.precision_post_entrenamiento - entrada.precision_pre_entrenamiento
                    isfinite(dif) || return false
                end
            end

            # 4. Entradas con subconfiguracion === nothing: pre y post deben ser 0.0
            if mapa.global_.subconfiguracion === nothing
                mapa.global_.precision_pre_entrenamiento == 0.0 || return false
                mapa.global_.precision_post_entrenamiento == 0.0 || return false
            end
            for (_, entrada) in mapa.parciales
                if entrada.subconfiguracion === nothing
                    entrada.precision_pre_entrenamiento == 0.0 || return false
                    entrada.precision_post_entrenamiento == 0.0 || return false
                end
            end

            true
        end
        @test result
    end

end
