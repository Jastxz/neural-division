using Test
using DivisionNeuronal
using Base.Threads: Atomic

@testset "Integración - Flujo completo" begin

    @testset "Flujo completo con red pequeña (2 entradas, 1 salida)" begin
        # Red: 2 entradas → 3 ocultas → 1 salida
        W1 = randn(Float64, 2, 3)
        W2 = randn(Float64, 3, 1)
        b1 = randn(Float64, 3)
        b2 = randn(Float64, 1)
        red = RedBase{Float64}([W1, W2], [b1, b2], 2, 1)

        datos = (
            entradas = Float64[0 0; 0 1; 1 0; 1 1],
            salidas = reshape(Float64[0, 1, 1, 0], 4, 1)
        )

        config = ConfiguracionDivision{Float64}(0.4)

        mapa = ejecutar_division(red, datos, config; epochs=0)

        # El mapa debe ser del tipo correcto
        @test mapa isa MapaDeSoluciones{Float64}

        # Con 1 salida, las claves parciales son solo [1]
        @test haskey(mapa.parciales, [1])
        @test length(mapa.parciales) == 1  # 2^1 - 1 = 1

        # La entrada global puede o no tener solución (depende de pesos aleatorios)
        @test mapa.global_ isa EntradaMapaSoluciones{Float64}

        # Si hay solución global, debe superar el umbral
        if mapa.global_.subconfiguracion !== nothing
            @test mapa.global_.precision > 0.4
        end

        # Si hay solución parcial, debe superar el umbral
        if mapa.parciales[[1]].subconfiguracion !== nothing
            @test mapa.parciales[[1]].precision > 0.4
        end
    end

    @testset "Cancelación durante exploración devuelve mapa válido parcial" begin
        # Red más grande para tener más iteraciones: 3 entradas → 4 ocultas → 2 salidas
        W1 = randn(Float64, 3, 4)
        W2 = randn(Float64, 4, 2)
        b1 = randn(Float64, 4)
        b2 = randn(Float64, 2)
        red = RedBase{Float64}([W1, W2], [b1, b2], 3, 2)

        datos = (
            entradas = randn(Float64, 8, 3),
            salidas = Float64.(rand(8, 2) .> 0.5)
        )

        config = ConfiguracionDivision{Float64}(0.3)

        señal = Atomic{Bool}(false)
        evaluadas_al_parar = Ref(0)

        # Callback que activa la señal de parada tras 3 evaluaciones
        callback = function(progreso::ProgresoExploracion)
            evaluadas_al_parar[] = progreso.evaluadas
            if progreso.evaluadas >= 3
                señal[] = true
            end
        end

        mapa = ejecutar_division(red, datos, config;
            callback_progreso=callback,
            señal_parada=señal,
            epochs=0
        )

        # El mapa debe ser válido
        @test mapa isa MapaDeSoluciones{Float64}

        # Debe tener las claves parciales correctas (2^2 - 1 = 3)
        @test length(mapa.parciales) == 3
        @test haskey(mapa.parciales, [1])
        @test haskey(mapa.parciales, [2])
        @test haskey(mapa.parciales, [1, 2])

        # La exploración se detuvo antes de completar todas las subconfiguraciones
        total = (2^3 - 1) * (2^2 - 1)  # 7 * 3 = 21
        @test evaluadas_al_parar[] < total

        # Ninguna entrada debe estar corrupta
        @test mapa.global_ isa EntradaMapaSoluciones{Float64}
        for (k, v) in mapa.parciales
            @test v isa EntradaMapaSoluciones{Float64}
            if v.subconfiguracion !== nothing
                @test v.precision > 0.0
            end
        end
    end

    @testset "Flujo con entrenamiento verifica métricas pre/post" begin
        # Red: 2 entradas → 3 ocultas → 1 salida
        W1 = randn(Float64, 2, 3)
        W2 = randn(Float64, 3, 1)
        b1 = randn(Float64, 3)
        b2 = randn(Float64, 1)
        red = RedBase{Float64}([W1, W2], [b1, b2], 2, 1)

        # Datos AND: linealmente separable, fácil de aprender
        datos = (
            entradas = Float64[0 0; 0 1; 1 0; 1 1],
            salidas = reshape(Float64[0, 0, 0, 1], 4, 1)
        )

        config = ConfiguracionDivision{Float64}(0.2)

        mapa = ejecutar_division(red, datos, config; epochs=50)

        @test mapa isa MapaDeSoluciones{Float64}

        # Verificar que las entradas con subconfiguración tienen métricas de entrenamiento
        if mapa.global_.subconfiguracion !== nothing
            @test 0.0 <= mapa.global_.precision_pre_entrenamiento <= 1.0
            @test 0.0 <= mapa.global_.precision_post_entrenamiento <= 1.0
        end

        for (k, entrada) in mapa.parciales
            if entrada.subconfiguracion !== nothing
                @test 0.0 <= entrada.precision_pre_entrenamiento <= 1.0
                @test 0.0 <= entrada.precision_post_entrenamiento <= 1.0
            end
        end
    end

    @testset "Sin solución global emite @info" begin
        # Red: 2 entradas → 2 ocultas → 1 salida
        W1 = randn(Float64, 2, 2)
        W2 = randn(Float64, 2, 1)
        b1 = randn(Float64, 2)
        b2 = randn(Float64, 1)
        red = RedBase{Float64}([W1, W2], [b1, b2], 2, 1)

        datos = (
            entradas = Float64[0 0; 0 1; 1 0; 1 1],
            salidas = reshape(Float64[0, 1, 1, 0], 4, 1)
        )

        # Umbral imposible de alcanzar con pesos aleatorios sin entrenar
        config = ConfiguracionDivision{Float64}(1.0)

        # Verificar que se emite @info cuando no hay solución global
        mapa = @test_logs (:info, "Ninguna subconfiguración alcanzó el umbral de acierto para la solución global") ejecutar_division(red, datos, config; epochs=0)

        @test mapa isa MapaDeSoluciones{Float64}
        @test mapa.global_.subconfiguracion === nothing
    end

end
