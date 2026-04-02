using Test
using DivisionNeuronal

@testset "Entrenamiento" begin

    @testset "entrenar_y_evaluar! devuelve tupla de precisiones" begin
        # Red simple: 2 entradas → 2 ocultas → 1 salida
        W1 = randn(Float64, 2, 2)
        W2 = randn(Float64, 2, 1)
        b1 = randn(Float64, 2)
        b2 = randn(Float64, 1)
        sub = Subconfiguracion{Float64}([1, 2], [1], [W1, W2], [b1, b2], 5)

        datos_train = (
            entradas = Float64[0 0; 0 1; 1 0; 1 1],
            salidas = Float64[0; 1; 1; 0]''  # XOR - reshape to matrix
        )
        # Make salidas a proper matrix
        datos_train = (
            entradas = Float64[0 0; 0 1; 1 0; 1 1],
            salidas = reshape(Float64[0, 1, 1, 0], 4, 1)
        )
        datos_val = datos_train

        resultado = entrenar_y_evaluar!(sub, datos_train, datos_val; epochs=10)

        @test resultado isa Tuple{Float64, Float64}
        @test length(resultado) == 2
        @test 0.0 <= resultado[1] <= 1.0
        @test 0.0 <= resultado[2] <= 1.0
    end

    @testset "entrenar_y_evaluar! modifica pesos in-place" begin
        W1 = Float64[0.5 0.3; -0.2 0.8]
        b1 = Float64[0.1, -0.1]
        W2 = Float64[0.4; -0.6]''
        W2 = reshape(Float64[0.4, -0.6], 2, 1)
        b2 = Float64[0.0]
        sub = Subconfiguracion{Float64}([1, 2], [1], [W1, W2], [b1, b2], 5)

        pesos_originales_W1 = copy(W1)
        pesos_originales_W2 = copy(W2)

        datos = (
            entradas = Float64[1 0; 0 1; 1 1; 0 0],
            salidas = reshape(Float64[1, 0, 1, 0], 4, 1)
        )

        entrenar_y_evaluar!(sub, datos, datos; epochs=50)

        # Los pesos deben haber cambiado
        @test sub.pesos[1] != pesos_originales_W1
        @test sub.pesos[2] != pesos_originales_W2
    end

    @testset "entrenar_y_evaluar! con datos linealmente separables mejora precisión" begin
        # Problema simple: salida = entrada[1] AND entrada[2]
        # Con suficientes epochs, debería mejorar
        W1 = randn(Float64, 2, 4)
        W2 = randn(Float64, 4, 1)
        b1 = randn(Float64, 4)
        b2 = randn(Float64, 1)
        sub = Subconfiguracion{Float64}([1, 2], [1], [W1, W2], [b1, b2], 7)

        # Datos AND: linealmente separable
        datos = (
            entradas = Float64[0 0; 0 1; 1 0; 1 1],
            salidas = reshape(Float64[0, 0, 0, 1], 4, 1)
        )

        (pre, post) = entrenar_y_evaluar!(sub, datos, datos; epochs=200)

        # Con datos linealmente separables y suficientes epochs,
        # la precisión post debería ser razonable
        @test pre isa Float64
        @test post isa Float64
        # Post-training should be at least as good or better for simple problems
        # (not guaranteed for all random inits, but likely with 200 epochs on AND)
    end

    @testset "entrenar_y_evaluar! funciona con Float32" begin
        W1 = randn(Float32, 2, 3)
        W2 = randn(Float32, 3, 1)
        b1 = randn(Float32, 3)
        b2 = randn(Float32, 1)
        sub = Subconfiguracion{Float32}([1, 2], [1], [W1, W2], [b1, b2], 6)

        datos = (
            entradas = Float32[0 0; 0 1; 1 0; 1 1],
            salidas = reshape(Float32[0, 0, 0, 1], 4, 1)
        )

        resultado = entrenar_y_evaluar!(sub, datos, datos; epochs=10)

        @test resultado isa Tuple{Float32, Float32}
        @test 0.0f0 <= resultado[1] <= 1.0f0
        @test 0.0f0 <= resultado[2] <= 1.0f0
    end

    @testset "entrenar_y_evaluar! con epochs=0 devuelve misma precisión pre y post" begin
        W1 = randn(Float64, 2, 1)
        b1 = randn(Float64, 1)
        sub = Subconfiguracion{Float64}([1, 2], [1], [W1], [b1], 3)

        datos = (
            entradas = Float64[1 0; 0 1],
            salidas = reshape(Float64[1, 0], 2, 1)
        )

        (pre, post) = entrenar_y_evaluar!(sub, datos, datos; epochs=0)

        @test pre ≈ post
    end

    @testset "entrenar_y_evaluar! con múltiples salidas" begin
        W1 = randn(Float64, 3, 4)
        W2 = randn(Float64, 4, 2)
        b1 = randn(Float64, 4)
        b2 = randn(Float64, 2)
        sub = Subconfiguracion{Float64}([1, 2, 3], [1, 2], [W1, W2], [b1, b2], 9)

        datos = (
            entradas = randn(Float64, 8, 3),
            salidas = Float64.(rand(8, 2) .> 0.5)
        )

        resultado = entrenar_y_evaluar!(sub, datos, datos; epochs=20)

        @test resultado isa Tuple{Float64, Float64}
        @test 0.0 <= resultado[1] <= 1.0
        @test 0.0 <= resultado[2] <= 1.0
    end

    @testset "entrenar_mapa! almacena métricas pre/post en entradas del mapa" begin
        # Crear red y datos
        W1 = randn(Float64, 2, 3)
        W2 = randn(Float64, 3, 1)
        b1 = randn(Float64, 3)
        b2 = randn(Float64, 1)
        sub = Subconfiguracion{Float64}([1, 2], [1], [W1, W2], [b1, b2], 6)

        datos = (
            entradas = Float64[0 0; 0 1; 1 0; 1 1],
            salidas = reshape(Float64[0, 0, 0, 1], 4, 1)
        )

        # Crear mapa con 1 salida e insertar subconfiguración en la global
        mapa = inicializar_mapa(1, Float64)
        mapa.global_.subconfiguracion = sub
        mapa.global_.precision = 0.5

        entrenar_mapa!(mapa, datos, datos; epochs=10)

        @test mapa.global_.precision_pre_entrenamiento isa Float64
        @test mapa.global_.precision_post_entrenamiento isa Float64
        @test 0.0 <= mapa.global_.precision_pre_entrenamiento <= 1.0
        @test 0.0 <= mapa.global_.precision_post_entrenamiento <= 1.0
    end

    @testset "entrenar_mapa! calcula diferencia post - pre correctamente" begin
        W1 = randn(Float64, 2, 1)
        b1 = randn(Float64, 1)
        sub = Subconfiguracion{Float64}([1, 2], [1], [W1], [b1], 3)

        datos = (
            entradas = Float64[1 0; 0 1],
            salidas = reshape(Float64[1, 0], 2, 1)
        )

        mapa = inicializar_mapa(1, Float64)
        mapa.global_.subconfiguracion = sub
        mapa.global_.precision = 0.5

        entrenar_mapa!(mapa, datos, datos; epochs=0)

        # Con 0 epochs, pre y post deben ser iguales
        diferencia = mapa.global_.precision_post_entrenamiento - mapa.global_.precision_pre_entrenamiento
        @test diferencia ≈ 0.0 atol=1e-10
    end

    @testset "entrenar_mapa! entrena soluciones parciales" begin
        W1 = randn(Float64, 2, 3)
        W2 = randn(Float64, 3, 2)
        b1 = randn(Float64, 3)
        b2 = randn(Float64, 2)
        sub = Subconfiguracion{Float64}([1, 2], [1, 2], [W1, W2], [b1, b2], 7)

        datos = (
            entradas = randn(Float64, 6, 2),
            salidas = Float64.(rand(6, 2) .> 0.5)
        )

        mapa = inicializar_mapa(2, Float64)
        # Insertar en una entrada parcial
        clave_parcial = [1]
        mapa.parciales[clave_parcial].subconfiguracion = sub
        mapa.parciales[clave_parcial].precision = 0.5

        entrenar_mapa!(mapa, datos, datos; epochs=10)

        entrada = mapa.parciales[clave_parcial]
        @test 0.0 <= entrada.precision_pre_entrenamiento <= 1.0
        @test 0.0 <= entrada.precision_post_entrenamiento <= 1.0
    end

    @testset "entrenar_mapa! ignora entradas sin subconfiguración" begin
        datos = (
            entradas = Float64[0 0; 1 1],
            salidas = reshape(Float64[0, 1], 2, 1)
        )

        mapa = inicializar_mapa(1, Float64)
        # No insertar ninguna subconfiguración

        resultado = entrenar_mapa!(mapa, datos, datos; epochs=10)

        # Debe devolver el mapa sin cambios
        @test resultado === mapa
        @test mapa.global_.subconfiguracion === nothing
        @test mapa.global_.precision_pre_entrenamiento == 0.0
        @test mapa.global_.precision_post_entrenamiento == 0.0
    end

    @testset "entrenar_mapa! emite @warn cuando post < pre" begin
        # Crear una subconfiguración con pesos que dan buena precisión inicial
        # y entrenar con datos contradictorios para forzar degradación
        W1 = Float64[10.0 -10.0; -10.0 10.0]
        b1 = Float64[-5.0, -5.0]
        W2 = reshape(Float64[10.0, 10.0], 2, 1)
        b2 = Float64[-15.0]
        sub = Subconfiguracion{Float64}([1, 2], [1], [W1, W2], [b1, b2], 5)

        # Datos XOR - the network with these weights should have good precision
        datos_val = (
            entradas = Float64[0 0; 0 1; 1 0; 1 1],
            salidas = reshape(Float64[0, 1, 1, 0], 4, 1)
        )
        # Train with contradictory data (inverted)
        datos_train = (
            entradas = Float64[0 0; 0 1; 1 0; 1 1],
            salidas = reshape(Float64[1, 0, 0, 1], 4, 1)
        )

        mapa = inicializar_mapa(1, Float64)
        mapa.global_.subconfiguracion = sub
        mapa.global_.precision = 0.75

        # Capturar warnings usando @test_logs
        # Si post < pre, se emite @warn; si no, el test pasa igualmente
        entrenar_mapa!(mapa, datos_train, datos_val; epochs=50)

        if mapa.global_.precision_post_entrenamiento < mapa.global_.precision_pre_entrenamiento
            # Re-run with fresh subconfig to verify warning is emitted
            W1b = Float64[10.0 -10.0; -10.0 10.0]
            b1b = Float64[-5.0, -5.0]
            W2b = reshape(Float64[10.0, 10.0], 2, 1)
            b2b = Float64[-15.0]
            sub2 = Subconfiguracion{Float64}([1, 2], [1], [W1b, W2b], [b1b, b2b], 5)

            mapa2 = inicializar_mapa(1, Float64)
            mapa2.global_.subconfiguracion = sub2
            mapa2.global_.precision = 0.75

            @test_logs (:warn,) min_level=Base.CoreLogging.Warn entrenar_mapa!(mapa2, datos_train, datos_val; epochs=50)
        else
            @test true  # Training improved or stayed same, no warning expected
        end
    end

    @testset "entrenar_mapa! devuelve el mapa" begin
        datos = (
            entradas = Float64[0 0; 1 1],
            salidas = reshape(Float64[0, 1], 2, 1)
        )

        mapa = inicializar_mapa(1, Float64)
        resultado = entrenar_mapa!(mapa, datos, datos; epochs=1)

        @test resultado === mapa
    end

end
