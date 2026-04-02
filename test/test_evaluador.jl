using Test
using DivisionNeuronal

@testset "Evaluador" begin

    @testset "forward_pass con una capa produce salida en (0,1)" begin
        # Red simple: 2 entradas → 1 salida
        W = Float64[0.5 -0.3; 0.2 0.8]'  # 2×1 → transposed for clarity
        # Actually let's be explicit: 2 inputs, 1 output
        W1 = Float64[0.5; -0.3]''  # This is tricky, let's use reshape
        W1 = reshape(Float64[0.5, -0.3], 2, 1)
        b1 = Float64[0.0]
        sub = Subconfiguracion{Float64}([1, 2], [1], [W1], [b1], 3)

        entrada = Float64[1.0 0.0; 0.0 1.0; 1.0 1.0]  # 3 muestras × 2 features
        resultado = DivisionNeuronal.forward_pass(sub, entrada)

        @test size(resultado) == (3, 1)
        # Sigmoid output should be in (0, 1)
        @test all(0.0 .< resultado .< 1.0)
    end

    @testset "forward_pass con múltiples capas" begin
        # Red: 2 entradas → 3 ocultas → 2 salidas
        W1 = randn(2, 3)
        W2 = randn(3, 2)
        b1 = randn(3)
        b2 = randn(2)
        sub = Subconfiguracion{Float64}([1, 2], [1, 2], [W1, W2], [b1, b2], 7)

        entrada = randn(5, 2)  # 5 muestras × 2 features
        resultado = DivisionNeuronal.forward_pass(sub, entrada)

        @test size(resultado) == (5, 2)
        @test all(0.0 .< resultado .< 1.0)
    end

    @testset "evaluar con datos perfectos da precisión 1.0" begin
        # Crear una subconfiguración que produce salidas > 0.5 para todas las muestras
        # Usamos pesos grandes positivos y bias grande positivo para forzar sigmoid ≈ 1.0
        W1 = Float64[10.0 10.0; 10.0 10.0]  # 2×2
        b1 = Float64[10.0, 10.0]
        sub = Subconfiguracion{Float64}([1, 2], [1, 2], [W1], [b1], 4)

        # Datos donde las salidas esperadas son todas 1.0
        datos = (
            entradas = ones(Float64, 4, 2),
            salidas = ones(Float64, 4, 2)
        )

        resultado = evaluar(sub, datos, [1, 2])

        @test resultado.precision_global ≈ 1.0
        # Todas las precisiones parciales deben ser 1.0
        for (_, prec) in resultado.precisiones_parciales
            @test prec ≈ 1.0
        end
    end

    @testset "evaluar con datos incorrectos da precisión 0.0" begin
        # Pesos grandes positivos → sigmoid ≈ 1.0, pero targets son 0.0
        W1 = Float64[10.0 10.0; 10.0 10.0]
        b1 = Float64[10.0, 10.0]
        sub = Subconfiguracion{Float64}([1, 2], [1, 2], [W1], [b1], 4)

        datos = (
            entradas = ones(Float64, 4, 2),
            salidas = zeros(Float64, 4, 2)  # Targets son 0, predicciones serán ≈1
        )

        resultado = evaluar(sub, datos, [1, 2])

        @test resultado.precision_global ≈ 0.0
    end

    @testset "evaluar calcula precisiones parciales para todos los subconjuntos" begin
        # Red simple con 3 salidas totales, subconfig cubre salidas [1, 2]
        W1 = Float64[10.0 10.0; 10.0 10.0]
        b1 = Float64[10.0, 10.0]
        sub = Subconfiguracion{Float64}([1, 2], [1, 2], [W1], [b1], 4)

        datos = (
            entradas = ones(Float64, 3, 2),
            salidas = ones(Float64, 3, 3)  # 3 salidas totales
        )

        indices_totales = [1, 2, 3]
        resultado = evaluar(sub, datos, indices_totales)

        # Debe haber 2^3 - 1 = 7 subconjuntos
        @test length(resultado.precisiones_parciales) == 7

        # Subconjuntos que solo contienen índices cubiertos por la subconfig ([1], [2], [1,2])
        # deben tener precisión 1.0 (predicciones ≈ 1.0, targets = 1.0)
        @test resultado.precisiones_parciales[[1]] ≈ 1.0
        @test resultado.precisiones_parciales[[2]] ≈ 1.0
        @test resultado.precisiones_parciales[[1, 2]] ≈ 1.0

        # Subconjuntos que contienen índice 3 (no cubierto) → precisión 0.0
        @test resultado.precisiones_parciales[[3]] ≈ 0.0
    end

    @testset "evaluar con subconfig parcial y salidas no cubiertas" begin
        # Subconfig solo cubre salida [1] de un problema con salidas [1, 2]
        W1 = reshape(Float64[10.0, 10.0], 2, 1)  # 2×1
        b1 = Float64[10.0]
        sub = Subconfiguracion{Float64}([1, 2], [1], [W1], [b1], 3)

        datos = (
            entradas = ones(Float64, 3, 2),
            salidas = ones(Float64, 3, 2)
        )

        resultado = evaluar(sub, datos, [1, 2])

        # Precisión global: solo evalúa la salida [1] que la subconfig cubre
        # La subconfig produce ≈1.0 para salida 1, target es 1.0 → correcto
        @test resultado.precision_global ≈ 1.0

        # Parcial [1]: cubierta, predicción correcta
        @test resultado.precisiones_parciales[[1]] ≈ 1.0
        # Parcial [2]: no cubierta → 0.0
        @test resultado.precisiones_parciales[[2]] ≈ 0.0
    end

    @testset "ResultadoEvaluacion tiene tipo correcto" begin
        W1 = randn(Float32, 2, 1)
        b1 = randn(Float32, 1)
        sub = Subconfiguracion{Float32}([1, 2], [1], [W1], [b1], 3)

        datos = (
            entradas = randn(Float32, 5, 2),
            salidas = rand(Float32, 5, 1) .> 0.5f0 .|> Float32
        )

        resultado = evaluar(sub, datos, [1])
        @test resultado isa ResultadoEvaluacion{Float32}
        @test resultado.precision_global isa Float32
    end

    @testset "subconjuntos_no_vacios genera todos los subconjuntos" begin
        subconj = DivisionNeuronal.subconjuntos_no_vacios([1, 2, 3])
        @test length(subconj) == 7  # 2^3 - 1
        # Verificar que contiene los esperados
        @test [1] in subconj
        @test [2] in subconj
        @test [3] in subconj
        @test [1, 2] in subconj
        @test [1, 3] in subconj
        @test [2, 3] in subconj
        @test [1, 2, 3] in subconj
    end

    @testset "subconjuntos_no_vacios con un solo elemento" begin
        subconj = DivisionNeuronal.subconjuntos_no_vacios([1])
        @test length(subconj) == 1
        @test subconj == [[1]]
    end

    @testset "es_mejor" begin
        # Helper para crear subconfiguraciones con n_neuronas_activas específico
        function hacer_sub(n_neuronas::Int)
            W = reshape(Float64[1.0], 1, 1)
            b = Float64[0.0]
            Subconfiguracion{Float64}([1], [1], [W], [b], n_neuronas)
        end

        @testset "devuelve true si actual.subconfiguracion es nothing" begin
            nueva = hacer_sub(5)
            actual = EntradaMapaSoluciones{Float64}(nothing, 0.0, 0.0, 0.0)
            @test es_mejor(nueva, 0.8, actual) == true
        end

        @testset "devuelve true si nueva tiene menos neuronas activas" begin
            nueva = hacer_sub(3)
            sub_actual = hacer_sub(5)
            actual = EntradaMapaSoluciones{Float64}(sub_actual, 0.9, 0.0, 0.0)
            @test es_mejor(nueva, 0.5, actual) == true
        end

        @testset "devuelve true si mismas neuronas y nueva tiene mayor precisión" begin
            nueva = hacer_sub(5)
            sub_actual = hacer_sub(5)
            actual = EntradaMapaSoluciones{Float64}(sub_actual, 0.7, 0.0, 0.0)
            @test es_mejor(nueva, 0.8, actual) == true
        end

        @testset "devuelve false si mismas neuronas y nueva tiene menor precisión" begin
            nueva = hacer_sub(5)
            sub_actual = hacer_sub(5)
            actual = EntradaMapaSoluciones{Float64}(sub_actual, 0.9, 0.0, 0.0)
            @test es_mejor(nueva, 0.8, actual) == false
        end

        @testset "devuelve false si mismas neuronas e igual precisión" begin
            nueva = hacer_sub(5)
            sub_actual = hacer_sub(5)
            actual = EntradaMapaSoluciones{Float64}(sub_actual, 0.8, 0.0, 0.0)
            @test es_mejor(nueva, 0.8, actual) == false
        end

        @testset "devuelve false si nueva tiene más neuronas activas" begin
            nueva = hacer_sub(7)
            sub_actual = hacer_sub(5)
            actual = EntradaMapaSoluciones{Float64}(sub_actual, 0.5, 0.0, 0.0)
            @test es_mejor(nueva, 0.9, actual) == false
        end
    end

end
