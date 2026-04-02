using Test
using DivisionNeuronal

@testset "Validación" begin

    @testset "validar_red_base" begin
        @testset "acepta red válida con una capa" begin
            pesos = [rand(Float64, 3, 2)]
            biases = [rand(Float64, 2)]
            red = RedBase(pesos, biases, 3, 2)
            @test validar_red_base(red) === nothing
        end

        @testset "acepta red válida con múltiples capas" begin
            # 3 entradas → 4 ocultas → 2 salidas
            pesos = [rand(Float64, 3, 4), rand(Float64, 4, 2)]
            biases = [rand(Float64, 4), rand(Float64, 2)]
            red = RedBase(pesos, biases, 3, 2)
            @test validar_red_base(red) === nothing
        end

        @testset "rechaza red con pesos vacíos" begin
            red = RedBase(Matrix{Float64}[], Vector{Float64}[], 3, 2)
            @test_throws RedBaseNoInicializadaError validar_red_base(red)
        end

        @testset "rechaza red con número distinto de capas pesos/biases" begin
            pesos = [rand(Float64, 3, 4), rand(Float64, 4, 2)]
            biases = [rand(Float64, 4)]  # falta un bias
            red = RedBase(pesos, biases, 3, 2)
            @test_throws RedBaseNoInicializadaError validar_red_base(red)
        end

        @testset "rechaza red con dimensiones inconsistentes entre capas" begin
            # cols de capa 1 (4) != filas de capa 2 (5)
            pesos = [rand(Float64, 3, 4), rand(Float64, 5, 2)]
            biases = [rand(Float64, 4), rand(Float64, 2)]
            red = RedBase(pesos, biases, 3, 2)
            @test_throws RedBaseNoInicializadaError validar_red_base(red)
        end

        @testset "rechaza red con bias de dimensión incorrecta" begin
            pesos = [rand(Float64, 3, 4)]
            biases = [rand(Float64, 3)]  # debería ser 4
            red = RedBase(pesos, biases, 3, 4)
            @test_throws RedBaseNoInicializadaError validar_red_base(red)
        end

        @testset "acepta red con Float32" begin
            pesos = [rand(Float32, 2, 3)]
            biases = [rand(Float32, 3)]
            red = RedBase(pesos, biases, 2, 3)
            @test validar_red_base(red) === nothing
        end
    end

    @testset "validar_neuronas" begin
        @testset "acepta valores positivos" begin
            @test validar_neuronas(1, 1) === nothing
            @test validar_neuronas(5, 3) === nothing
            @test validar_neuronas(100, 50) === nothing
        end

        @testset "rechaza n_entradas < 1" begin
            @test_throws NeuronasInvalidasError validar_neuronas(0, 3)
            @test_throws NeuronasInvalidasError validar_neuronas(-1, 3)
        end

        @testset "rechaza n_salidas < 1" begin
            @test_throws NeuronasInvalidasError validar_neuronas(3, 0)
            @test_throws NeuronasInvalidasError validar_neuronas(3, -5)
        end

        @testset "rechaza ambos < 1" begin
            @test_throws NeuronasInvalidasError validar_neuronas(0, 0)
        end
    end

    @testset "validar_umbral" begin
        @testset "acepta valores en rango [0.0, 1.0]" begin
            @test validar_umbral(0.0) === nothing
            @test validar_umbral(0.4) === nothing
            @test validar_umbral(0.5) === nothing
            @test validar_umbral(1.0) === nothing
        end

        @testset "rechaza valores negativos" begin
            @test_throws UmbralFueraDeRangoError validar_umbral(-0.1)
            @test_throws UmbralFueraDeRangoError validar_umbral(-1.0)
        end

        @testset "rechaza valores mayores que 1.0" begin
            @test_throws UmbralFueraDeRangoError validar_umbral(1.1)
            @test_throws UmbralFueraDeRangoError validar_umbral(2.0)
        end

        @testset "devuelve valor por defecto 0.4 sin argumentos" begin
            @test validar_umbral() == 0.4
        end
    end

end
