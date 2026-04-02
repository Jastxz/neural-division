using Test
using DivisionNeuronal

@testset "GeneradorDeSubconfiguraciones" begin

    @testset "bitmask_a_indices" begin
        @test bitmask_a_indices(0b001, 3) == [1]
        @test bitmask_a_indices(0b010, 3) == [2]
        @test bitmask_a_indices(0b011, 3) == [1, 2]
        @test bitmask_a_indices(0b111, 3) == [1, 2, 3]
        @test bitmask_a_indices(0b101, 3) == [1, 3]
        @test bitmask_a_indices(1, 1) == [1]
    end

    @testset "length devuelve (2^n_ent - 1) × (2^n_sal - 1)" begin
        red_2_1 = RedBase{Float64}(
            [randn(2, 3), randn(3, 1)],
            [randn(3), randn(1)],
            2, 1
        )
        gen = GeneradorDeSubconfiguraciones(red_2_1)
        @test length(gen) == (2^2 - 1) * (2^1 - 1)  # 3 × 1 = 3

        red_3_2 = RedBase{Float64}(
            [randn(3, 4), randn(4, 2)],
            [randn(4), randn(2)],
            3, 2
        )
        gen2 = GeneradorDeSubconfiguraciones(red_3_2)
        @test length(gen2) == (2^3 - 1) * (2^2 - 1)  # 7 × 3 = 21
    end

    @testset "iteración produce exactamente length(gen) elementos" begin
        red = RedBase{Float64}(
            [randn(3, 4), randn(4, 2)],
            [randn(4), randn(2)],
            3, 2
        )
        gen = GeneradorDeSubconfiguraciones(red)
        elementos = collect(gen)
        @test length(elementos) == length(gen)
        @test length(elementos) == 21
    end

    @testset "cada elemento es un par de vectores de Int no vacíos" begin
        red = RedBase{Float64}(
            [randn(2, 3), randn(3, 2)],
            [randn(3), randn(2)],
            2, 2
        )
        gen = GeneradorDeSubconfiguraciones(red)
        for (idx_ent, idx_sal) in gen
            @test !isempty(idx_ent)
            @test !isempty(idx_sal)
            @test eltype(idx_ent) == Int
            @test eltype(idx_sal) == Int
        end
    end

    @testset "índices están dentro de los rangos válidos" begin
        red = RedBase{Float64}(
            [randn(3, 4), randn(4, 2)],
            [randn(4), randn(2)],
            3, 2
        )
        gen = GeneradorDeSubconfiguraciones(red)
        for (idx_ent, idx_sal) in gen
            @test all(1 .<= idx_ent .<= 3)
            @test all(1 .<= idx_sal .<= 2)
        end
    end

    @testset "no hay duplicados en la enumeración" begin
        red = RedBase{Float64}(
            [randn(3, 4), randn(4, 2)],
            [randn(4), randn(2)],
            3, 2
        )
        gen = GeneradorDeSubconfiguraciones(red)
        elementos = collect(gen)
        @test length(unique(elementos)) == length(elementos)
    end

    @testset "caso mínimo: 1 entrada, 1 salida" begin
        red = RedBase{Float64}(
            [randn(1, 1)],
            [randn(1)],
            1, 1
        )
        gen = GeneradorDeSubconfiguraciones(red)
        @test length(gen) == 1
        elementos = collect(gen)
        @test length(elementos) == 1
        @test elementos[1] == ([1], [1])
    end

    @testset "soporta Float32" begin
        red = RedBase{Float32}(
            [randn(Float32, 2, 3), randn(Float32, 3, 1)],
            [randn(Float32, 3), randn(Float32, 1)],
            2, 1
        )
        gen = GeneradorDeSubconfiguraciones(red)
        @test length(gen) == 3
        elementos = collect(gen)
        @test length(elementos) == 3
    end

end


@testset "extraer_subconfiguracion" begin

    @testset "red con una capa oculta: recorta primera y última capa correctamente" begin
        # Red: 3 entradas → 4 ocultas → 2 salidas
        W1 = Float64[1 2 3 4; 5 6 7 8; 9 10 11 12]  # 3×4
        W2 = Float64[1 2; 3 4; 5 6; 7 8]              # 4×2
        b1 = Float64[10, 20, 30, 40]                   # 4
        b2 = Float64[100, 200]                         # 2
        red = RedBase{Float64}([W1, W2], [b1, b2], 3, 2)

        sub = extraer_subconfiguracion(red, [1, 3], [2])
        @test sub !== nothing
        @test sub.indices_entrada == [1, 3]
        @test sub.indices_salida == [2]

        # Primera capa: filas 1 y 3 de W1
        @test sub.pesos[1] == W1[[1, 3], :]
        # Bias de capa oculta: completo
        @test sub.biases[1] == b1
        # Última capa: columna 2 de W2
        @test sub.pesos[2] == W2[:, [2]]
        # Bias de salida: solo índice 2
        @test sub.biases[2] == b2[[2]]
    end

    @testset "red sin capas ocultas (una sola capa)" begin
        # Red: 2 entradas → 3 salidas directamente
        W = Float64[1 2 3; 4 5 6]  # 2×3
        b = Float64[10, 20, 30]     # 3
        red = RedBase{Float64}([W], [b], 2, 3)

        sub = extraer_subconfiguracion(red, [2], [1, 3])
        @test sub !== nothing
        @test sub.indices_entrada == [2]
        @test sub.indices_salida == [1, 3]
        @test sub.pesos[1] == W[[2], [1, 3]]
        @test sub.biases[1] == b[[1, 3]]
    end

    @testset "preserva exactamente los pesos originales" begin
        W1 = randn(3, 5)
        W2 = randn(5, 4)
        W3 = randn(4, 2)
        b1 = randn(5)
        b2 = randn(4)
        b3 = randn(2)
        red = RedBase{Float64}([W1, W2, W3], [b1, b2, b3], 3, 2)

        sub = extraer_subconfiguracion(red, [1, 2], [1])
        @test sub !== nothing
        # Primera capa: filas 1,2
        @test sub.pesos[1] == W1[[1, 2], :]
        # Capa oculta intermedia: completa
        @test sub.pesos[2] == W2
        # Última capa: columna 1
        @test sub.pesos[3] == W3[:, [1]]
        # Biases ocultos completos
        @test sub.biases[1] == b1
        @test sub.biases[2] == b2
        # Bias de salida recortado
        @test sub.biases[3] == b3[[1]]
    end

    @testset "capas ocultas se mantienen completas" begin
        W1 = randn(4, 3)
        W2 = randn(3, 5)
        W3 = randn(5, 2)
        red = RedBase{Float64}([W1, W2, W3], [randn(3), randn(5), randn(2)], 4, 2)

        sub = extraer_subconfiguracion(red, [2], [1, 2])
        @test sub !== nothing
        # Capa oculta intermedia debe ser idéntica
        @test size(sub.pesos[2]) == size(W2)
        @test sub.pesos[2] == W2
    end

    @testset "n_neuronas_activas se calcula correctamente" begin
        # Red: 3 entradas → 4 ocultas → 2 salidas
        W1 = randn(3, 4)
        W2 = randn(4, 2)
        red = RedBase{Float64}([W1, W2], [randn(4), randn(2)], 3, 2)

        sub = extraer_subconfiguracion(red, [1, 3], [2])
        @test sub !== nothing
        # 2 entradas + 4 ocultas + 1 salida = 7
        @test sub.n_neuronas_activas == 7
    end

    @testset "n_neuronas_activas sin capas ocultas" begin
        W = randn(2, 3)
        red = RedBase{Float64}([W], [randn(3)], 2, 3)

        sub = extraer_subconfiguracion(red, [1], [2, 3])
        @test sub !== nothing
        # 1 entrada + 0 ocultas + 2 salidas = 3
        @test sub.n_neuronas_activas == 3
    end

    @testset "n_neuronas_activas con múltiples capas ocultas" begin
        # Red: 3 entradas → 5 ocultas → 4 ocultas → 2 salidas
        W1 = randn(3, 5)
        W2 = randn(5, 4)
        W3 = randn(4, 2)
        red = RedBase{Float64}([W1, W2, W3], [randn(5), randn(4), randn(2)], 3, 2)

        sub = extraer_subconfiguracion(red, [1, 2, 3], [1])
        @test sub !== nothing
        # 3 entradas + 5 ocultas + 4 ocultas + 1 salida = 13
        @test sub.n_neuronas_activas == 13
    end

    @testset "devuelve nothing para red sin pesos" begin
        red = RedBase{Float64}(Matrix{Float64}[], Vector{Float64}[], 0, 0)
        result = extraer_subconfiguracion(red, Int[], Int[])
        @test result === nothing
    end

    @testset "soporta Float32" begin
        W1 = randn(Float32, 2, 3)
        W2 = randn(Float32, 3, 1)
        red = RedBase{Float32}([W1, W2], [randn(Float32, 3), randn(Float32, 1)], 2, 1)

        sub = extraer_subconfiguracion(red, [1], [1])
        @test sub !== nothing
        @test eltype(sub.pesos[1]) == Float32
        @test eltype(sub.biases[1]) == Float32
    end

    @testset "selección completa devuelve pesos equivalentes" begin
        W1 = randn(3, 4)
        W2 = randn(4, 2)
        b1 = randn(4)
        b2 = randn(2)
        red = RedBase{Float64}([W1, W2], [b1, b2], 3, 2)

        sub = extraer_subconfiguracion(red, [1, 2, 3], [1, 2])
        @test sub !== nothing
        @test sub.pesos[1] == W1
        @test sub.pesos[2] == W2
        @test sub.biases[1] == b1
        @test sub.biases[2] == b2
        # 3 + 4 + 2 = 9
        @test sub.n_neuronas_activas == 9
    end

end
