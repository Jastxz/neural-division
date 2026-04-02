using Test
using DivisionNeuronal

@testset "Mapa de Soluciones" begin

    @testset "inicializar_mapa" begin
        @testset "n_salidas = 1" begin
            mapa = inicializar_mapa(1, Float64)
            # Global entry exists with nothing
            @test mapa.global_.subconfiguracion === nothing
            @test mapa.global_.precision == 0.0
            # 2^1 - 1 = 1 partial entry: [1]
            @test length(mapa.parciales) == 1
            @test haskey(mapa.parciales, [1])
            @test mapa.parciales[[1]].subconfiguracion === nothing
        end

        @testset "n_salidas = 2" begin
            mapa = inicializar_mapa(2, Float64)
            @test mapa.global_.subconfiguracion === nothing
            # 2^2 - 1 = 3 partial entries: [1], [2], [1,2]
            @test length(mapa.parciales) == 3
            @test haskey(mapa.parciales, [1])
            @test haskey(mapa.parciales, [2])
            @test haskey(mapa.parciales, [1, 2])
            for (_, entrada) in mapa.parciales
                @test entrada.subconfiguracion === nothing
                @test entrada.precision == 0.0
                @test entrada.precision_pre_entrenamiento == 0.0
                @test entrada.precision_post_entrenamiento == 0.0
            end
        end

        @testset "n_salidas = 3" begin
            mapa = inicializar_mapa(3, Float64)
            # 2^3 - 1 = 7 partial entries
            @test length(mapa.parciales) == 7
            expected_keys = [[1], [2], [1,2], [3], [1,3], [2,3], [1,2,3]]
            for key in expected_keys
                @test haskey(mapa.parciales, key)
            end
        end

        @testset "Float32 type parameter" begin
            mapa = inicializar_mapa(2, Float32)
            @test mapa.global_.precision isa Float32
            @test mapa.global_.precision == Float32(0)
        end
    end

    @testset "actualizar_si_mejor!" begin
        # Helper to create a simple Subconfiguracion
        function make_subconfig(T, n_neuronas; idx_ent=[1], idx_sal=[1])
            pesos = [randn(T, length(idx_ent), length(idx_sal))]
            biases = [randn(T, length(idx_sal))]
            Subconfiguracion{T}(idx_ent, idx_sal, pesos, biases, n_neuronas)
        end

        @testset "actualiza global cuando supera umbral y es mejor" begin
            mapa = inicializar_mapa(2, Float64)
            subconfig = make_subconfig(Float64, 3; idx_sal=[1, 2])
            resultado = ResultadoEvaluacion{Float64}(
                0.8,  # precision_global
                Dict{Vector{Int}, Float64}([1] => 0.9, [2] => 0.7, [1,2] => 0.8)
            )
            actualizar_si_mejor!(mapa, subconfig, resultado, 0.5)
            @test mapa.global_.subconfiguracion === subconfig
            @test mapa.global_.precision == 0.8
        end

        @testset "no actualiza global cuando no supera umbral" begin
            mapa = inicializar_mapa(2, Float64)
            subconfig = make_subconfig(Float64, 3; idx_sal=[1, 2])
            resultado = ResultadoEvaluacion{Float64}(
                0.3,  # below threshold
                Dict{Vector{Int}, Float64}([1] => 0.9, [2] => 0.7, [1,2] => 0.3)
            )
            actualizar_si_mejor!(mapa, subconfig, resultado, 0.5)
            @test mapa.global_.subconfiguracion === nothing
        end

        @testset "actualiza parciales que superan umbral" begin
            mapa = inicializar_mapa(2, Float64)
            subconfig = make_subconfig(Float64, 3; idx_sal=[1, 2])
            resultado = ResultadoEvaluacion{Float64}(
                0.3,
                Dict{Vector{Int}, Float64}([1] => 0.9, [2] => 0.3, [1,2] => 0.8)
            )
            actualizar_si_mejor!(mapa, subconfig, resultado, 0.5)
            # [1] should be updated (0.9 > 0.5)
            @test mapa.parciales[[1]].subconfiguracion === subconfig
            @test mapa.parciales[[1]].precision == 0.9
            # [2] should NOT be updated (0.3 <= 0.5)
            @test mapa.parciales[[2]].subconfiguracion === nothing
            # [1,2] should be updated (0.8 > 0.5)
            @test mapa.parciales[[1,2]].subconfiguracion === subconfig
            @test mapa.parciales[[1,2]].precision == 0.8
        end

        @testset "no reemplaza si nueva no es mejor (más neuronas)" begin
            mapa = inicializar_mapa(1, Float64)
            # First: insert a good subconfig with 3 neurons
            sub1 = make_subconfig(Float64, 3)
            res1 = ResultadoEvaluacion{Float64}(
                0.8,
                Dict{Vector{Int}, Float64}([1] => 0.8)
            )
            actualizar_si_mejor!(mapa, sub1, res1, 0.5)
            @test mapa.global_.subconfiguracion === sub1

            # Second: try with worse subconfig (5 neurons, same precision)
            sub2 = make_subconfig(Float64, 5)
            res2 = ResultadoEvaluacion{Float64}(
                0.8,
                Dict{Vector{Int}, Float64}([1] => 0.8)
            )
            actualizar_si_mejor!(mapa, sub2, res2, 0.5)
            # Should still be sub1
            @test mapa.global_.subconfiguracion === sub1
        end

        @testset "reemplaza si nueva es más simple" begin
            mapa = inicializar_mapa(1, Float64)
            sub1 = make_subconfig(Float64, 5)
            res1 = ResultadoEvaluacion{Float64}(
                0.8,
                Dict{Vector{Int}, Float64}([1] => 0.8)
            )
            actualizar_si_mejor!(mapa, sub1, res1, 0.5)

            # Simpler subconfig (2 neurons)
            sub2 = make_subconfig(Float64, 2)
            res2 = ResultadoEvaluacion{Float64}(
                0.7,
                Dict{Vector{Int}, Float64}([1] => 0.7)
            )
            actualizar_si_mejor!(mapa, sub2, res2, 0.5)
            @test mapa.global_.subconfiguracion === sub2
            @test mapa.global_.precision == 0.7
        end

        @testset "empate en neuronas: mayor precisión gana" begin
            mapa = inicializar_mapa(1, Float64)
            sub1 = make_subconfig(Float64, 3)
            res1 = ResultadoEvaluacion{Float64}(
                0.7,
                Dict{Vector{Int}, Float64}([1] => 0.7)
            )
            actualizar_si_mejor!(mapa, sub1, res1, 0.5)

            # Same neurons, higher precision
            sub2 = make_subconfig(Float64, 3)
            res2 = ResultadoEvaluacion{Float64}(
                0.9,
                Dict{Vector{Int}, Float64}([1] => 0.9)
            )
            actualizar_si_mejor!(mapa, sub2, res2, 0.5)
            @test mapa.global_.subconfiguracion === sub2
            @test mapa.global_.precision == 0.9
        end
    end
end
