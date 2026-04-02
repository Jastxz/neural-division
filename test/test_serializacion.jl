using DivisionNeuronal
using Test

@testset "Serialización y Deserialización" begin
    # Helper: crear un MapaDeSoluciones de prueba con datos reales
    function crear_mapa_prueba(::Type{T}) where T
        mapa = inicializar_mapa(2, T)

        # Crear una subconfiguración de prueba
        pesos = [T[0.5 0.3; 0.1 0.7], T[0.2 0.8; 0.4 0.6]]
        biases = [T[0.1, 0.2], T[0.3, 0.4]]
        subconfig = Subconfiguracion{T}(
            [1, 2], [1, 2], pesos, biases, 6
        )

        # Actualizar la entrada global
        mapa.global_.subconfiguracion = subconfig
        mapa.global_.precision = T(0.85)

        # Actualizar una entrada parcial
        entrada_parcial = mapa.parciales[[1]]
        entrada_parcial.subconfiguracion = subconfig
        entrada_parcial.precision = T(0.9)

        return mapa
    end

    @testset "serializar y deserializar - ida y vuelta" begin
        mapa_original = crear_mapa_prueba(Float64)
        ruta = tempname() * ".jld2"

        try
            serializar(mapa_original, ruta)
            @test isfile(ruta)

            mapa_cargado = deserializar(ruta)
            @test mapa_cargado isa MapaDeSoluciones{Float64}

            # Verificar solución global
            @test mapa_cargado.global_.precision ≈ mapa_original.global_.precision
            @test mapa_cargado.global_.subconfiguracion !== nothing
            @test mapa_cargado.global_.subconfiguracion.indices_entrada == [1, 2]
            @test mapa_cargado.global_.subconfiguracion.indices_salida == [1, 2]
            @test mapa_cargado.global_.subconfiguracion.n_neuronas_activas == 6

            # Verificar pesos preservados
            for i in eachindex(mapa_original.global_.subconfiguracion.pesos)
                @test mapa_cargado.global_.subconfiguracion.pesos[i] ≈ mapa_original.global_.subconfiguracion.pesos[i]
            end

            # Verificar soluciones parciales
            @test length(mapa_cargado.parciales) == length(mapa_original.parciales)
            @test mapa_cargado.parciales[[1]].precision ≈ 0.9
        finally
            isfile(ruta) && rm(ruta)
        end
    end

    @testset "serializar y deserializar - Float32" begin
        mapa_original = crear_mapa_prueba(Float32)
        ruta = tempname() * ".jld2"

        try
            serializar(mapa_original, ruta)
            mapa_cargado = deserializar(ruta)

            @test mapa_cargado isa MapaDeSoluciones{Float32}
            @test mapa_cargado.global_.precision ≈ Float32(0.85)
        finally
            isfile(ruta) && rm(ruta)
        end
    end

    @testset "serializar y deserializar - mapa vacío" begin
        mapa = inicializar_mapa(3, Float64)
        ruta = tempname() * ".jld2"

        try
            serializar(mapa, ruta)
            mapa_cargado = deserializar(ruta)

            @test mapa_cargado isa MapaDeSoluciones{Float64}
            @test mapa_cargado.global_.subconfiguracion === nothing
            @test length(mapa_cargado.parciales) == 7  # 2^3 - 1
        finally
            isfile(ruta) && rm(ruta)
        end
    end

    @testset "deserializar - archivo inexistente" begin
        @test_throws ErrorException deserializar("/tmp/no_existe_$(rand(UInt32)).jld2")
    end

    @testset "deserializar - archivo corrupto" begin
        ruta = tempname() * ".jld2"
        try
            write(ruta, "datos corruptos que no son JLD2")
            @test_throws ErrorException deserializar(ruta)
        finally
            isfile(ruta) && rm(ruta)
        end
    end

    @testset "formatear - mapa con soluciones" begin
        mapa = crear_mapa_prueba(Float64)
        resultado = formatear(mapa)

        @test resultado isa String
        @test !isempty(resultado)
        @test occursin("Mejor Subred Global", resultado)
        @test occursin("Soluciones Parciales", resultado)
        @test occursin("Neuronas activas: 6", resultado)
        @test occursin("0.85", resultado)
    end

    @testset "formatear - mapa vacío" begin
        mapa = inicializar_mapa(2, Float64)
        resultado = formatear(mapa)

        @test resultado isa String
        @test occursin("Sin solución encontrada", resultado)
    end

    @testset "doble ida y vuelta produce resultado equivalente" begin
        mapa_original = crear_mapa_prueba(Float64)
        ruta1 = tempname() * ".jld2"
        ruta2 = tempname() * ".jld2"

        try
            # Primera ida y vuelta
            serializar(mapa_original, ruta1)
            mapa_intermedio = deserializar(ruta1)

            # Segunda ida y vuelta
            serializar(mapa_intermedio, ruta2)
            mapa_final = deserializar(ruta2)

            # Verificar equivalencia
            @test mapa_final.global_.precision ≈ mapa_original.global_.precision
            @test mapa_final.global_.subconfiguracion.n_neuronas_activas == mapa_original.global_.subconfiguracion.n_neuronas_activas
            @test length(mapa_final.parciales) == length(mapa_original.parciales)

            for (clave, entrada_orig) in mapa_original.parciales
                @test haskey(mapa_final.parciales, clave)
                entrada_final = mapa_final.parciales[clave]
                @test entrada_final.precision ≈ entrada_orig.precision
                if entrada_orig.subconfiguracion !== nothing
                    @test entrada_final.subconfiguracion !== nothing
                    @test entrada_final.subconfiguracion.indices_entrada == entrada_orig.subconfiguracion.indices_entrada
                end
            end
        finally
            isfile(ruta1) && rm(ruta1)
            isfile(ruta2) && rm(ruta2)
        end
    end
end
