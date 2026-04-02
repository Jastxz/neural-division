using Test
using DivisionNeuronal
using Base.Threads: Atomic

@testset "Progreso y Cancelación" begin

    @testset "reportar_progreso - callback nothing no hace nada" begin
        mapa = inicializar_mapa(2, Float64)
        # No debe lanzar error
        reportar_progreso(5, 10, mapa, nothing)
    end

    @testset "reportar_progreso - mapa vacío" begin
        mapa = inicializar_mapa(2, Float64)
        resultado = Ref{ProgresoExploracion}()

        callback = p -> (resultado[] = p)
        reportar_progreso(3, 9, mapa, callback)

        p = resultado[]
        @test p.evaluadas == 3
        @test p.total == 9
        @test p.soluciones_globales == 0
        @test p.soluciones_parciales == 0
    end

    @testset "reportar_progreso - con solución global" begin
        mapa = inicializar_mapa(2, Float64)

        # Simular una solución global
        sub = Subconfiguracion{Float64}(
            [1], [1, 2],
            [ones(Float64, 3, 1), ones(Float64, 2, 3)],
            [ones(Float64, 3), ones(Float64, 2)],
            6
        )
        mapa.global_.subconfiguracion = sub
        mapa.global_.precision = 0.8

        resultado = Ref{ProgresoExploracion}()
        callback = p -> (resultado[] = p)
        reportar_progreso(5, 10, mapa, callback)

        p = resultado[]
        @test p.soluciones_globales == 1
        @test p.soluciones_parciales == 0
    end

    @testset "reportar_progreso - con soluciones parciales" begin
        mapa = inicializar_mapa(2, Float64)

        sub = Subconfiguracion{Float64}(
            [1], [1],
            [ones(Float64, 3, 1), ones(Float64, 1, 3)],
            [ones(Float64, 3), ones(Float64, 1)],
            5
        )

        # Asignar solución parcial a la clave [1]
        mapa.parciales[[1]].subconfiguracion = sub
        mapa.parciales[[1]].precision = 0.7

        # Asignar solución parcial a la clave [2]
        sub2 = Subconfiguracion{Float64}(
            [1], [2],
            [ones(Float64, 3, 1), ones(Float64, 1, 3)],
            [ones(Float64, 3), ones(Float64, 1)],
            5
        )
        mapa.parciales[[2]].subconfiguracion = sub2
        mapa.parciales[[2]].precision = 0.6

        resultado = Ref{ProgresoExploracion}()
        callback = p -> (resultado[] = p)
        reportar_progreso(7, 9, mapa, callback)

        p = resultado[]
        @test p.soluciones_globales == 0
        @test p.soluciones_parciales == 2
    end

    @testset "reportar_progreso - con global y parciales" begin
        mapa = inicializar_mapa(2, Float64)

        sub_global = Subconfiguracion{Float64}(
            [1, 2], [1, 2],
            [ones(Float64, 3, 2), ones(Float64, 2, 3)],
            [ones(Float64, 3), ones(Float64, 2)],
            7
        )
        mapa.global_.subconfiguracion = sub_global
        mapa.global_.precision = 0.9

        sub_parcial = Subconfiguracion{Float64}(
            [1], [1],
            [ones(Float64, 3, 1), ones(Float64, 1, 3)],
            [ones(Float64, 3), ones(Float64, 1)],
            5
        )
        mapa.parciales[[1]].subconfiguracion = sub_parcial
        mapa.parciales[[1]].precision = 0.75

        resultado = Ref{ProgresoExploracion}()
        callback = p -> (resultado[] = p)
        reportar_progreso(9, 9, mapa, callback)

        p = resultado[]
        @test p.evaluadas == 9
        @test p.total == 9
        @test p.soluciones_globales == 1
        @test p.soluciones_parciales == 1
    end

    @testset "reportar_progreso - porcentaje evaluadas/total" begin
        mapa = inicializar_mapa(2, Float64)
        resultado = Ref{ProgresoExploracion}()
        callback = p -> (resultado[] = p)

        reportar_progreso(0, 100, mapa, callback)
        @test resultado[].evaluadas == 0
        @test resultado[].total == 100

        reportar_progreso(50, 100, mapa, callback)
        @test resultado[].evaluadas == 50
        @test resultado[].total == 100

        reportar_progreso(100, 100, mapa, callback)
        @test resultado[].evaluadas == 100
        @test resultado[].total == 100
    end

    @testset "debe_parar - señal no activada" begin
        señal = Atomic{Bool}(false)
        @test debe_parar(señal) == false
    end

    @testset "debe_parar - señal activada" begin
        señal = Atomic{Bool}(true)
        @test debe_parar(señal) == true
    end

    @testset "debe_parar - activación dinámica" begin
        señal = Atomic{Bool}(false)
        @test debe_parar(señal) == false

        señal[] = true
        @test debe_parar(señal) == true
    end

end
