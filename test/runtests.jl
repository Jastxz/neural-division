using Test

@testset "DivisionNeuronal" begin
    include("test_validacion.jl")
    include("test_generador.jl")
    include("test_evaluador.jl")
    include("test_mapa_soluciones.jl")
    include("test_serializacion.jl")
    include("test_progreso.jl")
    include("test_entrenamiento.jl")
    include("test_propiedades.jl")
    include("test_integracion.jl")
end
