# Excepciones tipadas para el módulo DivisionNeuronal

struct RedBaseNoInicializadaError <: Exception
    msg::String
end

struct NeuronasInvalidasError <: Exception
    msg::String
end

struct UmbralFueraDeRangoError <: Exception
    msg::String
end
