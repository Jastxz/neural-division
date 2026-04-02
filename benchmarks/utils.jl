"""
Utilidades compartidas para los benchmarks del Método de la División Neuronal.
"""

using DivisionNeuronal
using Random
using Printf
using Statistics

"""
Crea una RedBase{Float64} con pesos aleatorios.
"""
function crear_red(capas::Vector{Int}; seed::Int=42)
    Random.seed!(seed)
    n_entradas = capas[1]
    n_salidas = capas[end]
    pesos = Matrix{Float64}[]
    biases = Vector{Float64}[]
    for i in 1:(length(capas) - 1)
        push!(pesos, randn(Float64, capas[i], capas[i + 1]))
        push!(biases, randn(Float64, capas[i + 1]))
    end
    RedBase{Float64}(pesos, biases, n_entradas, n_salidas)
end

"""
Genera todas las combinaciones binarias de `n` bits como matriz (2^n × n).
"""
function tabla_verdad(n::Int)
    filas = 2^n
    m = zeros(Float64, filas, n)
    for i in 0:(filas - 1)
        for j in 0:(n - 1)
            m[i + 1, n - j] = Float64((i >> j) & 1)
        end
    end
    m
end

const SEEDS_20 = [42, 123, 7, 2024, 31415, 1, 99, 256, 777, 1337,
                  5555, 8080, 9999, 12345, 54321, 65536, 100000,
                  271828, 314159, 999999]

const SEEDS_100 = vcat(SEEDS_20, collect(200001:200080))

# --- Problemas ---

function problema_xor()
    e = tabla_verdad(2)
    s = reshape(Float64[0, 1, 1, 0], 4, 1)
    (entradas=e, salidas=s)
end

function problema_and()
    e = tabla_verdad(2)
    s = reshape(Float64[0, 0, 0, 1], 4, 1)
    (entradas=e, salidas=s)
end

function problema_or()
    e = tabla_verdad(2)
    s = reshape(Float64[0, 1, 1, 1], 4, 1)
    (entradas=e, salidas=s)
end

function problema_nand()
    e = tabla_verdad(2)
    s = reshape(Float64[1, 1, 1, 0], 4, 1)
    (entradas=e, salidas=s)
end

function problema_multi_logica()
    e = tabla_verdad(2)
    s = Float64[0 0 0; 0 1 1; 0 1 1; 1 1 0]
    (entradas=e, salidas=s)
end

function problema_paridad_3bits()
    e = tabla_verdad(3)
    s = reshape(Float64[0, 1, 1, 0, 1, 0, 0, 1], 8, 1)
    (entradas=e, salidas=s)
end

function problema_encoder_4a2()
    e = Float64[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]
    s = Float64[0 0; 0 1; 1 0; 1 1]
    (entradas=e, salidas=s)
end

"""
Ejecuta la división para un problema con una semilla dada.
Devuelve NamedTuple con métricas o nothing si no encontró solución global.
"""
function ejecutar_una(datos, capas::Vector{Int}, seed::Int; umbral=0.5, epochs=2000)
    red = crear_red(capas; seed=seed)
    config = ConfiguracionDivision{Float64}(umbral)
    t = @elapsed mapa = ejecutar_division(red, datos, config;
        datos_entrenamiento=datos, epochs=epochs)
    g = mapa.global_
    if g.subconfiguracion !== nothing
        return (encontrada=true,
                neuronas=g.subconfiguracion.n_neuronas_activas,
                n_entradas=length(g.subconfiguracion.indices_entrada),
                prec_exp=g.precision,
                prec_pre=g.precision_pre_entrenamiento,
                prec_post=g.precision_post_entrenamiento,
                tiempo=t,
                mapa=mapa)
    else
        return (encontrada=false, neuronas=0, n_entradas=0,
                prec_exp=0.0, prec_pre=0.0, prec_post=0.0,
                tiempo=t, mapa=mapa)
    end
end
