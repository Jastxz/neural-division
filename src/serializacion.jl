"""
Serialización, deserialización y formateo del MapaDeSoluciones.

Usa JLD2 para persistencia a disco en formato nativo de Julia,
garantizando preservación exacta de tipos paramétricos.
"""

using JLD2

"""
    serializar(mapa::MapaDeSoluciones{T}, ruta::String) where T

Serializa el `MapaDeSoluciones` completo a un archivo JLD2 en la ruta especificada.
"""
function serializar(mapa::MapaDeSoluciones{T}, ruta::String) where T
    jldsave(ruta; mapa=mapa)
end

"""
    deserializar(ruta::String)::MapaDeSoluciones

Carga y reconstruye un `MapaDeSoluciones` desde un archivo JLD2.
Lanza un error descriptivo si el archivo no existe, es inválido o está corrupto.
"""
function deserializar(ruta::String)::MapaDeSoluciones
    try
        data = load(ruta)
        if !haskey(data, "mapa")
            error("El archivo no contiene un MapaDeSoluciones válido (clave 'mapa' no encontrada)")
        end
        mapa = data["mapa"]
        if !(mapa isa MapaDeSoluciones)
            error("El objeto deserializado no es un MapaDeSoluciones")
        end
        return mapa
    catch e
        if e isa ErrorException
            rethrow(e)
        end
        error("No se pudo deserializar el archivo '$ruta': $e")
    end
end

"""
    formatear(mapa::MapaDeSoluciones{T})::String where T

Genera una representación legible (pretty print) del `MapaDeSoluciones`.
Muestra la solución global y cada solución parcial con su precisión
y número de neuronas activas.
"""
function formatear(mapa::MapaDeSoluciones{T})::String where T
    io = IOBuffer()

    println(io, "╔══════════════════════════════════════════╗")
    println(io, "║        Mapa de Soluciones ($T)        ║")
    println(io, "╠══════════════════════════════════════════╣")

    # Referencia completa
    println(io, "║ Referencia Completa:")
    _formatear_entrada(io, mapa.referencia_completa)

    # Solución global
    println(io, "║──────────────────────────────────────────")
    println(io, "║ Mejor Subred Global:")
    _formatear_entrada(io, mapa.global_)

    # Soluciones parciales
    claves_ordenadas = sort(collect(keys(mapa.parciales)), by=k -> (length(k), k))
    if !isempty(claves_ordenadas)
        println(io, "║──────────────────────────────────────────")
        println(io, "║ Soluciones Parciales:")
        for clave in claves_ordenadas
            entrada = mapa.parciales[clave]
            println(io, "║   Salidas $clave:")
            _formatear_entrada(io, entrada, "    ")
        end
    end

    println(io, "╚══════════════════════════════════════════╝")

    return String(take!(io))
end

"""
    _formatear_entrada(io::IOBuffer, entrada::EntradaMapaSoluciones{T}, indent::String="  ") where T

Formatea una entrada individual del mapa de soluciones.
"""
function _formatear_entrada(io::IOBuffer, entrada::EntradaMapaSoluciones{T}, indent::String="  ") where T
    if entrada.subconfiguracion === nothing
        println(io, "║ $(indent)Sin solución encontrada")
    else
        sc = entrada.subconfiguracion
        println(io, "║ $(indent)Entradas: $(sc.indices_entrada)")
        println(io, "║ $(indent)Salidas:  $(sc.indices_salida)")
        println(io, "║ $(indent)Neuronas activas: $(sc.n_neuronas_activas)")
        println(io, "║ $(indent)Precisión: $(entrada.precision)")
        if entrada.precision_pre_entrenamiento != T(0) || entrada.precision_post_entrenamiento != T(0)
            println(io, "║ $(indent)Pre-entrenamiento:  $(entrada.precision_pre_entrenamiento)")
            println(io, "║ $(indent)Post-entrenamiento: $(entrada.precision_post_entrenamiento)")
        end
    end
end
