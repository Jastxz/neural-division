"""
Progreso y cancelación cooperativa para el Método de la División Neuronal.

Proporciona funciones para reportar el progreso de la exploración de subconfiguraciones
y para comprobar si se ha solicitado la parada del proceso.
"""

using Base.Threads: Atomic

"""
    reportar_progreso(evaluadas::Int, total::Int, mapa::MapaDeSoluciones{T}, callback::Union{Nothing, Function}) where T

Construye un `ProgresoExploracion` con el estado actual de la exploración y lo pasa al callback.

- El porcentaje se calcula como `evaluadas / total`
- Cuenta soluciones globales: 1 si la entrada global tiene subconfiguración, 0 en caso contrario
- Cuenta soluciones parciales: número de entradas parciales con subconfiguración no vacía

Si `callback` es `nothing`, no hace nada.
"""
function reportar_progreso(evaluadas::Int, total::Int, mapa::MapaDeSoluciones{T}, callback::Union{Nothing, Function}) where T
    callback === nothing && return

    # Contar soluciones globales (0 o 1)
    soluciones_globales = mapa.global_.subconfiguracion !== nothing ? 1 : 0

    # Contar soluciones parciales
    soluciones_parciales = count(e -> e.subconfiguracion !== nothing, values(mapa.parciales))

    progreso = ProgresoExploracion(evaluadas, total, soluciones_globales, soluciones_parciales)
    callback(progreso)
end

"""
    debe_parar(señal_parada::Atomic{Bool})::Bool

Comprueba si se ha solicitado la parada cooperativa del proceso.

Devuelve `true` si `señal_parada` está activada, `false` en caso contrario.
Se debe llamar entre iteraciones del bucle principal de exploración.
"""
function debe_parar(señal_parada::Atomic{Bool})::Bool
    return señal_parada[]
end
