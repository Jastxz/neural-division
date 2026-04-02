"""
Motor de División Neuronal — orquestador principal del proceso.

Coordina la generación, evaluación y almacenamiento de subconfiguraciones,
con soporte para cancelación cooperativa, reporte de progreso y entrenamiento
final de las subredes encontradas.
"""

using Base.Threads: Atomic

"""
    ejecutar_division(red_base, datos_validacion, config; ...) -> MapaDeSoluciones{T}

Punto de entrada principal del Método de la División Neuronal.

Parámetros posicionales:
- `red_base::RedBase{T}`: Red neuronal inicializada con pesos aleatorios.
- `datos_validacion`: NamedTuple con campos `entradas::Matrix{T}` y `salidas::Matrix{T}`.
- `config::ConfiguracionDivision{T}`: Configuración con el umbral de acierto.

Parámetros con nombre:
- `datos_entrenamiento`: Datos para entrenar las subredes (por defecto = `datos_validacion`).
- `callback_progreso`: Función `(ProgresoExploracion) -> nothing` llamada tras cada evaluación.
- `señal_parada`: `Atomic{Bool}` para cancelación cooperativa.
- `epochs`: Número de épocas de entrenamiento (por defecto 100).

Devuelve un `MapaDeSoluciones{T}` con las mejores subconfiguraciones encontradas.
"""
function ejecutar_division(
    red_base::RedBase{T},
    datos_validacion,
    config::ConfiguracionDivision{T};
    datos_entrenamiento = datos_validacion,
    callback_progreso::Union{Nothing, Function} = nothing,
    señal_parada::Atomic{Bool} = Atomic{Bool}(false),
    epochs::Int = 1000,
    lr::T = T(0.001),
    batch_size::Int = 32,
    paciencia::Int = 50
)::MapaDeSoluciones{T} where T
    # 1. Validar entradas
    validar_red_base(red_base)
    validar_neuronas(red_base.n_entradas, red_base.n_salidas)
    validar_umbral(config.umbral_de_acierto)

    # 2. Inicializar mapa de soluciones con referencia completa
    indices_entrada_totales = collect(1:red_base.n_entradas)
    indices_salida_totales = collect(1:red_base.n_salidas)
    referencia = extraer_subconfiguracion(red_base, indices_entrada_totales, indices_salida_totales)
    mapa = inicializar_mapa(red_base.n_salidas, T; referencia=referencia)

    # 3. Crear generador e iterar sobre subconfiguraciones
    generador = GeneradorDeSubconfiguraciones(red_base)
    total = length(generador)
    evaluadas = 0

    for (idx_ent, idx_sal) in generador
        # Comprobar señal de parada
        if debe_parar(señal_parada)
            break
        end

        # Extraer subconfiguración
        subconfig = extraer_subconfiguracion(red_base, idx_ent, idx_sal)
        if subconfig === nothing
            continue
        end

        # Evaluar
        resultado = evaluar(subconfig, datos_validacion, indices_salida_totales)

        # Actualizar mapa si es mejor
        actualizar_si_mejor!(mapa, subconfig, resultado, config.umbral_de_acierto)

        evaluadas += 1

        # Reportar progreso
        reportar_progreso(evaluadas, total, mapa, callback_progreso)
    end

    # 4. Informar si no se encontró solución global
    if mapa.global_.subconfiguracion === nothing
        @info "Ninguna subconfiguración alcanzó el umbral de acierto para la solución global"
    end

    # 5. Entrenar subredes (siempre, incluso tras cancelación parcial)
    entrenar_mapa!(mapa, datos_entrenamiento, datos_validacion;
        epochs=epochs, lr=lr, batch_size=batch_size, paciencia=paciencia)

    return mapa
end
