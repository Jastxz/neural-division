"""
Funciones de validación para el módulo DivisionNeuronal.

Validan la Red Base, neuronas de entrada/salida y umbral de acierto
antes de iniciar el proceso de división neuronal.
"""

"""
    validar_red_base(red::RedBase)

Valida que la Red Base contenga pesos inicializados con dimensiones consistentes.

Verifica:
- Que el vector de pesos no esté vacío
- Que el número de capas de pesos y biases coincida
- Que las dimensiones entre capas consecutivas sean consistentes
  (columnas de capa i == filas de capa i+1)
- Que cada bias tenga longitud igual al número de columnas de su capa

Lanza `RedBaseNoInicializadaError` si alguna condición no se cumple.
"""
function validar_red_base(red::RedBase)
    # Verificar que los pesos no estén vacíos
    if isempty(red.pesos)
        throw(RedBaseNoInicializadaError(
            "La Red_Base debe contener pesos inicializados con dimensiones consistentes"))
    end

    # Verificar que el número de capas de pesos y biases coincida
    if length(red.pesos) != length(red.biases)
        throw(RedBaseNoInicializadaError(
            "La Red_Base debe contener pesos inicializados con dimensiones consistentes"))
    end

    # Verificar dimensiones consistentes entre capas consecutivas
    for i in 1:(length(red.pesos) - 1)
        cols_actual = size(red.pesos[i], 2)
        filas_siguiente = size(red.pesos[i + 1], 1)
        if cols_actual != filas_siguiente
            throw(RedBaseNoInicializadaError(
                "La Red_Base debe contener pesos inicializados con dimensiones consistentes"))
        end
    end

    # Verificar que cada bias coincida con las columnas de su capa de pesos
    for i in 1:length(red.pesos)
        cols = size(red.pesos[i], 2)
        if length(red.biases[i]) != cols
            throw(RedBaseNoInicializadaError(
                "La Red_Base debe contener pesos inicializados con dimensiones consistentes"))
        end
    end

    return nothing
end

"""
    validar_neuronas(n_entradas, n_salidas)

Valida que el número de neuronas de entrada y salida sean enteros positivos (≥ 1).

Lanza `NeuronasInvalidasError` si alguno de los valores es menor que 1.
"""
function validar_neuronas(n_entradas, n_salidas)
    if n_entradas < 1 || n_salidas < 1
        throw(NeuronasInvalidasError(
            "El número de neuronas de entrada y salida debe ser un entero positivo"))
    end
    return nothing
end

"""
    validar_umbral(umbral)

Valida que el umbral de acierto esté en el rango [0.0, 1.0].

Lanza `UmbralFueraDeRangoError` si el valor está fuera del rango.
"""
function validar_umbral(umbral)
    if umbral < 0.0 || umbral > 1.0
        throw(UmbralFueraDeRangoError(
            "El Umbral_De_Acierto debe estar entre 0.0 y 1.0"))
    end
    return nothing
end

"""
    validar_umbral()

Versión sin argumentos que devuelve el valor por defecto de 0.4.
"""
function validar_umbral()
    return 0.4
end
