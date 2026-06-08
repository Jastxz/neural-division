# Plan de Implementación: Método de la División Neuronal

## Visión General

Implementación incremental en Julia del Método de la División Neuronal. Cada tarea construye sobre las anteriores, comenzando por los tipos base y la validación, avanzando por el generador y evaluador, hasta integrar el motor completo con serialización, progreso y entrenamiento. Se usa PropCheck.jl para tests de propiedades.

## Tareas

- [x] 1. Definir tipos base y excepciones
  - [x] 1.1 Crear el módulo `DivisionNeuronal` con los tipos paramétricos `RedBase{T}`, `Subconfiguracion{T}`, `ResultadoEvaluacion{T}`, `EntradaMapaSoluciones{T}`, `MapaDeSoluciones{T}`, `ConfiguracionDivision{T}` y `ProgresoExploracion`
    - Crear archivo `src/DivisionNeuronal.jl` como módulo principal
    - Crear archivo `src/tipos.jl` con todas las structs paramétricas según el diseño
    - Todos los tipos deben ser paramétricos en `T <: AbstractFloat`
    - `EntradaMapaSoluciones{T}` debe ser `mutable struct`
    - _Requisitos: 1.1, 1.2, 3.1, 5.1_

  - [x] 1.2 Crear las excepciones tipadas `RedBaseNoInicializadaError`, `NeuronasInvalidasError` y `UmbralFueraDeRangoError`
    - Crear archivo `src/errores.jl`
    - Cada excepción hereda de `Exception` y contiene un campo `msg::String`
    - _Requisitos: 1.3, 1.4, 3.3_

  - [x] 1.3 Implementar funciones de validación: `validar_red_base(red::RedBase)`, `validar_neuronas(n_entradas, n_salidas)` y `validar_umbral(umbral)`
    - Crear archivo `src/validacion.jl`
    - `validar_red_base`: verificar que pesos no estén vacíos, dimensiones consistentes entre capas, biases coincidan con dimensiones de pesos
    - `validar_neuronas`: verificar que ambos valores sean enteros positivos (≥ 1)
    - `validar_umbral`: verificar que el valor esté en [0.0, 1.0]; si no se proporciona, usar 0.4 por defecto
    - _Requisitos: 1.1, 1.3, 1.4, 3.1, 3.2, 3.3_

  - [x] 1.4 Escribir test de propiedad para validación de Red Base
    - **Propiedad 1: Validación de Red Base**
    - Implementar generador `gen_red_base(T, max_entradas, max_salidas, max_capas_ocultas)` en `test/generadores.jl`
    - Verificar que redes válidas son aceptadas y redes con pesos vacíos o dimensiones inconsistentes son rechazadas
    - **Valida: Requisitos 1.1, 1.3**

  - [ ]* 1.5 Escribir test de propiedad para validación del umbral de acierto
    - **Propiedad 15: Validación del umbral de acierto**
    - Implementar generador `gen_umbral()` en `test/generadores.jl`
    - Verificar que valores en [0.0, 1.0] son aceptados y valores fuera de rango son rechazados
    - **Valida: Requisitos 3.1, 3.3**


- [x] 2. Implementar el Generador de Subconfiguraciones
  - [x] 2.1 Implementar `GeneradorDeSubconfiguraciones{T}` como iterador lazy
    - Crear archivo `src/generador.jl`
    - Implementar `Base.iterate(gen::GeneradorDeSubconfiguraciones{T}, state)` para el protocolo de iteración de Julia
    - Iterar sobre el producto cartesiano de `powerset(1:n_entradas)` × `powerset(1:n_salidas)` excluyendo conjuntos vacíos
    - Implementar `Base.length(gen)` que devuelva `(2^n_entradas - 1) × (2^n_salidas - 1)`
    - _Requisitos: 2.1_

  - [x] 2.2 Implementar la extracción de subred: función `extraer_subconfiguracion(red_base::RedBase{T}, indices_entrada, indices_salida)::Subconfiguracion{T}`
    - Recortar matrices de pesos por filas/columnas según los índices seleccionados
    - Preservar exactamente los pesos originales de la Red_Base
    - Mantener capas ocultas completas, recortando solo la primera capa (filas de entrada) y la última capa (columnas de salida)
    - Calcular `n_neuronas_activas` como suma de neuronas de entrada + ocultas + salida seleccionadas
    - Descartar subconfiguraciones sin conexiones válidas (matrices resultantes vacías)
    - _Requisitos: 2.2, 2.3, 2.4_

  - [x] 2.3 Escribir test de propiedad para registro de límites de entrada/salida
    - **Propiedad 2: Registro de límites de entrada/salida**
    - Verificar que todas las subconfiguraciones generadas tienen índices de entrada en `1:n_entradas` e índices de salida en `1:n_salidas`
    - **Valida: Requisitos 1.2**

  - [x] 2.4 Escribir test de propiedad para enumeración exhaustiva
    - **Propiedad 3: Enumeración exhaustiva**
    - Verificar que el generador produce exactamente `(2^n_entradas - 1) × (2^n_salidas - 1)` subconfiguraciones (menos las descartadas)
    - **Valida: Requisitos 2.1**

  - [ ]* 2.5 Escribir test de propiedad para preservación de pesos
    - **Propiedad 4: Preservación de pesos**
    - Verificar que los pesos de cada subconfiguración coinciden exactamente con los valores correspondientes de la Red_Base
    - **Valida: Requisitos 2.2, 2.3**

- [x] 3. Checkpoint - Verificar tipos, validación y generador
  - Asegurar que todos los tests pasan, preguntar al usuario si surgen dudas.


- [x] 4. Implementar el Evaluador
  - [x] 4.1 Implementar la función `evaluar(subconfig::Subconfiguracion{T}, datos_validacion, indices_salida_totales::Vector{Int})::ResultadoEvaluacion{T}`
    - Crear archivo `src/evaluador.jl`
    - Realizar forward pass con los pesos de la subconfiguración sobre los datos de validación
    - Calcular precisión global considerando todas las salidas
    - Calcular precisión parcial para cada subconjunto no vacío de salidas del problema
    - Devolver `ResultadoEvaluacion{T}` con ambas métricas
    - _Requisitos: 4.1, 4.2, 4.3_

  - [x] 4.2 Implementar la función de comparación `es_mejor(nueva::Subconfiguracion, nueva_prec::T, actual::EntradaMapaSoluciones{T})::Bool`
    - Añadir en `src/evaluador.jl`
    - Si la entrada actual es `nothing`, devolver `true`
    - Si `nueva.n_neuronas_activas < actual.subconfiguracion.n_neuronas_activas`, devolver `true`
    - Si neuronas iguales y `nueva_prec > actual.precision`, devolver `true`
    - En otro caso, devolver `false`
    - _Requisitos: 5.2, 5.3, 5.4, 5.5_

  - [x] 4.3 Escribir test de propiedad para completitud de evaluación
    - **Propiedad 5: Completitud de evaluación**
    - Implementar generador `gen_datos_validacion(n_entradas, n_salidas, n_muestras)` en `test/generadores.jl`
    - Verificar que el evaluador calcula precisión global y precisiones parciales para todos los subconjuntos
    - **Valida: Requisitos 4.1, 4.2, 4.3**

  - [x] 4.4 Escribir test de propiedad para candidatura por umbral
    - **Propiedad 6: Candidatura por umbral**
    - Verificar que una subconfiguración es candidata si y solo si su precisión supera el umbral
    - **Valida: Requisitos 4.4, 4.5**

  - [ ]* 4.5 Escribir test de propiedad para la función es_mejor
    - **Propiedad 9: Función de comparación es_mejor**
    - Verificar la lógica de comparación: menor neuronas gana, empate se resuelve por mayor precisión
    - **Valida: Requisitos 5.4, 5.5**


- [x] 5. Implementar el Mapa de Soluciones
  - [x] 5.1 Implementar `inicializar_mapa(n_salidas::Int, T::Type)::MapaDeSoluciones{T}` y `actualizar_si_mejor!(mapa, subconfig, resultado, umbral)`
    - Crear archivo `src/mapa_soluciones.jl`
    - `inicializar_mapa`: crear entrada global y una entrada parcial por cada subconjunto no vacío de `1:n_salidas`, todas con `subconfiguracion = nothing`
    - `actualizar_si_mejor!`: para la solución global y cada solución parcial, comprobar si la nueva subconfiguración supera el umbral y es mejor que la almacenada usando `es_mejor`
    - _Requisitos: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [x] 5.2 Escribir test de propiedad para inicialización del Mapa de Soluciones
    - **Propiedad 7: Inicialización del Mapa de Soluciones**
    - Verificar que el mapa inicializado contiene exactamente `2^n - 1` claves para `n` salidas
    - **Valida: Requisitos 5.1**

  - [x] 5.3 Escribir test de propiedad para invariante de mejor-es-más-simple
    - **Propiedad 8: Invariante de mejor-es-más-simple**
    - Verificar que tras procesar una secuencia de subconfiguraciones, el mapa siempre contiene la más simple que supera el umbral
    - **Valida: Requisitos 5.2, 5.3, 5.4, 5.5**

  - [ ]* 5.4 Escribir test de propiedad para completitud del resultado
    - **Propiedad 10: Completitud del resultado**
    - Verificar que cada entrada no vacía del mapa contiene subconfiguración, precisión y número de neuronas activas
    - **Valida: Requisitos 6.2, 6.3**

- [x] 6. Checkpoint - Verificar evaluador y mapa de soluciones
  - Asegurar que todos los tests pasan, preguntar al usuario si surgen dudas.


- [x] 7. Implementar Serialización y Deserialización
  - [x] 7.1 Implementar `serializar(mapa::MapaDeSoluciones{T}, ruta::String)`, `deserializar(ruta::String)::MapaDeSoluciones` y `formatear(mapa::MapaDeSoluciones{T})::String`
    - Crear archivo `src/serializacion.jl`
    - Usar JLD2 para serialización/deserialización a disco
    - `serializar`: guardar el mapa completo en formato JLD2
    - `deserializar`: cargar y reconstruir el `MapaDeSoluciones` desde archivo, lanzar error si el archivo es inválido o corrupto
    - `formatear`: generar representación legible (pretty printer) del mapa
    - _Requisitos: 7.1, 7.2, 7.3_

  - [x] 7.2 Escribir test de propiedad para ida y vuelta de serialización
    - **Propiedad 11: Ida y vuelta de serialización**
    - Implementar generador `gen_mapa_soluciones(n_salidas)` en `test/generadores.jl`
    - Verificar que serializar → deserializar → serializar produce resultado equivalente al original
    - **Valida: Requisitos 7.1, 7.2, 7.3, 7.4**

- [x] 8. Implementar Progreso y Cancelación
  - [x] 8.1 Implementar el reporte de progreso con `ProgresoExploracion` y el mecanismo de cancelación cooperativa con `Atomic{Bool}`
    - Crear archivo `src/progreso.jl`
    - Implementar función `reportar_progreso(evaluadas, total, mapa, callback)` que construya `ProgresoExploracion` y llame al callback
    - El porcentaje se calcula como `evaluadas / total`
    - Contar soluciones globales (entrada global no vacía) y parciales (entradas parciales no vacías) del mapa actual
    - Implementar comprobación de `señal_parada::Atomic{Bool}` entre iteraciones del bucle principal
    - _Requisitos: 8.1, 8.2, 8.3_

  - [x] 8.2 Escribir test de propiedad para precisión del reporte de progreso
    - **Propiedad 12: Precisión del reporte de progreso**
    - Verificar que el porcentaje reportado es `evaluadas / total` y los conteos coinciden con el estado del mapa
    - **Valida: Requisitos 8.1, 8.2**

  - [x] 8.3 Escribir test de propiedad para cancelación ordenada
    - **Propiedad 13: Cancelación ordenada**
    - Verificar que tras señal de parada, el mapa devuelto es válido y sin entradas corruptas
    - **Valida: Requisitos 8.3**


- [x] 9. Implementar Entrenamiento y Métricas Pre/Post
  - [x] 9.1 Implementar `entrenar_y_evaluar!(subconfig::Subconfiguracion{T}, datos_entrenamiento, datos_validacion; epochs=100)::Tuple{T, T}`
    - Crear archivo `src/entrenamiento.jl`
    - Medir precisión pre-entrenamiento usando `evaluar`
    - Entrenar la subconfiguración (descenso de gradiente simple sobre los pesos)
    - Medir precisión post-entrenamiento
    - Devolver tupla `(precision_pre, precision_post)`
    - _Requisitos: 9.1, 9.2_

  - [x] 9.2 Implementar almacenamiento de métricas pre/post en `EntradaMapaSoluciones` y cálculo de diferencia
    - Actualizar `precision_pre_entrenamiento` y `precision_post_entrenamiento` en cada entrada del mapa tras el entrenamiento
    - Calcular diferencia `post - pre` para cada subconfiguración
    - Emitir `@warn` si la precisión post-entrenamiento es inferior a la pre-entrenamiento
    - _Requisitos: 9.3, 9.4, 9.5_

  - [x] 9.3 Escribir test de propiedad para métricas pre/post entrenamiento
    - **Propiedad 14: Métricas pre/post entrenamiento**
    - Verificar que tras entrenamiento, cada entrada contiene precisión pre, post y diferencia calculada
    - **Valida: Requisitos 9.1, 9.2, 9.3, 9.4**

- [x] 10. Checkpoint - Verificar serialización, progreso y entrenamiento
  - Asegurar que todos los tests pasan, preguntar al usuario si surgen dudas.


- [x] 11. Implementar el Motor de División (integración)
  - [x] 11.1 Implementar `ejecutar_division(red_base, datos_validacion, config; callback_progreso, señal_parada)::MapaDeSoluciones{T}`
    - Crear archivo `src/motor.jl`
    - Validar `red_base` con `validar_red_base`, validar neuronas con `validar_neuronas`, validar umbral con `validar_umbral`
    - Inicializar `MapaDeSoluciones` con `inicializar_mapa`
    - Crear `GeneradorDeSubconfiguraciones` e iterar sobre todas las subconfiguraciones
    - Para cada subconfiguración: evaluar con `evaluar`, actualizar mapa con `actualizar_si_mejor!`, reportar progreso
    - Comprobar `señal_parada` entre iteraciones; si activada, salir del bucle y devolver mapa parcial
    - Tras completar exploración: entrenar subredes del mapa con `entrenar_y_evaluar!`
    - Informar con `@info` si ninguna subconfiguración alcanzó el umbral para la solución global
    - _Requisitos: 1.1, 1.2, 1.3, 1.4, 2.1, 3.1, 3.2, 4.1, 4.4, 4.5, 5.1, 5.2, 5.3, 6.1, 6.4, 8.1, 8.2, 8.3, 9.1, 9.2_

  - [x] 11.2 Conectar el módulo `DivisionNeuronal` exportando todas las funciones e interfaces públicas
    - Actualizar `src/DivisionNeuronal.jl` con `include` de todos los archivos fuente
    - Exportar: `ejecutar_division`, `serializar`, `deserializar`, `formatear`, `RedBase`, `ConfiguracionDivision`, `MapaDeSoluciones`, `ProgresoExploracion`
    - _Requisitos: 6.1, 6.2, 6.3_

  - [x] 11.3 Escribir tests unitarios de integración del flujo completo
    - Test con red pequeña (2 entradas, 1 salida) verificando flujo completo de división
    - Test de cancelación durante exploración verificando mapa válido parcial
    - Test de flujo con entrenamiento verificando métricas pre/post
    - Test de caso sin solución global verificando mensaje informativo
    - _Requisitos: 1.1, 2.1, 4.1, 5.1, 6.1, 6.4, 8.3, 9.1_

- [x] 12. Configurar estructura de tests
  - [x] 12.1 Crear `test/runtests.jl` y archivos de test según la estructura del diseño
    - Crear `test/runtests.jl` que incluya todos los archivos de test
    - Crear `test/generadores.jl` con todos los generadores de PropCheck.jl (si no existen ya)
    - Organizar tests en archivos: `test_validacion.jl`, `test_generador.jl`, `test_evaluador.jl`, `test_mapa_soluciones.jl`, `test_serializacion.jl`, `test_progreso.jl`, `test_entrenamiento.jl`, `test_propiedades.jl`
    - Cada test de propiedad debe incluir comentario `# Feature: neural-division, Property N: [descripción]`
    - Configurar PropCheck.jl con mínimo 100 iteraciones por propiedad
    - _Requisitos: todos_

- [x] 13. Checkpoint final - Verificar integración completa
  - Asegurar que todos los tests pasan, preguntar al usuario si surgen dudas.

## Notas

- Las tareas marcadas con `*` son opcionales y pueden omitirse para un MVP más rápido
- Cada tarea referencia los requisitos específicos para trazabilidad
- Los checkpoints aseguran validación incremental
- Los tests de propiedades validan propiedades universales de corrección
- Los tests unitarios validan ejemplos concretos y casos borde
- Se usa PropCheck.jl como librería de property-based testing
