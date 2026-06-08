# Documento de Requisitos: Método de la División Neuronal

## Introducción

El Método de la División Neuronal es un enfoque de búsqueda y poda de arquitecturas de redes neuronales. Partiendo de una red inicializada con pesos aleatorios, el método explora de forma exhaustiva todas las posibles subconfiguraciones de dicha red. El objetivo es encontrar subredes más simples que resuelvan el problema completo o partes específicas del mismo (subconjuntos de salidas). Las subredes resultantes se entrenan al final del proceso. El resultado es una o varias redes neuronales eficientes y ajustadas al problema concreto. La implementación se realizará en Julia.

## Glosario

- **Red_Base**: Red neuronal inicializada con pesos aleatorios que sirve como punto de partida para el proceso de división.
- **Subconfiguración**: Subconjunto de neuronas y conexiones extraído de la Red_Base, que forma una subred funcional capaz de realizar predicciones.
- **Mapa_De_Soluciones**: Estructura de datos tipo diccionario cuyas claves son la solución global y cada subconjunto posible de salidas del problema, y cuyos valores son la mejor Subconfiguración encontrada para cada clave.
- **Umbral_De_Acierto**: Valor numérico configurable entre 0.0 y 1.0 que define la precisión mínima que una Subconfiguración debe alcanzar para ser considerada válida.
- **Solución_Global**: Entrada en el Mapa_De_Soluciones que representa la Subconfiguración que resuelve el problema completo (todas las salidas).
- **Solución_Parcial**: Entrada en el Mapa_De_Soluciones que representa la Subconfiguración que resuelve un subconjunto específico de salidas.
- **Generador_De_Subconfiguraciones**: Componente que enumera todas las posibles Subconfiguraciones de la Red_Base.
- **Evaluador**: Componente que mide la precisión de una Subconfiguración sobre un conjunto de datos de validación.
- **Motor_De_División**: Componente principal que orquesta el proceso completo de generación, evaluación y almacenamiento de Subconfiguraciones.

## Requisitos

### Requisito 1: Definición de la Red Base

**Historia de Usuario:** Como investigador, quiero proporcionar una red neuronal inicializada junto con el número máximo de neuronas de entrada y salida, para que el sistema pueda iniciar el proceso de división neuronal.

#### Criterios de Aceptación

1. WHEN el usuario proporciona una Red_Base inicializada, THEN THE Motor_De_División SHALL validar que la Red_Base contiene pesos inicializados y una arquitectura definida.
2. WHEN el usuario especifica el número máximo de neuronas de entrada y de salida, THEN THE Motor_De_División SHALL registrar dichos valores como límites para la generación de Subconfiguraciones.
3. IF la Red_Base proporcionada no contiene pesos inicializados, THEN THE Motor_De_División SHALL devolver un error descriptivo indicando que la red debe estar inicializada previamente.
4. IF el número de neuronas de entrada o salida especificado es menor que 1, THEN THE Motor_De_División SHALL devolver un error indicando que los valores deben ser enteros positivos.

### Requisito 2: Generación de Subconfiguraciones

**Historia de Usuario:** Como investigador, quiero que el sistema genere todas las posibles subconfiguraciones de la red base, para poder explorar de forma exhaustiva las subredes candidatas.

#### Criterios de Aceptación

1. WHEN el Motor_De_División inicia el proceso de exploración, THE Generador_De_Subconfiguraciones SHALL enumerar todas las combinaciones posibles de subconjuntos de neuronas de entrada y neuronas de salida dentro de los límites definidos.
2. THE Generador_De_Subconfiguraciones SHALL producir cada Subconfiguración como una subred funcional capaz de realizar predicciones, preservando los pesos originales de la Red_Base.
3. THE Generador_De_Subconfiguraciones SHALL incluir las conexiones de las capas ocultas correspondientes a las neuronas de entrada y salida seleccionadas en cada Subconfiguración.
4. IF una Subconfiguración generada resulta en una red sin conexiones válidas entre entrada y salida, THEN THE Generador_De_Subconfiguraciones SHALL descartar dicha Subconfiguración.

### Requisito 3: Configuración del Umbral de Acierto

**Historia de Usuario:** Como investigador, quiero definir un umbral de acierto configurable, para controlar la calidad mínima exigida a las subconfiguraciones candidatas.

#### Criterios de Aceptación

1. THE Motor_De_División SHALL aceptar un Umbral_De_Acierto como parámetro de configuración con un valor entre 0.0 y 1.0.
2. IF el usuario no proporciona un Umbral_De_Acierto, THEN THE Motor_De_División SHALL utilizar un valor por defecto de 0.4.
3. IF el Umbral_De_Acierto proporcionado está fuera del rango 0.0 a 1.0, THEN THE Motor_De_División SHALL devolver un error indicando el rango válido.

### Requisito 4: Evaluación de Subconfiguraciones

**Historia de Usuario:** Como investigador, quiero que cada subconfiguración sea evaluada contra un conjunto de datos de validación, para determinar su capacidad de resolver el problema completo o parcial.

#### Criterios de Aceptación

1. WHEN una Subconfiguración es generada, THE Evaluador SHALL calcular la precisión de dicha Subconfiguración sobre el conjunto de datos de validación proporcionado.
2. THE Evaluador SHALL calcular la precisión global de la Subconfiguración considerando todas las salidas del problema.
3. THE Evaluador SHALL calcular la precisión parcial de la Subconfiguración para cada subconjunto posible de salidas del problema.
4. WHEN la precisión global de una Subconfiguración supera el Umbral_De_Acierto, THE Evaluador SHALL marcar dicha Subconfiguración como candidata a Solución_Global.
5. WHEN la precisión parcial de una Subconfiguración para un subconjunto de salidas supera el Umbral_De_Acierto, THE Evaluador SHALL marcar dicha Subconfiguración como candidata a Solución_Parcial para ese subconjunto.

### Requisito 5: Almacenamiento en el Mapa de Soluciones

**Historia de Usuario:** Como investigador, quiero que las mejores subconfiguraciones se almacenen en un mapa de soluciones, para obtener al final del proceso las redes más eficientes para cada parte del problema.

#### Criterios de Aceptación

1. THE Motor_De_División SHALL inicializar el Mapa_De_Soluciones con una clave para la Solución_Global y una clave para cada subconjunto posible de salidas.
2. WHEN una Subconfiguración candidata a Solución_Global es más simple que la Solución_Global almacenada actualmente y supera el Umbral_De_Acierto, THE Motor_De_División SHALL reemplazar la Solución_Global almacenada por la nueva Subconfiguración.
3. WHEN una Subconfiguración candidata a Solución_Parcial para un subconjunto de salidas es más simple que la Solución_Parcial almacenada actualmente para ese subconjunto y supera el Umbral_De_Acierto, THE Motor_De_División SHALL reemplazar la Solución_Parcial almacenada por la nueva Subconfiguración.
4. THE Motor_De_División SHALL definir "más simple" como una Subconfiguración con menor número total de neuronas activas.
5. IF dos Subconfiguraciones tienen el mismo número de neuronas activas, THEN THE Motor_De_División SHALL conservar la que tenga mayor precisión.

### Requisito 6: Resultado del Proceso de División

**Historia de Usuario:** Como investigador, quiero obtener como resultado final las redes neuronales optimizadas, para poder utilizarlas en producción o análisis posterior.

#### Criterios de Aceptación

1. WHEN el proceso de exploración finaliza, THE Motor_De_División SHALL devolver el Mapa_De_Soluciones completo con todas las mejores Subconfiguraciones encontradas.
2. THE Motor_De_División SHALL incluir en el resultado la precisión alcanzada por cada Subconfiguración almacenada en el Mapa_De_Soluciones.
3. THE Motor_De_División SHALL incluir en el resultado el número de neuronas activas de cada Subconfiguración almacenada.
4. IF el Mapa_De_Soluciones no contiene ninguna Subconfiguración válida para la Solución_Global, THEN THE Motor_De_División SHALL informar al usuario de que ninguna Subconfiguración alcanzó el Umbral_De_Acierto para el problema completo.

### Requisito 7: Serialización y Deserialización de Resultados

**Historia de Usuario:** Como investigador, quiero poder guardar y cargar los resultados del proceso de división, para poder reutilizarlos sin repetir el cómputo.

#### Criterios de Aceptación

1. THE Motor_De_División SHALL serializar el Mapa_De_Soluciones a un formato de archivo persistente.
2. WHEN se proporciona un archivo serializado válido, THE Motor_De_División SHALL deserializar el contenido y reconstruir el Mapa_De_Soluciones original.
3. THE Motor_De_División SHALL formatear el Mapa_De_Soluciones serializado de vuelta a un archivo válido (pretty printer).
4. FOR ALL Mapa_De_Soluciones válidos, serializar y luego deserializar y luego serializar de nuevo SHALL producir un resultado equivalente al original (propiedad de ida y vuelta).

### Requisito 8: Progreso y Monitorización del Proceso

**Historia de Usuario:** Como investigador, quiero poder monitorizar el progreso del proceso de exploración, para estimar el tiempo restante y verificar que el sistema está funcionando correctamente.

#### Criterios de Aceptación

1. WHILE el Motor_De_División está explorando Subconfiguraciones, THE Motor_De_División SHALL reportar el progreso como porcentaje de Subconfiguraciones evaluadas respecto al total.
2. WHILE el Motor_De_División está explorando Subconfiguraciones, THE Motor_De_División SHALL reportar el número de Soluciones_Globales y Soluciones_Parciales encontradas hasta el momento.
3. WHEN el usuario solicita detener el proceso, THE Motor_De_División SHALL finalizar la exploración de forma ordenada y devolver el Mapa_De_Soluciones con los resultados obtenidos hasta ese momento.

### Requisito 9: Medición de Rendimiento Antes y Después del Entrenamiento

**Historia de Usuario:** Como investigador, quiero medir el rendimiento de las subredes resultantes antes y después de entrenarlas, para poder comparar y validar la mejora obtenida por el entrenamiento.

#### Criterios de Aceptación

1. WHEN el proceso de exploración finaliza y se obtienen las Subconfiguraciones del Mapa_De_Soluciones, THE Evaluador SHALL medir la precisión de cada Subconfiguración antes de aplicar el entrenamiento.
2. WHEN las Subconfiguraciones del Mapa_De_Soluciones son entrenadas, THE Evaluador SHALL medir la precisión de cada Subconfiguración después del entrenamiento.
3. THE Motor_De_División SHALL almacenar la precisión pre-entrenamiento y la precisión post-entrenamiento de cada Subconfiguración en el Mapa_De_Soluciones.
4. THE Motor_De_División SHALL calcular la diferencia de precisión entre pre-entrenamiento y post-entrenamiento para cada Subconfiguración almacenada.
5. IF la precisión post-entrenamiento de una Subconfiguración es inferior a la precisión pre-entrenamiento, THEN THE Motor_De_División SHALL emitir una advertencia indicando que el entrenamiento no produjo mejora para dicha Subconfiguración.
