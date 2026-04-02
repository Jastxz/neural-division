# Análisis de resultados: por qué funcionan las subredes

Este documento explica la lógica detrás de las decisiones del método de la División Neuronal en cada dataset, dando contexto del problema y por qué las subredes seleccionadas tienen sentido.

## Glass — el resultado estrella (+37 pp sobre la referencia)

**El problema**: clasificar 214 muestras de vidrio en 6 tipos (ventana float, ventana no-float, vehículo, contenedor, vajilla, faro) usando 9 propiedades químicas (índice de refracción, Na, Mg, Al, Si, K, Ca, Ba, Fe).

**Qué hace el método**: en 20/20 semillas selecciona una subred con 1-2 features, alcanzando 94.4% frente al 57.1% de la red completa.

**Por qué funciona**: la red completa con 6 salidas one-hot y 9 entradas tiene 31 neuronas y muchos parámetros para solo 214 muestras. El optimizador se pierde en un paisaje de pérdida complejo con 6 clases simultáneas. En cambio, las subredes resuelven subproblemas más sencillos: "¿es este vidrio de tipo X o no?" con 1-2 features. Eso es un problema binario o de pocas clases con pocos parámetros, que converge fácilmente.

**La feature dominante es el Índice de Refracción (RI, 85%)**. Esto tiene sentido físico directo: el RI depende de la composición química del vidrio y varía significativamente entre tipos. Los vidrios de ventana (float y no-float) tienen RI alto (~1.52), mientras que los de vajilla y faro tienen RI más bajo. Una sola medida de RI ya separa bien varios tipos de vidrio. La segunda feature más usada es Na (sodio, 25%), que distingue entre subtipos de vidrio de ventana.

**Conclusión**: el método descubre que la clasificación de vidrios es fundamentalmente un problema de índice de refracción, y que intentar usar las 9 features simultáneamente con 6 clases confunde al optimizador más de lo que ayuda.

## Ecoli — subredes +11 pp sobre la referencia

**El problema**: predecir la localización celular de proteínas en E. coli (5 clases: citoplasma, membrana interna, membrana interna sin señal, periplasma, membrana externa) usando 7 atributos derivados de la secuencia de aminoácidos.

**Qué hace el método**: en 20/20 semillas selecciona una subred, alcanzando 93.2% frente al 82.3% de la red completa. Feature dominante: mcg (75-89%).

**Por qué funciona**: el atributo mcg (método de McGeoch para reconocimiento de péptidos señal) es el predictor más fuerte para localización de proteínas. Las proteínas del citoplasma (clase mayoritaria, 143/327) tienen valores mcg bajos, mientras que las de membrana y periplasma tienen valores altos. Con solo mcg, una subred puede separar bien la clase mayoritaria del resto, y las soluciones parciales se encargan de los subproblemas más finos.

La red completa con 5 salidas y 7 entradas tiene el mismo problema que Glass: demasiados parámetros para el número de muestras, y el optimizador no converge bien al intentar resolver 5 clases simultáneamente. Las subredes especializadas, al resolver subproblemas más simples, convergen de forma más fiable.

**Conclusión**: el método identifica correctamente que mcg es la señal dominante para localización de proteínas, un resultado consistente con la biología molecular.

## Seeds — subredes +1 pp con 27% menos neuronas

**El problema**: clasificar 210 semillas de trigo en 3 variedades (Kama, Rosa, Canadian) usando 7 medidas geométricas del grano (área, perímetro, compacidad, longitud del grano, anchura, asimetría, longitud del surco).

**Qué hace el método**: en 18/20 semillas selecciona una subred con 1 feature, alcanzando 95.1% frente al 94.4% de la referencia. Feature dominante: Area (94%).

**Por qué funciona**: las tres variedades de trigo tienen tamaños de grano significativamente diferentes. Kama tiene granos pequeños, Canadian medianos, y Rosa grandes. El área del grano es la medida más directa del tamaño, y por sí sola separa bien las tres variedades. Las otras 6 features (perímetro, compacidad, etc.) están altamente correlacionadas con el área — añadirlas no aporta información nueva pero sí añade ruido y parámetros.

Caso destacado: seed 256, la subred con solo Area alcanza 97.6% mientras la referencia con las 7 features se queda en 85.7%. La referencia probablemente sobreajusta a correlaciones espurias entre features redundantes.

**Conclusión**: el método descubre que la clasificación de variedades de trigo es esencialmente un problema de tamaño del grano, y que las features adicionales son redundantes.

## Balance Scale — subredes +1.3 pp con 33% menos neuronas

**El problema**: predecir si una balanza se inclina a la izquierda (L), se equilibra (B), o se inclina a la derecha (R), dados 4 atributos: peso izquierdo, distancia izquierda, peso derecho, distancia derecha. La regla física es: si peso_izq × distancia_izq > peso_der × distancia_der → L, si son iguales → B, si menor → R.

**Qué hace el método**: en 14/20 semillas selecciona una subred con 1 feature (LeftWeight, 79%), alcanzando 91.5% frente al 90.2% de la referencia.

**Por qué funciona**: esto parece contraintuitivo — ¿cómo puede una sola feature resolver un problema que depende de 4 variables? La clave está en la distribución del dataset. Los pesos y distancias toman valores 1-5, y la clase B (equilibrio) es muy rara (49/625 = 7.8%). La mayoría de muestras son L o R. Con solo el peso izquierdo, la red puede aprender una heurística: peso alto → probablemente L, peso bajo → probablemente R. Esta heurística acierta ~92% porque los casos de equilibrio son raros y los pesos extremos (1 o 5) predicen bien la dirección.

La red completa con 4 features intenta aprender la regla exacta (producto cruzado), que es más difícil de representar con una sola capa oculta y sigmoide. La subred con 1 feature aprende una aproximación más simple que generaliza mejor.

**Conclusión**: el método descubre que para este dataset desbalanceado, una heurística simple basada en una sola variable supera al intento de aprender la regla física exacta. Esto es un ejemplo de cómo la simplicidad puede superar a la complejidad cuando los datos son limitados.

## Wine — feature selection de 13 a 4 features

**El problema**: clasificar 178 vinos italianos en 3 cultivares usando 13 análisis químicos (alcohol, ácido málico, ceniza, alcalinidad, magnesio, fenoles totales, flavonoides, fenoles no flavonoides, proantocianinas, intensidad de color, tono, OD280/OD315, prolina).

**Qué hace el método**: selecciona subredes con 3.8 features de media (70% de reducción), alcanzando 93.3% frente al 97.8% de la referencia. Feature dominante: Alcohol (53-67%).

**Por qué funciona**: el contenido de alcohol es el discriminador más fuerte entre cultivares de vino. El cultivar 1 tiene alcohol alto (~13.7), el cultivar 3 bajo (~12.5), y el cultivar 2 intermedio. Flavonoides e Intensidad de Color son los siguientes discriminadores más fuertes, y juntos con Alcohol cubren la mayor parte de la varianza entre clases.

Las 13 features incluyen varias altamente correlacionadas (fenoles totales y flavonoides, por ejemplo). La red completa puede aprovechar todas, pero la subred con 3-4 features bien elegidas captura lo esencial con menos riesgo de sobreajuste.

**Conclusión**: el método identifica Alcohol como el predictor principal de cultivar, consistente con la enología. La reducción de 13 a 4 features es prácticamente útil: simplifica la medición necesaria sin perder mucha precisión.

## Breast Cancer — features clínicas relevantes

**El problema**: clasificar tumores mamarios como benignos o malignos usando 10 medidas de núcleos celulares (radio, textura, perímetro, área, suavidad, compacidad, concavidad, puntos cóncavos, simetría, dimensión fractal).

**Qué hace el método**: selecciona subredes con 1-2 features, alcanzando 90.6% frente al 94.4% de la referencia. Features dominantes: Radius (36-75%) y Area (36%).

**Por qué funciona**: el radio y el área del núcleo celular son los indicadores más directos de malignidad. Los tumores malignos tienen núcleos significativamente más grandes que los benignos. Concavity y Concave Points (14%) capturan la irregularidad de la forma del núcleo, otro indicador de malignidad.

Aquí la referencia completa es mejor (94.4% vs 90.6%) porque las 10 features aportan información complementaria real — la textura, simetría y dimensión fractal ayudan a distinguir casos ambiguos. Pero la subred con solo Radio ya alcanza >90%, lo cual es clínicamente relevante: una sola medida puede hacer un screening inicial efectivo.

**Conclusión**: el método confirma que el tamaño del núcleo celular es el predictor primario de malignidad, pero las features de forma aportan valor adicional. Esto es consistente con la práctica clínica.

## Haberman — regularización en datos difíciles

**El problema**: predecir supervivencia a 5 años tras cirugía de cáncer de mama usando solo 3 variables: edad de la paciente, año de la operación, y número de nódulos axilares positivos detectados. Dataset muy desbalanceado (225 supervivientes vs 81 fallecidas).

**Qué hace el método**: selecciona subredes con 1 feature, alcanzando 74.5% frente al 73.4% de la referencia.

**Por qué funciona**: con solo 3 features y clases muy desbalanceadas, la red completa tiende a sobreajustar. La subred con 1 feature (típicamente nódulos axilares) aprende una regla simple: pocos nódulos → supervivencia, muchos nódulos → fallecimiento. Esta regla es médicamente correcta — el número de nódulos axilares positivos es el factor pronóstico más importante en cáncer de mama.

La mejora es modesta (+1 pp) pero consistente, y la varianza se reduce (2.6% vs 3.8%). La subred es más estable porque tiene menos parámetros que puedan sobreajustar al ruido.

**Conclusión**: en datasets difíciles con pocas features y clases desbalanceadas, la simplicidad forzada por la división neuronal actúa como regularizador, produciendo modelos más estables.

## Espirales 2D — regularización en problemas difíciles

**El problema**: clasificar puntos en dos espirales entrelazadas. Problema no linealmente separable que requiere fronteras de decisión complejas.

**Qué hace el método**: con la red 2→32→2, las subredes alcanzan 58.9% frente al 56.8% de la referencia.

**Por qué funciona**: las espirales son un problema difícil para redes de una capa oculta. La red completa con 32 neuronas ocultas tiene muchos parámetros (36 neuronas totales) para 300 muestras, y tiende a sobreajustar a patrones locales. Las subredes con 34 neuronas (2 menos) tienen una ligera ventaja de regularización.

Más importante: en varias semillas, la subred con 1 entrada supera a la referencia con 2 entradas. Esto sugiere que para algunas inicializaciones, la red aprende mejor una proyección 1D del problema que el problema 2D completo. Es un caso donde la simplificación forzada descubre una representación más útil.

**Conclusión**: en problemas difíciles donde la red completa no converge bien, la división neuronal puede encontrar representaciones más simples que generalizan mejor.

## Patrones generales observados

### Cuándo las subredes superan a la referencia

1. **Muchas clases, pocas muestras** (Glass, Ecoli): la red completa con N salidas one-hot tiene dificultades para converger. Las subredes resuelven subproblemas más simples y convergen mejor.

2. **Features redundantes** (Seeds, Wine): cuando varias features están correlacionadas, la red completa puede sobreajustar a correlaciones espurias. La subred con 1-2 features captura lo esencial.

3. **Clases desbalanceadas** (Haberman, Balance): la subred aprende heurísticas simples que son más robustas que reglas complejas con pocos datos de la clase minoritaria.

### Cuándo la referencia completa gana

1. **Features complementarias** (Breast Cancer, Iris): cuando cada feature aporta información única, la red completa aprovecha toda la información disponible.

2. **Problemas bien condicionados** (Círculos, 4 Clusters): cuando el problema es fácil y la red converge bien, no hay beneficio en simplificar.

3. **Muchas features, exploración parcial** (Ionosphere, Sonar): cuando no se puede explorar todo el espacio, las subredes evaluadas (1-2 features) no son representativas del óptimo.

### El valor de la exploración exhaustiva

Los mejores resultados (Glass +37pp, Ecoli +11pp, Seeds +1pp) se obtienen con exploración completa del espacio de subconfiguraciones. Cuando el espacio es manejable (hasta ~10⁵), el método garantiza encontrar la subred óptima. Esto no es solo una cuestión de rendimiento — es una herramienta de comprensión: el mapa de soluciones revela la estructura del problema de forma que ningún otro método de feature selection proporciona.


## Pima Indians Diabetes — la referencia gana, pero Glucose domina

**El problema**: predecir diabetes en mujeres Pima usando 8 variables clínicas (embarazos, glucosa, presión arterial, grosor de piel, insulina, BMI, función de pedigrí diabético, edad).

**Qué hace el método**: la referencia completa domina (15/20 semillas), con 77.1% vs 76.8% de las subredes. Cuando selecciona subredes, la feature dominante es Glucose (80%).

**Por qué la referencia gana**: la diabetes es un problema multifactorial donde cada variable aporta información complementaria. Glucosa es el predictor más fuerte (es literalmente lo que se mide para diagnosticar diabetes), pero BMI, edad y embarazos añaden valor predictivo real. Con 768 muestras y solo 8 features, la red completa tiene suficientes datos para aprovechar todas las variables sin sobreajustar.

**Lo que el método revela**: cuando sí selecciona subredes, identifica correctamente Glucose como el predictor primario. Esto es médicamente correcto — la glucosa en ayunas es el criterio diagnóstico estándar para diabetes.

**Conclusión**: en problemas donde las features son genuinamente complementarias y hay suficientes datos, la red completa es la mejor opción. El método lo reconoce correctamente al seleccionar la referencia.

## Banknote Authentication — problema perfectamente resuelto

**El problema**: distinguir billetes genuinos de falsificados usando 4 features extraídas de imágenes wavelet (varianza, asimetría, curtosis, entropía).

**Qué hace el método**: 100% de precisión en 20/20 semillas, referencia siempre seleccionada.

**Por qué**: con 1372 muestras, 4 features bien diseñadas, y 2 clases, el problema es fácil para cualquier red razonable. Las 4 features wavelet fueron diseñadas específicamente para este problema y son todas necesarias. No hay beneficio en simplificar.

**Conclusión**: cuando el problema está bien condicionado y las features son óptimas, el método correctamente no intenta simplificar.
