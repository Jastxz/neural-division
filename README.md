# Método de la División Neuronal

Implementación en Julia del Método de la División Neuronal: un enfoque de búsqueda exhaustiva de arquitecturas de subredes dentro de una red neuronal inicializada con pesos aleatorios.

## La idea

Partimos de dos principios:

1. El ser humano tiende a crear soluciones complejas para problemas simples.
2. Dividir para vencer: si un problema se puede descomponer en subproblemas más sencillos, cada uno puede resolverse con una red más pequeña.

El método toma una red neuronal inicializada (no entrenada) y explora **todas** las posibles subconfiguraciones — combinaciones de subconjuntos de neuronas de entrada y salida. Para cada subconfiguración evalúa su precisión con pesos aleatorios y guarda las más simples que superan un umbral configurable. Tras la exploración, entrena las subredes encontradas y selecciona la mejor opción comparando contra la red completa.

El resultado es un **Mapa de Soluciones**: un diccionario donde cada clave es un subconjunto de salidas del problema, y el valor es la subred más eficiente encontrada para resolverlo.

### Doble objetivo: resolver y comprender

El método no solo busca redes eficientes — también es una herramienta para **comprender la estructura del problema**. El mapa de soluciones responde preguntas que otros métodos no plantean:

- **¿Qué variables son realmente necesarias?** Si una subred con 1 de 13 features alcanza el 93% de precisión, esa feature es el predictor dominante. En Wine, el método descubre que Alcohol es la variable clave; en Glass, el Índice de Refracción; en Seeds, el Área del grano.
- **¿Se puede descomponer el problema?** Si el mapa contiene soluciones parciales para subconjuntos de salidas, el problema tiene estructura interna. En el multi-salida AND+OR+XOR, el método identifica que XOR necesita ambas entradas mientras AND y OR pueden resolverse con una.
- **¿La red completa es necesaria?** En varios datasets (Glass, Ecoli, Seeds), las subredes superan a la red completa porque actúan como regularizador implícito. Esto revela que el problema es más simple de lo que la red completa asume.

Esta capacidad de análisis es especialmente valiosa cuando la exploración es exhaustiva: con espacios manejables (hasta ~10⁵ subconfiguraciones), el método garantiza encontrar la subred óptima y proporciona un mapa completo de cómo se relacionan las entradas con las salidas.

## Instalación

```julia
using Pkg
Pkg.develop(path=".")
```

Requisitos: Julia 1.10+ con JLD2 para serialización.

## Uso rápido

```julia
using DivisionNeuronal

# Crear red base: 4 entradas → 16 ocultas → 3 salidas
pesos = [randn(4, 16), randn(16, 3)]
biases = [randn(16), randn(3)]
red = RedBase{Float64}(pesos, biases, 4, 3)

# Datos (entradas y salidas como matrices)
datos = (entradas = rand(100, 4), salidas = rand(Bool, 100, 3) .|> Float64)

# Configuración
config = ConfiguracionDivision{Float64}(0.4)  # umbral de acierto

# Ejecutar división
mapa = ejecutar_division(red, datos, config;
    datos_entrenamiento = datos,
    epochs = 1000,
    lr = 0.01)

# Seleccionar la mejor solución (referencia completa vs subredes)
mejor = seleccionar_mejor(mapa)
println("Tipo: $(mejor.tipo), Precisión: $(mejor.precision), Neuronas: $(mejor.neuronas)")

# Ver el mapa completo
println(formatear(mapa))
```

## Arquitectura

```
src/
├── DivisionNeuronal.jl    # Módulo principal
├── tipos.jl               # Tipos paramétricos (RedBase, Subconfiguracion, MapaDeSoluciones...)
├── errores.jl             # Excepciones tipadas
├── validacion.jl          # Validación de entradas
├── generador.jl           # Generador lazy de subconfiguraciones (iterador por bitmask)
├── evaluador.jl           # Forward pass con sigmoide, cálculo de precisión global y parcial
├── comparador.jl          # Criterio de simplicidad (menos neuronas, mayor precisión)
├── mapa_soluciones.jl     # Inicialización y actualización del mapa
├── entrenamiento.jl       # Adam optimizer con mini-batches y early stopping
├── motor.jl               # Orquestador principal (ejecutar_division)
├── seleccion.jl           # Selección de mejor solución (score precisión/coste)
├── serializacion.jl       # Persistencia JLD2 y pretty printing
└── progreso.jl            # Reporte de progreso y cancelación cooperativa
```

### Flujo del proceso

1. **Validación**: verifica que la red base tiene pesos consistentes, neuronas válidas y umbral en rango.
2. **Referencia completa**: extrae la subconfiguración con todas las entradas y salidas como baseline.
3. **Exploración exhaustiva**: itera sobre todas las combinaciones de subconjuntos de entrada × subconjuntos de salida (excluyendo vacíos). Para cada una, extrae la subred, evalúa con forward pass, y actualiza el mapa si supera el umbral y es más simple que la almacenada.
4. **Entrenamiento**: entrena con Adam optimizer (mini-batches, early stopping) la referencia completa, la mejor subred global, y todas las soluciones parciales.
5. **Selección**: compara todas las opciones con un score que pesa 80% precisión y 20% ahorro de neuronas, y devuelve la mejor.

### Criterio de simplicidad

Una subconfiguración A es mejor que B si:
- A tiene menos neuronas activas (entrada + ocultas + salida), o
- Mismas neuronas y A tiene mayor precisión.

## Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

374 tests incluyendo:
- Tests unitarios por módulo
- 11 tests de propiedades con PropCheck.jl (100 iteraciones cada uno)
- Tests de integración del flujo completo

## Resultados experimentales

### Problemas lógicos

| Problema | Red | Tasa éxito | Precisión post | Mejora entrenamiento |
|----------|-----|-----------|----------------|---------------------|
| XOR | 2→16→1 | 10% | 100% | +25 pp |
| Paridad 3 bits | 3→8→1 | 11% | 100% | +37.5 pp |
| AND+OR+XOR (multi) | 2→8→3 | 81% | 75-100% | hasta +25 pp |

Observaciones:
- XOR y Paridad son problemas equilibrados (50/50), por lo que pocas subconfiguraciones superan el umbral 0.5 con pesos aleatorios. Cuando lo hacen, el entrenamiento siempre alcanza 100%.
- En el problema multi-salida, el método diferencia correctamente las salidas: AND/OR/NAND se resuelven con 1 entrada, XOR necesita 2.
- La tasa de éxito escala con el número de neuronas ocultas (más "lotería" favorable).

### Sensibilidad al umbral (Paridad 3 bits, 3→6→1, 100 semillas)

| Umbral | Tasa éxito | Neuronas | Entradas | Post entren. |
|--------|-----------|----------|----------|-------------|
| 0.3 | 100% | 8.0 | 1.0 | 50% |
| 0.4 | 100% | 8.0 | 1.0 | 50% |
| 0.5 | 9% | 10.0 | 3.0 | 100% |
| 0.6 | 9% | 10.0 | 3.0 | 100% |
| 0.7 | 0% | - | - | - |

Hay un punto de inflexión claro: umbrales bajos seleccionan subredes triviales (1 entrada, sin capacidad de mejora); umbrales en 0.5-0.6 fuerzan subredes con todas las entradas necesarias que luego alcanzan 100%.

### Dataset Iris (4 entradas, 3 clases, 150 muestras)

| Red | Ref. completa | Mejor seleccionada | Tipo seleccionado | Ahorro neuronas |
|-----|--------------|-------------------|-------------------|----------------|
| 4→8→3 | 97.8% ± 2.9% | 96.7% ± 3.7% | subred 55% | 20% (12 vs 15) |
| 4→16→3 | 97.8% ± 3.1% | 97.8% ± 3.5% | subred 40% | 9% (21 vs 23) |
| 4→32→3 | 98.2% ± 2.8% | 98.7% ± 2.5% | subred 20% | 3% (38 vs 39) |

Observaciones:
- Con la red más pequeña (4→8→3), el método es más efectivo: en 11/20 semillas selecciona una subred con 33% menos neuronas y precisión comparable.
- Caso destacado: seed 1337, la subred de 10 neuronas alcanza 100% en test mientras la referencia completa solo llega a 90%.
- Las soluciones parciales revelan que Versicolor (la clase más difícil) genera subredes especializadas.

### Dataset Wine (13 entradas, 3 clases, 178 muestras)

| Red | Ref. completa | Mejor seleccionada | Tipo seleccionado | Ahorro neuronas |
|-----|--------------|-------------------|-------------------|----------------|
| 13→16→3 | 97.8% ± 2.8% | 93.3% ± 4.3% | subred 85% | 34% (21 vs 32) |
| 13→32→3 | 98.1% ± 2.6% | 96.2% ± 3.7% | subred 60% | 15% (41 vs 48) |

Observaciones:
- **Feature selection automática**: las subredes usan en media 3.8 de 13 entradas, descartando el 70% de las features.
- Feature más seleccionada: Alcohol (53-67%), seguida de Flavanoids y Color Intensity. Coincide con la literatura.

### Breast Cancer Wisconsin (10 features mean, 2 clases, 569 muestras)

| Red | Ref. completa | Mejor seleccionada | Tipo seleccionado | Ahorro neuronas |
|-----|--------------|-------------------|-------------------|----------------|
| 10→16→2 | 94.4% | 90.6% | subred 70% | 25% (21 vs 28) |
| 10→32→2 | 94.7% | 93.1% | subred 40% | 9% (40 vs 44) |

Features más relevantes: Radius (36-75%) y Area (36%), seguidas de Concavity y Concave Points.

### Datos sintéticos 2D (300 muestras)

| Problema | Red | Ref. completa | Mejor seleccionada | Observación |
|----------|-----|--------------|-------------------|-------------|
| Lunas | 2→8→2 | 89.5% | 89.6% | Referencia domina |
| Círculos | 2→8→2 | 100% | 100% | Problema fácil |
| Espirales | 2→32→2 | 56.8% | **58.9%** | Subredes superan referencia |
| 4 Clusters | 2→8→4 | 100% | 100% | Linealmente separable |

En espirales, las subredes superan a la referencia: **regularización implícita**.

### Seeds (7 features, 3 clases, 210 muestras)

| Red | Ref. completa | Mejor seleccionada | Tipo seleccionado | Ahorro neuronas |
|-----|--------------|-------------------|-------------------|----------------|
| 7→16→3 | 94.4% | **95.1%** | subred 90% | 27% (19 vs 26) |
| 7→32→3 | 94.6% | **95.5%** | subred 85% | 17% (35 vs 42) |

Subredes superan a la referencia. Feature dominante: Area (94%).

### Glass (9 features, 6 clases, 214 muestras) — resultado estrella

| Red | Ref. completa | Mejor seleccionada | Tipo seleccionado | Ahorro neuronas |
|-----|--------------|-------------------|-------------------|----------------|
| 9→16→6 | 57.1% | **94.4%** | subred 100% | 42% (18 vs 31) |
| 9→32→6 | 57.6% | **94.8%** | subred 100% | 28% (34 vs 47) |

Las subredes son **masivamente mejores** (+37 pp). La red completa con 6 clases no converge bien, pero las subredes especializadas sí. Feature dominante: Refractive Index (85%). En 20/20 semillas, la subred gana.

### Haberman Survival (3 features, 2 clases, 306 muestras)

| Red | Ref. completa | Mejor seleccionada | Tipo seleccionado |
|-----|--------------|-------------------|-------------------|
| 3→8→2 | 73.4% | **74.5%** | subred 65% |
| 3→16→2 | 72.9% | **74.2%** | subred 75% |

Subredes superan a la referencia con menor varianza. Regularización implícita.

### Ionosphere y Sonar (exploración parcial)

| Dataset | Red | Ref. completa | Mejor seleccionada | Exploración |
|---------|-----|--------------|-------------------|-------------|
| Ionosphere (34 feat) | 34→32→2 | 92.1% | 88.9% | 10000 / 51.5×10⁹ |
| Sonar (60 feat) | 60→32→2 | 82.9% | 79.4% | 10000 / 3.4×10¹⁸ |

Con exploración parcial, solo se evalúan subredes con 1-2 entradas. **La exploración exhaustiva es el punto fuerte del método** — cuando es factible, garantiza encontrar la subred óptima y revelar la estructura del problema.

### Adult Census Income (6 features continuas, 2 clases, ~30K train)

| Red | Ref. completa | Mejor seleccionada | Tipo seleccionado |
|-----|--------------|-------------------|-------------------|
| 6→16→2 | 82.0% | 80.8% | referencia 70% |
| 6→32→2 | 81.9% | 81.7% | referencia 90% |

Con 30K muestras y features complementarias, la referencia domina. Cuando las subredes ganan, identifican Capital-Gain y Education-Num como predictores clave — consistente con la sociología económica.

### Sobre MNIST y CIFAR

MNIST (784 entradas) y CIFAR-10 (3072 entradas) tienen espacios de subconfiguraciones astronómicos para exploración exhaustiva. Dos enfoques prometedores:

1. **Exploración acotada**: limitar la búsqueda a subconfiguraciones con el 40-60% de las entradas. No es exhaustiva, pero puede revelar qué regiones de la imagen son más informativas para cada clase.
2. **División sobre representaciones**: aplicar la división neuronal sobre features extraídas por una capa previa (convolucional, PCA), reduciendo a 10-30 entradas manejables.

### MNIST con PCA (10-15 componentes, 10 semillas)

| Variante | Ref. completa | Mejor seleccionada | Componentes usados |
|----------|--------------|-------------------|-------------------|
| 10 PCA, 0 vs 1 | 99.9% | 98.0% | 3.7 / 10 |
| 10 PCA, 0 vs 1 vs 7 | 99.1% | 97.5% | 2.0 / 10 |
| 15 PCA, 0 vs 1 | 99.9% | 99.5% | 4.2 / 15 |

El método descubre que distinguir dígitos 0, 1 y 7 solo necesita 2 componentes PCA. Con 15 componentes, una subred de 20 neuronas alcanza 100% (seed 42) superando a la referencia de 33 neuronas.

### Hallazgos principales

1. **Comprensión del problema**: el método revela la estructura interna de los datos. En Glass descubre que el Índice de Refracción es el predictor dominante (85%); en Seeds, que el Área del grano separa las variedades (94%); en Pima, que la Glucosa es el factor clave (80%). Estos descubrimientos coinciden con el conocimiento experto de cada dominio, validando el método como herramienta de análisis.

2. **Feature selection emergente**: como subproducto de buscar subredes simples, el método identifica qué entradas son necesarias y cuáles son redundantes. En Wine reduce de 13 a 4 features; en Cancer identifica Radio y Área como predictores primarios de malignidad.

3. **Regularización implícita**: en Glass (+37pp), Ecoli (+11pp), Seeds (+1pp), Haberman (+1pp) y Espirales (+2pp), las subredes superan a la referencia completa. Menos parámetros = mejor generalización, especialmente con muchas clases o pocas muestras.

4. **Descomposición de problemas**: el mapa de soluciones parciales muestra cómo se descompone el problema. En el multi-salida AND+OR+XOR, identifica que XOR necesita ambas entradas mientras AND/OR se resuelven con una. Cada subconjunto de salidas tiene su subred óptima.

5. **Saber cuándo no simplificar**: en Pima Diabetes y Banknote, el método selecciona correctamente la red completa, reconociendo que todas las features aportan información complementaria. Esto es tan informativo como encontrar subredes — confirma que el problema requiere todas las variables.

6. **Exploración exhaustiva como garantía**: con espacios manejables (hasta ~10⁵ subconfiguraciones), el método garantiza la subred óptima y proporciona un mapa completo de la relación entradas-salidas. El valor real está en la búsqueda completa.

Para un análisis detallado de cada dataset, ver [benchmarks/ANALISIS.md](benchmarks/ANALISIS.md).

## Benchmarks

Los scripts de benchmarks están en `benchmarks/`:

Para un análisis detallado de por qué las subredes funcionan en cada dataset, ver [benchmarks/ANALISIS.md](benchmarks/ANALISIS.md).

```bash
# Problemas lógicos básicos
julia --project=. benchmarks/run_benchmarks.jl

# Experimentos individuales
julia --project=. benchmarks/exp1_umbral.jl        # Sensibilidad al umbral
julia --project=. benchmarks/exp2_escalado.jl       # Escalado de red base
julia --project=. benchmarks/exp3_multisalida.jl    # Descomposición multi-salida
julia --project=. benchmarks/exp4_comparacion.jl    # Comparación con red completa
julia --project=. benchmarks/exp5_estadisticas.jl   # Estadísticas detalladas (100 semillas)
julia --project=. benchmarks/exp6_iris.jl           # Dataset Iris
julia --project=. benchmarks/exp7_wine.jl           # Dataset Wine
julia --project=. benchmarks/exp8_cancer.jl         # Breast Cancer Wisconsin
julia --project=. benchmarks/exp9_sinteticos2d.jl   # Datos sintéticos 2D
julia --project=. benchmarks/exp10_seeds_glass.jl   # Seeds y Glass
julia --project=. benchmarks/exp11_mas_datasets.jl  # Ionosphere, Sonar, Haberman
julia --project=. benchmarks/exp12_ecoli_balance.jl # Ecoli y Balance Scale
julia --project=. benchmarks/exp13_pima_banknote.jl # Pima Diabetes y Banknote
julia --project=. benchmarks/exp14_adult.jl         # Adult Census Income
julia --project=. benchmarks/exp14b_adult_acotado.jl # Adult exploración acotada
julia --project=. benchmarks/exp15_mnist_pca.jl     # MNIST con PCA
```

## Licencia

Ver [LICENSE](LICENSE).
