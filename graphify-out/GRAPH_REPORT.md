# Graph Report - .  (2026-06-03)

## Corpus Check
- Corpus is ~36,386 words - fits in a single context window. You may not need a graph.

## Summary
- 218 nodes · 264 edges · 47 communities (28 shown, 19 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]

## God Nodes (most connected - your core abstractions)
1. `main()` - 9 edges
2. `tabla_verdad()` - 7 edges
3. `tabla_verdad()` - 7 edges
4. `exp15()` - 6 edges
5. `exp9_sinteticos()` - 6 edges
6. `normalizar!()` - 5 edges
7. `exp17()` - 5 edges
8. `normalizar_entradas()` - 5 edges
9. `exp10()` - 4 edges
10. `exp12()` - 4 edges

## Surprising Connections (you probably didn't know these)
- None detected - all connections are within the same source files.

## Import Cycles
- None detected.

## Communities (47 total, 19 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.24
Nodes (16): crear_red(), ejecutar_benchmark(), imprimir_resultado(), DivisionNeuronal, Printf, Random, Statistics, main() (+8 more)

### Community 1 - "Community 1"
Cohesion: 0.20
Nodes (13): crear_red(), ejecutar_una(), DivisionNeuronal, Printf, Random, Statistics, problema_and(), problema_multi_logica() (+5 more)

### Community 2 - "Community 2"
Cohesion: 0.27
Nodes (11): aplicar_pca(), combinations_iter(), CombIter, ejecutar_mnist_pca(), exp15(), filtrar_clases(), Int, LinearAlgebra (+3 more)

### Community 3 - "Community 3"
Cohesion: 0.30
Nodes (11): cargar_ecoli_datos(), cargar_glass_datos(), cargar_seeds_datos(), cargar_wine_datos(), entrenar_con_features(), erfc(), exp17(), normalizar!() (+3 more)

### Community 4 - "Community 4"
Cohesion: 0.33
Nodes (8): a_cpu(), a_dispositivo(), _cuda_disponible(), inicializar_gpu!(), _rocm_disponible(), seleccionar_backend!(), _verificar_cuda(), _verificar_rocm()

### Community 5 - "Community 5"
Cohesion: 0.50
Nodes (8): ejecutar_sintetico(), exp9_sinteticos(), generar_4clusters(), generar_circulos(), generar_espirales(), generar_lunas(), normalizar_entradas(), split_sintetico()

### Community 6 - "Community 6"
Cohesion: 0.43
Nodes (6): cargar_adult_11(), combinations_iter(), CombIter, exp14b(), Atomic, Int

### Community 7 - "Community 7"
Cohesion: 0.60
Nodes (5): cargar_glass(), cargar_seeds(), ejecutar_dataset(), exp10(), split_estratificado()

### Community 8 - "Community 8"
Cohesion: 0.53
Nodes (5): cargar_csv_generico(), ejecutar_con_presupuesto(), exp11(), Atomic, split_strat()

### Community 9 - "Community 9"
Cohesion: 0.60
Nodes (5): cargar_balance(), cargar_ecoli(), ejecutar_dataset(), exp12(), split_strat()

### Community 10 - "Community 10"
Cohesion: 0.53
Nodes (5): cargar_adult_para_embeddings(), combinations_iter(), CombIter, exp18(), Int

### Community 11 - "Community 11"
Cohesion: 0.60
Nodes (5): Exception, NeuronasInvalidasError, RedBaseNoInicializadaError, UmbralFueraDeRangoError, String

### Community 12 - "Community 12"
Cohesion: 0.70
Nodes (4): cargar_csv_simple(), ejecutar_dataset(), exp13(), split_strat()

### Community 13 - "Community 13"
Cohesion: 0.60
Nodes (4): cargar_cancer(), exp8_cancer(), Atomic, split_cancer()

### Community 14 - "Community 14"
Cohesion: 0.40
Nodes (4): Atomic, DivisionNeuronal, PropCheck, Test

### Community 15 - "Community 15"
Cohesion: 0.83
Nodes (3): cargar_monks(), ejecutar_monks(), exp16()

### Community 16 - "Community 16"
Cohesion: 0.83
Nodes (3): cargar_iris(), exp6_iris(), split_datos()

### Community 17 - "Community 17"
Cohesion: 0.83
Nodes (3): cargar_wine(), exp7_wine(), split_datos()

### Community 21 - "Community 21"
Cohesion: 0.50
Nodes (3): Atomic, DivisionNeuronal, Test

### Community 23 - "Community 23"
Cohesion: 0.50
Nodes (3): Atomic, DivisionNeuronal, Test

## Knowledge Gaps
- **49 isolated node(s):** `Atomic`, `Atomic`, `Int`, `CodecZlib`, `LinearAlgebra` (+44 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **19 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What connects `Atomic`, `Atomic`, `Int` to the rest of the system?**
  _49 weakly-connected nodes found - possible documentation gaps or missing edges._