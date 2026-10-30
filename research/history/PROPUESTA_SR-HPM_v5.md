# SR-HPM v5: dónde queda margen y qué hacer después

Fecha: 28 oct. 2026. Parte del estado medido en [f3/README_F5.md](f3/README_F5.md) y [HERRAMIENTA.md](HERRAMIENTA.md).

**Estado de partida:** Qwen3.8-27B UD-Q3_K_XL en el portátil (RTX 3060 6 GB, Ryzen 7 5800H, DDR4-3200).
- **Generación:** 8,5 tok/s con contexto corto y 5,9 con 10k tokens. Exacto frente a llama.cpp (KLD 0).
- **Prefill en el servidor:** 115–133 tok/s; tarda ~75 s en dar la primera palabra con 10k tokens.

## 1. Dónde se va el tiempo hoy

Una pasada de verificación (4 tokens, de los que se aceptan unos 3,25) tarda ~0,39 s:

| Parte | Tiempo | ¿Margen? |
|---|---|---|
| Productos matriz-vector en CPU (capas 22–63) | ~0,24 s | Ya van al ancho de banda medido (~33 GB/s). Solo baja con menos bytes o más ancho de banda |
| Otras operaciones de CPU (GatedDeltaNet, estados, normalizaciones) | ~0,045 s | Pequeño (~2 % quitando la copia de estados) |
| GPU: 22 capas, cabeza de salida y borrador | ~0,08 s | Los tipos IQ a 4 columnas van a 70–80 GB/s. Probado un kernel propio: sin mejora |
| Sincronización y muestreo | ~0,02 s | — |

Coste medido de verificar M tokens en CPU, relativo a 1 token:

| M | 1 | 4 | 6 | 8 |
|---|---|---|---|---|
| Coste | 1,00 | 1,09 | 1,41 | ~1,9 |

**Hasta 4 tokens la verificación es casi gratis; más allá se encarece rápido.** De este dato salen casi todas las decisiones siguientes.

## 2. Qué dice la investigación reciente y cómo encaja aquí

| Técnica | Idea | Encaje en este portátil |
|---|---|---|
| **DDTree / TAPS** (árboles para borradores de difusión por bloques, 2026) | Verificar varias ramas candidatas del borrador DFlash en una pasada, eligiendo el árbol que maximiza tokens aceptados entre coste | Medio. Con la tabla de coste de arriba, solo compensa un árbol de 5 nodos. DFlash2 ya da los 16 mejores candidatos por posición, así que construirlo es barato. Estimado +3–6 % |
| **Saguaro**, "speculative speculative decoding" (ICLR 2026, +30 % en GPU) | Preparar el siguiente borrador mientras se verifica, para los resultados más probables | Bajo. Solo esconde los ~20 ms del borrador cuando se acierta el resultado. +2–4 % |
| **ShadowSpec** (ICML 2026, +17–54 %) | Usar la GPU ociosa durante la carga de pesos para ampliar el árbol de borrador | Bajo aquí. El límite es el coste de verificar más de 4 tokens en CPU, no la GPU |
| **Recuperación del contexto** ("Draft Less, Retrieve More", SuffixDecoding) | Completar el borrador con n-gramas que ya aparecen en el contexto | Alto en código y textos largos, que es justo donde la aceptación cae (0,42–0,50). Con respuestas cortas `ngram-mod` no se activó |
| **TEAL / R-Sparse** (dispersión de activaciones sin reentrenar, 40–50 %) | Saltar columnas de pesos con activación casi nula | Descartado. Con 4 tokens la unión de canales activos es casi densa (ya lo medimos: H10), y exigiría reorganizar los pesos por columnas |
| **ik_llama.cpp** (cuantizaciones intercaladas por filas, IQ-K) | Kernels de CPU para tipos IQ, 5× más rápidos que llama.cpp de serie | Ya lo tenemos cubierto: SRV + C4 van al ancho de banda de memoria. Lo que sí aporta son sus **tipos IQ-K**, de mejor calidad por bit (ver P3) |
| **SpecOffload / DuoDecoding** (borrador y objetivo en dispositivos distintos, en paralelo) | Solapar borrador y verificación entre CPU y GPU | Ya es nuestra arquitectura (borrador en GPU, verificación sobre todo en CPU). El solape completo lo impide la dependencia de DFlash de las características del objetivo |

## 3. Propuestas ordenadas por ganancia y coste

| # | Propuesta | Ganancia estimada | Esfuerzo | Riesgo |
|---|---|---|---|---|
| **P1** | **Linux nativo, sin WDDM.** +1 GB de VRAM (≈5 capas más), sincronización CPU↔GPU más barata (hace viable el canal PCIe F6) y sin desbordamiento silencioso de VRAM | **+15–25 %** | Arranque dual y compilar allí | Medio |
| **P2** | **Páginas grandes (2 MB) para los pesos de CPU.** Menos fallos de TLB al recorrer 8,5 GB por pasada. Asignación con `MEM_LARGE_PAGES` en ggml-cpu | **+5–15 %** | Bajo en código. El privilegio "Bloquear páginas en memoria" lo tiene que conceder el usuario | Bajo |
| **P3** | **Recuantizar las capas de CPU con un formato de mejor calidad por bit** (IQ-K o trellis tipo QTIP/EXL3), con su kernel SRV. Misma calidad que Q3_K_XL con 15–20 % menos bytes | **+12–17 %** | Alto. ~60 GB de disco libre para partir del FP8, matriz de importancia y validación de KLD | Medio |
| **P4** | **Recuperación de n-gramas del contexto** combinada con DFlash (en código y contexto largo) y, después, **adaptación del borrador** a tu uso (F7) | **+25–35 % en código y contexto largo** si la aceptación sube a ~0,7 | La recuperación es media; la adaptación del borrador, alta | Bajo / medio |
| **P5** | **Árbol pequeño de verificación** (5 nodos, estilo TAPS, con la tabla de coste medida) | +3–6 % | Medio. Máscara de atención de ancestros y estados recurrentes por rama (lo más delicado en un modelo híbrido) | Medio |
| **P6** | **Pre-borrador en paralelo** (Saguaro) con la GPU ociosa | +2–4 % | Medio | Bajo |
| **P7** | **Prefill del servidor.** Va a 115–133 tok/s frente a 212–245 del benchmark. La diferencia viene de la inyección del prompt en el borrador (trozos de 16) y de la VRAM justa. Objetivo ~200 tok/s: de ~75 a ~45 s para 10k tokens. Además, reutilizar la caché del prompt entre turnos | Tiempo hasta la primera palabra −30–40 % | Medio | Bajo |
| **P8** | Evitar la copia de estados recurrentes (`GET_ROWS`) cuando la secuencia no cambia | ~2 % | Bajo | Bajo |
| **P9** | **Kernels con VNNI / AVX-512** (Zen 4 e Intel de 12.ª gen. en adelante) | No aplica a este 5800H. En ese hardware haría casi gratis verificar 8 tokens, y con ello los árboles grandes (P5) | Medio | Bajo |

**Combinación realista en este portátil sin tocar el hardware:** P2 + P4 (recuperación) + P5 + P6 + P8 dan ~+15–25 % con contexto corto, es decir, de ~8,5 a ~10–10,5 tok/s. En código y contexto largo la mejora sería mayor.

**Con P1 (Linux)** se añadiría otro ~+15–25 %, hasta unos 12 tok/s.

## 4. Lo probado y descartado en esta fase (para no repetirlo)

| Idea | Resultado |
|---|---|
| Kernel CUDA que decodifica una vez para varias columnas (tipos IQ) | 0,92–1,07×. El compilador ya comparte la decodificación |
| Retícula IQ en memoria compartida de GPU | 0,85–1,12× |
| Ruta split-KV de llama.cpp ampliada a varios tokens | Sin efecto: la atención estaba limitada por cálculo. Resuelto después con el kernel GQA (+48 % con 10k tokens) |
| Atención de prefill descargada a GPU | Sin efecto: no era el cuello del prefill |
| Lotes de 2048 en el servidor | Salta una aserción interna (`batch.slot_batched`) o desborda la VRAM. Se usan lotes de 1024 |
| Más VRAM para el borrador (`--spec-draft-n-ubatch 128`) | Peor: desbordamiento de VRAM |
| Borrador adaptativo (`p-min`) y n = 4–7 | Peor que n = 3 fijo |
| Capas en GPU elegidas tensor a tensor | Peor que por capas contiguas (5,1 frente a 5,7), por las idas y vueltas CPU↔GPU |

## Fuentes

- [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/pdf/2602.06036)
- [DDTree: Accelerating Speculative Decoding with Block Diffusion Draft Trees](https://arxiv.org/pdf/2604.12989)
- [TAPS: Target-Aware Prefix Tree Selection](https://arxiv.org/pdf/2606.00487)
- [Draft Less, Retrieve More: Hybrid Tree Construction](https://arxiv.org/pdf/2605.20104)
- [Speculative Speculative Decoding (Saguaro)](https://arxiv.org/abs/2603.03251)
- [ShadowSpec (ICML 2026)](https://icml.cc/virtual/2026/75131)
- [SpecOffload](https://arxiv.org/pdf/2505.10259)
- [DuoDecoding](https://arxiv.org/pdf/2503.00784)
- [SpecExec](https://arxiv.org/html/2406.02532v2)
- [TEAL: Training-Free Activation Sparsity](https://arxiv.org/pdf/2408.14690)
- [R-Sparse](https://arxiv.org/pdf/2504.19449)
- [ik_llama.cpp, comparación de rendimiento en CPU](https://github.com/ikawrakow/ik_llama.cpp/discussions/164)

## 5. Estado tras implementarlas (28 oct. 2026)

| # | Propuesta | Estado | Resultado medido / condición |
|---|---|---|---|
| P1 | Linux nativo | **En la herramienta**: `srhpm/platform.py` detecta Linux, reserva 0,3 GiB de VRAM en lugar de 1 GiB (más capas en GPU) y pide páginas enormes transparentes. `srhpm/build_linux.sh` compila con el parche | No probado aquí (Windows). El canal PCIe (F6) sigue sin implementar: en Linux sería el siguiente paso |
| P2 | Páginas grandes | **Implementado** en ggml (`GGML_LARGE_PAGES=1`): `VirtualAlloc(MEM_LARGE_PAGES)` en Windows, `madvise(MADV_HUGEPAGE)` en Linux. Si no se puede, vuelve a la reserva normal y lo dice en el log. El lanzador lo activa si detecta el privilegio | Aquí falta el privilegio: comprobado el modo de respaldo, no la ganancia |
| P3 | Recuantización para la máquina | **En la herramienta**: `srhpm/requant.py` (BF16, matriz de importancia de tu texto, mezcla de tipos con kernel SR-HPM, KLD frente a BF16). Solo se ejecuta con ≥ 60 GB libres y los pesos originales | Plan generado y probado en seco; `llama-quantize` y `llama-imatrix` compilados |
| P4 | Recuperación de n-gramas | **Opción** `--retrieval` (n-gramas de 4 tokens que proponen 3, antes que DFlash; llama.cpp ya les da prioridad) | En chat y resúmenes **nunca encuentra coincidencias**: 0 borradores, sin ganancia. Útil solo en cargas repetitivas (editar código, citar). No se activa por defecto |
| P5 | Árbol pequeño de verificación | **No implementado** | En un modelo híbrido cada rama necesita su propio estado recurrente: verificación multisecuencia en el servidor y copia del estado ganador. Para +3–6 % estimado no compensa |
| P6 | Pre-borrador en paralelo | **No viable con DFlash2** | El borrador usa las capas [6, 20, 34, 48, 62] del objetivo. La 62 sale al final de la verificación, así que no se puede adelantar |
| P7 | Prefill del servidor | **Parcial**: lotes de 1024 (+5–10 %) en el lanzador; atención de prefill descargada a GPU (sin efecto, se mantiene por ser correcta) | El borrador sigue costando ~1/3 del prefill (193 → 131 tok/s con 10k). Descartado como causa: inyección del prompt (probada una ventana de 1024 tokens, sin efecto) y VRAM |
| P8 | Copia de estados recurrentes | **No implementado** | ≤ 2 % (por debajo del ruido de ±5 %) y arriesga corromper estados con la reutilización de grafos |
| P9 | Kernels VNNI | **Implementado**: `vpdpwssd` (AVX-VNNI o AVX512-VNNI) sustituye `vpmaddwd` + `vpaddd` con el mismo resultado. Se activa compilando en una CPU con VNNI. El lanzador detecta la CPU | Compila para AVX-VNNI y Zen 4 (genera 129 `vpdpwssd`); no ejecutable en este Zen 3 |

**Hallazgo durante la implementación.** Para que el prefill pudiera subir los pesos a la GPU, el búfer de CPU dejó de usar el reordenado de serie de llama.cpp, y los K-quants (Q4_K, Q5_K, Q6_K, ~1,2 GB en CPU) volvieron a la ruta lenta: −10 % en generación. Solución: **kernels SR-HPM para Q4_K, Q5_K y Q6_K**, con el término de mínimo calculado a partir de las sumas de bloque de las activaciones y sin la operación de signo para los códigos sin signo. Recuperan la velocidad del reordenado de serie (8,22 frente a 8,25 tok/s) sin perder el prefill rápido. Exactitud frente a `vec_dot`: Q5_K idéntico, Q6_K 2,6·10⁻⁶ y Q4_K 3,5·10⁻⁴ en el peor caso, por el orden de las sumas del mínimo.

**Ruido de medida.** Tras horas de carga continua, el mismo portátil mide entre 7,5 y 8,7 tok/s con la misma configuración. Las comparaciones de esta sección se hicieron por parejas en la misma sesión.
