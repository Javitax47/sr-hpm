# SR-HPM v2 — Diseño corregido
## Objetivo: Qwen3.8-27B en este portátil, con la máxima inteligencia posible y a decenas de tokens por segundo

**Base:** [AUDITORIA.md](AUDITORIA.md).

**Convención de cifras:**
- **[med]** medido en esta máquina.
- **[cal]** cota calculada a partir de bytes y ancho de banda.
- **[est]** estimación con supuestos explícitos.
- **[ext]** dato de terceros, no verificado aquí.

---

## 1. Puntos de partida

### 1.1 Máquina

| Componente | Valor |
|---|---|
| GPU | RTX 3060 Laptop: 6 GB, **315 GB/s** [med], L2 3 MB |
| RAM | 32 GB DDR4-3200, doble canal (2 módulos) [med]: 51 GB/s pico; **30,5 GB/s** de lectura real con 4–16 hilos (17,7 GB/s con 1 hilo) [med] |
| CPU | Ryzen 7 5800H, 8 núcleos Zen 3, AVX2 [med] |
| PCIe | 3.0 x16, ~12 GB/s [med enlace; ancho de banda est] |

### 1.2 Modelo: Qwen3.8-27B ([ficha oficial](https://huggingface.co/Qwen/Qwen3.8-27B)) [ext]

- **Tipo:** modelo **denso** de 27 B parámetros. Son 28 B si se cuentan relleno y la torre de visión.
- **Capas:** 64. Siguen el patrón `16 × (3 × [Gated DeltaNet → FFN] + 1 × [Gated Attention → FFN])`, es decir, **48 capas de atención lineal** con estado constante y **16 de atención completa**.
- **Dimensiones:**
  - Oculta 5 120; FFN 17 408.
  - Atención completa: 24 cabezas Q y **4 KV** de dimensión 256.
  - Vocabulario de 248 k tokens.
- **Cabeza MTP** (predicción de varios tokens) incluida.
- **Contexto:** 262 k nativo.
- **Razonamiento:** *thinking* activado por defecto, con `reasoning_effort` low / medium / xhigh.

**Reparto aproximado de parámetros** [cal, a partir de las dimensiones]:

| Componente | Parámetros | Cuota |
|---|---|---|
| FFN | 64 × 3 × 5 120 × 17 408 ≈ **17,1 B** | ~2/3 |
| Mezcladores (DeltaNet + atención) | ≈ **7 B** | |
| Cabeza LM | ≈ **1,3 B** | |
| Embedding | ≈ **1,3 B** | Se lee solo una fila por token: prácticamente gratis |

**Por token se leen unos 25,5 B parámetros.**

### 1.3 Lo que el propio modelo ya resuelve del informe original

| Pilar original | Qwen3.8-27B ya lo trae |
|---|---|
| III. Memoria de contexto constante | 48 de 64 capas son Gated DeltaNet con **estado fijo**. Es la versión entrenada y funcional de la «memoria holográfica» (C3/C4 de la auditoría). Solo 16 capas guardan KV: 2 × 16 × 4 × 256 × 1 B ≈ **32 KB/token en q8**. Con 32 k tokens son ~1 GB; con 128 k, ~4 GB. |
| II. Cómputo «elástico» | La cabeza **MTP** permite decodificación especulativa: el modelo propone varios tokens y los verifica en una sola lectura de pesos. Es la versión correcta de la profundidad elástica. |
| I. Pocos bytes por token | **No.** Es denso: cada token lee los ~25,5 B parámetros. Aquí está todo el problema. |

---

## 2. La física del objetivo

En batch 1 cada token lee todos los pesos activos una vez. Lo que no quepa en VRAM se lee desde la RAM y se calcula en la CPU, a ~30 GB/s [med], 10 veces más despacio que la VRAM. Mandarlo por el PCIe sería aún peor: 12 GB/s.

$$t_\text{token} \approx \frac{G}{315\ \text{GB/s}} + \frac{S - G}{30\ \text{GB/s}}$$

donde *S* son los bytes leídos por token y *G* los bytes alojados en VRAM. De los 6 GB caben **G ≈ 4,5 GB** de pesos. El resto se va en:
- contexto CUDA (~0,4 GB);
- KV de 16 k tokens en q8 (~0,5 GB);
- estados de DeltaNet (~0,1–0,2 GB);
- búferes de cómputo (~0,4 GB).

**Consecuencia:** casi todo el tiempo se va en la parte de RAM. Para ir rápido hay que reducir los bytes que quedan fuera de la GPU, y eso solo se consigue bajando los bits por peso, que es justo lo que resta inteligencia. Mientras el modelo sea denso, **velocidad e inteligencia compiten por el mismo recurso**.

### 2.1 Tabla de decisión en este portátil

Supuestos: G = 4,5 GB en GPU, embedding en RAM y fuera de la cuenta, MTP con ganancia de ×1,3–1,6. La ganancia por MTP es menor que en una GPU completa, porque verificar varios tokens en la CPU se vuelve en parte limitado por cálculo.

| Cuantización (Unsloth Dynamic) | Tamaño [ext] | KLD medio [ext] | Calidad [ext] | tok/s sin MTP [cal] | tok/s con MTP [est] |
|---|---|---|---|---|---|
| UD-IQ1_S | 6,2 GB | ≈ 0,43 | Degradada; falla en llamadas a herramientas | ~17 | **~23–28** |
| **UD-Q2_K_XL** | 9,8 GB | ≈ 0,09 | Usable en chat; herramientas marginales | ~5,6 | **~7–9** |
| **UD-Q3_K_XL** | 13,2 GB | ≈ 0,026 | Cercana al original | ~3,5 | **~5–6** |
| UD-Q4_K_XL | ~16–17 GB | ≈ 0,01–0,02 | ≈ original | ~2,5 | ~3–4 |

Dos datos de terceros encajan con este cálculo. Un portátil con 12 GB estima ~6 tok/s con Q2_K, y quien probó *offload* parcial en 16 GB cayó a ~1 tok/s ([guía Qwen 3.6](https://codersera.com/blog/how-to-run-qwen-3-6-locally-2026/amp/), [willitrunai](https://willitrunai.com/es/can-run/qwen-3.6-27b-on-rtx-3060-12gb)).

**Medición real (F0-b, 28 oct 2026, llama.cpp con Vulkan, ajuste automático de capas):**
- UD-Q3_K_XL va a **2,63 tok/s**: un 25 % por debajo de los ~3,5 calculados.
- Con la MTP baja a **1,89 tok/s**. Es decir, **en este portátil la MTP resta velocidad** y no suma los ×1,3–1,6 supuestos en la tabla: verificar varios tokens en la parte que corre en CPU queda limitado por cálculo.
- Hay que leer la tabla como cotas superiores. Ver [f0/README_F0.md](f0/README_F0.md).

**Conclusión sin rodeos:** en este portátil, **Qwen3.8-27B no puede ir a decenas de tok/s con su inteligencia casi intacta**. El techo físico para la calidad alta (Q3) son ~5–6 tok/s. Las «decenas» solo llegan con 1–2 bits, perdiendo justo lo que lo hace valioso. Ninguna técnica de software cambia esto, porque lo impone el ancho de banda de la RAM. Hay que mover una de las tres variables:

| Variable | Opción |
|---|---|
| Bytes por token | §3 (exprimir el portátil) y §4-B (modelo MoE) |
| Ancho de banda | §4-A (GPU con más VRAM) |
| Tokens necesarios por respuesta | §3.5 (gestionar el *thinking*) |

---

## 3. Plan A: exprimir Qwen3.8-27B en el portátil (máxima inteligencia, ~5–9 tok/s)

### 3.1 Colocación de tensores (la decisión más importante)

Regla: en VRAM va lo que tiene más estado o es pequeño y sensible. En RAM van las FFN, que son matmuls densas y regulares y en la CPU se ejecutan bien.

**En GPU:**
- Todos los mezcladores (DeltaNet y atención: ~7 B parámetros, ~2,5 GB a ~2,9 bpw).
- La cabeza LM (~0,9 GB).
- Las normas.
- KV y estados de DeltaNet.
- Las FFN completas de tantas capas como quepan con el espacio restante (~1 GB).

**En RAM y CPU:**
- Las FFN del resto de capas.
- El embedding.

En llama.cpp / ik_llama.cpp esto se hace con `-ngl 99`, que pone todo en GPU, más `-ot "blk\.(N..63)\.ffn_.*=CPU"`, que devuelve a la CPU las FFN de las capas indicadas. Se ajusta N hasta llenar la VRAM sin OOM.

**Por qué así:** mover un mezclador a la CPU mueve también su estado y su KV, y obliga a ida y vuelta de activaciones por capa. Las FFN son las que mejor toleran la CPU.

### 3.2 Cuantización mixta orientada a la calidad

- Partir de **UD-Q3_K_XL** (máxima inteligencia razonable) o **UD-Q2_K_XL** (compromiso).
- Mejora: cuantización **asimétrica por ubicación**:
  - Tensores en GPU a más bits (Q4–Q5): su coste en tiempo es 8 veces menor.
  - FFN en RAM a menos bits (IQ2/IQ3 con imatrix).

  Cada bit ahorrado en RAM vale 10 veces más que uno ahorrado en VRAM. **[est]** Con mezcladores y cabeza en Q4–Q5 en GPU y FFN de CPU en ~2,7 bpw, se obtiene la calidad de Q3_K_XL en lo sensible con un tamaño en RAM cercano a Q2: **~7–9 tok/s con MTP**.
- Validar con KLD frente a la versión Q8, no con perplejidad sola (Unsloth documenta casos en que la perplejidad engaña: [benchmarks Qwen3.5](https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks)).
- Probar los formatos trellis de ik_llama.cpp (IQ2_KT/IQ3_KT): mejor calidad por bit y buenos kernels de CPU.

### 3.3 Decodificación especulativa con la MTP nativa

- Activar la MTP del propio modelo: llama.cpp añadió soporte MTP para Qwen 3.6 [ext]. Hay que verificar el soporte para 3.8.
- Con un borrador de 2–3 tokens, la parte de GPU verifica casi gratis. En la parte de CPU, verificar k tokens a la vez reutiliza cada peso leído, pero el 5800H acaba limitado por cálculo. Ganancia esperada **×1,3–1,6** [est]; en GPU completa sería ×2–2,5.
- **No** usar un modelo borrador aparte: robaría VRAM a los pesos, que es el recurso más escaso.

### 3.4 Afinado de sistema

- **Hilos:** `-t 8` (núcleos físicos). Plan de energía de alto rendimiento y portátil enchufado.
- **Flash-attention** activada; KV en `q8_0` (pérdida despreciable gracias a las pocas capas con KV).
- **Contexto declarado** al tamaño real que se use (16–32 k): cada GB de KV ahorrado vuelve a la GPU como pesos.
- **Ancho de banda de la RAM:** medido en 30,5 GB/s (lectura secuencial con 4–16 hilos). Por encima de 4 hilos no mejora: el límite es la memoria, no la CPU. Para acercarse al pico de 51 GB/s haría falta RAM más rápida, y el portátil probablemente no admite más que DDR4-3200.
- Los *prompts* largos (*prefill*) se procesan en GPU transmitiendo pesos por PCIe en lotes grandes: son unos segundos por cada 8 k tokens [est]. Es aceptable.

### 3.5 Gestionar el *thinking* (la palanca más infravalorada)

La latencia que nota el usuario depende de los tokens por respuesta tanto como de los tok/s. En modo `xhigh` el modelo puede generar miles de tokens de razonamiento: 3 000 tokens a 8 tok/s son 6 minutos.

| Tipo de petición | `reasoning_effort` |
|---|---|
| Por defecto | `medium` |
| Problemas difíciles | `xhigh` |
| Chat y extracción | `low`, o `enable_thinking: false` |

Con esto se suele reducir el tiempo por respuesta más que pasando de Q3 a Q2, sin tocar los pesos.

**Resultado esperado del Plan A:** Qwen3.8-27B a **~5–6 tok/s (Q3) o ~7–9 tok/s (Q2 / mixta)** [est], con contexto de 32 k. No llega a «decenas».

---

## 4. Planes para ir más rápido

### 4-A. Cambiar el ancho de banda: GPU con ≥ 16–24 GB (mantiene la máxima inteligencia)

| Hardware | Cuantización que cabe | Roofline [cal] | Con MTP en GPU (×2–2,5) [est] |
|---|---|---|---|
| RTX 4060 Ti 16 GB (288 GB/s) | Q3_K_XL (13 GB) | ~22 tok/s | ~40–50 |
| **RTX 3090 24 GB usada (936 GB/s)** | Q4_K_XL (~16,5 GB) | ~57 tok/s | **~60–78 [ext]** |
| RTX 5090 32 GB (1,8 TB/s) | Q6–Q8 | ~65–100 tok/s | >100 |

Notas:
- El portátil (Ryzen 5000, probablemente sin Thunderbolt/USB4) no admite una eGPU razonable. Esto implica un sobremesa, un segundo equipo en la red local con un servidor compatible con la API de OpenAI, o un servidor en la nube.
- La RTX 3090 es la mejor relación inteligencia/velocidad/coste para este objetivo.

### 4-B. Cambiar los bytes por token: modelo MoE de la familia (en este portátil)

[Qwen3.6-35B-A3B](https://www.dsebastien.net/qwen3-6-35b-a3b/) tiene 35 B de parámetros totales y **~3 B activos** [ext].
- Colocación: atención, router y expertos compartidos en GPU; expertos en RAM (`--n-cpu-moe`).
- Lectura por token: ≈ 1,7 GB en Q4, de los que ~1,4 GB son expertos en RAM. **Roofline ≈ 20 tok/s [cal]**; real **~12–20** [est]. Una caché de expertos calientes en VRAM la sube.
- No se ha encontrado un MoE de la familia 3.8. Si se publica, sería el candidato natural.
- **Coste:** menos inteligencia que el 27B denso, sobre todo en código y tareas agénticas, donde Qwen sitúa al 27B por encima [ext].

### 4-C. Combinación recomendada si no se cambia de hardware

Dos modelos detrás del mismo servidor:
- **MoE A3B** por defecto, a ~12–20 tok/s, para el día a día.
- **Qwen3.8-27B en Q3/mixta** bajo demanda, para lo difícil.

Los 32 GB de RAM no admiten ambos residentes con cuantizaciones altas: 13 GB + ~20 GB superan lo útil. Opciones: el MoE en Q3/IQ3 (~14 GB) más el 27B en Q2/mixta (~10 GB), o cargar bajo demanda.

---

## 5. Líneas de I+D (mejoras sobre el estado del arte; no garantizadas)

Cada una ataca el término dominante, los bytes que hay que leer desde RAM:

| Línea | Idea | Ganancia potencial [est] | Riesgo |
|---|---|---|---|
| **Esparsidad de activaciones en FFN de CPU** (TEAL, PowerInfer) | Saltar filas/columnas con activación ≈ 0. Sin reentrenar, 40–50 % de esparsidad en FFN SwiGLU [ext, otros modelos] | Hasta ×1,5–1,8 en la parte de CPU | Requiere un kernel de CPU propio y validación de calidad; en batch 1 la lectura dispersa rinde peor |
| **Poda estructurada + reentrenamiento corto** (Minitron, ShortGPT) | Quitar ~20–25 % de capas o de anchura de FFN y recuperar con destilación | ~×1,3 | Pérdida de inteligencia; coste de entrenamiento (nube) |
| **Cuantización consciente de ubicación** (§3.2 llevado al límite) | Asignar bits por tensor minimizando KLD con la restricción `tiempo = G/315 + S_ram/30` | ×1,2–1,4 frente a una cuantización uniforme de igual calidad | Bajo; es optimización de configuración |
| **Salida temprana como borrador** (LayerSkip) | Usar las primeras capas (las de GPU) como borrador en lugar de la MTP | Complementa a la MTP | Requiere ajuste fino |

Ninguna de estas líneas lleva el 27B en Q3 de ~5–6 a «decenas» por sí sola. Combinadas (esparsidad + bits por ubicación + MTP), el rango optimista es **~10–14 tok/s** [est] con calidad cercana a Q3.

---

## 6. Plan de validación (benchmark)

1. **Medidas base:**
   - Ancho de banda de la RAM.
   - tok/s del 27B en UD-Q2_K_XL y UD-Q3_K_XL con la colocación de §3.1, sin MTP y con MTP.
   - Prompts de 512 y 8 k tokens.
2. **Calidad frente a Q8** (ejecutado lento, como referencia):
   - KLD medio sobre 1 M tokens de texto variado.
   - GSM8K y MMLU-Pro (subconjuntos).
   - LiveCodeBench, subconjunto pequeño.
   - Una prueba de llamada a herramientas.
3. **Métrica que importa:** tiempo por respuesta = tokens generados / tok/s, con `reasoning_effort` low / medium / xhigh.
4. **Criterios de aceptación:**
   - KLD ≤ 0,05 (equivale a Q3).
   - Pérdida en evaluaciones ≤ 2 puntos frente a Q8.
   - Reportar siempre tok/s y KLD juntos.
5. **Comparar** con el plan 4-B (MoE A3B) en las mismas pruebas. Así se decide con datos si el salto de inteligencia del 27B justifica ir 3–4 veces más lento.

---

## 7. Conclusión

- Qwen3.8-27B ya incorpora lo bueno del proyecto original: estado de contexto casi constante (DeltaNet), KV pequeño y MTP para especular.
- Lo que no puede cambiar es que es **denso**: cada token lee ~25 B parámetros. Con 6 GB de VRAM, la mayoría tiene que venir de una RAM a ~30 GB/s [med].
- **Máxima inteligencia en este portátil:** ~5–9 tok/s (Q3 / mixta + MTP + colocación óptima + *thinking* gestionado).
- **Decenas de tok/s con esa inteligencia:** exige más ancho de banda, es decir, una GPU de 16–24 GB, que daría ~40–78 tok/s.
- **Más velocidad en este portátil:** exige un modelo MoE (~12–20 tok/s), con algo menos de inteligencia. Las «decenas» reales en este equipo solo llegan con cuantización de 1 bit.
