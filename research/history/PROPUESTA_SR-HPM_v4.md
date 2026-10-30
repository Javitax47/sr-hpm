# SR-HPM v4 — Motor especulativo tri-canal con verificación por lotes en CPU
### Qwen3.8-27B en un portátil con RTX 3060 6 GB + 32 GB DDR4, rediseñado a partir de lo medido

**Fecha:** 28 oct 2026.
**Sustituye a:** [PROPUESTA_SR-HPM_v3.md](PROPUESTA_SR-HPM_v3.md). La v3 queda como antecedente; su kernel GPU se reutiliza aquí.
**Convención:** [med] medido en esta máquina · [cal] calculado con el modelo calibrado · [est] estimado · [ext] dato de terceros.

---

## 1. Qué hemos aprendido (todo medido en este portátil)

| # | Hecho | Fuente |
|---|---|---|
| H1 | Qwen3.8-27B UD-Q3_K_XL con llama.cpp: **2,6 tok/s** | F0-b |
| H2 | La especulación con llama.cpp **no acelera, o frena**: MTP n = 8: 1,9 tok/s; DFlash2 n = 2 / 3 / 7: 2,6 / 2,5 / 2,5 tok/s | F0-b, este trabajo |
| H3 | Los borradores son buenos. **DFlash2** (2 B parámetros, 1,1 GB): **τ = 4,48 tokens por pasada** (p ≈ 0,82); 5,9–7,2 en código y matemáticas, 2,7–3,8 en prosa. IQ1_S: τ ≈ 4,4 | F0-b + DFlash2 |
| H4 | La causa de H2: en llama.cpp, verificar M tokens en CPU cuesta **2,1–2,7× con M = 4** y **3,3–3,9× con M = 8** | `llama-bench` |
| H5 | **No es un límite físico.** El motor propio (formato Q4S, pool adaptativo), en una **pasada completa sobre 8 GB**, cuesta **1,07–1,15× con M = 4 y 1,46–1,79× con M = 8**. Con M = 1 lee a **36 GB/s**, con *overhead* de orquestación casi nulo. El micro inicial sin escalas daba 1,18× / 1,77× | **F2 / F2b** |
| H6 | PCIe en Gen3 **x8**: **6,7 GB/s** | F1 |
| H7 | El streaming PCIe solapado con el cálculo de GPU funciona: eficiencia **0,99**. El kernel GPU (B+R)·X rinde **7,4 TFLOPS** | F1 |
| H8 | La RAM rinde 30,5 GB/s (27,5 en el kernel); la CPU, ~0,5–0,7 TOPS int8 con lotes | F1/F2 |
| H9 | Un borrador obtenido como subproducto del cuantizador del objetivo no sirve (p = 7 %). Ha de estar entrenado u optimizado para sí mismo | F0 |
| H10 | El culling de columnas de la FFN no sirve: esparsidad de la unión ≈ 0 % | F0 |

**Diagnóstico:** el cuello no es el ancho de banda ni el borrador, sino **el cálculo de la verificación en CPU**. Toda la v4 sale de ahí.

## 2. Idea central

> **En hardware de consumo con el modelo en la RAM, la especulación solo acelera si la verificación de M tokens cuesta casi lo mismo que la de 1. En esta CPU eso es posible (H5), pero no con los kernels actuales. La v4 construye la decodificación alrededor de esa propiedad y añade el PCIe como segundo canal de pesos sin coste de cálculo.**

```
             VRAM 6 GB (315 GB/s)                       RAM 32 GB (27,5 GB/s)
 ┌───────────────────────────────────────┐   ┌───────────────────────────────────────┐
 │ Borrador DFlash2 (1,1 GB, SOTA)       │   │ Capas "CPU" en formato de planos       │
 │ Capas "residentes" del objetivo       │   │ de 4 bits (C1): la CPU las calcula     │
 │ (~3,5 GB) + KV/estados DeltaNet       │   │ in situ para M = n+1 tokens            │
 │ Anillo de 3 capas "streamed" ◄────────┼───┤ Capas "PCIe": el mismo formato,        │
 └───────────────────────────────────────┘   │ enviadas a la GPU durante el cálculo    │
            ▲  6,7 GB/s, solapado            │ de la CPU (C2)                          │
            └────────────────────────────────┴───────────────────────────────────────┘
 Pasada:  borrador (n tokens, 1 forward de DFlash2) → verificar M = n+1 tokens capa a capa
          [GPU residente] → [CPU: kernel por lotes C1] ↔ [GPU: capas recibidas por PCIe]
          → aceptar prefijo → re-anclar el borrador → n se adapta al contenido (C3)
```

## 3. Componentes

### Del estado del arte (adoptados, no reivindicados)

| Componente | Qué aporta | Fuente |
|---|---|---|
| **DFlash2**: borrador por difusión de bloques, condicionado por los estados del objetivo | τ = 4,48 medido aquí; pequeño (1,1 GB en Q4_K_M) | [DFlash](https://arxiv.org/pdf/2602.06036), [DFlash2 Qwen3.8-27B](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF) |
| **SuffixDecoding**: borrador sin modelo, a partir del contexto y el historial | Mucha aceptación en código y tareas agénticas que copian contexto; corre en CPU y cuesta casi nada | [SuffixDecoding](https://arxiv.org/pdf/2411.04975) |
| **Verificación en CPU, borrador en GPU** | Precedente del reparto heterogéneo | [Dovetail](https://arxiv.org/html/2412.18934v2) |
| **Especulación para ocultar el offload** | Precedente del uso del PCIe | [SpecExec](https://arxiv.org/html/2406.02532), [SpecOffload](https://arxiv.org/html/2505.10259v3) |
| **Kernels de matmul por lotes en CPU** (tinyBLAS de llamafile, *repack* de ggml) | Precedente de C1, orientado a prompts grandes | [llamafile sgemm](https://redirect.github.com/Mozilla-Ocho/llamafile/releases/tag/0.8.2) |

### Propios

**C1. Verificador de CPU por lotes pequeños (M ≤ 16)** — *medido en F2.*
- Formato de pesos en «planos» de 4 bits, de modo que el desempaquetado es un desplazamiento sobre bloques contiguos.
- Se combinan `vpmaddubsw` y `vpmaddwd`, cada bloque de pesos se lee una sola vez y se reutiliza para los M tokens, y se procesan 2 filas con bloques de 4 tokens.
- Novedad: el objetivo de diseño no es el *prefill* (M grande, donde ya existen kernels buenos), sino el **régimen M = 2–8 de la verificación especulativa**, donde llama.cpp pierde ×2–3.
- Resultado: coste de M = 4 de **1,18×** frente a 2,1–2,7×.

**C2. Tri-canal con streaming solapado al cálculo de la CPU.**
- Las capas del objetivo se reparten en tres clases: residentes en VRAM, calculadas por la CPU desde la RAM y **enviadas por PCIe a la GPU mientras la CPU calcula otras capas**.
- Las capas se ejecutan en orden, pero la DMA de las capas «PCIe» no depende de los datos, así que se adelanta durante el tramo de CPU.
- La GPU tiene cálculo de sobra (H7), así que cada GB enviado por PCIe es un GB que la CPU no tiene que leer ni calcular.
- El reparto óptimo se resuelve por pasada, igualando el tiempo de PCIe con el de CPU (`srhpm.budget.v4_tok_s`). Con los valores medidos van **~3–3,6 GB por PCIe** en cada pasada.

**C3. Longitud de borrador según el coste y el contenido** (control de tasa, como en vídeo).
- Por pasada se elige n para maximizar τ(n, p̂) / T(n). p̂ es la aceptación reciente (media móvil por tipo de contenido) y T(n) es **la curva de coste medida** de C1 (1,00 / 1,08 / 1,18 / 1,77…).
- Con p alto (código, JSON), n = 7. Con p bajo (prosa), n = 3–4.
- La diferencia con EAGLE-2 o SuffixDecoding, que adaptan la longitud por confianza del borrador, es que aquí se optimiza **tiempo de pared con un coste de verificación no lineal y medido**.

**C4. Formato de pesos único para los dos canales.**
- El mismo formato de planos de 4 bits lo consumen el kernel de CPU (C1) y el kernel Triton de GPU (v3/F1, 7,4 TFLOPS).
- Una capa puede **migrar entre canales en tiempo real** sin recodificarse. Por ejemplo, cuando el contexto crece y el KV ocupa VRAM, o cuando la carga de la CPU cambia.

**C5. Borrador adaptado al dominio del usuario.**
- Cada token verificado es una etiqueta gratuita.
- En segundo plano se ajusta un LoRA del borrador DFlash2 con las salidas verificadas del propio objetivo: autodestilación en línea, que extiende la idea de *Online Speculative Decoding*.
- Se combina con SuffixDecoding para el texto que repite el contexto.
- Objetivo: llevar p hacia el ~0,9 que ya se mide en código.
- [est] Es la palanca más incierta.

## 4. Rendimiento previsto

### 4.1 Con el motor propio F2b (formato Q4S, 13,4 GB, calidad entre Q4_0 y Q4_K)

Curva de coste medida: M = 1/2/3/4/8 → 1,00 / 1,08 / 1,10 / 1,11 / 1,63, a 36 GB/s. 3,5 GB residentes. PCIe a 6,7 GB/s. n ≤ 7.

| *Overhead* por pasada | Canales | Media | Código / matemáticas | Prosa |
|---|---|---|---|---|
| 0,08 s (conservador, el de llama.cpp) | CPU + VRAM | 8,4 | 12,1 | 6,5 |
| 0,08 s | **Tri-canal** | **10,8** | **15,7** | **8,0** |
| 0,03 s [est] (el motor propio no tiene *overhead* medible en las GEMM) | Tri-canal | 11,9 | 17,3 | 9,1 |

Todo en tok/s [cal]. Supone **×4–4,5 sobre lo actual (2,6 tok/s)**, con calidad cercana a Q4. Las cifras de §4.2 (curva del micro inicial) quedan como cota inferior.

### 4.2 Versión anterior: micro F2 sin escalas, modelo calibrado con H1–H8

Supuestos:
- 3,5 GB residentes (tras el borrador y la KV).
- CPU a 27,5 GB/s con la curva de coste de C1 (F2).
- PCIe a 6,7 GB/s.
- Overhead de 0,08 s por pasada, calibrado con llama.cpp: con el mismo modelo, sin especulación, salen 2,6 tok/s frente a 2,6 medidos.
- n ≤ 7 (bloque de DFlash2).
- p sacado de τ medido: media 4,48; código ~6,5; prosa ~3,2.

| Objetivo | Canales | Media | Código / matemáticas | Prosa |
|---|---|---|---|---|
| Q3 (3,9 bpw, 12,4 GB) | CPU (C1) + VRAM | 6,9 | 9,8 | 5,4 |
| **Q3 (3,9 bpw)** | **Tri-canal (C1 + C2)** | **9,7** | **14,0** | **7,1** |
| Q4 (4,5 bpw, 14,0 GB), más calidad | CPU + VRAM | 6,0 | 8,5 | 4,7 |
| **Q4 (4,5 bpw)** | **Tri-canal** | **8,4** | **12,1** | **6,2** |

Todo en tok/s [cal]. Referencias [med]: llama.cpp 2,6 tok/s; llama.cpp + DFlash2, 2,5–2,6 tok/s.

**Lectura honesta:**
- **×3–4 sobre lo actual**, con calidad Q3 o incluso Q4. Q4 está por encima de lo que permitía cualquier propuesta anterior en esta máquina.
- **Decenas de tok/s sostenidas: no.** Solo se rozan en código y matemáticas (12–14). El techo lo fija τ ≤ 1/(1−p) junto con el tiempo de leer de RAM lo que no cabe en VRAM: ~0,25 s por pasada como mínimo.
- Con C5 (p → 0,9 en el dominio del usuario), la media podría subir a ~12–15 [est]. No está medido.

## 5. Ruta de implementación

La vía más corta hasta algo que puedas usar es **integrarlo en llama.cpp**, que ya tiene DFlash2, MTP, `llama-server` y el envío de operaciones a la GPU:

| Fase | Trabajo | Criterio para seguir |
|---|---|---|
| **F3** | C1 dentro de ggml: un nuevo tipo de *repack* para AVX2 con M pequeño (Q4_0 y planos). Medir `llama-bench` pp1…pp8 con CPU sola | Coste de pp4/pp1 ≤ 1,4× (hoy 2,0–2,7) |
| **F4** | DFlash2 de extremo a extremo con F3, n = 3–7, sin C2 | ≥ 5 tok/s de media (hoy 2,6). Si no se llega: **abandonar** |
| F5 | C3, control de n según contenido, en `common/speculative` | +10–20 % sobre F4 |
| F6 | C2: streaming PCIe de capas solapado con el tramo de CPU. Extiende el envío de operaciones a la GPU con un anillo de 3 capas | ≥ 8 tok/s de media |
| F7 | C5: LoRA del borrador por autodestilación + SuffixDecoding | p̂ medido en tu uso ≥ 0,85 |

Coste de entrada bajo: F3 es un kernel de ~200 líneas más su registro en ggml, y F4 usa piezas que ya existen.

**Resultado F3/F4 (28 oct. 2026, [f3/README_F3.md](f3/README_F3.md)): las dos se cumplen.**
- **F3:** con el *repack* de llama.cpp activado (`--no-host -lm none`), M = 4 cuesta 1,12× lo que M = 1 con la ruta de serie y 1,07× con el kernel Q4S.
- **F4:** DFlash2 n = 3 da **6,30 tok/s** de media, frente a 2,47 sin borrador.
- **Q4S frente a la ruta de serie:** gana un 15–16 % con n = 5–7 (5,82 frente a 5,04) y pierde un 9 % con n = 3. El mejor punto sigue siendo la ruta de serie con n = 3.
- **Exactitud de Q4S:** no es bit-exacto. KLD 0,0047, PPL +0,6 %, causado por la escala de activación por cada 256 columnas.
- **Para F5/F6:** Q4S solo aporta cuando la pasada verifica ≥ 6 tokens.

**Resultado F5 (28 oct. 2026, [f3/README_F5.md](f3/README_F5.md)):**
- Qwen3.8-27B UD-Q3_K_XL llega a **8,6 tok/s de media** (de 2,7 a 10,9 según el prompt), exacto frente a llama.cpp (KLD 0).
- Lo que más aporta, por orden: kernels SRV, capas de GPU elegidas por coste, `--no-op-offload`, C4 y la escritura directa de instantáneas GDN.
- F6 (PCIe) se descarta en este portátil: cada cambio CPU↔GPU cuesta ~1,7 ms con WDDM y la VRAM útil es de 5,1 GB, así que la ganancia neta estimada es de ~5 %.

## 6. Qué se descarta y por qué

| Descartado | Motivo medido |
|---|---|
| Núcleo «residente en SRAM», memoria holográfica, atractores (v1) | Ver [AUDITORIA.md](AUDITORIA.md) |
| Borrador como subproducto anidado del objetivo (v3, F0) | p = 7 % (H9) |
| Verificación solo por PCIe (v3) | 6,7 GB/s (H6): ~7,5 tok/s como mucho y requiere cuantización propia. v4 la reutiliza como canal secundario |
| Culling de columnas de la FFN | Esparsidad de la unión ≈ 0 (H10) |
| MTP como borrador principal | τ similar al de DFlash2, pero propone token a token. DFlash2 genera el bloque en un solo forward |

## 7. Si «decenas» es imprescindible

Con este portátil no se puede de forma sostenida (§4). Con hardware [cal/ext]:
- **GPU de 16 GB** (p. ej. RTX 5060 Ti/4060 Ti, ~288–448 GB/s): Q3/Q4 del 27B entero en VRAM → ~20–30 tok/s sin especulación; con DFlash2 (τ ≈ 4,5) y verificación en GPU (barata), **~50–80**.
- **RTX 3090 de 24 GB usada:** Q4–Q5 con DFlash2, **> 60 tok/s** [ext: 60–78 con especulación].
- En ese hardware, C3 y C5 siguen aportando; C1 y C2 dejan de hacer falta.
