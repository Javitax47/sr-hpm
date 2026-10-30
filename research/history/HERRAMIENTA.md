# SR-HPM como herramienta: uso, escalado y límites

## Qué es

Un conjunto de cambios sobre llama.cpp ([f3/llama-srhpm.patch](f3/llama-srhpm.patch)) y un lanzador
([srhpm/launch.py](srhpm/launch.py)). Sirve para ejecutar **modelos densos más grandes que la VRAM** en hardware de consumo, repartiendo el trabajo entre GPU y CPU y usando decodificación especulativa.

Ninguna pieza depende del modelo concreto: el lanzador lee el GGUF y decide el reparto.

```bash
python -m srhpm.launch MODELO.gguf --draft BORRADOR.gguf --ctx 8192 --run
```

| Caso detectado | Qué hace |
|---|---|
| El modelo cabe en VRAM | Todo en GPU (`-ngl 99`). Las técnicas de CPU no aplican |
| MoE que no cabe | Expertos en CPU, el resto en GPU (`-ot exps=CPU`). Sin especulación por defecto |
| Denso que no cabe | Ruta SR-HPM, descrita abajo |

Componentes de la ruta SR-HPM:
- **Kernels de CPU para lotes pequeños** (`GGML_SRV`). Cada bloque de pesos se decodifica una vez para todos los tokens que se verifican. Cubren IQ4_XS, IQ3_S, IQ3_XXS, Q3_K, IQ2_*, Q4_K, Q5_K y Q6_K, y dan resultados **idénticos bit a bit** a llama.cpp (KLD 0).
- **C4** (`GGML_SRV_C4`). IQ3 se reescribe sin pérdida a códigos de 4 bits, que se decodifican con una sola tabla (`pshufb`).
- **Capas de GPU elegidas por coste** (`LLAMA_GPU_LAYERS`). Se pone en GPU el tramo contiguo de capas con más tiempo de CPU por byte.
- **Cabeza de salida en GPU** cuando el borrador la comparte (DFlash).
- **Instantáneas de estado recurrente escritas en su sitio** (`SRHPM_GDN_DIRECT`), en modelos híbridos.
- **Atención agrupada por GQA en CPU** para lotes pequeños (`SRHPM_FA_GQA`).
- **Op offload corregido** (`GGML_CUDA_SRHPM_TUNE`). Solo los productos de matrices de lotes grandes se suben a la GPU (prefill rápido) y ya no hay un ida y vuelta por capa al generar. Además, el cambio de MMVQ a MMQ se ajusta por tipo.

## Estrategias según la máquina (detección automática, `srhpm/platform.py`)

| Condición detectada | Qué aplica la herramienta |
|---|---|
| Siempre (modelo denso más grande que la VRAM) | Kernels SR-HPM, C4, capas de GPU por coste, atención GQA, op offload corregido y lotes de 1024 en el prefill |
| Siempre, con borrador DFlash | Borrador adaptativo: 3 tokens fijos y hasta 7 si el selector está seguro (≥ 0,9) |
| Siempre, con GPU en modo híbrido | *Keepalive* de la GPU (`GGML_CUDA_KEEPALIVE=200`) para que no baje a reposo entre ráfagas |
| Modelo con capas recurrentes (GatedDeltaNet) | Instantáneas perezosas: 2 estados completos por pasada y *replay* exacto al deshacer (`SRHPM_GDN_LAZY=1`) |
| Privilegio "Bloquear páginas en memoria" (Windows) o páginas enormes transparentes (Linux) | `GGML_LARGE_PAGES=1`: pesos en páginas de 2 MB |
| Linux | 0,3 GiB de reserva de VRAM en lugar de 1 GiB, así que caben más capas en GPU; compilación con `srhpm/build_linux.sh` |
| CPU con AVX-VNNI o AVX512-VNNI (Intel de 12.ª gen. en adelante, Zen 4/5) | Aviso para compilar ggml-cpu en esa máquina; los kernels usan entonces `vpdpwssd` |
| ≥ 60 GB libres y los pesos originales (`--source`) | Plan de recuantización para esa máquina (`srhpm/requant.py`): BF16, matriz de importancia de tu texto, mezcla de tipos con kernel SR-HPM y comprobación de KLD |
| `--retrieval` (cargas repetitivas) | Recuperación de n-gramas del contexto antes que el borrador |

En esta máquina (Windows, Zen 3, sin el privilegio y sin espacio) solo se aplica la primera fila. Las demás están implementadas y se activan solas cuando se cumple la condición. Su ganancia no se ha podido medir aquí.

## Lo medido en este portátil

RTX 3060 Laptop 6 GB, Ryzen 7 5800H, 32 GB DDR4-3200.

| Modelo | Reparto | Generación (tok/s) | Prefill (tok/s) |
|---|---|---|---|
| Qwen3.8-27B UD-Q3_K_XL, llama.cpp de serie | `-ngl 17`, DFlash2 | 2,7 | 220 |
| **Qwen3.8-27B UD-Q3_K_XL, SR-HPM** (ajuste manual) | 22 capas + salida, DFlash2 | **8,5–8,7** | 115–168 |
| Qwen3.8-27B, SR-HPM con el lanzador automático | 20 capas + salida | 8,2 | — |
| **Qwen3.8-27B, SR-HPM F6** (borrador adaptativo, IQ4_XS emparejado, keepalive, GDN perezoso; [f3/README_F6.md](f3/README_F6.md)) | 20 capas + salida | **~9,7** (4 de 5 prompts > 10) | 115–131 |
| K2-Horizon 36B-A4B (MoE, 21 GB), de serie | expertos en CPU | 15,6 | 174 |
| jevk5-4B (cabe en VRAM) | todo en GPU | 75 | 2.777 |
| MiniCPM5-2B (cabe en VRAM) | todo en GPU | 136 | 5.172 |

En el MoE, las técnicas SR-HPM no mejoran: por token solo se leen los ~4B de parámetros activos, y al verificar varios tokens se activan expertos distintos.

## Contexto largo (Qwen3.8-27B, SR-HPM)

| Prompt | Prefill | Hasta la primera palabra | Generación antes | **Generación con atención GQA** | Aceptación |
|---|---|---|---|---|---|
| 525 tokens | 87 tok/s | 6 s | 7,5 tok/s | **7,7 tok/s** | 0,71 |
| 2.425 tokens | 113 tok/s | 21 s | 5,9 tok/s | **6,1 tok/s** | 0,50 |
| 9.658 tokens | 117 tok/s | 83 s | 4,0 tok/s | **5,9 tok/s** | 0,53 |

**Atención en CPU para la verificación** (`SRHPM_FA_GQA=1`).
- **Problema:** con 10k tokens, la atención de las 12 capas de atención que van en CPU llegaba al 33–36 % de la pasada. llama.cpp procesaba cada fila (token × cabeza) por separado: las 24 filas que comparten cabeza K/V en la GQA de Qwen3.8 convertían de f16 y multiplicaban los mismos datos 24 veces.
- **Solución:** el kernel nuevo agrupa esas 24 filas en un mosaico. Cada bloque K/V se convierte una sola vez y se calcula como un GEMM pequeño. La caché KV se reparte en trozos entre los hilos (softmax incremental por trozo y reducción por fila).
- **Resultado:** la atención baja al ~1 % del tiempo y la generación con 10k de contexto sube un **48 %**.
- **Validación:** pasa los 4042 tests de `FLASH_ATTN_EXT` de `test-backend-ops` contra CUDA. Con prompts cortos no cambia nada: 8,46 tok/s, frente a 8,5.

Lo que queda con contexto largo es la aceptación del borrador, que baja en textos largos.

## Escalado estimado a otro hardware

Calculado con [srhpm/scaling.py](srhpm/scaling.py), calibrado con la medida del portátil (predice 8,5 frente a 8,6 medido). Es una estimación, no una medida.

| Modelo | Hardware | Sin especulación | SR-HPM |
|---|---|---|---|
| Qwen3.8-27B Q3_K_XL | Este portátil | 3,3 | 8,5 |
| | Solo CPU, DDR4 | 2,4 | 6,3 |
| | Portátil RTX 4060 8 GB + DDR5-5600 | 5,4 | 12,9 |
| | Sobremesa RTX 4060 Ti 16 GB + DDR5 | 14,5 | 22,7 |
| | RTX 3090/4090 24 GB | 30,9 | 46,7 |
| Denso 70B Q3_K_XL | Este portátil (justo de RAM) | 1,2 | 3,3 |
| | Sobremesa 16 GB + DDR5 | 2,9 | 7,1 |
| | RTX 3090/4090 24 GB + DDR5 | 4,8 | 11,3 |

La ganancia de SR-HPM es mayor cuanto más peso queda en CPU: ×2,5 en este portátil, frente a ×1,5 cuando casi todo cabe en VRAM.

## Límites conocidos

- **Borrador:** hace falta uno compatible con el modelo (un DFlash, o un modelo pequeño de la misma familia). Sin borrador, la ganancia se reduce a los kernels y al reparto.
- **Plataforma:** los kernels de CPU son AVX2 (x86). En ARM (Mac, Snapdragon) haría falta portarlos a NEON.
- **Windows:** reserva ~1 GB de VRAM, y si la VRAM se llena, el controlador desborda a memoria del sistema sin avisar (más lento). El lanzador deja margen.
