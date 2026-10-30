# F3/F4: kernel Q4S dentro de llama.cpp y DFlash2 de extremo a extremo

**Equipo:** RTX 3060 Laptop 6 GB, Ryzen 7 5800H, 32 GB DDR4-3200.
**Modelo:** Qwen3.8-27B, `speedtest-Q4_0.gguf` (15 GB).
**Borrador:** DFlash2 Q4_K_M, 1,1 GB.
**Base:** llama.cpp `d05706b`, compilado con CUDA sm_86 y backend CPU AVX2.

## Qué se ha hecho

| Pieza | Archivo |
|---|---|
| Kernel Q4S como nuevo *repack* de ggml-cpu. Se activa con `GGML_Q4S=1` y solo afecta a los tensores Q4_0 con K % 256 = 0 | `ggml/src/ggml-cpu/q4s-verify.inc` (234 líneas) y `repack.cpp` (+10). Parche completo en [llama-q4s.patch](llama-q4s.patch) |
| Batería F4 con `llama-server` | [run_f4.sh](run_f4.sh), que usa [f0/f0b_spec.py](../f0/f0b_spec.py) |
| Medición de exactitud (KLD frente a la ruta de serie) | [run_kld.sh](run_kld.sh), resultado en [results/f3_kld.json](results/f3_kld.json) |

Formato Q4S, el mismo que en F2b:
- Los códigos de 4 bits se empaquetan por pares de bloques de 32 columnas.
- Cada bloque guarda una sub-escala int16 relativa al máximo del súper-bloque de 256 columnas. Esa sub-escala se aplica dentro de `vpmaddwd`.
- La conversión a fp32 se hace una sola vez por súper-bloque.
- Las activaciones se pasan a u8 (q + 128) con una escala por cada 256 columnas.
- El término de desplazamiento se precalcula por fila.
- Los tokens se procesan en tiles fijos de 4/3/2/1.

## Hallazgo previo: el *repack* de llama.cpp estaba apagado

El kernel de serie para M pequeño (Q4_0 8x8 de `repack.cpp`) **no se usa** si los pesos de CPU están en mmap (`CPU_Mapped`) o en memoria anclada para CUDA (`CUDA_Host`). Hay que lanzar con **`-lm none --no-host`**; el log debe mostrar `CPU_REPACK model buffer size = …`.

Solo con eso, DFlash2 n = 3 pasa de 2,6 tok/s (F0-b) a **6,3 tok/s**. Es la mayor mejora de esta fase, y no necesita código propio.

## F3: coste por lote, solo CPU

Medido con `llama-bench`, `-ngl 0`, *repack* activado. La tabla da segundos por lote de M tokens:

| Configuración | M=1 | M=2 | M=3 | M=4 | M=8 |
|---|---|---|---|---|---|
| 8 hilos, de serie | 0,474 | 0,658 | 0,840 | 0,687 | 1,196 |
| 8 hilos, Q4S | 0,510 | 0,562 | 0,596 | 0,679 | 1,024 |
| 16 hilos, de serie | 0,493 | 0,629 | 0,759 | **0,554** | 0,952 |
| 16 hilos, Q4S | 0,559 | 0,599 | **0,588** | 0,600 | **0,831** |

El criterio de F3 era coste(M=4)/coste(M=1) ≤ 1,4. Se cumple con ambos: 1,12 de serie y 1,07 con Q4S.

Q4S gana en M = 2, 3 y 8. La ruta de serie gana en M = 1 y M = 4: su GEMM 4×8 aprovecha bien M = 4, mientras que con M = 2 o 3 vuelve a un GEMV por token.

## F4: DFlash2 de extremo a extremo

Configuración:
- `llama-server` con `-ngl 14` (el resto de capas en CPU) y `-np 1`.
- Borrador en la GPU, `-tb 16`.
- 5 prompts × 200 tokens, temperatura 0.

| Borrador | De serie (tok/s) | Q4S (tok/s) | Aceptación | Tokens aceptados por pasada |
|---|---|---|---|---|
| ninguno | 2,47 | 2,36 | — | — |
| DFlash2 n = 3 | **6,30** | 5,71 | 0,76 | 2,29 |
| DFlash2 n = 5 | 5,04 | **5,82** | 0,62–0,65 | 3,1–3,3 |
| DFlash2 n = 7 | 5,00 | **5,78** | 0,52 | 3,66 |

- **F4 se cumple**: 6,30 tok/s de media, por encima del umbral de 5. Son 2,55× sin borrador y 2,4× F0-b.
- Q4S gana un 15–16 % con n = 5 y n = 7, donde la pasada verifica 6 u 8 tokens. Pierde un 9 % con n = 3, porque M = 4 es justo el caso en que la ruta de serie es más rápida.
- **El mejor punto sigue siendo la ruta de serie con n = 3.** Con Q4S el máximo es 5,82 (n = 5), un 8 % por debajo.
- Hay que usar `-np 1`. Con los 4 *slots* por defecto, n ≥ 5 se queda sin VRAM al cargar el borrador.
- El prompt de código (el n.º 2) es el peor en todas las configuraciones: 3–4,5 tok/s, con aceptación del 30–45 %. Ahí entra F5 (n según el contenido).

Tampoco hay modo híbrido barato. Q4S reescribe los pesos en su propio formato, así que usar la ruta de serie para M = 1 y 4 obligaría a guardar dos copias de 11 GB.

## Exactitud: Q4S no es bit-exacto

Resultado de `llama-perplexity --kl-divergence` con 8 fragmentos de 512 tokens, solo CPU:

| Métrica | Valor |
|---|---|
| PPL de serie / Q4S | 7,439 / 7,483 (+0,60 % ± 0,25 %) |
| KLD media / mediana / máxima | 0,0047 / 0,0024 / 0,55 |
| Mismo token top-1 | 96,4 % ± 0,4 % |

Causa: la ruta de serie cuantiza las activaciones a Q8 con una escala por cada 32 columnas. Q4S usa una sola escala por cada 256, y un valor atípico le quita resolución a las otras 255 columnas. Esto explica que la aceptación con n = 3 sea algo menor (0,750 frente a 0,763).

El error es pequeño, pero no despreciable: equivale a una fracción apreciable del que ya introduce pasar de Q4_K a Q4_0. Para corregirlo hay que usar escalas por cada 64 columnas, al precio de una conversión a fp32 más por bloque. Ver la sección siguiente.

## Conclusión y próximos pasos

1. **La configuración utilizable hoy, sin código propio**, es esta:

   ```
   llama-server -m <Q4_0> -md <DFlash2> --spec-type draft-dflash --draft-max 3 \
     -ngl 14 -ngld 99 -np 1 -tb 16 --no-host -lm none
   ```

   Da unos 6,3 tok/s con Qwen3.8-27B en este portátil.
2. Q4S solo compensa si la pasada verifica 6 tokens o más. Eso pasa con F5 (n adaptativo; a veces n grande) y con F6, donde la GPU absorbe capas y el tramo de CPU pesa menos, y aun así el n óptimo crece.
3. Para que Q4S compita en todos los casos:
   - acelerar el tile T = 4 hasta ≤ 0,55 s;
   - pasar las escalas de activación a una por cada 64 columnas para acercar la KLD a 0.
