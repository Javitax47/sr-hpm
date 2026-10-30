# F5: Qwen3.8-27B (UD-Q3_K_XL) de extremo a extremo en el portátil

**Equipo:**
- RTX 3060 Laptop 6 GB. Con la reserva de WDDM quedan **~5,1 GB útiles**.
- Ryzen 7 5800H.
- 32 GB DDR4-3200.

**Objetivo:** Qwen3.8-27B con la calidad de UD-Q3_K_XL (no Q4_0 recuantizado) a más de 10 tok/s.

**Medida:**
- `llama-server` con el borrador DFlash2 (Q4_K_M) en GPU y n = 3.
- 5 prompts × 200 tokens, temperatura 0.
- Batería en [run_f4.sh](run_f4.sh); resultados en `f0/results/f4_*.json`.

## Progresión

Media de los 5 prompts. Cada fila añade un cambio a la anterior:

| Paso | tok/s | Qué se cambió |
|---|---|---|
| llama.cpp de serie, *repack* activado, `-ngl 17` | 2,70 | — |
| + kernels **SRV** (IQ4_XS, IQ3_S, IQ3_XXS, Q3_K, IQ2_*) | 5,72 | Kernel propio para lotes pequeños: decodifica cada bloque una vez para todos los tokens. Además: clang en lugar de MSVC y reparto dinámico de filas |
| + capas 0–23 y cabeza de salida en GPU | 6,77 | `LLAMA_GPU_LAYERS` elige qué capas van a la GPU, en vez de las N últimas. Las capas cargadas de IQ3 son las que más tiempo de CPU cuestan por byte |
| + `--no-op-offload` | 7,19 | llama.cpp descargaba la normalización de cada capa de CPU a la GPU (64 cortes CPU↔GPU por token; ahora 4) |
| + cruce MMVQ→MMQ por tipo | 7,23 | En esta GPU, los K-quants con 3 tokens o más van más rápido con tensor cores (`GGML_CUDA_SRHPM_TUNE`) |
| + 22 capas en vez de 24 | 7,0–7,2 | Con 24 capas Windows desbordaba la VRAM a memoria del sistema y el borrador tardaba 65 ms; con 22, 18 ms |
| + **C4** | 8,13 | IQ3_S e IQ3_XXS se reescriben sin pérdida a códigos de 4 bits (`pshufb` en lugar de 8 búsquedas en la retícula): 1,5–1,8× más rápido |
| + **instantáneas GDN directas** | 8,62 (repetición 8,39) | Las 4 instantáneas de estado por capa se escriben directamente en la caché (antes: salida y luego copia, ~1,1 GB de tráfico por pasada) |
| + activaciones idénticas bit a bit a Q8_K (versión final) | **8,69 / 8,57** | Exactitud: KLD 0 frente a llama.cpp de serie |

Velocidad por prompt en la versión final: **9,5 / 9,2 / 6,0 (código) / 10,9 / 10,2** tok/s.

Como referencia, Q4_0 (recuantizado desde Q3_K_XL, peor calidad) con todas las mejoras aplicables da 6,50.

## Exactitud

- **Kernels SRV con C4 frente a la ruta de serie** (KLD de `llama-perplexity` en lotes de 4 tokens, [run_kld_srv.sh](run_kld_srv.sh)): **KLD 0,000000**, perplejidad 7,12767 frente a 7,12768. Coincide en el token más probable el 99,9 % de las veces.
- **Comprobación interna** (`GGML_SRV_CHECK=1`): en todas las llamadas, la salida es idéntica bit a bit al `vec_dot` de ggml.
- **Fallo encontrado y corregido:** la primera versión cuantizaba las activaciones con `127/|max|`, frente a `-127/max` de la ruta de serie, y daba KLD 0,005. Ahora replica `quantize_row_q8_K_ref`.
- **Instantáneas GDN directas:** texto y borradores idénticos con y sin el cambio.
- **C4:** salida idéntica bit a bit a la del kernel IQ3 nativo ([srv_bench.cpp](srv_bench.cpp)).

## Dónde se va una pasada hoy (~0,39 s para 4 tokens verificados, unos 3,25 aceptados)

| Parte | Tiempo | Estado |
|---|---|---|
| Productos matriz-vector en CPU (capas 22–63) | ~0,24 s | Al límite del ancho de banda medido (~33 GB/s): IQ4_XS 0,13 s y IQ3_S (C4) 0,095 s |
| Otras operaciones de CPU (GDN, atención, get_rows) | ~0,045 s | — |
| GPU: 22 capas del objetivo | ~0,06 s | Los tipos IQ con 4 columnas van a 70–80 GB/s, por las búsquedas en la retícula |
| GPU: borrador más cabeza de salida | ~0,03 s | — |
| Sincronización, construcción del grafo y muestreo | ~0,01–0,02 s | — |

## Por qué no se alcanzan los 11,9 tok/s de la predicción de v4

La predicción suponía cuatro cosas:
1. Verificación casi plana en coste. **Se cumple:** la parte de CPU está limitada por memoria.
2. Canal PCIe con +30 %. **No compensa en este portátil.**
   - Cada cambio CPU↔GPU cuesta ~1,7 ms con WDDM.
   - El anillo de staging resta capas residentes de una VRAM que ya está llena.
   - Estimación neta: ~+5 %.
3. 30 GB/s para 12,45 GB. **Se cumple casi exacto.**
4. 0,03 s de sobrecoste por pasada. **La realidad son ~0,10 s:** la GPU decodifica lento los tipos IQ y solo hay 5,1 GB de VRAM útil.

## Qué falta para superar los 10 tok/s de media

| Opción | Ganancia estimada | Coste |
|---|---|---|
| ~~Kernel CUDA "decodificar una vez" para los tipos IQ en GPU~~ | **Probado: 0 %** (ver abajo) | — |
| Páginas grandes de 2 MB para los pesos de CPU | +5–15 % (menos fallos de TLB) | El usuario tiene que concederse el privilegio de Windows "Bloquear páginas en memoria" |
| Borrador recuantizado (~0,9 GB) para meter 1–2 capas más | +2–4 % | Riesgo de que baje un poco la aceptación |
| Borrador adaptado al dominio (F7), sobre todo para código | El prompt de código está en 6 tok/s con un 42 % de aceptación | Entrenamiento por autodestilación |
| Más ancho de banda (DDR5 de sobremesa) o más VRAM | ×1,5–3 | Hardware |

## Experimento: kernel CUDA para los tipos IQ residentes en GPU (descartado)

- **C4 tal cual en la GPU.** No compensa: ocupa un 27–43 % más de bytes y la VRAM ya es el límite. Se perderían ~3 capas en GPU, que costarían en CPU tanto como lo que se ahorra.
- **Variante equivalente sin coste de VRAM.** Es MMVQ con decodificación única: cada sub-bloque se decodifica una vez y se multiplica por las N columnas, con resultados idénticos bit a bit. Medido con `test-backend-ops` (4096 × 14336, n = 1/2/4/8): **0,92–1,07×** frente a la ruta de serie con n = 4. El compilador ya comparte la decodificación entre columnas.
- **Retícula copiada a memoria compartida.** También 0,85–1,12×, sin mejora con n = 4.
- **Conclusión.** Con 1 columna, IQ3_S ya va a ~170 GB/s, la mitad del ancho de banda de la VRAM, y el coste extra con 4 columnas no viene de la decodificación ni de la latencia de la tabla. Los tipos IQ en GPU suponen ~43 ms de ~390 por pasada. Aunque se exprimieran, el techo es de ~3 %. Se revirtió el código.

## Cómo lanzarlo

```bash
GGML_SRV=1 GGML_SRV_C4=1 SRHPM_GDN_DIRECT=1 GGML_CUDA_SRHPM_TUNE=1 \
LLAMA_GPU_LAYERS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,out \
llama-server -m Qwen3.8-27B-UD-Q3_K_XL.gguf -ngl 23 -c 4096 -fa on -t 8 -tb 16 -np 1 \
  --no-host -lm none --no-op-offload --jinja \
  --spec-type draft-dflash -md Qwen3.8-27B-DFlash2-Q4_K_M.gguf -ngld 99 \
  --spec-draft-n-max 3 --spec-draft-n-min 3 --spec-draft-n-ubatch 16
```

Requisitos:
- La DLL `ggml-cpu.dll` compilada con clang (`build_cpu_clang.bat`).
- El parche [llama-srhpm.patch](llama-srhpm.patch) aplicado sobre llama.cpp `d05706b`.
