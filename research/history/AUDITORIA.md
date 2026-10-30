# Auditoría técnica del informe «SR-HPM»

**Objeto:** *Informe técnico definitivo de proyecto SR-HPM* (oct. 2026).
**Método:** cada afirmación cuantitativa se recalculó con [srhpm/budget.py](srhpm/budget.py); cada afirmación sobre el comportamiento de la memoria y la dinámica se puso a prueba con experimentos numéricos ([srhpm/hpm.py](srhpm/hpm.py), [srhpm/dynamics.py](srhpm/dynamics.py), tests en [tests/test_srhpm.py](tests/test_srhpm.py)). Las mediciones de GPU se tomaron en el hardware objetivo exacto del informe: **RTX 3060 Laptop, 6 GB, bus de 192 bits, 336 GB/s nominales, 3 MB de L2**.
**Resultados en bruto:** [results/results.json](results/results.json) (presupuesto, E1, E2). Identificadores: E1 memoria holográfica, E2 dinámica del núcleo, E3 ancho de banda GPU, E5 profundidad adaptativa (prototipo entrenado; el código y sus resultados se retiraron del repositorio, se conserva solo la conclusión).

## Veredicto

El problema que plantea el informe es real: el *memory wall* en batch 1, el coste uniforme por token y el crecimiento del KV-cache. Las tres ideas también son legítimas: profundidad recurrente con pesos compartidos, parada adaptativa y memoria de estado constante. Pero **ninguno de los tres pilares funciona tal como está especificado**, y las cifras de rendimiento de la sección 5 **superan el límite físico que el propio informe deriva en la sección 2**:

| Pilar | Lo que afirma el informe | Lo que se ha medido o derivado |
|---|---|---|
| I. Núcleo «residente en SRAM» | 1,8 GB residentes en L2/SRAM, tráfico de bus ≈ 0 | La L2 tiene 3 MB (600× menos). El núcleo se relee de VRAM **en cada iteración**: k × 1,8 GB por token |
| II. Profundidad elástica | Atractor de punto fijo; los tokens fáciles convergen en 1–3 ciclos | Con W ortogonal + residual no hay punto fijo útil. La parada se dispara por el crecimiento de la norma (k·Δ ≈ cte.), igual para todas las entradas (CV 2 %). Con ReLU/GELU la norma crece ×10¹⁶ |
| III. Memoria holográfica de 32 KB | Contexto de 1 M tokens en O(1) | La regla escrita encadena convoluciones: desborda fp32 al cabo de 22 tokens, ignora el orden y colapsa a ~3 modos de Fourier. Recuperación = azar. El límite teórico de una memoria de 32 KB está en unos 170 pares clave→valor |

Severidad: **C** = crítica (invalida una conclusión), **M** = mayor (cifra o diseño erróneo que hay que rehacer), **m** = menor.

---

## Hallazgos críticos

### C1. El núcleo no puede residir en L2/SRAM; el tráfico de bus por token se multiplica por k en lugar de anularse
- **Informe (§4.1, §5, §6-F3):** $W_\text{base}$ (1,8 GB) «residente de forma estática… en la jerarquía L2/SRAM»; «tráfico en bus por token ≈ 0 GB».
- **Hechos:** el GA106 tiene **3 MB de L2** y 30 SM × 100 KB = **3 MB de memoria compartida** en total. 1,8 GB es **600 veces** la L2. El mayor núcleo cuadrado int4 que cabe en L2 tiene d ≈ 2 449 (6 M de parámetros). La VRAM no tiene una «región de más alta velocidad».
- **Medido (E3):** iteración encadenada `x = tanh(W x)` en fp16. Cuando W pasa de 3 MB, el ancho de banda efectivo se satura en **≈ 315 GB/s**, es decir, en DRAM. Cada iteración relee W entero.
- **Consecuencia:** el tráfico por token es **k × |W|**. Con k = 35 (token «analítico») son **63 GB/token**, 13 veces más que el Llama-3-8B Q4 al que se pretendía superar. La recurrencia **empeora** el *memory wall* en lugar de eliminarlo, salvo que el núcleo quepa en la L2 (≤ ~6 M de parámetros en int4).
- **Corrección:** presupuestar `t_token = (B_estático + k·B_núcleo)/BW`. Hay dos salidas honestas: (a) un núcleo diminuto residente en L2 con kernel persistente, o (b) aceptar el coste k× y vender la arquitectura como **cómputo adaptativo**, no como ahorro de ancho de banda. Ver v2 §5.

### C2. Las velocidades de la tabla §5 violan la cota de ancho de banda (incluso con las cifras del propio informe)
Con 1,8 GB por iteración y 336 GB/s (sin cabeza LM, sin overhead, es decir, en el caso más optimista):

| k (iteraciones) | Cota física (tok/s) | Informe (tok/s) |
|---|---|---|
| 1 | 187 | 180–260 |
| 2 | 93 | ″ |
| 3 | 62 | ″ |
| 25 | 7,5 | 15–30 |
| 50 | 3,7 | 15–30 |

Medido (E3, 315 GB/s efectivos): k = 1 → 175 tok/s; k = 3 → 58; k = 35 → 5,0; k = 50 → 3,5. **Las cifras «sintácticas» y «analíticas» quedan entre 2× y 8× por encima de lo físicamente posible.**

### C3. La regla de actualización holográfica no es una memoria: encadena *bindings*
- **Informe (§4.3):** $\Psi_t = \Psi_{t-1} \circledast e^{i\phi(x_t)}$.
- Convolucionar de forma repetida es **multiplicar espectros**: $\mathcal F(\Psi_t) = \mathcal F(\Psi_0)\prod_j \mathcal F(e^{i\phi(x_j)})$. No se superpone nada; se compone todo. Medido en E1 (d = 4096):
  1. **Desbordamiento numérico:** $\|\Psi\|$ crece ×√d = **×64 por token**. En fp32 aparecen `inf` **en el token 22**. La regla no aguanta ni una frase.
  2. **Ceguera al orden:** el producto es conmutativo. Una permutación de 200 tokens da un Ψ idéntico (diferencia relativa ~1e-11). «El perro muerde al hombre» = «el hombre muerde al perro».
  3. **Colapso espectral**, incluso renormalizando en cada paso: la razón de participación (modos de Fourier activos) cae de ≈2 060 a ≈41 en 10 tokens y a **≈3 en 100 tokens** (sigue en 3 a los 1 000). Todo el estado se reduce a un puñado de sinusoides.
  4. **Sin recuperación:** test de pertenencia AUC 0,70 (producto escalar) y 0,44 (pico de correlación), frente a 1,0 de la superposición FHRR; recuperación clave→valor **0 %** para N = 16…256 (azar = 0,02 %). La memoria FHRR correcta recupera el 100 % con N ≤ 128.
- **Errores de dominio asociados:** (m4) $e^{i\phi}$ tiene módulo unitario en el dominio temporal, así que su FFT **no** es unitaria, y eso causa el crecimiento √d. (m3) $\Psi \circledast \bar q$ no es la correlación: la correlación HRR usa la involución $q(-n)$.
- **Corrección:** superposición de pares ligados con olvido, $M_t = \lambda M_{t-1} + k(x_t)\odot v(x_t)$, con claves fasoriales (FHRR, Plate 1995), y lectura $\hat v = \overline{k(q)}\odot M_t$ más *cleanup*. Mejor aún: memoria matricial con regla delta (§C4).

### C4. «Contexto de 1 000 000 tokens en 32 KB» es imposible en términos de información
- **Cota de casillero:** 32 KB = 262 144 bits. Incluso con compresión ideal a 3 bits/token caben ≤ 87 000 tokens, y sin comprimir (log₂ 32 000 ≈ 15 bits) ≤ 17 500. Recordar 1 M tokens exige ≥ 3 Mbit.
- **Capacidad real de FHRR** (fórmula en `budget.fhrr_capacity`, verificada en E1): con d = 4096 y un vocabulario de 32 k caben **≈ 170 pares clave→valor** recuperables al 99 %. Medido en E1 (d = 4096, V = 4096, fórmula → ~199 pares): 100 % hasta N = 128, 99 % con N = 256, 69 % con N = 512 y 21 % con N = 1024.
- Es la limitación conocida de todo modelo de estado fijo (Mamba, RWKV, atención lineal): ver Arora et al. 2023 (*Zoology*/MQAR) y Jelassi et al. 2024 (*Repeat After Me*).
- **Corrección:** presentar la memoria como **compresión con pérdida y capacidad O(d)**. Para recuperación exacta, híbrido con atención de ventana deslizante (2–4 k tokens). Si se quiere estado constante con más capacidad, usar estado **matricial** (regla delta con compuerta: d_k × d_v por cabeza), que almacena hasta d_k asociaciones casi exactas por cabeza en vez de ruido cruzado √N.

### C5. La dinámica del núcleo no tiene atractor, la parada es un artefacto y la «restricción unitaria» no evita la explosión
- **Informe (§4.1, §4.2, §6-F1):** $h_{k+1} = \sigma(W(h_k\odot P_k)) + h_k$, $W = \exp(A - A^\top)$, «radio espectral 1… impide la explosión de gradientes independientemente de k».
- **Análisis:** un punto fijo exige $\sigma(W(h\odot P)) = 0$. Con tanh o GELU y W ortogonal (núcleo trivial), **el único punto fijo es h = 0**, que no codifica nada. Además, W ortogonal es una **isometría**: no contrae, y un atractor necesita contracción. El Pilar II contradice al Pilar I-F1.
- **Medido (E2, d = 256, 64 iteraciones, P ~ N(1, 0,1)):**
  - tanh: la norma crece ×53 y el paso absoluto final es ≈ 16, así que el estado no converge. Aun así, el criterio relativo Δ < 0,05 **se dispara en k ≈ 26 para todas las entradas (CV = 2 %)**, porque Δ_k ≈ 1,25/k cae solo por el crecimiento de ‖h‖. La «profundidad elástica» es una función de ε, no del token.
  - ReLU/GELU: la norma crece **×10¹⁶** y la parada no se dispara nunca.
  - Jacobiano del producto de 64 pasos: **10^3,7 (tanh), 10^16,8 (ReLU)**. La ortogonalidad de W (error 1e-15) no impide la explosión, porque el Jacobiano por paso es $I + D_\sigma W \operatorname{diag}(P_k)$, con norma de hasta 1 + max|P|.
  - P_k cambia con k, así que el sistema no es autónomo y «converger a un punto fijo» ni siquiera está bien definido.
- **Corrección (validada en E2):** contracción amortiguada con inyección de entrada (estilo DEQ): $h_{k+1} = (1-\alpha)h_k + \alpha\, W_2\tanh(W_1(h_k\odot P_k) + U u)$ con $\|W_i\|_2 \le 1$ (normalización espectral), $|P_k| \le \gamma < 1$ y P_k fijo a partir de k ≥ n_mod. Por Banach hay un punto fijo **único y dependiente de la entrada**: convergencia desde condiciones iniciales distintas a 4e-16 y Jacobiano 10^-60,7 tras 256 iteraciones. Ojo: en E2 P_k varía en cada paso y el criterio ε = 1e-3 no llegó a dispararse en 256 iteraciones; por eso la corrección fija P_k para k ≥ n_mod.

### C6. El flujo de datos desconecta los pilares: el núcleo nunca ve el contexto
- En el diagrama §4, la memoria Ψ se consulta **después** del núcleo y justo antes de emitir el token. El núcleo itera solo sobre $h_0 = \text{Proy}(x_t)$, es decir, delibera sobre el token actual **sin contexto**, y la ecuación $h_{k+1}$ no tiene ningún término de lectura de memoria.
- Tampoco existe mezcla entre tokens fuera de Ψ. Tal como está, el modelo es un clasificador del token actual seguido de un sesgo contextual.
- **Corrección:** mezclador de memoria **antes** del núcleo (prelude) y **lectura de la memoria en cada iteración** del núcleo con una lectura 1-Lipschitz. En el diseño final esto lo resuelven modelos híbridos ya entrenados (ver v2).

---

## Hallazgos mayores

### M1. Dimensiones incompatibles entre sí
«$W_\text{base}\in\mathbb R^{d\times d}$ con 1,2 B parámetros» implica **d = 34 641**. Pero $P_k$ «8–16 KB» implica d = 4 096–8 192, y la HPM usa d = 4 096. Con d = 4 096, W tiene **16,8 M** de parámetros, no 1,2 B. Con d = 34 641, cada $P_k$ ocupa 69 KB y los 64 vectores 4,4 MB (no «< 1 MB»).

### M2. Bytes de pesos mal calculados
1,2 B en int4 = **0,6 GB**, no 1,8 GB (eso serían 12 bits/parámetro). En ternario (1,58 b) son **0,24 GB**, o 0,30 GB empaquetado a 2 bits, no 1,1 GB (7,3 bits/parámetro).

### M3. Embedding y cabeza LM omitidos
Con d = 34 641 y vocabulario de 32 k, la cabeza tiene **1,11 B de parámetros** (0,55 GB int4), tantos como el núcleo, y se lee completa en cada token. Con 128 k de vocabulario son 4,4 B. No aparece en ninguna tabla.

### M4. Fórmula de KV-cache errónea (resultado correcto por casualidad)
$2\cdot2\cdot n_\text{layers}\cdot d_\text{model}\cdot s$ ignora GQA: da **16 GiB** para Llama-3-8B a 32 k. La cifra real (8 cabezas KV × 128) es **4,0 GiB** en fp16, 2,0 en q8 y 1,0 en q4. Con KV en q4, Llama-3-8B Q4 a 32 k queda en el límite de 6 GB, no en un «OOM irrecuperable». Fórmula correcta: $2\cdot n_\text{layers}\cdot n_\text{kv}\cdot d_\text{head}\cdot s\cdot b$.

### M5. Errores en la tabla del estado del arte (§3, §5)
- «DeepSeek Distill» (R1-Distill-Qwen/Llama) son modelos **densos**: no tienen MoE ni MLA. MoE + MLA real en tamaño pequeño es DeepSeek-V2-Lite (16 B / 2,4 B activos), que en Q4 ocupa ~9 GB y no cabe en 6 GB.
- «14 000 M (MoE) en 4,9 GB»: 14 B a ~4,5 bits son **7,9 GB**.
- MoD: en batch 1, una capa saltada **no** se lee, así que sí ahorra ancho de banda.
- TTT: la actualización interna es un paso de gradiente de forma cerrada sobre un mini-lote, no una retropropagación completa.
- MTP: su ganancia principal es la decodificación especulativa (~1,8×), que **supera** la cota de 67 tok/s de §2.1 con los mismos pesos (ver m2).

### M6. Arte previo no citado; la originalidad está sobredeclarada
Pesos compartidos en profundidad con parada adaptativa: Universal Transformers + ACT (Dehghani 2018; Graves 2016), ALBERT (2019), DEQ (Bai 2019), PonderNet (2021), *Looped Transformers*, **Huginn / recurrent depth** (Geiping et al. 2025: 3,5 B, núcleo recurrente con prelude/coda), **Relaxed Recursive Transformers** (Bae et al. 2024: LoRA por iteración, que generaliza $P_k$) y Mixture-of-Recursions (2025). Memoria: HRR/FHRR (Plate 1995), Hrrformer (2023), atención lineal como *fast weights* (Schlag 2021), DeltaNet / Gated DeltaNet (Yang 2024), Mamba-2. Híbridos: Griffin/RecurrentGemma, Jamba, Samba.

### M7. $P_k$ es un escalado de columnas, no una modulación «espectral» ni «ortogonal»
$W\operatorname{diag}(P_k)$ comparte el espacio de filas y la base de salida de W en todas las «capas virtuales». Cada capa virtual aporta d grados de libertad frente a d² de una capa real. «64 capas virtuales» no equivalen a 64 capas en expresividad, y no interviene ninguna transformada de Fourier. Además, el conocimiento almacenable escala con los parámetros **únicos** (~2 bits/parámetro, Allen-Zhu & Li 2024). Iterar añade cómputo, no memoria factual.

### M8. Nada liga la profundidad con la «entropía» del token
El criterio ‖Δ‖ < ε mide convergencia, no dificultad. E2 muestra que en el sistema original lo fija el crecimiento de la norma y en uno contractivo la tasa de contracción. **E5 (entrenado, juguete de 0,5 M parámetros, tarea de saltos encadenados en contexto)** muestra algo peor: el modelo **esquiva el bucle**. La precisión con k = 1 es idéntica a la de k = 32 y el número de iteraciones es independiente de la dificultad (ρ de Spearman ≈ 0,01–0,10). Ocurrió igual con núcleo contractivo y no contractivo, y aun quitando los atajos de profundidad fija. La adaptatividad tiene que entrenarse y forzarse (k aleatorio, sin atajos), y medirse.

### M9. La Fase 1 no cabe en 6 GB ni en 36 h tal como está escrita
Para 50 M de parámetros en un W cuadrado (d = 7 071): BPTT a través de 64 iteraciones con batch 8 × 512 tokens necesita **≈ 14,8 GB** de activaciones fp32. Y `matrix_exp` de 7 071² cuesta **≈ 10,6 TFLOP por paso** de forward, ~1 s solo en la exponencial en esta GPU, más el backward. **Corrección:** backprop truncado (las últimas 4–8 iteraciones) o gradiente implícito (DEQ), normalización espectral, Cayley o Householder en lugar de `expm`, y núcleo con anchura MLP (d × 4d) en lugar de un único d × d.

### M10. La destilación por trayectorias de la Fase 2 está mal planteada
(a) MiniCPM-V es un modelo **visión-lenguaje**. (b) Los anchos no coinciden (maestro 2 304 o 3 584 frente a d del alumno), así que hace falta un proyector. (c) Fijar k(l) contradice la profundidad elástica. (d) Las activaciones del maestro dependen de atención sobre todo el contexto, algo que 32 KB no reproducen. **Corrección:** KD sobre logits (y opcionalmente sobre estados finales) con k aleatorio, maestro de texto con el mismo tokenizador (Qwen2.5-1.5B/3B o Llama-3.2-1B), e inicialización del núcleo desde capas del maestro (Relaxed Recursive Transformers).

### M11. Fase 3: kernel con W en memoria compartida
Solo es viable para núcleos ≤ ~3 MB (memoria compartida total). Además se midió que el **overhead de lanzamiento en Windows (WDDM) ronda los 55 µs por iteración** (GEMV + tanh) para núcleos pequeños. Para k grande ese overhead domina, así que hacen falta CUDA Graphs o un kernel persistente fusionado.

---

## Hallazgos menores

- **m1** «64 capas = 14 GB»: 64 capas de Llama-8B ocupan 27,9 GB en fp16 o 7,0 GB en int4. El informe no indica la precisión.
- **m2** «El bus prohíbe superar 67 tok/s»: solo es cierto en batch 1 **sin decodificación especulativa**.
- **m3** $\operatorname{Re}(\Psi\circledast\bar q)$ no es la correlación circular (falta la involución de índices).
- **m4** Módulo unitario en el dominio equivocado (ver C3).
- **m5** «Termodinámico», «System 2», «demuestra»: no hay contenido termodinámico y nada se demuestra. Son hipótesis.
- **m6** «GPU 6 GB, Ampere/Ada, 192 bits»: la RTX 4050 Laptop (6 GB) tiene un bus de **96 bits** (~192 GB/s). Las cifras solo valen para la RTX 3060 Laptop y la RTX 2060.
- **m7** cuFFT «< 5 µs»: el kernel puede tardar eso, pero el lanzamiento medido es mayor. Además sobra: en FHRR la memoria vive en el dominio de frecuencia y bastan operaciones elemento a elemento.
- **m8** «VRAM libre 3,4–4,1 GB» ignora el contexto CUDA (~0,3–0,5 GB) y las activaciones.
- **m9** «Tokens sintácticos sub-milisegundo»: una sola lectura de 1,8 GB a 315 GB/s tarda 5,7 ms.

---

## Lo que sí es correcto
- Aritmética del roofline §2.1 (4,5 GB / 300 GB/s → 66,7 tok/s).
- Diagnóstico general: en batch 1 la decodificación está limitada por memoria, y el KV-cache crece O(s).
- $\exp(A - A^\top)$ es ortogonal (verificado: error 1,4e-15).
- Tamaño de un vector complejo de 4 096 componentes en fp32: 32 768 B.
- Idea de cómputo adaptativo por token y de estado de contexto constante: es una dirección de investigación válida, y v2 la conserva.

Ver [INFORME_SR-HPM_v2.md](INFORME_SR-HPM_v2.md) para el diseño corregido orientado al objetivo (LLM grandes, rápidos, en 6 GB).
