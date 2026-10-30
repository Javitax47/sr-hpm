# F6: de ~8 a ~9,7 tok/s en Qwen3.8-27B (28 oct. 2026)

Mismo portátil (RTX 3060 Laptop 6 GB, Ryzen 7 5800H, DDR4-3200) y mismo modelo (UD-Q3_K_XL) y borrador (DFlash2).
Todas las comparaciones van **por parejas en la misma tanda**: el portátil se calienta y la misma configuración varía
entre 7,3 y 9,3 tok/s según la hora. Banco: [qb.sh](qb.sh) + [dec_bench.py](dec_bench.py) (5 prompts de F0, 200 tokens,
voraz). El hash del texto indica si la salida cambia.

## Resultado

| Paso | Ganancia medida (pareja en la misma tanda) | ¿Cambia la salida? |
|---|---|---|
| Arreglo de Q4_K/Q5_K con ≥ 4 tokens | +4–6 % | No |
| Borrador adaptativo (bloque de 8, se guardan 3 y se amplía hasta 7 si la confianza ≥ 0,9) | +7–10 % (τ 3,25 → 4,24) | Sí, por redondeo: verificar 5–8 tokens usa otra ruta |
| IQ4_XS emparejado (2 bloques por carga, activaciones permutadas) | +5–7 % | Sí, por redondeo (~1e-7 relativo) |
| GPU *keepalive* (kernel de 1 warp que impide que la GPU baje a reposo) | +0–5 %; el borrador pasa de 19 a 11 ms por pasada | No |
| Escrituras no temporales de las instantáneas GDN | +0,3–1 % | No |
| Instantáneas GDN perezosas (2 estados completos por pasada + *replay* exacto) | +4 % | No (idéntica, mismo hash) |

**Con el lanzador (`srhpm/launch.py --run`) y los 10 prompts de F0:** 9,36 tok/s de media (6,0–12,6 según el prompt; 5 de 10 por encima de 10).

**Configuración final:** ~9,7 tok/s de media en frío (de 8,05 al empezar la sesión en las mismas condiciones). En 4 de
los 5 prompts se pasa de 10 tok/s (10,6–12,6). El resumen en castellano se queda en ~6 tok/s porque el borrador acierta
poco en prosa española.

## Qué se encontró

- **Q4_K/Q5_K con 4+ tokens iban 5× más lentos** en el kernel propio: un bucle escalar por token para el término del mínimo.
  Con un `vpmaddwd` el coste con 4 tokens baja de 5,5× a 1,1×.
- **IQ4_XS (53 % de los bytes de CPU) ya estaba limitado por cálculo con 4 tokens:** 30 GB/s frente a 38 de techo de la RAM.
  Medido el techo real con [bw_bench.cpp](bw_bench.cpp): 38–41 GB/s.
- **La GPU pasa la mayor parte de cada pasada ociosa** y el driver la baja a 600–870 MHz (P3) entre ráfagas.
- **La verificación de 4 tokens es casi gratis, la de 8 no:** de ahí que compense ampliar el borrador solo cuando la confianza es alta.
- **Las instantáneas GDN escribían 64 KB por cabeza y token:** 400–800 MB por pasada para poder deshacer tokens rechazados.
  Ahora solo se escriben el estado tras el primer token y el final. De los intermedios se guarda la actualización de rango 1
  (k, δ, g; ~1 KB), y si se rechazan tokens se rehace exactamente. De paso se elimina la copia de 3 MB por capa (`GET_ROWS`).

## Probado y descartado

| Idea | Resultado |
|---|---|
| 12 o 16 hilos para la generación | Igual que 8 |
| *Prefetch* software en el kernel | Sin efecto: el prefetcher hardware ya lo cubre |
| Ruta "offset" sin `vpsignb` (pesos sin signo + suma de activaciones) | Igual de rápida con 4 tokens; no es bit-exacta. Queda tras `GGML_SRV_OFF=1` |
| Mantener en registros hasta 8 tokens | Igual que el búfer en L1 |
| Umbral de ampliación 0,85 / 0,95, 2 o 4 borradores fijos | Dentro del ruido o peor (4 fijos: −10 %) |
| 20 capas en GPU con `-ub 1024` | Peor: desborda la VRAM |
| `-ub 512` con 20 o 21 capas en GPU | Igual o peor |
| C4 emparejado (como IQ4_XS) | Sin ganancia clara (−6 % a +4 %); queda tras `GGML_SRV_PAIR=2` |
| nsys 2021 para la línea temporal de la GPU | No capturó kernels; se usaron los temporizadores del servidor |

## Variables nuevas (las pone `srhpm/launch.py`)

| Variable | Qué hace |
|---|---|
| `SRHPM_DRAFT_KEEP=3`, `SRHPM_DRAFT_PEXT=0.9` (+ `--spec-draft-n-max 7 --spec-draft-n-min 1`) | Borrador adaptativo |
| `GGML_SRV_PAIR=0` | Vuelve al kernel IQ4_XS sin emparejar (bit-idéntico a ggml) |
| `GGML_CUDA_KEEPALIVE=200` | Kernel de mantenimiento de 200 µs |
| `SRHPM_GDN_LAZY=1` | Instantáneas perezosas (solo CPU, una secuencia) |
| `SRHPM_GDN_NT=0` | Desactiva las escrituras no temporales |
| `GGML_SRV_OFF=1`, `GGML_SRV_PF=<bytes>` | Experimentos descartados, apagados por defecto |

## Estrategias de fondo evaluadas (29 oct. 2026)

Objetivo: un salto grande (20 tok/s), no un 3–5 % más. Cada idea se midió o se simuló antes de construirla.

| Estrategia | Cómo se evaluó | Resultado |
|---|---|---|
| Compresión sin pérdida de los pesos | Entropía de los códigos ([entropy.py](entropy.py)) | IQ4_XS: 3,91 bits de 4; IQ3_S: 8,75 de 9. Como mucho un **3 %** menos de bytes. Descartada |
| Árboles de borradores (DDTree) | Simulación con 611 pasadas reales: retícula de DFlash2 y tokens aceptados ([tree_sim.py](tree_sim.py)) | Con la curva de coste actual, +5 % como máximo (árbol adaptativo). Si verificar fuese gratis, +61 %. **El límite es el coste por token** |
| Verificación más barata: hilos ayudantes SMT que traen los pesos a L2 | [srv_bench.cpp](srv_bench.cpp) `helper` | Peor en todos los casos: el kernel está limitado por cálculo, no por la memoria |
| Techo de cálculo del Zen 3 | [peak_bench.cpp](peak_bench.cpp) | 1,48 ciclos por 32 pesos × token con `vpsignb`, 1,10 sin él. El kernel actual va a 2,3 / 2,0. Con un kernel perfecto caben ~6–9 tokens a la sombra de la memoria, no 16 |
| Longitud de borrador por coste esperado (`SRHPM_DRAFT_COST`) | 10 prompts, A/B | Igual que el umbral fijo (9,44 frente a 9,45 / 9,43) |
| Cuello del salto de 4 a 5 tokens en GPU (MMVQ hasta 8) | llama-bench pp4–pp8 | Sin cambio: el salto viene de la CPU |
| MoE a partir del modelo denso (MoEfication, CMoE) | Análisis | Choca con la especulación: al verificar 4–8 tokens la unión de expertos cubre casi todo el FFN. Además pierde calidad sin reentrenar |

**Conclusión.** En este portátil y con estos pesos, el cuello es el cálculo int8 por token del Zen 3 (sin VNNI). Eso limita la verificación a ~4–6 tokens por pasada, y con ella el beneficio de árboles y borradores largos. Lo que todavía puede dar un salto:

1. **Recuantización consciente del kernel:** menos bytes y tipos sin paso de signo (Q4_K o C4 en lugar de IQ4_XS), con KLD medido frente a BF16. Necesita ~60 GB y los pesos originales.
2. **Borrador autodestilado:** τ +15–25 %. Necesita horas de generación y entrenamiento con LoRA en la GPU de 6 GB.
3. **Un MoE de calidad equivalente:** K2-Horizon 36B-A4B ya va a 15,6 tok/s de serie. Falta medir su calidad.
4. **Kernel de verificación afinado** hasta ~1,2× del techo: +15–20 %, trabajo largo.
5. **Hardware con VNNI** (Zen 4/5, Intel de 12.ª gen. o posterior) o más VRAM: cambia la curva de coste y hace rentables los árboles.

## Hipótesis originales probadas (29 oct. 2026, sesión autónoma)

Cada idea se probó con un experimento que podía refutarla antes de construirla entera.

| Hipótesis | Prueba | Resultado |
|---|---|---|
| **Verificación por diferencias:** los tokens de una pasada se parecen, así que `W·(xₜ − x₀)` sería barato | Volcado de activaciones reales ([delta_stats.py](delta_stats.py)) | **Refutada.** Las diferencias son el 50–95 % de la norma, no hay bloques nulos y solo un 4–8 % cabe en 4 bits. El 66–75 % de la energía va en una dirección común (activaciones masivas), sin utilidad para la velocidad |
| **El calor limita:** sin espera activa de hilos (`--poll 0`) ni *keepalive*, sobraría margen térmico | Frecuencia de CPU y GPU durante el banco ([thermo.ps1](thermo.ps1)) | **Refutada.** `--poll 0` no cambia nada y la CPU se mantiene al 106–108 % del nominal. Sin *keepalive* se pierde un 3–4 %: confirma el *keepalive* |
| **Las activaciones no caben en L1** con 8 tokens y K grande | [srv_bench.cpp](srv_bench.cpp) `l2k`, K = 1024 a 17408 | **Refutada.** Los ciclos no dependen de K |
| **IQ4_XS sin paso de signo** (pesos +128 y activaciones de 7 bits, con los canales atípicos aparte) | Error de cuantización ([outlier_stats.py](outlier_stats.py)) y kernel `iq4u` | Precisión: con K = 5120, 7 bits + atípicos aparte es *más* preciso que los 8 bits actuales (0,0184 frente a 0,0199). Velocidad: **sin ganancia**, el núcleo está limitado por instrucciones por ciclo, no por la tubería de multiplicación |
| **Desempaquetar escalas de IQ4_XS es caro** | IQ4_XS recodificado sin pérdida al formato C4 | Exacto, pero solo gana con 1–2 tokens |
| **Presión de registros** | [peak_bench.cpp](peak_bench.cpp), construcción paso a paso | En el kernel sintético, mover las sumas en coma flotante a memoria baja de 3,07 a 2,53 ciclos. En el kernel real empeora: hipótesis no transferible |
| **Guarda de estáticos en el bucle interno** | Ensamblador del kernel | **Confirmada:** `static const __m256i` dentro de `table()` metía una comprobación TLS y una posible llamada en cada bloque (~13 de ~45 instrucciones). Con tablas constantes, C4 con 1 token va de 10,8 a 4,9 ciclos. En el modelo apenas cambia: con 4 tokens manda la memoria |
| **La iGPU como segundo verificador** (filas repartidas CPU + iGPU en las pasadas de 5–8 tokens) | Ancho de banda: [igpu_bw.cpp](igpu_bw.cpp). Kernel OpenCL: [igpu_mv.cpp](igpu_mv.cpp). Integración: `ggml-cpu/srv-igpu.inc` | Por separado, prometedora: la iGPU lee la RAM a 38 GB/s y con la CPU suman 34,8 GB/s. El kernel v5 (filas intercaladas, 2–4 filas por hilo) llega a 31,6 GB/s con 4 tokens y 18,7 con 8, igual o más que la CPU. **En el modelo pierde (4,7 frente a 8,5 tok/s):** con los 8 núcleos calculando, la iGPU se queda sin presupuesto de potencia del APU y su kernel va 5,5 veces más lento (15,7 ms frente a 2,8 ms por llamada). Queda implementada y apagada (`GGML_SRV_IGPU=1`). Tendría sentido con una iGPU con presupuesto propio o en otra máquina |

**Lecciones:**
- **El portátil va al límite de memoria comprometida:** 39,3 GB (32 GB de RAM + 8 GB de paginación), y otros programas ya ocupan ~15 GB. El servidor solo llega a 36,6 GB. [guard.sh](guard.sh) mata el servidor antes de que un fallo de memoria arrastre a otros procesos.
- **Más espacio de paginación** (de 8 a 16–24 GB) daría margen a experimentos con copias adicionales. Es una configuración del sistema que tiene que cambiar el usuario.

## Cuantización pensada para el kernel (29 oct. 2026)

Partiendo de `Qwen3.8-27B-Q8_0.gguf` (29 GB, referencia casi sin pérdida) y de la matriz de importancia de Unsloth.
Las mezclas se generan con [mix.py](mix.py) a partir de los tipos de UD-Q3_K_XL. El KLD se mide frente a Q8_0 con 16 × 512
tokens de tu corpus.

**La observación de partida:** en la ruta de CPU, IQ3_S/IQ3_XXS se leen recodificados a C4 (4,375 bpw), más bytes que IQ4_XS
(4,25 bpw). En las capas de CPU, IQ3 cuesta como IQ4 y da menos calidad. En cambio, Q3_K se lee nativo a 3,44 bpw.

| Modelo | KLD medio | Mismo token top | Bytes CPU por pasada | tok/s (10 prompts, sesión) |
|---|---|---|---|---|
| UD-Q3_K_XL (Unsloth, desde BF16) | 0,0427 | — | 9,42 GB | 8,66–8,77 |
| `repro` (misma mezcla, desde Q8_0) | 0,0468 | 89,9 % | 9,42 GB | — |
| **`cpu_iq4`** (IQ3 → IQ4_XS en capas de CPU) | **0,0345** | **91,7 %** | 9,31 GB | 8,5–8,7 |
| `cpu_q3k` (IQ3 → Q3_K en capas de CPU) | 0,0483 | — | 8,61 GB | 9,02 |
| **`speed1`** (IQ3 del FFN → Q3_K, IQ3 de atención → IQ4_XS) | **0,0430** | 90,9 % | 8,80 GB | 8,77–8,80 |
| `speed2` (`speed1` + gate/up IQ4_XS → Q3_K) | 0,0516 | 89,4 % | 8,46 GB | 8,73 (borrador acierta menos) |

**Kernels mejorados por el camino:**
- **Tablas constantes en lugar de `static const __m256i`:** se quitó una guarda TLS por bloque. C4 con 1 token pasa de 10,8 a 4,9 ciclos.
- **Escalas de Q3_K desempaquetadas con 3 instrucciones vectoriales** en vez de ~50 escalares.
- **Ruta sin signo de Q3_K y C4:** el código guardado ya es w + desplazamiento. Las escalas de grupo salen en un vector por superbloque (`group_scales`) en vez de dos escrituras escalares por bloque. Q3_K con 4 tokens baja de 3,22 a 2,87 ciclos; C4, de 2,39 a 2,04.
- **Escalas de IQ4_XS desempaquetadas en vectorial** (~12 instrucciones en vez de ~64). IQ4_XS emparejado con 4 tokens baja de 2,89 a 2,61 ciclos (−10 %); con 8, de 2,59 a 2,38. El resultado es el mismo y la comprobación contra la referencia escalar pasa.

**Conclusiones:**
- **`cpu_iq4` es la opción de calidad:** un 19 % menos de pérdida que el original a la misma velocidad.
- **`speed1` es la opción de velocidad a calidad igual al original.**
- **Q3_K solo convierte sus bytes en velocidad con el kernel optimizado y la ruta sin signo** (`GGML_SRV_OFF=1`).

## Prefill con borrador: de −35 % a −9 % (29 oct. 2026)

Con DFlash, el procesado del prompt iba un 35 % más lento que sin borrador: 95 frente a 143 tok/s, y 38 frente a 25 s
hasta el primer token con un prompt de 3646 tokens. Medido con [prefill_prof.sh](prefill_prof.sh), que acepta las
variables `MT`, `DUB`, `NGLD` y `NODRAFT_SKIP`.

| Hipótesis | Prueba | Resultado |
|---|---|---|
| La inyección de features en el borrador, en trozos de 16 tokens (`--spec-draft-n-ubatch 16`) | `DUB=64`; `SRHPM_SKIP_INJECT=1` | **Refutada.** Sin inyectar va igual |
| La extracción de capas del objetivo para el borrador | `SRHPM_NO_EXTRACT=1` | **Refutada** |
| Las instantáneas de *rollback* del estado recurrente (`n_rs_seq > 0`) | `SRHPM_FORCE_RS=7` sin borrador | **Confirmada.** Sin borrador, el prefill baja igual: 143 → 98 tok/s |

Dentro de esas instantáneas había **dos costes independientes:**
1. **Estado de la convolución:** se guardaban K = 8 copias por capa, cada una con stride sobre todo el ubatch. Juntas costaban
   ~5 s por prompt. Ahora se copian una sola vez las últimas K + 2 columnas a un tensor pequeño y las 8 instantáneas salen de
   ahí. Es exacto (`SRHPM_PF_TAIL=0` lo desactiva).
2. **GDN en las capas de CPU:** el op lazy y el direct escriben en la caché del host. Por eso solo existen en CPU y van token
   a token. El op normal, en cambio, el planificador lo manda a la GPU en lotes grandes, donde corre por trozos con MMA.
   Ahora, en ubatches de más de 64 tokens:
   - el op lazy hace solo el primer token. Así reconstruye el estado si había un *rollback* pendiente. Con K = 1 no escribe
     instantáneas y marca el registro lazy como obsoleto;
   - el op normal hace el resto y escribe las K instantáneas completas.

   `SRHPM_GDN_PF_SPLIT=<n>` cambia el umbral; `0` lo desactiva.

| Prompt de 3646 tokens | Prefill | Hasta el primer token | Texto generado (64 tokens, voraz) |
|---|---|---|---|
| Sin borrador | 144,6 tok/s | 25,2 s | `1728eee315` |
| Con borrador, antes | 98,0 tok/s | 37,2 s | `1728eee315` |
| **Con borrador, ahora** | **131,8 tok/s** | **27,7 s** | `1728eee315` |

**La decodificación no cambia:** 10 prompts, mismo hash (`2a2f878e1a`) y 9,21 / 9,03 frente a 9,10 / 9,22 tok/s. Las dos
correcciones están activas por defecto y no hace falta tocar el lanzador.

**`speed1_gq3`** (IQ3_S → Q3_K en las capas de GPU) frente a `speed1`: 9,24 / 9,22 frente a 9,11 / 8,30 tok/s. τ también
sube (de 3,97 a 4,06) porque el texto generado es otro, y eso explica casi toda la diferencia. Su KLD es 0,0435, frente a 0,0430.
Se queda como alternativa, sin ganancia clara.

**Trampa de compilación:** `build_llama_inc.bat` (MSVC) vuelve a enlazar `ggml-cpu.dll` cuando cambian sus fuentes y pisa
la de clang-cl, que trae los kernels SRV con `-march=native`. Ahora el script copia la de clang al terminar.

## La dGPU como segundo verificador por PCIe (29 oct. 2026)

**Idea:** mientras corren las capas de CPU, la GPU dedicada está ociosa. La GPU calcula las primeras filas de cada matmul
IQ4_XS / Q3_K de esas capas y lee una copia anclada (*pinned*) de esas filas directamente de la RAM por PCIe
(*zero copy*, sin ocupar VRAM). Mientras tanto, la CPU calcula el resto. Usa el mismo esquema de reparto que el intento con
la iGPU, que se descartó porque se quedaba sin potencia.

- **Medido aparte** ([dgpu/dgpu_bw.cpp](dgpu/dgpu_bw.cpp)): el enlace es PCIe 3.0 x8 y da 6,7 GB/s. Con la CPU calculando a
  T = 8 a la vez, la CPU va a 22,0 GB/s y la GPU suma 7,3: **29,3 GB/s en total**, sin frenar a la CPU.
- **Kernels** ([dgpu/dgpu.cu](dgpu/dgpu.cu), cargados en tiempo de ejecución como `srhpm_dgpu.dll`): un warp por fila, que
  lee la fila con cargas consecutivas de 4 bytes a memoria compartida. Activaciones en float. La comprobación de
  [dgpu/dgpu_test.cpp](dgpu/dgpu_test.cpp) contra la decuantización de ggml da un error relativo de 1e-7, a 5–6 GB/s
  incluyendo todos los gastos.
- **En el modelo depende de M:** con M = 4, el caso más frecuente, la CPU ya lee a **35 GB/s, casi el techo de la RAM**, y
  la GPU solo compite con ella (890 → 1120 µs por matmul IQ4_XS). Con M = 6–8 la CPU está limitada por cálculo (20–25 GB/s)
  y la GPU sí suma: con M = 8, 1533 → 1309 µs (−15 %). Por eso el reparto va en una tabla por M:
  `GGML_SRV_DGPU=0,0,0,0,0,0.12,0.16,0.2`.
- Las filas de la GPU usan activaciones exactas en float, no int8. Son *más* precisas que las de la CPU, así que el texto
  cambia un poco.

| 10 prompts | tok/s | τ | Tiempo por pasada |
|---|---|---|---|
| Sin dGPU | 9,47 / 9,57 | 3,97 | 0,417 s |
| dGPU, tabla A | 10,17 / 10,16 (r12); 9,99 / 10,03 (r13) | 4,08 | 0,401–0,408 s |
| dGPU, tabla B (más reparto, desde M = 5) | 9,52 / 9,57 | 3,91 | 0,409 s |

**Ganancia real por pasada: −2 a −4 %.** El resto de la subida a ~10 tok/s viene de τ, que cambia con el texto generado. La
tabla B tiene el mismo coste por pasada que la A. Bajar el umbral de ampliación (`PEXT 0,85`) no mejora.

**Perfil de una pasada** (`GGML_OP_PROF=1`, 4 prompts, ~0,45 s por pasada):

| Parte | Peso |
|---|---|
| Matmuls de CPU (IQ4_XS 55 %, Q3_K 27 %, Q4_K/Q5_K/Q8_0 7 %) | ~76 % |
| Resto de la CPU (GDN 4 %, CPY 1,6 %, RMS_NORM 1,1 %, FA 1 %…) | ~10 % |
| GPU, borrador (11 ms), inyección (2 ms) y gastos fijos | ~14 % |

**Conclusión:** la pared sigue siendo la RAM en la pasada típica de 4 tokens. Lo que queda está en los bytes por pasada
(cuantización) o en τ (más tokens aceptados por pasada).

### Más hipótesis probadas (29–30 oct. 2026)

| Hipótesis | Prueba | Resultado |
|---|---|---|
| **Borrador híbrido n-grama + DFlash** (`--spec-type ngram-map-k,draft-dflash`; el primero que propone, gana) | 10 prompts, A/B con `ngram-map-k` y `ngram-mod` | **Sin efecto.** El n-grama nunca propone: mismo τ y mismo hash. Sin repeticiones en el contexto no hay coincidencias. Puede servir para resúmenes o edición de código; queda disponible mediante `SPEC` en qb.sh |
| **Camino de la retícula que maximiza la aceptación esperada** (programación dinámica hacia atrás, E = p₁(1 + p₂(1 + …)), en lugar del argmax paso a paso) | [tree_sim.py](tree_sim.py) con 611 pasadas reales (`dp chain`) | **Refutada.** +0,3 % de τ. Incluso un oráculo que corta justo en el primer fallo solo da +9 %. La política adaptativa ya está en +4–5 %: **el límite de τ es el propio borrador** |
| **Mezcla guiada por sensibilidad** (error de los pesos ponderado por la energía de activación de la imatrix, más una mochila entre IQ4_XS y Q3_K) | [sens_mix.py](sens_mix.py), calibrado con 4 mezclas medidas; después, cuantización y KLD | **El proxy falla entre clases de tensor.** Predijo un KLD de 0,042 para −5 % de bytes; el medido fue **0,0569**. Pasaba a Q3_K sobre todo `ffn_down` y `ssm_out`, que escriben directamente en el flujo residual. Ajustando un peso ×16 para esas clases, el ajuste cuadra con las 5 mezclas, pero el beneficio baja a −1,8 % de bytes con la misma calidad. **speed1 ya está casi en la frontera de Pareto** de IQ4_XS/Q3_K |
| **IQ3_S nativo** (3,44 bpw sin pasar por C4) | srv_bench `iq3` | Con T = 4 cuesta 1,24 frente a 0,79–0,86 de C4: limitado por cálculo. Descartado |
| **Páginas grandes** (menos fallos de TLB al leer 8,8 GB por pasada) | Requiere el privilegio "Bloquear páginas en memoria" | No concedido. Es una configuración del sistema que decide el usuario. Ganancia esperada pequeña: el kernel ya llega al ~90 % del techo medido |
| **Una o dos capas más en GPU** con la VRAM que queda libre (~0,5–0,7 GB, medida con `cudaMemGetInfo`) | qb.sh (`-ub 512 -c 4096`) y [lb.sh](lb.sh) (configuración del lanzador: `-ub 1024 -c 8192`) | Con `-ub 512`, 21 capas bajan un 3 % el tiempo por pasada; 22 empeoran. Con la configuración del lanzador, 19 capas dan 9,4 tok/s, 21 dan 8,8 y 22 dan 8,3. **En cuanto la VRAM libre baja de ~0,4 GB, WDDM lo frena todo.** El lanzador conserva su margen (5,0 GiB útiles) |

**El lanzador** ([srhpm/launch.py](../srhpm/launch.py)) activa ahora la tabla de la dGPU cuando encuentra `srhpm_dgpu.dll`
junto al servidor. También se ha corregido una importación rota (`NEED_GIB`) que venía de reescribir `requant.py`.

### Nuevos flujos de decodificación probados (30 oct. 2026)

Cada idea, antes de construirla, se sometió a una prueba barata que podía refutarla.

| Idea | Prueba | Resultado |
|---|---|---|
| **Corrección de bajo rango en la GPU ociosa:** capas de CPU a ~2,5 bits (−30 % de bytes) más U·(Vᵀx) del error de cuantización, guardado en la VRAM libre y calculado en paralelo | [lowrank_test.py](lowrank_test.py): SVD ponderada con la imatrix del residuo frente a Q8_0, en 15 matrices reales de UD-IQ1_S | **Refutada.** El residuo es casi ruido: el rango 256 solo quita entre el 10 y el 20 % del error. Por ejemplo, IQ2_S en `ffn_down`: 4,7e-2 → 3,8e-2, frente a ~2e-2 de Q3_K y ~5e-3 de IQ4_XS. De paso se mide el coste en calidad: IQ2_S tiene ~2,3 veces el error de Q3_K |
| **Adaptar el borrador en línea con lo que revela la verificación** (un reordenador de los 16 candidatos: puntuación, rango, posición, copia o inducción del contexto) | [rerank_test.py](rerank_test.py): entrenado con la mitad de 611 pasadas y evaluado con la otra mitad | **Refutada:** τ 4,098 frente a 4,105 voraz. Pero el techo es enorme: en el 15–19 % de las posiciones el token verdadero está entre los 16 sin ser el primero, y **un reordenador perfecto daría τ 6,33 (+54 %)** |
| **La media red de GPU reordena el borrador:** las capas 0–19 (casi gratis) calculan el estado oculto y una *logit lens* restringida a los 16 candidatos reordena la retícula antes de que verifique la CPU (sin pérdida) | [lens_test.py](lens_test.py): volcado por pasada de la entrada de las capas 20, 40 y 62 (`GGML_SRV_DUMP_NAMES`), con la retícula y la aceptación | **Refutada.** La lens de la capa 20 acierta el 10,9 % (el azar da 6,25 %) y la combinación rescata el 19,7 % de las posiciones de ruptura, frente al 18,4 %. Ni con la capa 62 pasa del 31 %: **lo que distingue a esos candidatos solo aparece en las capas profundas, las que están en la CPU** |

**Lo que dejan claro estas pruebas:** el borrador casi siempre tiene el token correcto entre sus 16 candidatos, y saber cuál es
cuesta lo mismo que la verificación completa. La forma de aprovechar ese techo sin reentrenar es verificar varios candidatos
en la misma pasada (un árbol). Por eso compensa abaratar los lotes de 6–8 tokens (dGPU) y vale la pena repetir la simulación
de árboles con la curva de coste actual.
| **Árboles de borrador con la curva de coste real** (medida dentro del servidor con `SRHPM_PASS_LOG=1`, con la dGPU activa: T(4) = 344–358 ms, T(6) ≈ 446–484, T(8) ≈ 504–512; sin la dGPU, T(8) = 563) | [tree_sim.py](tree_sim.py) con `SRHPM_T_PASS`. Se añade la política de **dos cadenas como dos secuencias**, que es la única forma de árbol que admite un modelo híbrido con capas recurrentes | **Refutada.** El mejor árbol ideal da 0,94 y el árbol adaptativo 1,033, lo mismo que la cadena adaptativa (1,031). Las dos cadenas dan 0,59–0,67: el prefijo duplicado infla M. En un modelo con estado recurrente, los árboles no salen a cuenta en este hardware |
| **Longitud de borrador por valor esperado con la nueva curva** (`SRHPM_DRAFT_COST`; la simulación predecía +3 %) | 10 prompts, dos parejas | **Empate:** 9,64 / 9,74 frente a 9,77 / 9,59 de KEEP=3/PEXT=0,9 |
| **Capas de CPU a 2 bits** (`lite1`: speed1 con `ffn_gate`/`ffn_up` de CPU en IQ2_S, 2,56 bpw; −13 % de bytes) | Cuantización desde Q8_0, KLD y 10 prompts × 2 | **Pierde en las dos cosas:** KLD **0,094** (2,2 veces el de speed1) y mismo token top 86,1 % (frente a 90,9 %). Velocidad **7,84 frente a 9,61 tok/s**: IQ2_S se decodifica con una rejilla y está limitado por cálculo, así que los bytes ahorrados no compensan. Con los kernels actuales, bajar bits no da velocidad |

### Aceptación relajada (con pérdida, ajustable): `SRHPM_RELAX=δ` (30 oct. 2026)

Además del argmax del objetivo, se acepta el token del borrador si su logit está a menos de δ nats del máximo. En la posición
donde se rompe la cadena, el borrador suele proponer la segunda opción del objetivo, con una probabilidad parecida; esa es
la parte del techo de +54 % que se puede aprovechar sin calcular nada más. El coste se mide como nats por token generado
(log-probabilidad que se cede frente al argmax) y se puede comparar con el KLD que ya cuesta la cuantización: speed1 tiene
0,043 frente a Q8_0.

| δ | tok/s (10 prompts) | τ | Aceptación | Coste (nats por token) |
|---|---|---|---|---|
| 0 (exacta) | 9,80 | 4,08 | 0,728 | 0 |
| 0,5 | 9,85 | 4,13 | 0,756 | 0,004 |
| **1** | **10,40 (+6 %)** | 4,45 | 0,807 | **0,018** |
| 2 | 10,70 (+9 %) | 4,67 | 0,841 | 0,053 |
| 3 | 10,84 (+11 %) | 4,66 | 0,844 | 0,075 |

Desactivada por defecto. Se activa con `srhpm/launch.py --relax 1.0`.

**Longitud de borrador con δ = 1:** KEEP 3 / PEXT 0,8 da 10,26 tok/s (τ 4,70), frente a 9,93 / 9,05 de KEEP 3 / PEXT 0,9,
medidos al principio y al final de la misma tanda: la deriva es mayor que la diferencia. KEEP 4 empeora (9,54 / 8,34). Se
queda KEEP 3 / PEXT 0,9.

### Reducir trabajo con pérdida controlada, sin entrenar (30 oct. 2026)

| Idea | Prueba | Resultado |
|---|---|---|
| **Esparsidad de activaciones con una máscara compartida por la pasada** (estilo TEAL: no leer los bloques de pesos cuya entrada es ~0 en todos los tokens de la pasada) | [sparsity_test.py](sparsity_test.py) con volcados reales por pasada de las entradas de `attn_qkv` (capas 20, 40 y 62) y de `ffn_down` (capas 30, 45 y 60) | **Refutada.** Con bloques de 32 o de 256, que es como están cuantizados los pesos y por tanto la granularidad que ahorra bytes, saltarse solo el 10 % de los bloques ya cuesta 5–10 veces el error de IQ4_XS (`ffn_down` de la capa 30: 2,4e-2 frente a 5e-3). La esparsidad de TEAL existe elemento a elemento, no por bloques, y la máscara compartida es solo un poco peor que la de cada token |
| **Quitar capas redundantes** (ShortGPT / Gromov et al.) | Distancia angular entre las entradas de capas consecutivas ([layer_redundancy.py](layer_redundancy.py)), después `SRHPM_SKIP_LAYERS` y KLD | **Refutada sin re-entrenar.** Ninguna capa está "vacía": cada una gira el estado 0,06–0,14 π. Quitando 30–31 (−4,5 % de bytes) el KLD es 0,108; con 29–32, 0,178; con 50–51, 0,220; speed1 tiene 0,043. Solo podría salir a cuenta curando el modelo con un ajuste (LoRA) |
| **Cabeza de salida en Q4_K** (−165 MB de VRAM, casi una capa más en GPU) | KLD y A/B entre 21 y 22 capas | KLD 0,0464 frente a 0,0435. La velocidad no se pudo medir: la máquina iba cargada por otros procesos (7–8,7 tok/s en todas las configuraciones) |

**Conclusión de esta ronda:** en este modelo, los atajos sin entrenamiento cuestan más calidad de la que ahorran. Lo que
queda en la literatura con margen grande (cabezas MTP o EAGLE-3, un borrador destilado sobre este mismo modelo cuantizado,
quitar capas y curar después) requiere entrenar.

### Pendiente: adaptar el selector de DFlash2 a este objetivo (preparado, sin medir)

El borrador tiene el token correcto entre sus 16 candidatos casi siempre, y elegir bien entre ellos daría hasta τ 6,3. El
elemento que elige es el **selector**:

    s(p, c) = unario[c] + E_sig[c] · (E_prev[p] ⊙ gate),    con gate = W·h de rango 256

Es pequeño, así que se puede reajustar en la GPU de 6 GB con lo que revela la verificación. Queda preparado:
- `SRHPM_LATTICE_EXTRA=1` añade al registro de la retícula el `gate` y los logits unarios de cada posición (en `src/models/dflash.cpp` y en `common/speculative.cpp`);
- [collect.py](collect.py) + [collect.sh](collect.sh) generan cientos de respuestas voraces con prompts variados del corpus;
- [selector_fit.py](selector_fit.py) reconstruye las puntuaciones, ajusta una corrección de 256×256 sobre `gate` y escalas por posición, y mide τ con pasadas apartadas.

La recogida se detuvo a las 116 pasadas al cerrar la fase. Para retomarla: `NP=150 SRHPM_LATTICE_EXTRA=1
SRHPM_LATTICE_LOG=... SRHPM_ACCEPT_LOG=... bash f3/collect.sh <etiqueta>` (~50 min) y luego
`python -m f3.selector_fit lat.jsonl acc.jsonl`.

## Estado al cerrar la fase F6 (30 oct. 2026)

| Configuración | tok/s (10 prompts) | Calidad frente a Q8_0 |
|---|---|---|
| UD-Q3_K_XL al empezar F6 | ~8,0 | KLD 0,043 |
| **speed1 + dGPU** (exacta; `srhpm/launch.py`) | **~9,6–10** | KLD 0,043, mismo token top 90,9 % |
| speed1 + dGPU + `--relax 1.0` | ~10,4 | + 0,018 nats por token |
| Prefill con borrador (3646 tokens) | 98 → 132 tok/s | Idéntica |

Lo que no funcionó está documentado más arriba, cada cosa con su prueba.

### Microoptimizaciones sin entrenamiento (30 oct. 2026)

| Idea | Prueba | Resultado |
|---|---|---|
| **Operaciones pequeñas en un solo hilo, sin la barrera entre dos seguidas** (`GGML_CPU_SOLO=1` en `ggml-cpu.c`). La hipótesis venía del perfil: cada operación pequeña cuesta ~20 µs con los 16 hilos de las pasadas de verificación, frente a 8 µs con 8 hilos | 10 prompts × 2 y perfil por operación | **Refutada.** La salida es idéntica (mismo hash), pero la velocidad no cambia (9,55 / 9,35 frente a 9,45 / 9,51) y cada operación pequeña se vuelve *más* cara: RMS_NORM 31 µs, CPY 40 µs. El coste no estaba en la barrera sino en los fallos de caché, y con 16 hilos esos fallos se reparten. Se deja apagado |

Con `-tb 8` las operaciones pequeñas bajan a 8 µs, pero la matmul empeora (1446 frente a 1258 µs por IQ4_XS): el SMT ayuda a
la matmul, así que se mantiene `-tb 16`. El GDN de CPU cuesta ~494 µs por capa y pasada (~16 ms por pasada, ~18 GB/s);
como mucho se ganaría un ~2 %, dentro del ruido térmico de las medidas.
| **Cabeza de salida exacta sobre una parte del vocabulario** (grupos k-means de las filas de `lm_head` con la cota c·h + r·‖h‖, para leer solo los grupos que pueden contener el argmax; la cabeza pasaría a la CPU y liberaría 0,87 GB de VRAM, ~4 capas más) | [head_prune_test.py](head_prune_test.py), con la entrada real de la cabeza (volcado por pasada) y k-means en la GPU | **Refutada.** ‖h‖ ≈ 157 y el radio de los grupos es 0,4–0,6, así que la cota suma 60–95 logits, frente a un máximo de ~20,5 y una separación de 3,2 entre los dos primeros. Hay que leer el **97–100 % de la cabeza** incluso con 16 384 grupos |

### Decodificación con contexto largo (30 oct. 2026)

Medido con [ctxdec.sh](ctxdec.sh): configuración de decodificación de qb.sh y un resumen en castellano de 525–4838 tokens
de contexto ([ctx_bench.py](ctx_bench.py)).

| Contexto | Decodificación | Aceptación | τ / tok/s |
|---|---|---|---|
| 525 tokens | 6,19 tok/s | 0,58 | 0,49 s |
| 2425 tokens | 6,24 tok/s | 0,55 | 0,51 s |
| 4838 tokens | 5,32 tok/s | 0,45 | 0,47 s |

- **El tiempo por pasada no crece con el contexto.** La bajada frente a los ~10 tok/s del banco viene de la tarea: el borrador
  acierta poco en resúmenes en castellano.
- **La aceptación relajada (δ = 1) no ayuda en esta tarea:** la aceptación sube de 0,55 a 0,62, pero la decodificación queda en
  6,29 frente a 6,26.
- **`-c 4096` frente a `-c 8192`:** la primera es un ~2 % más rápida (6,26 / 6,10 frente a 6,12 / 5,99, con el mismo texto). El
  lanzador usa 8192 por defecto; quien no necesite tanto contexto puede bajarlo con `--ctx 4096`.
- **Recuperación n-grama desde el contexto** (`ngram-simple`: 4 → 3 y una coincidencia, como `--retrieval` del lanzador) con
  2425 y 4838 tokens de contexto: aceptación y salida idénticas a las de sin ella. El n-grama no llega a proponer nada,
  porque el resumen parafrasea en lugar de copiar. **Sin efecto en esta tarea.**
