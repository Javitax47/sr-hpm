# SR-HPM v3 — Propuesta original
## *Speculative Residual · Hierarchical Precision Memory*
### Verificación especulativa con plano residual en streaming para ejecutar Qwen3.8-27B en 6 GB

**Objetivo:** Qwen3.8-27B en un portátil con RTX 3060 de 6 GB y 32 GB DDR4, con inteligencia de clase Q3 y **decenas de tokens por segundo**.
**Resultado de F0 (27 oct 2026): NO-GO** en su variante ingenua (B ternario como parte gruesa del cuantizador GPTQ del objetivo):
- p = 7 %;
- KLD del objetivo = 0,21 a 3,7 bpw;
- residuo de 1,99 bpw;
- culling sin efecto.

**F0-b (28 oct 2026, llama.cpp en este portátil):**
- El objetivo Q3_K_XL va a **2,63 tok/s**; con la MTP nativa baja a **1,89 tok/s**, porque la verificación en CPU queda limitada por cálculo.
- Un borrador UD-IQ1_S (~1,83 bpw) del mismo modelo alcanza **p ≈ 0,81**, con un 20–34 % en prosa y un 61–73 % en código o JSON.
- Con esos valores, el modelo de v3 predice **8–12 tok/s (×3–4,5)**: mejora real, pero no «decenas».

Veredicto: **GO condicional** para ×3–4; **NO-GO** para decenas de tok/s en este hardware. Detalle en [f0/README_F0.md](f0/README_F0.md).

**F1 (28 oct 2026, microbenchmarks):**
- El PCIe del portátil funciona en **Gen3 x8**: **6,7 GB/s** medidos, la mitad de lo supuesto.
- El solapamiento entre streaming y cálculo alcanza una eficiencia de **0,99**, así que el diseño de C2 funciona.
- El kernel Triton (B+R)·X es correcto, pero ingenuo: 2,3 TFLOPS.

Con hardware medido, v3 predice **~6–8 tok/s** con un kernel optimizado (×2,5–3 sobre los 2,6 tok/s medidos). Detalle en [f1/README_F1.md](f1/README_F1.md).
**Estado:** propuesta de investigación. El modelo de rendimiento está en [srhpm/budget.py](srhpm/budget.py) (`v3_tok_s`). Nada está implementado ni medido extremo a extremo.
**Convención:** [med] medido en esta máquina · [cal] calculado · [est] estimado · [hip] hipótesis por validar.

---

## 1. El problema en una línea

Hoy, con la mejor técnica conocida, el 27B denso va a **~5–6 tok/s** con calidad Q3 ([INFORME_SR-HPM_v2.md](INFORME_SR-HPM_v2.md) §2.1). La causa: cada token generado obliga a la CPU a leer ~8 GB de pesos desde una RAM de **30,5 GB/s** [med] y a multiplicarlos. La CPU también es un mal verificador: su pico fp32 es **~470 GFLOPS** [med], lo que hace inviable verificar en ella muchos tokens a la vez.

## 2. La idea central

Hay tres observaciones que, juntas, cambian el problema.

1. **Los pesos no dependen del token.** Si los mismos ~4–5 GB que «faltan» en la GPU sirven para verificar **64 tokens candidatos a la vez**, el coste por token cae 64 veces. El tráfico de pesos deja de crecer con los tokens.
2. **El canal correcto para esos bytes es el PCIe hacia la GPU, no la CPU.** La CPU tiene más ancho de banda de RAM (30 GB/s frente a ~12 GB/s del PCIe 3.0), pero no tiene cómputo: verificar 64 tokens sobre 25 B parámetros son 3,3 TFLOP, que a ~0,2–0,5 TFLOPS serían segundos. La GPU los hace en ~0,2 s. Los pesos que llegan por streaming se consumen en la GPU y **el coste del bus es independiente del tamaño del árbol** de candidatos.
3. **No hace falta enviar el modelo, solo lo que le falta al borrador.** Si la VRAM aloja una versión de baja precisión del **mismo** modelo (plano base, ~1,45 bits/peso), al verificador solo le falta el **residuo** de precisión (plano residual, ~1,8 bits/peso). Si además ese residuo se **entrena para ser comprimible**, lo que cruza el bus es aún menos.

De ahí sale la estrategia: **la VRAM guarda un «Qwen3.8-27B borroso» completo que propone árboles de tokens. Una vez por pasada, el residuo que lo vuelve nítido fluye por el PCIe y la GPU verifica el árbol entero contra el modelo nítido.** La salida es exactamente la del modelo objetivo (base + residuo, clase Q3). La calidad del borrador solo afecta a la velocidad, nunca a la respuesta.

```
          VRAM (6 GB)                                   RAM (32 GB)
 ┌──────────────────────────────────┐          ┌──────────────────────────────┐
 │ Plano BASE  ~1,45 bpw  (4,6 GB)  │          │ Plano RESIDUAL ~1,8 bpw      │
 │  = Qwen3.8-27B completo, borroso │          │ codificado por entropía      │
 │ + adaptadores de alineamiento    │          │ (~4,6 GB por pasada)         │
 │ + cabeza MTP nativa              │          └──────────────┬───────────────┘
 │ + estado DeltaNet/KV confirmado  │                         │ PCIe ~12 GB/s, capa a capa,
 │ + anillo de 3 capas residuales ◄─┼─────────────────────────┘ con independencia del texto
 └──────────────┬───────────────────┘
                │
   (1) BORRADOR: base + MTP genera un árbol de 16 niveles × 4 candidatos (64 nodos)
   (2) VERIFICACIÓN: por cada capa ℓ, W_ℓ = B_ℓ + R_ℓ (descomprimido en la GPU) aplicado a los 64 nodos
   (3) ACEPTACIÓN: el camino más largo que coincide con el objetivo + 1 token extra
   (4) RE-ANCLAJE: el borrador continúa desde el estado EXACTO del objetivo
   (5) SOLAPAMIENTO: mientras se verifica el árbol i, se redacta el árbol i+1 (optimista)
```

---

## 3. Componentes

### C1. Precisión jerárquica anidada (plano base + plano residual)
- Cada matriz se cuantiza como $W \approx B + R$. B es un cuantizador vectorial / trellis de ~1,45 bpw. R cuantiza el error $W - B$ con ~1,8 bpw adicionales, hasta llegar a un objetivo de **~3,25 bpw**.
- [hip] Con cuantizadores trellis actuales, 3,25 bpw totales rondan la calidad de UD-Q3_K_XL (KLD ≈ 0,03). El anidamiento cuesta algo de calidad frente a un cuantizador conjunto del mismo tamaño. Hay que medirlo (§6, F0).
- Se calibra **conjuntamente**: B se optimiza para la aceptación (que sea un buen borrador) y B + R para la fidelidad (que sea un buen objetivo).

### C2. Verificación residual por streaming, con coste independiente del árbol
- R vive en RAM, organizado **capa a capa en bloques contiguos**.
- Un hilo de copia DMA lo transmite a un **anillo de 3 capas** en VRAM (~0,22 GB), por delante del cálculo. Los pesos no dependen de los datos, así que el prefetch nunca espera a nada.
- La GPU descomprime R_ℓ, lo suma a B_ℓ de forma fusionada en el kernel y aplica la capa a los N nodos del árbol.
- Coste por pasada ≈ max(bytes de R / BW_PCIe, cómputo de N nodos). Con N = 64: ~0,38 s de bus frente a ~0,23 s de cómputo, solapados [cal].
- La CPU queda libre para servir el streaming y gestionar el árbol.

### C3. Residuo entrenado para ser comprimible (*rate-aware residual*)
- El cuello de botella es el **bus**, no el almacenamiento, así que se optimiza la **entropía** de R, no su anchura en bits.
- Durante la calibración de R se añade un término de tasa: se minimiza $\text{KLD} + \lambda \cdot H(R)$, de modo que los códigos residuales se concentren cerca de cero.
- R se almacena con **rANS por bloques** y se descomprime en la GPU a cientos de GB/s.
- [hip] Ganancia de 0,8× en bytes (a 1,8 bpw nominales llegan ~1,45 bits efectivos). La compresión sin pérdida de pesos ya está demostrada en BF16 (DFloat11, 2025). Aquí se aplica a un plano residual que **se entrena para ello**.

### C4. Borrador alineado y re-anclado
- **Adaptadores de alineamiento:**
  - LoRA de rango bajo (~50–100 MB en VRAM) sobre el plano B.
  - Se entrenan por destilación para maximizar la coincidencia top-1 con el objetivo B + R. Es un entrenamiento barato, porque el modelo entero ya existe y solo se ajustan adaptadores.
- **Re-anclaje:**
  - En cada pasada, el borrador **parte del estado exacto del objetivo**: KV de las 16 capas de atención y estado de las 48 DeltaNet calculados con B + R.
  - El error del borrador no se acumula entre pasadas; solo hay deriva dentro de un mismo árbol.
  - Un solo estado confirmado sirve a los dos, lo que además ahorra VRAM.
- **MTP nativa como multiplicador del borrador:** cada forward del plano base se extiende con la cabeza MTP de Qwen3.8 (2 tokens por forward). Así se reduce a la mitad el tiempo de redacción.

### C5. Verificación en árbol para capas Gated DeltaNet con registro de rango 1
- En las capas recurrentes no hay KV que enmascarar: cada nodo del árbol necesita el estado de **su camino**.
- La actualización de la regla delta es de **rango 1**: $S \leftarrow a\,S(I-\beta kk^\top)+\beta vk^\top$.
- **Verificación:** se recorre el árbol en profundidad aplicando y deshaciendo actualizaciones a lo largo de cada rama. Solo hacen falta tantos estados vivos como profundidad tenga el árbol (16), no tantos como nodos (64).
- **Confirmación:** se guarda por nodo solo $(a,\beta,k,v)$, unos 16 KB por capa. Tras aceptar, se **reproducen** las actualizaciones del camino ganador sobre el estado confirmado.
- **Memoria:** ~6 MB por árbol, frente a ~0,6 GB si se guardaran copias completas del estado por posición [cal]. En 6 GB esto decide si la técnica cabe o no.

### C6. Tubería solapada
- Mientras el bus y la GPU verifican el árbol *i*, la GPU redacta el árbol *i*+1 suponiendo que se acepte el camino principal.
- Si la suposición falla, se descarta y se vuelve a redactar desde el estado confirmado, que C5 hace barato.
- El rendimiento real queda entre las columnas «conservador» (serie) y «solapado» de §4.

---

## 4. Modelo de rendimiento [cal + hip]

Supuestos (calculados con `srhpm.budget.v3_tok_s`):
- 25,5 B de parámetros leídos por token; base 1,45 bpw (4,6 GB en VRAM).
- Objetivo 3,25 bpw; residuo comprimido ×0,8 (4,6 GB por pasada).
- PCIe 12 GB/s [est]; GPU 315 GB/s [med] y ~15 TFLOPS efectivos [est].
- Árbol de 16 × 4; 2 tokens por forward base (MTP); r_alt = 0,5.

La incógnita principal es **p**: la probabilidad de que el top-1 del borrador coincida con el del objetivo.

| p (coincidencia top-1) | Tokens aceptados / pasada | tok/s conservador | tok/s solapado |
|---|---|---|---|
| 0,70 | 3,8 | 7,3 | 9,9 |
| 0,80 | 5,4 | 10,3 | 13,9 |
| **0,85** | **6,7** | **12,9** | **17,3** |
| **0,90** | **8,7** | **16,8** | **22,6** |

**Sensibilidades** (p = 0,85):

| Cambio | tok/s (conservador / solapado) |
|---|---|
| Sin C3 (residuo sin comprimir) | 10,9 / 13,9 |
| Objetivo de clase Q2 (2,9 bpw) | 15,0 / 21,4 |
| **PCIe 4.0 (24 GB/s)** en otro equipo | 18,1 / 28,3 |
| Sin MTP en el borrador | 10,3 / 17,3 |

**Referencia en la misma máquina:** el mejor método conocido con calidad Q3 (offload CPU + MTP) da ~5–6 tok/s. **SR-HPM v3 lo multiplica por ×2–4.** Llega a **decenas de tok/s (≥ 17)** si p ≥ 0,85 con solapamiento, o si p ≥ 0,9.

**Por qué p puede ser alto** [hip]: BitNest midió un 95 % de aceptación entre W4 y W8 anidados. Aquí el salto es mayor (1,45 → 3,25 bpw), pero lo compensan tres mecanismos que BitNest no usa: adaptadores de alineamiento, re-anclaje en cada pasada y árbol con alternativas.

**VRAM** [cal], ajustada:

| Componente | Tamaño |
|---|---|
| Base | 4,62 GB |
| Anillo residual | 0,22 GB |
| KV q8 (8 k tokens) | 0,26 GB |
| Estado DeltaNet | 0,08 GB |
| Adaptadores | 0,08 GB |
| Contexto CUDA | ~0,4 GB |
| Búferes | ~0,2 GB |
| **Total** | **≈ 5,9 GB** |

Requiere que la pantalla no use la dGPU. Margen de reserva: base a 1,35 bpw.

---

## 5. Originalidad frente al estado del arte

| Trabajo | Qué hace | Qué añade SR-HPM v3 |
|---|---|---|
| [BitNest](https://arxiv.org/html/2610.02800v1) (2026) | Borrador = nibble alto W4, verificación W8; ambos planos **residentes en el dispositivo**; solo transformers; 1,5–1,6× | El plano residual **fuera del dispositivo y en streaming**, con coste independiente del árbol; residuo **entrenado para ser comprimible**; árbol con capas recurrentes; alineamiento y re-anclaje |
| [QSpec](https://arxiv.org/html/2410.11305v2), [Quasar](https://arxiv.org/pdf/2603.01399), [QuantSpec](https://arxiv.org/pdf/2502.10424) | Auto-especulación cuantizada (activaciones o KV) en la GPU | Lo mismo que en la fila anterior: separación de memoria por jerarquía de dispositivos |
| [Dovetail](https://arxiv.org/pdf/2412.18934) | Borrador en GPU, **objetivo completo en CPU** | Verificar en CPU no escala con el árbol (cómputo). v3 verifica en GPU y solo mueve el residuo |
| [SpecOffload](https://arxiv.org/html/2505.10259v3), SpecExec | Especulación para ocultar offload de pesos completos | v3 no transfiere el modelo, transfiere **lo que le falta al borrador**, comprimido |
| EAGLE / MTP / Sequoia | Borradores y árboles | Se usan como piezas (MTP nativa, árbol), no se reivindican |

**Contribución propia,** según la búsqueda bibliográfica realizada (no exhaustiva; hay que repetirla antes de publicar):
1. Precisión jerárquica **repartida entre la memoria del dispositivo y la del host**, con el residuo verificado por streaming y un coste por pasada independiente del número de candidatos.
2. Residuo cuantizado **con objetivo de tasa (entropía)** porque el recurso escaso es el bus.
3. Verificación en árbol y rollback exacto para **Gated DeltaNet mediante registro de actualizaciones de rango 1**, necesaria para aplicar todo lo anterior a modelos híbridos como Qwen3.8.
4. Re-anclaje del borrador al estado exacto del objetivo, que comparte el estado confirmado.

---

## 6. Plan de validación con criterios de abandono

| Fase | Experimento | Métrica | Seguir si… |
|---|---|---|---|
| **F0** (días, sin kernels) | Cuantizar Qwen3.8-27B como B + R. Medir, sobre 1 M tokens fuera de línea: KLD de B + R frente a bf16, **p** top-1 de B (con y sin adaptadores, con re-anclaje) y entropía de R | KLD(B+R), p, bits efectivos de R | KLD ≤ 0,04 **y** p ≥ 0,8 **y** R ≤ 1,5 bits efectivos |
| **F1** | Microbenchmark: PCIe H2D real; descompresión rANS en GPU; GEMM fusionada (B+R)·X con N = 64 | GB/s, TFLOPS | PCIe ≥ 11 GB/s; descompresión ≥ 5× PCIe |
| **F2** | Kernel de árbol DeltaNet con registro de rango 1 frente a la referencia secuencial | Error numérico, ms | Equivalencia exacta (fp32), < 5 ms por capa |
| **F3** | Sistema completo sobre llama.cpp / ik_llama.cpp o un runtime propio con CUDA Graphs | tok/s, salida idéntica al objetivo con *greedy* | **≥ 15 tok/s** con salida idéntica |
| **F4** | Evaluación: GSM8K, MMLU-Pro, LiveCodeBench (subconjunto), llamadas a herramientas, frente a Q8 y frente a la referencia de 5–6 tok/s | Precisión, tok/s, tiempo por respuesta | Pérdida ≤ 2 puntos frente a Q8 |

**F0 es lo único que decide la viabilidad, y no requiere escribir kernels.** Si p < 0,75 incluso con alineamiento, el techo cae a ~10 tok/s y la propuesta pasa a ser una mejora de ×2, no las «decenas».

---

## 7. Riesgos

| Riesgo | Efecto | Mitigación |
|---|---|---|
| p bajo con 1,45 bpw | Velocidad ×2 en lugar de ×3–4 | Adaptadores más grandes; árbol más ancho (el coste del bus no cambia); base a 1,6 bpw sacrificando contexto |
| Anidamiento peor que un cuantizador conjunto | Calidad por debajo de Q3 | Subir el objetivo a 3,5 bpw (cuesta ~10 % de velocidad) |
| PCIe real < 12 GB/s (portátil, Gen3) | Pasada más lenta | C3 más agresivo; medir en F1 |
| Overhead de sincronización por capa (64 capas) | +ms por pasada | Prefetch a 3 capas; CUDA Graphs por pasada |
| VRAM al límite | OOM | Base a 1,35 bpw; KV q4; contexto de 8 k |
| Prefill largo | Lento: el residuo entero por cada bloque | Bloques de prefill grandes (el streaming se amortiza igual que en el árbol) |

---

## 8. Conclusión

SR-HPM v3 conserva la intuición del proyecto original: un núcleo compacto y residente que trabaja más o menos según el token, y una memoria organizada por jerarquía. Ahora la sostiene un mecanismo físicamente posible. **La GPU guarda el modelo entero en baja precisión y propone; el bus solo transporta la precisión que falta, una vez por cada árbol de 64 candidatos.** Con p ≥ 0,85 el modelo predice **13–23 tok/s** para Qwen3.8-27B con calidad de clase Q3 en este portátil, frente a 5–6 tok/s hoy. Que se alcancen o no lo decide un experimento barato (F0) antes de invertir en kernels.
