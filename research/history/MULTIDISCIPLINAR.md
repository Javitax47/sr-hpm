# Técnicas de otros campos aplicadas a SR-HPM v3

**Criterio:** una técnica solo entra si reduce uno de los tres recursos que limitan [SR-HPM v3](PROPUESTA_SR-HPM_v3.md):

1. **Bytes que cruzan el PCIe por pasada.** Es el cuello de botella: ~0,38 s de cada pasada.
2. **Tokens aceptados por pasada,** que dependen de *p* (la tasa de acierto del borrador).
3. **VRAM** (5,9 GB de 6 ocupados).

Para cada técnica se dice qué resuelve, cómo encaja, la ganancia esperada [est] y si F0 puede medirla ya.

---

## 1. Lectura común: SR-HPM v3 es un códec predictivo

| Video (H.264/AV1) | SR-HPM v3 |
|---|---|
| Predicción (compensación de movimiento) | El **borrador** B predice tokens |
| Residuo transformado y codificado por entropía | El **plano R**, codificado por entropía |
| Control de tasa: bits donde importan | Bits de R donde importa el error del modelo |
| Árboles de partición de bloques (quadtree en HEVC) | Partición adaptativa de las matrices de pesos (§2) |
| Codificación aritmética con contexto (CABAC) | Codificación de R condicionada por B (§3) |
| Descarte de bloques que no se ven | Descarte de columnas de R que no influyen (§4) |

Por eso las herramientas de compresión de imagen, de video y de gráficos encajan de forma natural.

---

## 2. Quadtrees y optimización tasa-distorsión: bits de R allí donde importan

**De dónde viene:** compresión de imagen por quadtree (partición recursiva de bloques con poda tasa-distorsión, como en Shoham-Gersho, BFOS o la partición de bloques de HEVC) y LOD/mipmaps en gráficos.

**Aplicación:**
- Cada matriz se divide en *tiles* de 128 × 128. Cada tile puede subdividirse en 4 sub-tiles, de forma recursiva hasta 32 × 32.
- Cada hoja elige un nivel de refinamiento entre {0 (solo B), 3, 5, 9 niveles} minimizando $D + \lambda R$:
  - D es el error ponderado por la Hessiana de GPTQ, $\operatorname{tr}(\Delta W\, H\, \Delta W^\top)$, que ya se calcula.
  - R son los bits codificados.
- La poda es de abajo arriba: una hoja se fusiona con sus hermanas si así baja el coste.
- Un único λ global fija la tasa total. Cambiando λ se recorre la curva KLD–bits en lugar de probar configuraciones a mano.

**Por qué ayuda:** la sensibilidad de los pesos es muy desigual entre capas, cabezas y canales con activaciones atípicas. Con un R uniforme se gastan bits en zonas que no afectan a la salida.

| Ganancia [est] | Coste | ¿En F0? |
|---|---|---|
| 15–30 % menos bytes de R con el mismo KLD, es decir, +15–40 % de tok/s | Tabla de partición de ~2 bits por nodo (despreciable); kernel de descompresión con tamaño de bloque variable | Fase F0-b. El cuantizador necesita elegir el nivel por tile; la estructura de [f0/quant.py](f0/quant.py) ya lo admite |

**Variante fractal (autosimilitud):** la compresión fractal de imagen aprovecha que un bloque se parece a otro bloque transformado. En pesos de LLM, la similitud entre capas del mismo tipo es baja según la literatura. Predicción: **ganancia casi nula**. Prueba barata para descartarla: correlación entre los códigos R de la capa ℓ y la ℓ+4. No justifica más esfuerzo salvo sorpresa.

---

## 3. Análisis entrópico: codificar R con contexto

**De dónde viene:** codificación aritmética adaptativa por contexto (CABAC) y teoría de la información (H(X|Y) ≤ H(X)).

**Aplicación:** el residuo no es independiente del código base. Cuando B = 0, el peso original era pequeño y el residuo tiende a ser pequeño. Cuando |B| es máximo, puede haber saturación y un residuo grande. Si R se codifica con una tabla rANS **por cada valor de B**, que la GPU tiene ya en VRAM, los bits por peso bajan de H(R) a **H(R | B)**.

| Ganancia [est] | Coste | ¿En F0? |
|---|---|---|
| 5–15 % menos bits de R, sin ningún coste de calidad (es sin pérdida) | Una tabla rANS por contexto (3 en ternario) | **Sí, implementado:** F0 guarda el histograma conjunto (B, R) y reporta `residual_cond_entropy_bits` |

**Ampliación entrópica en tiempo de ejecución (árbol adaptativo):** el coste del bus por pasada es fijo, así que el árbol de 64 nodos debe repartirse donde más aceptación esperada aporta. La entropía del borrador en cada posición indica cuánto se puede extender: con poca entropía conviene una cadena profunda, con mucha conviene ramas anchas. Es una mochila sobre las probabilidades del borrador. EAGLE-2 ya hace árboles dinámicos por confianza; lo nuevo aquí es optimizar con un **presupuesto de nodos que no cuesta bus**. Ganancia [est]: +10–25 % de tokens por pasada. Es de runtime, no de F0.

---

## 4. Culling: no transportar lo que no va a influir

**De dónde viene:** gráficos 3D (frustum, occlusion y early-z culling: no procesar lo que no aparecerá en pantalla).

**Aplicación — culling predicho por el borrador:**
- En las FFN, la entrada de `down_proj` es la activación SwiGLU de 17 408 neuronas, y en cada token muchas son ≈ 0.
- La columna *j* del residuo de `down_proj` solo afecta a la salida si alguna de las 64 activaciones del árbol en la neurona *j* es grande. Si todas son pequeñas, esa columna de R está «ocluida».
- **El borrador ya conoce sus propias activaciones** para los 64 nodos antes de que empiece la verificación. Con ellas se predice qué columnas de R no hace falta transmitir. Es el equivalente de un *occlusion query* hecho con la geometría aproximada.
- Esto deja de ser exactamente sin pérdida: el objetivo pasa a ser «B + R salvo columnas descartadas». El error queda acotado por |x_j| · ‖R_{:,j}‖ y se controla con un umbral τ. Una red de seguridad conservadora: si la activación del objetivo en la verificación supera τ en alguna columna descartada, esa columna se pide bajo demanda. Eso añade latencia, pero recupera la exactitud.

**Clave para que compense:** la esparsidad de un token es alta, pero la de la **unión de 64 tokens** es mucho menor. Hay que medirla.

| Ganancia [est] | Coste | ¿En F0? |
|---|---|---|
| Si la unión deja ≥ 40 % de columnas de `down_proj` inactivas: −12 % de bytes totales de R (down_proj es ~1/3 de las FFN, y las FFN son ~2/3 del modelo) | Lógica de culling y la red de seguridad | **Sí, implementado:** F0 mide la fracción de columnas inactivas en la unión de ventanas de 64 tokens para varios τ (`ffn_union_inactive`) |

**Culling de ramas:** los nodos cuya rama ya se sabe rechazada se podrían descartar a mitad de la verificación. El ahorro sería de cómputo, que no es el cuello de botella, así que se descarta.

---

## 5. Heurísticas, cachés y control

| Técnica | Origen | Aplicación | Ganancia [est] | Decisión |
|---|---|---|---|---|
| Caché con reemplazo por valor (ARC / Belady) | Sistemas operativos | Fijar en la VRAM libre los tiles de R más caros de transmitir | VRAM libre ~0,1–0,3 GB frente a 4,6 GB de R por pasada: ≤ 5 % | Menor; solo si sobra VRAM |
| Control de tasa (*rate control*) | Codificación de video | Ajustar en tiempo real el tamaño y la forma del árbol según la aceptación medida en las últimas pasadas | Estabiliza el rendimiento entre tipos de texto (código, prosa, razonamiento) | **Sí**, en el runtime |
| Predicción de saltos | CPU | Redactar el árbol siguiente suponiendo que se acepta el camino principal (C6) | Ya incluido en el modelo | Incluido |
| Multigrid (V-cycle) | Métodos numéricos | Corrección en malla gruesa (B) más refinamiento fino (R). Ampliable a **3 niveles**: R1 por PCIe en cada pasada, R2 solo cuando la verificación no es concluyente | Solo vale si se acepta un objetivo variable (pérdida). Contradice la exactitud | Descartado para el objetivo principal |
| Rotación de Hadamard / incoherencia | Teoría de señales (QuIP#, QuaRot) | Rotar los pesos antes de cuantizar para que B y R sean gaussianos, sin valores atípicos | Mejor KLD para los mismos bits; **más p** | Ya es estado del arte: **incorporarlo en F0-b** |
| Recocido simulado / programación entera | Investigación operativa | Asignación global de bits por capa | Lo cubre la optimización tasa-distorsión de §2 con λ | Integrado en §2 |

---

## 6. Prioridad

| Prioridad | Técnica | Por qué |
|---|---|---|
| 1 | Codificación condicionada H(R|B) (§3) | Sin pérdida, gratuita, **ya se mide en F0** |
| 2 | Tasa-distorsión con quadtree (§2) + rotación de Hadamard (§5) | La mayor palanca sobre bytes y sobre p |
| 3 | Culling predicho por el borrador (§4) | Depende de la esparsidad de la unión, **que F0 ya mide** |
| 4 | Árbol entrópico adaptativo + control de tasa (§3, §5) | Runtime; suma tokens por pasada |
| — | Fractal, multigrid de 3 niveles, caché de tiles | Ganancia baja o contradicen la exactitud |

**Efecto combinado [est]** sobre el escenario p = 0,85 de la propuesta (17 tok/s con solapamiento):
- R baja ×0,9 (contexto) × 0,8 (tasa-distorsión) × 0,88 (culling) ≈ ×0,63 en bytes.
- El árbol adaptativo da +15 % de tokens por pasada.
- Resultado: **~24–28 tok/s** con solapamiento y ~16–18 en serie.

Son estimaciones encadenadas: cada factor debe confirmarse por separado (F0 cubre ya dos de ellos).
