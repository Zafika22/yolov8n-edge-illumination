# Evaluación de YOLOv8n bajo Iluminación Variable en Edge Devices

**IPD441 — Visión por Computador**, Prof. Nicolás Torres — Universidad Técnica Federico Santa María

Evaluación sistemática de YOLOv8n bajo cinco niveles de corrección gamma
(γ ∈ {0.3, 0.6, 1.0, 1.4, 1.8}) sobre COCO128, en simulación y en tres
niveles de hardware embebido real:

| Tier | Placa | Modelo | Formato | Resolución | Rol |
|---|---|---|---|---|---|
| 1 | XIAO ESP32S3 Sense | YOLOv8n INT8 | TFLite Micro | 96×96 | MCU sin acelerador |
| 2 | Grove Vision AI V2 | YOLOv8n INT8 | TFLite INT8 (NPU) | 192×192 | MCU + NPU Ethos-U55 |
| 3 | Raspberry Pi 4 | YOLOv8n FP32 | PyTorch | 640×640 | SBC de propósito general |

El documento completo (metodología, resultados, discusión) está en
[`paper/Articulo.pdf`](paper/Articulo.pdf).

> **Nota:** `paper/` contiene únicamente el PDF compilado (deliverable
> final); el código fuente LaTeX no se versiona en este repositorio.

---

## Resultados clave

- **H1 (degradación asimétrica) confirmada:** sub-exposición (γ=0.3) reduce
  mAP@0.5 en −21.9 % vs. −3.8 % de la sobre-exposición (γ=1.8). Punto de
  quiebre de viabilidad en γ ≈ 0.37.
- **H2 (tier mínimo viable) parcialmente confirmada:** ningún tier embebido
  cumple simultáneamente mAP@0.5 ≥ 0.5 y latencia ≤ 50 ms, pero el
  acelerador NPU dedicado (Tier 2) reduce el exceso de latencia a **1.72×**
  el umbral, frente a **154×** (Tier 1) y **20.7×** (Tier 3) sin
  aceleración dedicada. Un NPU es necesario pero no suficiente.
- **80 % de las clases** (57/71) caen bajo el umbral de mAP en γ=0.3.

| Configuración | Latencia | Razón umbral | Viable |
|---|---|---|---|
| Simulación CPU | 112.5 ms | 2.25× | No |
| Simulación MPS GPU | 9.2 ms | 0.18× | Sí (4/5 γ) |
| Tier 1 — ESP32S3 | 7,711 ms | 154× | No |
| Tier 2 — Grove NPU | 86.0 ms | 1.72× | No |
| Tier 3 — Raspberry Pi 4 | 1,035 ms | 20.7× | No |

---

## Estructura del repositorio

```
RepoFinal/
├── README.md
├── paper/
│   └── Articulo.pdf         # PDF final (código fuente LaTeX no versionado)
├── figures/                  # 19 imágenes usadas por el paper
├── data/
│   ├── raw/                  # Logs y JSON crudos por tier
│   └── processed/            # CSVs agregados usados en el análisis
├── datasets/                 # COCO128 + 5 variantes con gamma aplicado + COCO8
│   ├── coco128/
│   ├── coco128_gamma0.3/ ... coco128_gamma1.8/
│   └── coco8/
└── src/
    ├── tier1_esp32/
    │   └── tier1_Pipeline.py       # Pipeline serial + decodificación YOLOv8n + mAP
    ├── tier2_npu/
    │   ├── export_tier2_model.py       # Export YOLOv8n -> TFLite INT8 (192x192)
    │   └── tier2_latency_capture.py    # Captura de latencia vía serial (SSCMA)
    ├── tier3_rpi/
    │   └── README.md             # Nota: script real no disponible, ver más abajo
    ├── analysis/
    │   ├── AnalisisClases.py           # mAP por clase + heatmap
    │   ├── AnalisisConfidence.py       # Distribución de confidence scores
    │   ├── curvasSensibles.py          # Grilla fina de 18 puntos gamma
    │   ├── detectionVisualization.py   # Visualización de detecciones fallidas
    │   └── robustness.py               # Recall por tamaño de objeto
    └── notebooks/
        ├── Tarea2.ipynb           # Simulación baseline (CPU/MPS) + perturbación gamma
        └── Avances.ipynb          # Export y puesta en marcha del firmware Tier 1
```

> `datasets/` pesa ~61 MB. Si vas a versionar este repo con git, considera
> agregarlo a `.gitignore` y dejar documentado cómo regenerarlo (ver más
> abajo) en vez de commitearlo directamente — son datos derivados, no código.

---

## Cómo ejecutar

### 0. Entorno

Este proyecto usa dos entornos Python distintos:

```bash
# Entorno principal (análisis, Tier 3)
python3 -m venv venv
source venv/bin/activate
pip install ultralytics opencv-python numpy pandas matplotlib pyyaml pyserial

# Entorno de exportación (Python 3.12 — TensorFlow/coremltools
# son incompatibles con Python 3.14)
python3.12 -m venv venv_export312
source venv_export312/bin/activate
pip install ultralytics==8.4.67 tensorflow coremltools
```

> **Importante:** para la exportación a TFLite (Tier 1 y Tier 2) usar
> siempre `venv_export312`, no el entorno principal. En llamadas a
> subprocesos desde Jupyter, pasar `MPLBACKEND=Agg` explícitamente para
> evitar contaminación del backend de matplotlib.

### 1. Dataset y perturbación gamma

`datasets/coco128/` es el dataset base; `datasets/coco128_gamma{0.3,0.6,1.0,1.4,1.8}/`
son las 5 versiones con corrección gamma ya aplicada (usadas por Tier 1,
Tier 3 y el análisis explicativo). Se generan a partir de
`build_perturbed_dataset()`, definido en `src/notebooks/Tarea2.ipynb`.
`datasets/coco8/` es un subconjunto mínimo de 8 imágenes usado para pruebas
rápidas de pipeline antes de correr sobre las 128 imágenes completas.

### 2. Simulación (baseline)

Todo el flujo de simulación (CPU/MPS) está en
`src/notebooks/Tarea2.ipynb`: define `apply_gamma_correction()`,
`build_perturbed_dataset()` y `evaluate(mode, gamma, data_yaml)`, y termina
generando la tabla de resultados usada en `data/processed/metrics.csv`.
No existe un script `.py` equivalente independiente — correr el notebook
completo reproduce la simulación baseline.

### 3. Tier 1 — XIAO ESP32S3

```bash
# venv_export312 (Python 3.12)
python src/tier1_esp32/tier1_Pipeline.py --port /dev/cu.usbmodem1101 --gamma 1.0 --n-images 1
python src/tier1_esp32/tier1_Pipeline.py --port /dev/cu.usbmodem1101 --all-gammas
```

El export del modelo, la generación de `model_data.cc` y el flujo completo
de puesta en marcha del firmware están documentados paso a paso en
`src/notebooks/Avances.ipynb`.

⚠️ Ver **Problemas conocidos** más abajo. El firmware ArduTFLite (usado
para medir latencia) no expone el tensor de salida INT8; el firmware de
reemplazo (biblioteca oficial + protocolo binario propio, implementado en
este script) no pasó sus pruebas de verificación con `--n-images` antes
del cierre de esta etapa — por eso no hay mAP para Tier 1, solo latencia.

### 4. Tier 2 — Grove Vision AI V2 (NPU)

```bash
# 1. Exportar a TFLite INT8 (venv_export312)
#    Requiere ultralytics==8.4.67 exacto (no 8.4.83) para compatibilidad con Vela
python src/tier2_npu/export_tier2_model.py --weights yolov8n.pt --imgsz 192

# 2. Compilar con Vela para el NPU Ethos-U55 (fuera de este repo, requiere
#    clonar https://github.com/HimaxWiseEyePlus/YOLOv8_on_WE2)
vela --accelerator-config ethos-u55-64 --config himax_vela.ini \
     --system-config My_Sys_Cfg --memory-mode My_Mem_Mode_Parent \
     --output-dir ./tier2_vela_out results/tier2_export/yolov8n_full_integer_quant.tflite

# 3. Flashear el .tflite compilado al Grove Vision AI V2 vía SenseCraft AI Studio
#    (catálogo no incluye modelos de 80 clases — requiere subida manual de modelo custom)

# 4. Descubrir el formato del log antes de confiar en el parseo automático
python src/tier2_npu/tier2_latency_capture.py --discover --port /dev/tty.usbmodemXXXX

# 5. Capturar latencia (200 repeticiones, descarta warmup automáticamente)
python src/tier2_npu/tier2_latency_capture.py --port /dev/tty.usbmodemXXXX --gamma 1.0 --n 200
```

### 5. Tier 3 — Raspberry Pi 4

No se encontró un script independiente para Tier 3 entre los archivos del
proyecto — ver `src/tier3_rpi/README.md`. Los resultados se generaron
adaptando el mismo `evaluate()` de `src/notebooks/Tarea2.ipynb`, corrido
directamente en la Raspberry Pi con `device="cpu"` a 640×640px.

### 6. Análisis y figuras

```bash
python src/analysis/AnalisisClases.py          # mAP por clase + heatmap de sensibilidad
python src/analysis/AnalisisConfidence.py      # distribución de confidence scores
python src/analysis/curvasSensibles.py         # grilla fina de 18 puntos gamma
python src/analysis/detectionVisualization.py  # visualización de detecciones fallidas
python src/analysis/robustness.py              # recall por tamaño de objeto (small/medium/large)
```

Cada script imprime su carpeta de salida (`results/<nombre>/`) al correr;
revisa el docstring de cada uno para el detalle completo. Los PNG
resultantes son los que están hoy en `figures/`.

### 7. El paper

Este repositorio versiona solo `paper/Articulo.pdf` (el resultado final).
`figures/` en la raíz contiene las 19 imágenes generadas por los scripts
de `src/analysis/` que el documento usa — se mantienen aquí porque también
sirven como salida de referencia del pipeline de análisis, no solo como
insumo del paper.

---

## Problemas conocidos / troubleshooting

Estos son hallazgos metodológicos reales del proyecto, documentados también
en la sección de Limitaciones del paper.

### Tier 1 — dos firmwares, ninguno completo

El primer firmware de Tier 1 usa la biblioteca **ArduTFLite** (la
biblioteca oficial `Arduino_TensorFlowLite` no soporta ESP32-S3; falla
con `#error "unsupported board"` en `peripherals.h`). Con este firmware
se midió la latencia real (**7,711 ms/imagen**, vía monitor serial), pero
ArduTFLite no expone el tensor de salida INT8 cuantizado del modelo, así
que no permite decodificar predicciones ni calcular mAP.

Para resolverlo se desarrolló un **segundo firmware**, basado en la
biblioteca oficial `tflite-micro-arduino-examples` (que sí expone
`interpreter->output(0)->data.int8` directamente, comentando el chequeo de
placa no soportada en `peripherals.h:61`), junto con un protocolo binario
propio sobre serial implementado en `src/tier1_esp32/tier1_Pipeline.py`
(comando de inferencia + imagen → byte de sincronización `0xAA` +
latencia + tamaño + tensor de salida). **Este segundo firmware no pasó sus
pruebas de verificación:** las corridas de prueba con `--n-images 1` y
`--n-images 5` sobre 3 de los 5 gammas (`data/raw/tier1_gamma*.json`)
fallaron en el 100 % de los envíos — el dispositivo no respondió con el
protocolo binario esperado. Por eso el barrido completo de 128 imágenes ×
5 gammas nunca se ejecutó, y **Tier 1 solo tiene latencia, no mAP**.

**Pendiente para resolver:** depurar por qué el segundo firmware no
responde al protocolo binario (verificar que el sketch compilado
corresponda a la versión con el chequeo de placa comentado, revisar
baudrate/timeout, probar con `--n-images 1` y monitor serial abierto en
paralelo para ver qué llega realmente al puerto).

### Tier 2 — Bootloader corrupto

Durante la puesta en marcha del módulo WE2 (Grove Vision AI V2) el
bootloader se corrompió. Se recuperó vía I2C usando un ESP32-WROOM-32
adicional como programador externo. Si el módulo deja de responder o no
aparece en el puerto serial tras un flasheo fallido, este es el primer
diagnóstico a probar.

### Tier 2 — SenseCraft AI no tiene modelos de 80 clases

La plataforma SenseCraft AI (flujo sin código de Seeed) solo ofrece modelos
pre-entrenados de propósito específico (personas, vehículos, etc.), no un
detector COCO de 80 clases. Fue necesario exportar y compilar el modelo
manualmente:

- `ultralytics==8.4.67` exacto (no 8.4.83 — rompe compatibilidad con Vela)
- Usar `yolov8n_full_integer_quant.tflite`, **no** `yolov8n_int8.tflite`
  (a pesar del nombre, este último conserva entrada/salida float32)
- Compilar con `himax_vela.ini` para 100 % de cobertura de operadores NPU

### Tier 2 — Resolución forzada a 192×192

La resolución planeada originalmente era 320×320, pero el límite de SRAM
disponible para activaciones en el NPU Ethos-U55 obligó a reducir a
192×192.

### Tier 2 — mAP no comparable (auto-exposición)

El WE2 no permite inyectar imágenes externas (arquitectura de captura fija
sobre su sensor OV5647 en tiempo real) ni desactivar el control automático
de exposición/ganancia. Por esto, el mAP de Tier 2 **no es calculable de
forma comparable** contra COCO128; solo se reporta la latencia de
inferencia como resultado cuantitativo válido (ver `data/raw/tier2_raw_log.txt`).

### Entorno Python — CoreML bloqueado

`coremltools 9.0` es incompatible con Python 3.14, lo que impidió evaluar
el Neural Engine de Apple. Usar Python 3.12 (`venv_export312`) para
cualquier tarea de exportación.

---

## Datos

- `data/raw/` — logs y JSON directamente desde cada dispositivo (sin
  procesar): `tier1_gamma*.json`, `tier2_raw_log.txt`, `tier2_latency.csv`,
  `tier3_results.json`, `confidence_scores_raw.csv`.
- `data/processed/` — CSVs agregados por el pipeline de análisis:
  `metrics.csv`, `per_class_map_by_gamma.csv`, `per_class_summary.csv`,
  `map_fine_grid.csv`, `size_fine_grid.csv`, `recall_by_size.csv`.
- `datasets/` — imágenes de COCO128 (original + 5 variantes con corrección
  gamma aplicada) y COCO8 (subconjunto de prueba rápida).

---

## Citar este trabajo

```bibtex
@techreport{pizarro2026yolov8nedge,
  author = {Pizarro, Cristian},
  title  = {Evaluación de YOLOv8n bajo Condiciones de Iluminación Variable
            en Dispositivos de Borde: Análisis por Nivel de Hardware},
  institution = {Universidad Técnica Federico Santa María, IPD441},
  year   = {2026}
}
```

---

## Agradecimientos

Al profesor Nicolás Torres por la guía académica y retroalimentación
durante el desarrollo de este trabajo.