#!/usr/bin/env python3
"""
Exportacion YOLOv8n -> TFLite INT8 para Tier 2 (Grove Vision AI V2 / WiseEye2 Ethos-U55)
IPD441 - Evaluacion YOLOv8n bajo iluminacion variable en edge devices

Basado en el flujo oficial documentado por Himax:
https://github.com/HimaxWiseEyePlus/YOLOv8_on_WE2

IMPORTANTE - resolucion de entrada:
  Los ejemplos oficiales de Himax que pasan por el compilador Vela y corren
  en el Ethos-U55 usan imgsz=192, NO 320x320 como estaba en la tabla original
  del proyecto. Se ajusta aqui a 192 para maximizar compatibilidad con el NPU.
  Documentar este cambio en el paper como restriccion de hardware (igual que
  la resolucion 96x96 forzada en Tier 1).

Uso (con tu venv_export312, Python 3.12):
    /Users/zafika/Documents/Universidad/2026-1/Vision Por Computador/Tarea-2-Vision-Computador/venv_export312/bin/python \
        export_tier2_model.py --weights yolov8n.pt --imgsz 192

Salida esperada:
    yolov8n_saved_model/yolov8n_full_integer_quant.tflite

Siguiente paso (NO incluido en este script, requiere el repo de Himax):
    git clone https://github.com/HimaxWiseEyePlus/YOLOv8_on_WE2
    cd YOLOv8_on_WE2/vela
    pip install ethos-u-vela
    vela --accelerator-config ethos-u55-64 \
         --config himax_vela.ini \
         --system-config My_Sys_Cfg \
         --memory-mode My_Mem_Mode_Parent \
         --output-dir ./tier2_vela_out \
         /ruta/a/yolov8n_full_integer_quant.tflite

    Esto genera el *_vela.tflite que se sube al Grove Vision AI V2 via
    SenseCraft AI Studio (subida de modelo custom) o via xmodem/Himax AI
    web toolkit segun el metodo de flasheo que uses.
"""

import argparse
import shutil
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Exportar YOLOv8n a TFLite INT8 (COCO, 80 clases)")
    ap.add_argument("--weights", default="yolov8n.pt",
                     help="Ruta al checkpoint .pt (default: yolov8n.pt, COCO pre-entrenado)")
    ap.add_argument("--imgsz", type=int, default=192,
                     help="Resolucion de entrada. 192 = verificado por Himax para Ethos-U55. "
                          "320 podria fallar en Vela por limites de SRAM del NPU - probar con "
                          "precaucion y tener 192 como respaldo.")
    ap.add_argument("--data", default="coco128.yaml",
                     help="Dataset de calibracion para la cuantizacion INT8 (default: coco128.yaml, "
                          "el mismo dataset que ya usas en Simulacion/Tier1/Tier3 para consistencia)")
    ap.add_argument("--out-dir", default="results/tier2_export",
                     help="Carpeta donde copiar el .tflite final")
    args = ap.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        sys.exit(f"No se encontro el archivo de pesos: {weights_path}")

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit(
            "Falta ultralytics en este interprete. Verifica que estas corriendo esto con "
            "venv_export312 (el mismo que usas para TensorFlow/coremltools):\n"
            "  pip install ultralytics --break-system-packages"
        )

    print(f"Cargando {weights_path} ...")
    model = YOLO(str(weights_path))

    print(f"Exportando a TFLite INT8 (imgsz={args.imgsz}, calibracion={args.data}) ...")
    print("Esto puede tardar varios minutos (descarga TF, onnx2tf, calibracion INT8).")

    result_path = model.export(format="tflite", imgsz=args.imgsz, int8=True, data=args.data)

    print(f"\nExport reportado por Ultralytics en: {result_path}")

    # Ultralytics genera una carpeta <nombre>_saved_model/ con varios .tflite;
    # el que nos interesa es el full_integer_quant (INT8 puro, para Vela/Ethos-U55).
    saved_model_dir = weights_path.with_name(weights_path.stem + "_saved_model")
    candidates = []
    if saved_model_dir.exists():
        candidates = sorted(saved_model_dir.glob("*full_integer_quant*.tflite"))

    if not candidates:
        print(f"\nAVISO: no se encontro automaticamente el archivo full_integer_quant.tflite "
              f"en {saved_model_dir}. Revisa la carpeta manualmente:")
        if saved_model_dir.exists():
            for f in saved_model_dir.iterdir():
                print(f"  {f}")
        return

    src = candidates[0]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / src.name
    shutil.copy2(src, dst)

    print(f"\nListo. Modelo INT8 copiado a: {dst}")
    print(f"Tamano: {dst.stat().st_size / 1024:.1f} KB")
    print("\nSiguiente paso (requiere clonar el repo de Himax y el compilador Vela):")
    print("  git clone https://github.com/HimaxWiseEyePlus/YOLOv8_on_WE2")
    print("  cd YOLOv8_on_WE2/vela && pip install ethos-u-vela")
    print(f"  vela --accelerator-config ethos-u55-64 --config himax_vela.ini "
          f"--system-config My_Sys_Cfg --memory-mode My_Mem_Mode_Parent "
          f"--output-dir ./tier2_vela_out {dst.resolve()}")
    print("\nRevisa el reporte de Vela: 'Total SRAM used' idealmente < 1MB.")
    print("Es normal que algunas operaciones 'transpose' queden fuera del NPU y corran en CPU (Cortex-M55).")


if __name__ == "__main__":
    main()