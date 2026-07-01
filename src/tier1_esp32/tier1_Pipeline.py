#!/usr/bin/env python3
"""
3_tier1_pipeline.py
Pipeline de evaluación laptop ↔ XIAO ESP32S3 Sense (Tier 1).

Cambios respecto a versión anterior:
  - ArduTFLite envía output float32 (ya dequantizado) → se usa parse_output_float32()
  - SERIAL_TIMEOUT_S = 120 s para tolerar inferencia lenta en ESP32
  - --n-images para evaluar solo N imágenes (modo debug/test)

Uso:
    python 3_tier1_pipeline.py --port /dev/cu.usbmodem1101 --gamma 1.0 --n-images 1
    python 3_tier1_pipeline.py --port /dev/cu.usbmodem1101 --gamma 1.0
    python 3_tier1_pipeline.py --port /dev/cu.usbmodem1101 --all-gammas
"""

import argparse
import json
import struct
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import serial

# ── Configuración ─────────────────────────────────────────────────────────────
IMG_SIZE          = 96
CONF_THRESHOLD    = 0.25
IOU_NMS_THRESHOLD = 0.45
IOU_MAP_THRESHOLD = 0.50
N_COCO_CLASSES    = 80

BAUD_RATE         = 921600
SERIAL_TIMEOUT_S  = 120.0  # 2 min — ArduTFLite element-by-element es lento
BOOT_WAIT_S       = 2.5

CMD_INFER         = bytes([0x01])
RESP_OK           = 0xAA

ALL_GAMMAS        = [0.3, 0.6, 1.0, 1.4, 1.8]
RESULTS_DIR       = Path("results/tier1")

OUTPUT_ELEMS      = 84 * 189  # 15 876 floats = 63 504 bytes


# ═══════════════════════════════════════════════════════════════════════════════
# Comunicación serial
# ═══════════════════════════════════════════════════════════════════════════════

def connect_to_esp32(port: str) -> serial.Serial:
    print(f"Conectando a {port} @ {BAUD_RATE} baud…")
    ser = serial.Serial(port, BAUD_RATE, timeout=SERIAL_TIMEOUT_S)
    
    # Reset automático via DTR (mismo mecanismo que usa Arduino IDE)
    ser.setDTR(False)
    time.sleep(0.1)
    ser.setDTR(True)
    time.sleep(BOOT_WAIT_S)   # esperar arranque completo
    
    ser.reset_input_buffer()  # limpiar lo que llegó durante boot
    
 
    deadline = time.time() + 40
    while time.time() < deadline:
        if ser.in_waiting:
            line = ser.readline().decode("utf-8", errors="replace").strip()
            print(f"  [ESP32] {line}")
            if "FATAL" in line:
                ser.close()
                raise RuntimeError(f"ESP32 reportó error fatal: {line}")
            if line == "READY":
                print("✅ ESP32 listo\n")
                return ser
 
    raise TimeoutError("ESP32 no respondió con READY en 40 s")
 
 

def send_image_get_result(ser, image_rgb):
    assert image_rgb.shape == (IMG_SIZE, IMG_SIZE, 3)
    
    ser.write(CMD_INFER)
    ser.write(image_rgb.flatten().tobytes())
    ser.flush()

    # Leer byte de sincronización 0xAA
    sync = ser.read(1)
    if len(sync) < 1 or sync[0] != 0xAA:
        print(f"  ⚠️  Sync byte inválido: {sync.hex() if sync else 'vacío'}")
        return None

    # Leer header: latencia(4B) + out_size(2B)
    header = ser.read(6)
    if len(header) < 6:
        print("  ⚠️  Timeout leyendo header")
        return None

    latency_ms = struct.unpack_from("<I", header, 0)[0]
    out_size   = struct.unpack_from("<H", header, 4)[0]

    output_bytes = ser.read(out_size)
    if len(output_bytes) < out_size:
        print(f"  ⚠️  Output incompleto ({len(output_bytes)}/{out_size} bytes)")
        return None

    return {"latency_ms": latency_ms, "output_bytes": output_bytes}

# ═══════════════════════════════════════════════════════════════════════════════
# Preparación de imágenes
# ═══════════════════════════════════════════════════════════════════════════════

def apply_gamma(image_uint8: np.ndarray, gamma: float) -> np.ndarray:
    lut = np.array(
        [(i / 255.0) ** gamma * 255 for i in range(256)], dtype=np.uint8
    )
    return cv2.LUT(image_uint8, lut)


def load_and_prepare(image_path: Path, gamma: float) -> np.ndarray:
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"No se pudo leer: {image_path}")
    rgb     = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    return apply_gamma(resized, gamma)


# ═══════════════════════════════════════════════════════════════════════════════
# Parsing del output del modelo
# ═══════════════════════════════════════════════════════════════════════════════

def parse_output_float32(output_bytes: bytes) -> np.ndarray:
    """
    ArduTFLite dequantiza internamente → recibimos float32 directo.
    Convierte los bytes recibidos a array float32.
    """
    return np.frombuffer(output_bytes, dtype=np.float32)


def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    inter  = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union  = area_a + area_b - inter
    return inter / union if union > 1e-8 else 0.0


def nms(detections: np.ndarray, iou_threshold: float) -> np.ndarray:
    if len(detections) == 0:
        return detections
    idxs = np.argsort(detections[:, 4])[::-1]
    keep = []
    while len(idxs) > 0:
        best = idxs[0]
        keep.append(best)
        rest = idxs[1:]
        ious = np.array([iou(detections[best, :4], detections[j, :4]) for j in rest])
        idxs = rest[ious < iou_threshold]
    return detections[keep]


def decode_yolov8_output(
    output_float: np.ndarray,
    conf_threshold: float = CONF_THRESHOLD,
) -> np.ndarray:
    """
    YOLOv8n anchor-free @ 96×96 → [N, 6] (x1,y1,x2,y2,conf,cls_id).
    output_float: array 1D de 15 876 floats [84 × 189].
    """
    if output_float.size == 0:
        return np.empty((0, 6))

    n_anchors = output_float.size // 84
    output = output_float.reshape(84, n_anchors)

    boxes_cxcywh = output[:4, :]
    class_scores  = output[4:, :]

    max_scores = class_scores.max(axis=0)
    class_ids  = class_scores.argmax(axis=0)

    mask = max_scores >= conf_threshold
    if not mask.any():
        return np.empty((0, 6))

    b = boxes_cxcywh[:, mask].T
    s = max_scores[mask]
    c = class_ids[mask].astype(float)

    x1 = np.clip(b[:, 0] - b[:, 2] / 2, 0, 1)
    y1 = np.clip(b[:, 1] - b[:, 3] / 2, 0, 1)
    x2 = np.clip(b[:, 0] + b[:, 2] / 2, 0, 1)
    y2 = np.clip(b[:, 1] + b[:, 3] / 2, 0, 1)

    detections = np.column_stack([x1, y1, x2, y2, s, c])
    return nms(detections, IOU_NMS_THRESHOLD)


# ═══════════════════════════════════════════════════════════════════════════════
# Ground truth y cálculo de mAP
# ═══════════════════════════════════════════════════════════════════════════════

def load_gt_labels(label_path: Path) -> list[dict]:
    if not label_path.exists():
        return []
    labels = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        cx, cy, w, h = map(float, parts[1:5])
        labels.append({
            "cls_id": cls_id,
            "box": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
        })
    return labels


def compute_ap_11point(recalls: list[float], precisions: list[float]) -> float:
    ap = 0.0
    for thresh in np.linspace(0.0, 1.0, 11):
        p_at_r = [p for r, p in zip(recalls, precisions) if r >= thresh]
        ap += max(p_at_r) if p_at_r else 0.0
    return ap / 11.0


def compute_map50(
    all_preds: list[dict],
    all_gt: dict[str, list[dict]],
) -> dict:
    class_aps: dict[int, float] = {}

    for cls_id in range(N_COCO_CLASSES):
        preds = [p for p in all_preds if p["cls_id"] == cls_id]

        gt_by_img = {
            img_id: [g["box"] for g in gt_list if g["cls_id"] == cls_id]
            for img_id, gt_list in all_gt.items()
        }
        gt_by_img = {k: v for k, v in gt_by_img.items() if v}

        n_gt = sum(len(v) for v in gt_by_img.values())
        if n_gt == 0 or not preds:
            continue

        preds.sort(key=lambda p: p["conf"], reverse=True)

        matched: dict[str, set] = {img_id: set() for img_id in gt_by_img}
        tp_list, fp_list = [], []

        for pred in preds:
            img_id   = pred["img_id"]
            gt_boxes = gt_by_img.get(img_id, [])
            best_iou, best_idx = 0.0, -1

            for j, gt_box in enumerate(gt_boxes):
                score = iou(np.array(pred["box"]), np.array(gt_box))
                if score > best_iou:
                    best_iou, best_idx = score, j

            if best_iou >= IOU_MAP_THRESHOLD and best_idx not in matched.get(img_id, set()):
                tp_list.append(1)
                fp_list.append(0)
                matched.setdefault(img_id, set()).add(best_idx)
            else:
                tp_list.append(0)
                fp_list.append(1)

        tp_cum = np.cumsum(tp_list).astype(float)
        fp_cum = np.cumsum(fp_list).astype(float)
        recalls    = list(tp_cum / n_gt)
        precisions = list(tp_cum / (tp_cum + fp_cum + 1e-8))
        class_aps[cls_id] = compute_ap_11point(recalls, precisions)

    map50 = float(np.mean(list(class_aps.values()))) if class_aps else 0.0
    return {
        "map50": map50,
        "n_classes_with_detections": len(class_aps),
        "class_aps": {int(k): float(v) for k, v in class_aps.items()},
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluación por gamma
# ═══════════════════════════════════════════════════════════════════════════════

def find_coco128_paths() -> tuple[Path, Path]:
    try:
        from ultralytics import settings as ultra_settings
        datasets_dir = Path(ultra_settings["datasets_dir"])
    except Exception:
        datasets_dir = Path.home() / "datasets"

    images_dir = datasets_dir / "coco128" / "images" / "train2017"
    labels_dir = datasets_dir / "coco128" / "labels" / "train2017"

    if not images_dir.exists():
        raise FileNotFoundError(f"COCO128 no encontrado en {images_dir}")
    return images_dir, labels_dir


def evaluate_gamma(
    ser: serial.Serial,
    gamma: float,
    image_paths: list[Path],
    labels_dir: Path,
) -> dict:
    print(f"\n{'─'*56}")
    print(f"  γ = {gamma:.1f}  ({len(image_paths)} imágenes)")
    print(f"{'─'*56}")

    all_preds: list[dict] = []
    all_gt:    dict[str, list] = {}
    latencies: list[float] = []
    n_errors = 0

    for i, img_path in enumerate(image_paths):
        label_path = labels_dir / (img_path.stem + ".txt")
        img_id     = img_path.stem
        all_gt[img_id] = load_gt_labels(label_path)

        try:
            image = load_and_prepare(img_path, gamma)
        except FileNotFoundError as exc:
            print(f"  [{i+1:3d}] ERROR lectura: {exc}")
            n_errors += 1
            continue

        result = send_image_get_result(ser, image)
        if result is None:
            n_errors += 1
            if n_errors > 10:
                print("  ❌  Demasiados errores — abortar gamma")
                break
            continue

        latencies.append(float(result["latency_ms"]))

        # 1. Convertir los bytes crudos a un array firmado int8 (rango -128 a 127)
        output_int8 = np.frombuffer(result["output_bytes"], dtype=np.int8)
        
        # 2. Dequantizar manualmente usando los parámetros del modelo de salida (scale=0.006987, zero_point=-128)
        output_float = (output_int8.astype(np.float32) - (-128)) * 0.006987
        
        # 3. Decodificar las cajas de predicción usando el array float resultante
        detections   = decode_yolov8_output(output_float)

        for det in detections:
            all_preds.append({
                "img_id": img_id,
                "cls_id": int(det[5]),
                "conf":   float(det[4]),
                "box":    [float(v) for v in det[:4]],
            })

        # Progreso cada 8 imágenes (o en debug con pocas)
        if (i + 1) % 8 == 0 or len(image_paths) <= 5:
            med = np.median(latencies) if latencies else 0
            print(f"  [{i+1:3d}/{len(image_paths)}] "
                  f"latencia={result['latency_ms']} ms  "
                  f"mediana={med:.0f} ms  "
                  f"dets_acum={len(all_preds)}")

    map_result     = compute_map50(all_preds, all_gt)
    median_latency = float(np.median(latencies)) if latencies else None
    viable         = (map_result["map50"] >= 0.5) and (median_latency is not None and median_latency <= 50)

    print(f"\n  mAP@0.5         = {map_result['map50']:.4f}")
    if median_latency:
        print(f"  Latencia mediana = {median_latency:.0f} ms")
    else:
        print("  Latencia         = N/A")
    print(f"  Errores          = {n_errors}")
    print(f"  {'✅ VIABLE' if viable else '❌ NO VIABLE'} (mAP≥0.5 Y latencia≤50 ms)")

    return {
        "gamma":             gamma,
        "map50":             map_result["map50"],
        "n_classes":         map_result["n_classes_with_detections"],
        "median_latency_ms": median_latency,
        "latencies_ms":      latencies,
        "n_images":          len(image_paths),
        "n_errors":          n_errors,
        "n_predictions":     len(all_preds),
        "viable":            viable,
        "class_aps":         map_result["class_aps"],
    }

# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluación Tier 1 — XIAO ESP32S3 Sense"
    )
    parser.add_argument("--port",       required=True,
                        help="Puerto serial, ej: /dev/cu.usbmodem1101")
    parser.add_argument("--gamma",      type=float,
                        help="Gamma a evaluar (ej: 1.0)")
    parser.add_argument("--all-gammas", action="store_true",
                        help="Evaluar los 5 gammas en secuencia")
    parser.add_argument("--n-images",   type=int, default=None,
                        help="Limitar a N imágenes — útil para debug (ej: --n-images 1)")
    parser.add_argument("--tflite-dir", type=Path,
                        default=Path("yolov8n_saved_model"),
                        help="Directorio con output_quant.txt")
    return parser.parse_args()


def print_summary(results: list[dict]) -> None:
    print(f"\n{'═'*56}")
    print("  RESUMEN — Tier 1 (XIAO ESP32S3 Sense)")
    print(f"{'─'*56}")
    print(f"  {'γ':>5}  {'mAP@0.5':>8}  {'Latencia':>12}  {'Viable':>8}")
    print(f"  {'─'*5}  {'─'*8}  {'─'*12}  {'─'*8}")
    for r in results:
        lat        = f"{r['median_latency_ms']:.0f} ms" if r["median_latency_ms"] else "N/A"
        viable_str = "✅" if r["viable"] else "❌"
        print(f"  {r['gamma']:>5.1f}  {r['map50']:>8.4f}  {lat:>12}  {viable_str:>8}")
    print(f"{'═'*56}")
    n_viable = sum(1 for r in results if r["viable"])
    print(f"\n  Condiciones viables: {n_viable}/{len(results)}")
    print(f"  (Umbral: mAP@0.5 ≥ 0.5  Y  latencia ≤ 50 ms)")


def main() -> None:
    args = parse_args()

    if not args.gamma and not args.all_gammas:
        print("❌  Especificar --gamma VALOR o --all-gammas")
        sys.exit(1)

    gammas = ALL_GAMMAS if args.all_gammas else [args.gamma]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Localizar dataset
    images_dir, labels_dir = find_coco128_paths()
    image_paths = sorted(
        list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    )
    print(f"COCO128: {len(image_paths)} imágenes en {images_dir}")

    # Limitar imágenes si se usa --n-images (modo debug)
    if args.n_images:
        image_paths = image_paths[:args.n_images]
        print(f"⚠️   Modo debug: usando solo {args.n_images} imagen(es)")

    # Conectar al ESP32
    ser = connect_to_esp32(args.port)

    # Warmup — primera inferencia descartada
    print("Warmup (primera inferencia descartada)…")
    warmup_img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    warmup_result = send_image_get_result(ser, warmup_img)
    if warmup_result:
        print(f"Warmup completado — latencia warmup: {warmup_result['latency_ms']} ms\n")
    else:
        print("⚠️  Warmup sin respuesta — continuar igual\n")

    all_results: list[dict] = []
    try:
        for gamma in gammas:
            metrics = evaluate_gamma(ser, gamma, image_paths, labels_dir)
            all_results.append(metrics)

            out_file = RESULTS_DIR / f"tier1_gamma{gamma:.1f}.json"
            metrics_to_save = {k: v for k, v in metrics.items() if k != "class_aps"}
            metrics_to_save["latency_p50_ms"] = (
                float(np.percentile(metrics["latencies_ms"], 50))
                if metrics["latencies_ms"] else None
            )
            with open(out_file, "w") as f:
                json.dump(metrics_to_save, f, indent=2)
            print(f"\n  💾 Guardado: {out_file}")

    finally:
        ser.close()
        print("\nPuerto serial cerrado")

    print_summary(all_results)

    combined_file = RESULTS_DIR / "tier1_all_gammas.json"
    summary_data = [
        {
            "gamma":             r["gamma"],
            "map50":             r["map50"],
            "median_latency_ms": r["median_latency_ms"],
            "n_images":          r["n_images"],
            "n_errors":          r["n_errors"],
            "viable":            r["viable"],
        }
        for r in all_results
    ]
    with open(combined_file, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\n💾 Resultados: {combined_file}")


if __name__ == "__main__":
    main()