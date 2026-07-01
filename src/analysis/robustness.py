"""
robustness_by_size.py

Analiza la robustez de YOLOv8n frente a variación de gamma según el tamaño
de los objetos (small / medium / large), usando los umbrales estándar de COCO:

    Small:  área normalizada < (32/640)²   → objetos pequeños
    Medium: (32/640)² ≤ área < (96/640)²  → objetos medianos
    Large:  área ≥ (96/640)²              → objetos grandes

Para cada categoría y gamma calcula:
    - Recall (detección rate): TP / (TP + FN)
    - Precisión media: promedio de confidences de TPs
    - Distribución de objetos en COCO128

Matching GT→predicción: IoU ≥ 0.5 (mismo umbral que mAP@0.5).

Salida en: results/size_robustness/
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from ultralytics import YOLO
from ultralytics import settings as ultra_settings

# ── Configuración ──────────────────────────────────────────────────────────────

GAMMAS       = [0.3, 0.6, 1.0, 1.4, 1.8]
MODEL_PATH   = "yolov8n.pt"
IMAGE_SIZE   = 640
DEVICE       = "mps"
CONF_THRESH  = 0.25
IOU_THRESH   = 0.50
OUTPUT_DIR   = Path("results/size_robustness")

# Umbrales de tamaño COCO (normalizados a resolución 640×640)
SMALL_MAX  = (32  / IMAGE_SIZE) ** 2   # < 0.0025
MEDIUM_MAX = (96  / IMAGE_SIZE) ** 2   # < 0.0225

SIZE_LABELS  = ["Small\n(<32px)", "Medium\n(32–96px)", "Large\n(>96px)"]
SIZE_KEYS    = ["small", "medium", "large"]
SIZE_COLORS  = {"small": "#E53935", "medium": "#FB8C00", "large": "#43A047"}

PALETTE = {0.3: "#E53935", 0.6: "#FF7043", 1.0: "#1E88E5",
           1.4: "#43A047", 1.8: "#8E24AA"}

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


# ── Clasificación por tamaño ───────────────────────────────────────────────────

def classify_size(norm_w: float, norm_h: float) -> str:
    """Clasifica un bbox en small/medium/large según área normalizada."""
    area = norm_w * norm_h
    if area < SMALL_MAX:
        return "small"
    if area < MEDIUM_MAX:
        return "medium"
    return "large"


def load_gt_labels(label_dir: Path) -> dict[str, list[dict]]:
    """
    Carga etiquetas YOLO (.txt) desde un directorio.
    Retorna {filename_stem: [{"class_id", "cx", "cy", "w", "h", "size"}]}.
    """
    gt = {}
    for label_file in sorted(label_dir.glob("*.txt")):
        objects = []
        for line in label_file.read_text().strip().splitlines():
            if not line:
                continue
            parts = line.split()
            cls, cx, cy, w, h = int(parts[0]), *map(float, parts[1:5])
            objects.append({
                "class_id": cls,
                "cx": cx, "cy": cy, "w": w, "h": h,
                "size": classify_size(w, h),
            })
        gt[label_file.stem] = objects
    return gt


# ── Cálculo de IoU ─────────────────────────────────────────────────────────────

def iou_xyxy(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """IoU entre dos cajas en formato [x1, y1, x2, y2] (píxeles absolutos)."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def norm_to_xyxy(cx: float, cy: float, w: float, h: float,
                 img_w: int = IMAGE_SIZE, img_h: int = IMAGE_SIZE) -> np.ndarray:
    """Convierte bbox normalizado (cx,cy,w,h) a píxeles absolutos (x1,y1,x2,y2)."""
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return np.array([x1, y1, x2, y2])


# ── Matching GT → Predicción ───────────────────────────────────────────────────

def match_detections(
    gt_objects: list[dict],
    predictions: list[dict],
) -> tuple[list[dict], list[dict]]:
    """
    Empareja GT con predicciones usando IoU ≥ IOU_THRESH + mismo class_id.
    Retorna (true_positives, false_negatives).

    TP: objeto GT que tiene al menos una predicción coincidente.
    FN: objeto GT sin predicción coincidente (objeto perdido).
    """
    matched_gt = set()
    true_positives = []

    for pred in sorted(predictions, key=lambda x: -x["conf"]):
        best_iou  = 0.0
        best_gt_i = -1
        pred_box  = pred["box"]

        for i, gt in enumerate(gt_objects):
            if i in matched_gt:
                continue
            if gt["class_id"] != pred["class_id"]:
                continue
            gt_box = norm_to_xyxy(gt["cx"], gt["cy"], gt["w"], gt["h"])
            iou_val = iou_xyxy(pred_box.astype(float), gt_box)
            if iou_val > best_iou:
                best_iou  = iou_val
                best_gt_i = i

        if best_iou >= IOU_THRESH and best_gt_i >= 0:
            matched_gt.add(best_gt_i)
            true_positives.append({**gt_objects[best_gt_i], "conf": pred["conf"]})

    false_negatives = [gt for i, gt in enumerate(gt_objects) if i not in matched_gt]
    return true_positives, false_negatives


# ── Evaluación ─────────────────────────────────────────────────────────────────

def get_image_dir(gamma: float, datasets_dir: Path) -> Path:
    if gamma == 1.0:
        return datasets_dir / "coco128" / "images" / "train2017"
    return datasets_dir / f"coco128_gamma{gamma:.1f}" / "images" / "train2017"


def evaluate_by_size(
    model: YOLO,
    gt: dict[str, list[dict]],
    image_dir: Path,
    gamma: float,
) -> pd.DataFrame:
    """
    Para cada imagen corre predict(), hace matching con GT y acumula
    TPs y FNs por categoría de tamaño.
    Retorna DataFrame con métricas por tamaño para este gamma.
    """
    # Acumuladores
    counters = {size: {"tp": 0, "fn": 0, "confs": []} for size in SIZE_KEYS}

    for img_path in sorted(image_dir.glob("*.jpg")):
        stem = img_path.stem
        if stem not in gt:
            continue

        result = model.predict(
            source=str(img_path),
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            conf=CONF_THRESH,
            verbose=False,
        )[0]

        predictions = []
        if result.boxes is not None:
            for box in result.boxes:
                predictions.append({
                    "box":      box.xyxy[0].cpu().numpy(),
                    "conf":     float(box.conf.item()),
                    "class_id": int(box.cls.item()),
                })

        tps, fns = match_detections(gt[stem], predictions)

        for tp in tps:
            counters[tp["size"]]["tp"] += 1
            counters[tp["size"]]["confs"].append(tp["conf"])
        for fn in fns:
            counters[fn["size"]]["fn"] += 1

    rows = []
    for size in SIZE_KEYS:
        tp = counters[size]["tp"]
        fn = counters[size]["fn"]
        total = tp + fn
        recall = tp / total if total > 0 else 0.0
        avg_conf = np.mean(counters[size]["confs"]) if counters[size]["confs"] else 0.0
        rows.append({
            "gamma":    gamma,
            "size":     size,
            "tp":       tp,
            "fn":       fn,
            "total_gt": total,
            "recall":   recall,
            "avg_conf": avg_conf,
        })

    return pd.DataFrame(rows)


# ── Figuras ────────────────────────────────────────────────────────────────────

def plot_recall_by_size(df: pd.DataFrame, output_path: Path) -> None:
    """Recall por tamaño × gamma — el gráfico principal."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

    for ax, size, label in zip(axes, SIZE_KEYS, SIZE_LABELS):
        sub = df[df["size"] == size].sort_values("gamma")
        color = SIZE_COLORS[size]

        ax.plot(sub["gamma"], sub["recall"], "o-", color=color,
                linewidth=2.5, markersize=8, zorder=4)
        ax.fill_between(sub["gamma"], sub["recall"], alpha=0.12, color=color)

        # Anotar valores
        for _, row in sub.iterrows():
            ax.annotate(f'{row["recall"]:.2f}',
                        (row["gamma"], row["recall"]),
                        textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=9, color=color, fontweight="bold")

        ax.axhline(0.5, color="#999", linestyle="--", linewidth=1.2,
                   label="Umbral 50%", alpha=0.8)
        ax.axvline(1.0, color="#ccc", linestyle=":", linewidth=1)
        ax.set_title(label, fontsize=12, color=color, fontweight="bold")
        ax.set_xlabel("Gamma (γ)", fontsize=10)
        ax.set_xticks(GAMMAS)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("Recall (detección rate)", fontsize=11)

    fig.suptitle(
        "Robustez por tamaño de objeto — Recall vs Gamma\n"
        "YOLOv8n / COCO128 (IoU ≥ 0.5, conf ≥ 0.25)",
        fontsize=12, y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {output_path.name}")


def plot_recall_overlay(df: pd.DataFrame, output_path: Path) -> None:
    """Recall de los 3 tamaños en un solo gráfico para comparar directamente."""
    fig, ax = plt.subplots(figsize=(9, 5))

    for size, label, ls in zip(SIZE_KEYS, SIZE_LABELS,
                                ["-", "--", "-."]):
        sub = df[df["size"] == size].sort_values("gamma")
        label_clean = label.replace("\n", " ")
        ax.plot(sub["gamma"], sub["recall"], "o" + ls,
                color=SIZE_COLORS[size], linewidth=2.2, markersize=7,
                label=label_clean, zorder=4)

    ax.axhline(0.5, color="#999", linestyle=":", linewidth=1.2, alpha=0.8)
    ax.axvline(1.0, color="#ccc", linestyle=":", linewidth=1)
    ax.set_xlabel("Gamma (γ)", fontsize=12)
    ax.set_ylabel("Recall (detección rate)", fontsize=12)
    ax.set_title("Comparación de recall por tamaño de objeto vs gamma\n"
                 "Objetos pequeños son los más vulnerables a sub-exposición",
                 fontsize=11)
    ax.set_xticks(GAMMAS)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.grid(alpha=0.2)
    ax.legend(fontsize=10, loc="lower right")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {output_path.name}")


def plot_gt_distribution(gt: dict, output_path: Path) -> None:
    """Distribución de tamaños en el dataset COCO128."""
    counts = {"small": 0, "medium": 0, "large": 0}
    for objects in gt.values():
        for obj in objects:
            counts[obj["size"]] += 1
    total = sum(counts.values())

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        SIZE_LABELS,
        [counts[k] for k in SIZE_KEYS],
        color=[SIZE_COLORS[k] for k in SIZE_KEYS],
        zorder=3, alpha=0.85,
    )
    for bar, key in zip(bars, SIZE_KEYS):
        n = counts[key]
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 8,
                f"{n}\n({n/total*100:.1f}%)",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("N° de objetos GT", fontsize=11)
    ax.set_title("Distribución de tamaños en COCO128\n"
                 f"(total: {total} objetos en 128 imágenes)", fontsize=11)
    ax.grid(axis="y", alpha=0.25, zorder=0)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {output_path.name}")


def plot_conf_by_size(df: pd.DataFrame, output_path: Path) -> None:
    """Confidence promedio de TPs por tamaño × gamma."""
    fig, ax = plt.subplots(figsize=(9, 5))

    for size, label in zip(SIZE_KEYS, SIZE_LABELS):
        sub = df[df["size"] == size].sort_values("gamma")
        label_clean = label.replace("\n", " ")
        ax.plot(sub["gamma"], sub["avg_conf"], "o-",
                color=SIZE_COLORS[size], linewidth=2, markersize=7,
                label=label_clean, zorder=4)

    ax.axvline(1.0, color="#ccc", linestyle=":", linewidth=1)
    ax.set_xlabel("Gamma (γ)", fontsize=12)
    ax.set_ylabel("Confidence promedio (TPs)", fontsize=12)
    ax.set_title("Confidence promedio de detecciones correctas\npor tamaño de objeto",
                 fontsize=11)
    ax.set_xticks(GAMMAS)
    ax.set_ylim(0.2, 1.0)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {output_path.name}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets_dir = Path(ultra_settings["datasets_dir"])
    label_dir    = datasets_dir / "coco128" / "labels" / "train2017"

    print("Cargando etiquetas GT...")
    gt = load_gt_labels(label_dir)

    sizes = [obj["size"] for objs in gt.values() for obj in objs]
    for s in SIZE_KEYS:
        print(f"  {s}: {sizes.count(s)} objetos")

    print("\nGenerando distribución de tamaños...")
    plot_gt_distribution(gt, OUTPUT_DIR / "distribucion_tamanos.png")

    print("\nCargando modelo YOLOv8n...")
    model = YOLO(MODEL_PATH)

    all_frames = []
    for gamma in GAMMAS:
        image_dir = get_image_dir(gamma, datasets_dir)
        print(f"  Evaluando γ={gamma}...")
        frame = evaluate_by_size(model, gt, image_dir, gamma)
        all_frames.append(frame)
        for _, row in frame.iterrows():
            print(f"    {row['size']:6s} → recall={row['recall']:.3f}  "
                  f"({int(row['tp'])} TP / {int(row['total_gt'])} GT)")

    df = pd.concat(all_frames, ignore_index=True)
    df.to_csv(OUTPUT_DIR / "recall_by_size.csv", index=False)
    print(f"\nCSV guardado: {OUTPUT_DIR / 'recall_by_size.csv'}")

    print("\nGenerando figuras...")
    plot_recall_by_size(df,    OUTPUT_DIR / "recall_por_tamano.png")
    plot_recall_overlay(df,    OUTPUT_DIR / "recall_overlay.png")
    plot_conf_by_size(df,      OUTPUT_DIR / "confidence_por_tamano.png")

    print(f"\n✓ Análisis completo. Resultados en: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()