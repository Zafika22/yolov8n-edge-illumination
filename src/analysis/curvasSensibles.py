"""
sensitivity_curves.py

Evalúa YOLOv8n en una grilla fina de valores gamma para identificar
con precisión el punto de quiebre de mAP@0.5 y de recall por tamaño.

Grilla principal: γ ∈ {0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90,
                        1.00, 1.10, 1.20, 1.30, 1.40, 1.50, 1.60, 1.70, 1.80}

Las imágenes gamma intermedias se generan on-the-fly en /tmp/ y se
eliminan al final (no se guardan en el dataset permanente).

Salida en: results/sensitivity/
"""

import shutil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO
from ultralytics import settings as ultra_settings

# ── Configuración ──────────────────────────────────────────────────────────────

# Grilla fina de gammas (los 5 originales + intermedios)
GAMMAS_FINE = [
    0.10, 0.20, 0.30, 0.40, 0.50,
    0.60, 0.70, 0.80, 0.90, 1.00,
    1.10, 1.20, 1.30, 1.40, 1.50,
    1.60, 1.70, 1.80,
]
GAMMAS_ORIGINAL = [0.3, 0.6, 1.0, 1.4, 1.8]  # para marcar en los gráficos

MODEL_PATH  = "yolov8n.pt"
IMAGE_SIZE  = 640
DEVICE      = "mps"
TMP_DIR     = Path("/tmp/gamma_sensitivity")
OUTPUT_DIR  = Path("results/sensitivity")

VIABILITY_MAP  = 0.50
VIABILITY_LAT  = 50.0   # ms — para referencia, la latencia no cambia con gamma

# Umbrales de tamaño COCO
SMALL_MAX  = (32  / IMAGE_SIZE) ** 2
MEDIUM_MAX = (96  / IMAGE_SIZE) ** 2
IOU_THRESH = 0.50
CONF_THRESH = 0.25

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


# ── Generación on-the-fly de imágenes gamma ────────────────────────────────────

def apply_gamma(img_bgr: np.ndarray, gamma: float) -> np.ndarray:
    """Aplica corrección gamma: I_out = I_in^(1/gamma)."""
    inv_gamma = 1.0 / gamma
    lut = (np.arange(256) / 255.0) ** inv_gamma * 255.0
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return cv2.LUT(img_bgr, lut)


def generate_gamma_images(
    source_dir: Path, gamma: float, out_dir: Path
) -> None:
    """Genera imágenes con corrección gamma en out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for img_path in source_dir.glob("*.jpg"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        cv2.imwrite(str(out_dir / img_path.name), apply_gamma(img, gamma))


def build_tmp_yaml(gamma: float, img_dir: Path, label_dir: Path) -> Path:
    """Crea YAML temporal apuntando a imágenes gamma generadas on-the-fly."""
    yaml_data = {
        "path":  str(img_dir.parent.parent),
        "train": str(img_dir.relative_to(img_dir.parent.parent)),
        "val":   str(img_dir.relative_to(img_dir.parent.parent)),
        "names": {i: n for i, n in enumerate(COCO_NAMES)},
    }
    # Copiar etiquetas al lado de las imágenes si no existen
    lbl_target = img_dir.parent / "labels" / "train2017"
    if not lbl_target.exists():
        shutil.copytree(label_dir, lbl_target)

    yaml_path = Path(f"/tmp/sensitivity_gamma{gamma:.2f}.yaml")
    yaml_path.write_text(yaml.dump(yaml_data, allow_unicode=True))
    return yaml_path


# ── Evaluación de mAP ─────────────────────────────────────────────────────────

def evaluate_map(model: YOLO, data_yaml: Path) -> dict[str, float]:
    """Retorna mAP@0.5 global y por clase."""
    results = model.val(
        data=str(data_yaml),
        imgsz=IMAGE_SIZE,
        device=DEVICE,
        verbose=False,
        plots=False,
        save_json=False,
    )
    per_class = {}
    for pos, cls_idx in enumerate(results.box.ap_class_index):
        per_class[COCO_NAMES[int(cls_idx)]] = float(results.box.maps[pos])

    return {
        "map50":     float(results.box.map50),
        "map50_95":  float(results.box.map),
        "per_class": per_class,
    }


# ── Recall por tamaño ──────────────────────────────────────────────────────────

def classify_size(w: float, h: float) -> str:
    area = w * h
    if area < SMALL_MAX:
        return "small"
    if area < MEDIUM_MAX:
        return "medium"
    return "large"


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def norm_to_xyxy(cx, cy, w, h, s=IMAGE_SIZE) -> np.ndarray:
    return np.array([(cx-w/2)*s, (cy-h/2)*s, (cx+w/2)*s, (cy+h/2)*s])


def evaluate_recall_by_size(
    model: YOLO, gt: dict, img_dir: Path
) -> dict[str, float]:
    """Calcula recall (TP/total_GT) por categoría de tamaño."""
    counts = {s: {"tp": 0, "total": 0} for s in ["small", "medium", "large"]}

    for img_path in sorted(img_dir.glob("*.jpg")):
        stem = img_path.stem
        if stem not in gt:
            continue
        gt_objs = gt[stem]
        for obj in gt_objs:
            counts[obj["size"]]["total"] += 1

        result = model.predict(
            source=str(img_path), imgsz=IMAGE_SIZE,
            device=DEVICE, conf=CONF_THRESH, verbose=False,
        )[0]

        preds = []
        if result.boxes is not None:
            for box in result.boxes:
                preds.append({
                    "box": box.xyxy[0].cpu().numpy(),
                    "conf": float(box.conf.item()),
                    "class_id": int(box.cls.item()),
                })

        matched = set()
        for pred in sorted(preds, key=lambda x: -x["conf"]):
            for i, gt_obj in enumerate(gt_objs):
                if i in matched or gt_obj["class_id"] != pred["class_id"]:
                    continue
                gt_box = norm_to_xyxy(gt_obj["cx"], gt_obj["cy"],
                                       gt_obj["w"],  gt_obj["h"])
                if iou_xyxy(pred["box"].astype(float), gt_box) >= IOU_THRESH:
                    matched.add(i)
                    counts[gt_obj["size"]]["tp"] += 1
                    break

    return {
        s: (counts[s]["tp"] / counts[s]["total"]
            if counts[s]["total"] > 0 else 0.0)
        for s in ["small", "medium", "large"]
    }


def load_gt_labels(label_dir: Path) -> dict:
    gt = {}
    for lf in sorted(label_dir.glob("*.txt")):
        objs = []
        for line in lf.read_text().strip().splitlines():
            if not line:
                continue
            cls, cx, cy, w, h = int(line.split()[0]), *map(float, line.split()[1:5])
            objs.append({"class_id": cls, "cx": cx, "cy": cy, "w": w, "h": h,
                          "size": classify_size(w, h)})
        gt[lf.stem] = objs
    return gt


# ── Figuras ────────────────────────────────────────────────────────────────────

def gamma_color(gamma: float) -> str:
    """Color continuo: rojo (sub-exp) → azul (baseline) → verde (sobre-exp)."""
    if gamma <= 1.0:
        t = (gamma - 0.1) / 0.9
        r = int(220 - t * (220 - 30))
        g = int(50  + t * (130 - 50))
        b = int(50  + t * (220 - 50))
    else:
        t = (gamma - 1.0) / 0.8
        r = int(30  + t * (50  - 30))
        g = int(130 + t * (180 - 130))
        b = int(220 - t * (220 - 80))
    return f"#{r:02x}{g:02x}{b:02x}"


def plot_map_curve(df: pd.DataFrame, output_path: Path) -> None:
    """Curva principal: mAP@0.5 vs gamma con punto de quiebre anotado."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(df["gamma"], df["map50"], "o-", color="#1E88E5",
            linewidth=2.5, markersize=6, zorder=4, label="mAP@0.5")
    ax.fill_between(df["gamma"], df["map50"], VIABILITY_MAP,
                    where=df["map50"] >= VIABILITY_MAP,
                    alpha=0.10, color="#43A047", label="Zona viable")
    ax.fill_between(df["gamma"], df["map50"], VIABILITY_MAP,
                    where=df["map50"] < VIABILITY_MAP,
                    alpha=0.12, color="#E53935", label="Zona no viable")

    # Línea de umbral
    ax.axhline(VIABILITY_MAP, color="#E53935", linestyle="--",
               linewidth=1.4, alpha=0.8, label=f"Umbral mAP@0.5 = {VIABILITY_MAP}")

    # Marcar gammas originales del experimento
    orig = df[df["gamma"].isin(GAMMAS_ORIGINAL)]
    ax.scatter(orig["gamma"], orig["map50"], s=80, color="#FF6F00",
               zorder=5, label="Gammas del experimento original")

    # Anotar punto de quiebre (primer gamma donde mAP < 0.5)
    below = df[df["map50"] < VIABILITY_MAP]
    if not below.empty:
        breakpoint_gamma = below["gamma"].iloc[-1]  # último que falla (más alto)
        breakpoint_map   = below["map50"].iloc[-1]
        ax.annotate(
            f"Quiebre en γ≈{breakpoint_gamma:.2f}\n(mAP={breakpoint_map:.3f})",
            xy=(breakpoint_gamma, breakpoint_map),
            xytext=(breakpoint_gamma + 0.15, breakpoint_map - 0.06),
            arrowprops=dict(arrowstyle="->", color="#E53935", lw=1.5),
            fontsize=9, color="#E53935",
        )

    # Anotar valores en gammas originales
    for _, row in orig.iterrows():
        ax.annotate(f"{row['map50']:.3f}",
                    (row["gamma"], row["map50"]),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8.5, color="#FF6F00", fontweight="bold")

    ax.set_xlabel("Gamma (γ)", fontsize=12)
    ax.set_ylabel("mAP@0.5", fontsize=12)
    ax.set_title("Curva de sensibilidad fina — mAP@0.5 vs Gamma\n"
                 "YOLOv8n / COCO128 (grilla de 18 puntos)",
                 fontsize=12, pad=10)
    ax.set_xticks(GAMMAS_FINE)
    ax.set_xticklabels([f"{g:.1f}" for g in GAMMAS_FINE], fontsize=7.5, rotation=45)
    ax.set_ylim(0, 0.75)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {output_path.name}")


def plot_recall_size_curves(size_df: pd.DataFrame, output_path: Path) -> None:
    """Curvas de recall por tamaño en grilla fina."""
    size_colors = {"small": "#E53935", "medium": "#FB8C00", "large": "#43A047"}
    size_labels = {"small": "Small (<32px)", "medium": "Medium (32–96px)",
                   "large": "Large (>96px)"}

    fig, ax = plt.subplots(figsize=(10, 5))

    for size in ["small", "medium", "large"]:
        sub = size_df[size_df["size"] == size].sort_values("gamma")
        ax.plot(sub["gamma"], sub["recall"], "o-",
                color=size_colors[size], linewidth=2, markersize=5,
                label=size_labels[size], zorder=4)

    ax.axhline(0.5, color="#999", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.axvline(1.0, color="#ccc", linestyle=":", linewidth=1)

    # Marcar gammas originales
    for g in GAMMAS_ORIGINAL:
        ax.axvline(g, color="#FF6F00", linestyle="--", linewidth=0.7, alpha=0.5)

    ax.set_xlabel("Gamma (γ)", fontsize=12)
    ax.set_ylabel("Recall (IoU ≥ 0.5, conf ≥ 0.25)", fontsize=12)
    ax.set_title("Sensitivity curves por tamaño de objeto\n"
                 "Grilla fina de 18 puntos gamma",
                 fontsize=12, pad=10)
    ax.set_xticks(GAMMAS_FINE)
    ax.set_xticklabels([f"{g:.1f}" for g in GAMMAS_FINE], fontsize=7.5, rotation=45)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylim(-0.02, 0.55)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {output_path.name}")


def plot_degradation_rate(df: pd.DataFrame, output_path: Path) -> None:
    """
    Tasa de degradación: derivada de mAP respecto a gamma.
    Muestra en qué tramos cambia más rápido el rendimiento.
    """
    df_sorted = df.sort_values("gamma").reset_index(drop=True)
    dgamma = df_sorted["gamma"].diff()
    dmap   = df_sorted["map50"].diff()
    rate   = (dmap / dgamma).fillna(0)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Panel superior: mAP
    ax = axes[0]
    ax.plot(df_sorted["gamma"], df_sorted["map50"], "o-",
            color="#1E88E5", linewidth=2, markersize=5)
    ax.axhline(VIABILITY_MAP, color="#E53935", linestyle="--",
               linewidth=1.2, alpha=0.7, label=f"Umbral {VIABILITY_MAP}")
    ax.set_ylabel("mAP@0.5", fontsize=11)
    ax.set_title("mAP@0.5 y tasa de cambio por gamma\n"
                 "(derivada numérica — indica dónde la degradación es más rápida)",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

    # Panel inferior: tasa de cambio (derivada)
    ax2 = axes[1]
    colors = ["#E53935" if r < 0 else "#43A047" for r in rate]
    ax2.bar(df_sorted["gamma"], rate, width=0.08, color=colors,
            alpha=0.8, zorder=3)
    ax2.axhline(0, color="#333", linewidth=0.8)
    ax2.set_xlabel("Gamma (γ)", fontsize=11)
    ax2.set_ylabel("ΔmAP / Δγ", fontsize=11)
    ax2.set_xticks(GAMMAS_FINE)
    ax2.set_xticklabels([f"{g:.1f}" for g in GAMMAS_FINE],
                         fontsize=7.5, rotation=45)
    ax2.grid(axis="y", alpha=0.2)

    red_patch  = plt.Rectangle((0,0),1,1, color="#E53935", alpha=0.8)
    green_patch = plt.Rectangle((0,0),1,1, color="#43A047", alpha=0.8)
    ax2.legend([red_patch, green_patch],
               ["Degradación", "Mejora"], fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {output_path.name}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    datasets_dir = Path(ultra_settings["datasets_dir"])
    source_dir   = datasets_dir / "coco128" / "images" / "train2017"
    label_dir    = datasets_dir / "coco128" / "labels" / "train2017"

    print("Cargando etiquetas GT y modelo...")
    gt    = load_gt_labels(label_dir)
    model = YOLO(MODEL_PATH)

    map_records  = []
    size_records = []

    for gamma in GAMMAS_FINE:
        gamma_str = f"{gamma:.2f}"
        print(f"\nγ={gamma_str}...")

        # Generar imágenes gamma on-the-fly
        img_dir = TMP_DIR / f"gamma{gamma_str}" / "images" / "train2017"
        generate_gamma_images(source_dir, gamma, img_dir)

        # mAP@0.5 via model.val()
        data_yaml = build_tmp_yaml(gamma, img_dir, label_dir)
        metrics   = evaluate_map(model, data_yaml)
        map_records.append({"gamma": gamma, "map50": metrics["map50"],
                             "map50_95": metrics["map50_95"]})
        print(f"  mAP@0.5 = {metrics['map50']:.4f}")

        # Recall por tamaño via predict() + IoU matching
        recall = evaluate_recall_by_size(model, gt, img_dir)
        for size, r in recall.items():
            size_records.append({"gamma": gamma, "size": size, "recall": r})
        print(f"  Recall — small: {recall['small']:.3f}  "
              f"medium: {recall['medium']:.3f}  large: {recall['large']:.3f}")

    # Guardar CSVs
    map_df  = pd.DataFrame(map_records)
    size_df = pd.DataFrame(size_records)
    map_df.to_csv(OUTPUT_DIR / "map_fine_grid.csv",  index=False)
    size_df.to_csv(OUTPUT_DIR / "size_fine_grid.csv", index=False)

    # Figuras
    print("\nGenerando figuras...")
    plot_map_curve(map_df,               OUTPUT_DIR / "sensitivity_map_curve.png")
    plot_recall_size_curves(size_df,     OUTPUT_DIR / "sensitivity_recall_by_size.png")
    plot_degradation_rate(map_df,        OUTPUT_DIR / "sensitivity_degradation_rate.png")

    # Limpiar imágenes temporales
    shutil.rmtree(TMP_DIR)
    print(f"\n✓ Análisis completo. Resultados en: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()