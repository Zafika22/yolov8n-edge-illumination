"""
detection_visualization.py

Visualiza detecciones fallidas comparando γ=1.0 (baseline) vs γ=0.3 (sub-exposición).
Selecciona automáticamente las imágenes con mayor pérdida de detecciones para
mostrar los casos más representativos del colapso.

Salida en: results/detection_viz/
"""

from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
from ultralytics import settings as ultra_settings

# ── Configuración ──────────────────────────────────────────────────────────────

MODEL_PATH = "yolov8n.pt"
IMAGE_SIZE = 640
DEVICE = "mps"
CONF_THRESHOLD = 0.25   # umbral estándar de YOLO
N_COMPARISON_IMAGES = 6  # imágenes para el grid comparativo
OUTPUT_DIR = Path("results/detection_viz")

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

# Colores por condición
COLOR_BASELINE = (34, 197, 94)    # verde — detecciones en γ=1.0
COLOR_MISSED   = (239, 68, 68)    # rojo  — objetos perdidos en γ=0.3
COLOR_KEPT     = (59, 130, 246)   # azul  — detecciones que sobreviven en γ=0.3

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


# ── Utilidades ─────────────────────────────────────────────────────────────────

def get_image_dir(gamma: float, datasets_dir: Path) -> Path:
    if gamma == 1.0:
        return datasets_dir / "coco128" / "images" / "train2017"
    return datasets_dir / f"coco128_gamma{gamma:.1f}" / "images" / "train2017"


def run_predictions(model: YOLO, image_paths: list[Path]) -> dict[str, list]:
    """
    Corre predict() sobre una lista de imágenes.
    Retorna dict {filename: [{"box": xyxy, "conf": float, "class": str}]}.
    """
    results_map = {}
    for path in image_paths:
        result = model.predict(
            source=str(path),
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            conf=CONF_THRESHOLD,
            verbose=False,
        )[0]

        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                detections.append({
                    "box":       box.xyxy[0].cpu().numpy().astype(int),
                    "conf":      float(box.conf.item()),
                    "class":     COCO_NAMES[int(box.cls.item())],
                    "class_id":  int(box.cls.item()),
                })
        results_map[path.name] = detections

    return results_map


def draw_boxes(img: np.ndarray, detections: list, color: tuple,
               label_prefix: str = "") -> np.ndarray:
    """Dibuja bounding boxes con etiquetas sobre la imagen."""
    out = img.copy()
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = f"{label_prefix}{det['class']} {det['conf']:.2f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Fondo del texto
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return out


def select_most_affected_images(
    baseline_preds: dict, subexp_preds: dict, n: int
) -> list[str]:
    """
    Selecciona las N imágenes con mayor caída en número de detecciones
    entre baseline (γ=1.0) y sub-exposición (γ=0.3).
    Solo considera imágenes con al menos 2 detecciones en baseline.
    """
    drops = []
    for fname in baseline_preds:
        n_base = len(baseline_preds[fname])
        n_sub  = len(subexp_preds.get(fname, []))
        if n_base >= 2:
            drops.append((fname, n_base - n_sub, n_base, n_sub))

    drops.sort(key=lambda x: x[1], reverse=True)
    return [d[0] for d in drops[:n]]


# ── Figuras ────────────────────────────────────────────────────────────────────

def plot_comparison_grid(
    selected: list[str],
    baseline_dir: Path,
    subexp_dir: Path,
    baseline_preds: dict,
    subexp_preds: dict,
    output_path: Path,
) -> None:
    """
    Grid de N_COMPARISON_IMAGES filas × 2 columnas:
    izquierda = γ=1.0 con detecciones, derecha = γ=0.3 con detecciones.
    """
    n = len(selected)
    fig, axes = plt.subplots(n, 2, figsize=(13, 4.2 * n))
    if n == 1:
        axes = [axes]

    for row, fname in enumerate(selected):
        img_base = cv2.cvtColor(
            cv2.imread(str(baseline_dir / fname)), cv2.COLOR_BGR2RGB)
        img_sub  = cv2.cvtColor(
            cv2.imread(str(subexp_dir / fname)), cv2.COLOR_BGR2RGB)

        base_dets = baseline_preds[fname]
        sub_dets  = subexp_preds.get(fname, [])

        # Clases detectadas en cada condición
        base_classes = {d["class"] for d in base_dets}
        sub_classes  = {d["class"] for d in sub_dets}
        missed_classes = base_classes - sub_classes

        # Imagen baseline: todo en verde
        img_base_drawn = draw_boxes(img_base, base_dets, COLOR_BASELINE)

        # Imagen sub-exposición:
        # - detecciones que sobreviven en azul
        # - en baseline, marcar con rojo los que se perdieron
        sub_kept   = sub_dets
        base_lost  = [d for d in base_dets if d["class"] in missed_classes]

        img_sub_drawn = draw_boxes(img_sub,  sub_kept,  COLOR_KEPT)
        img_sub_drawn = draw_boxes(img_sub_drawn, base_lost, COLOR_MISSED)

        # Panel izquierdo
        ax_l = axes[row][0]
        ax_l.imshow(img_base_drawn)
        ax_l.set_title(
            f"γ=1.0 (baseline)  —  {len(base_dets)} detecciones\n{fname}",
            fontsize=9, pad=4)
        ax_l.axis("off")

        # Panel derecho
        ax_r = axes[row][1]
        ax_r.imshow(img_sub_drawn)
        lost_n = len(base_dets) - len(sub_dets)
        ax_r.set_title(
            f"γ=0.3 (sub-exposición)  —  {len(sub_dets)} detecciones "
            f"(−{lost_n} perdidas)",
            fontsize=9, pad=4)
        ax_r.axis("off")

        # Anotar clases perdidas
        if missed_classes:
            ax_r.text(
                0.01, 0.01,
                f"Perdidas: {', '.join(sorted(missed_classes))}",
                transform=ax_r.transAxes,
                fontsize=7.5, color="white", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#E53935", alpha=0.85),
            )

    # Leyenda global
    legend_patches = [
        mpatches.Patch(color=np.array(COLOR_BASELINE)/255, label="Detectado en γ=1.0"),
        mpatches.Patch(color=np.array(COLOR_KEPT)/255,     label="Detectado en γ=0.3"),
        mpatches.Patch(color=np.array(COLOR_MISSED)/255,   label="Perdido en γ=0.3 (posición desde γ=1.0)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        "Visualización de detecciones fallidas — YOLOv8n bajo sub-exposición (γ=0.3)\n"
        "Imágenes seleccionadas por mayor caída en número de detecciones",
        fontsize=12, y=1.005)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"  Grid guardado: {output_path.name}")


def plot_single_class_panels(
    baseline_dir: Path,
    subexp_dir: Path,
    baseline_preds: dict,
    subexp_preds: dict,
    target_classes: list[str],
    output_path: Path,
) -> None:
    """
    Panel enfocado en clases específicas (las más afectadas).
    Muestra la primera imagen disponible de cada clase donde ocurre pérdida.
    """
    examples = []  # (class_name, fname, base_dets, sub_dets)

    for cls in target_classes:
        for fname, base_dets in baseline_preds.items():
            base_cls = [d for d in base_dets if d["class"] == cls]
            sub_dets  = subexp_preds.get(fname, [])
            sub_cls   = [d for d in sub_dets if d["class"] == cls]

            # Buscar imagen donde se detecta en baseline pero se pierde en subexp
            if len(base_cls) > 0 and len(sub_cls) == 0:
                examples.append((cls, fname, base_dets, sub_dets))
                break

    if not examples:
        print("  No se encontraron ejemplos para el panel por clase.")
        return

    n = len(examples)
    fig, axes = plt.subplots(n, 2, figsize=(12, 3.8 * n))
    if n == 1:
        axes = [axes]

    for row, (cls, fname, base_dets, sub_dets) in enumerate(examples):
        img_base = cv2.cvtColor(
            cv2.imread(str(baseline_dir / fname)), cv2.COLOR_BGR2RGB)
        img_sub = cv2.cvtColor(
            cv2.imread(str(subexp_dir / fname)), cv2.COLOR_BGR2RGB)

        # Resaltar solo la clase objetivo
        target_base = [d for d in base_dets if d["class"] == cls]
        other_base  = [d for d in base_dets if d["class"] != cls]
        target_sub  = [d for d in sub_dets if d["class"] == cls]

        img_base_drawn = draw_boxes(img_base, other_base,  (160, 160, 160))
        img_base_drawn = draw_boxes(img_base_drawn, target_base, COLOR_BASELINE)

        img_sub_drawn  = draw_boxes(img_sub, sub_dets, COLOR_KEPT)
        # Mostrar en rojo dónde estaba el objeto en baseline
        img_sub_drawn  = draw_boxes(img_sub_drawn, target_base, COLOR_MISSED)

        ax_l = axes[row][0]
        ax_l.imshow(img_base_drawn)
        ax_l.set_title(f'γ=1.0 — "{cls}" detectado (conf={target_base[0]["conf"]:.2f})',
                       fontsize=9, pad=4)
        ax_l.axis("off")

        ax_r = axes[row][1]
        ax_r.imshow(img_sub_drawn)
        ax_r.set_title(f'γ=0.3 — "{cls}" NO detectado\n(caja roja = posición esperada)',
                       fontsize=9, pad=4)
        ax_r.axis("off")

    legend_patches = [
        mpatches.Patch(color=np.array(COLOR_BASELINE)/255, label="Clase objetivo — detectada"),
        mpatches.Patch(color=(0.63, 0.63, 0.63),           label="Otras clases"),
        mpatches.Patch(color=np.array(COLOR_MISSED)/255,   label="Posición esperada — NO detectada"),
        mpatches.Patch(color=np.array(COLOR_KEPT)/255,     label="Detecciones supervivientes"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4,
               fontsize=9, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        "Fallos de detección por clase — γ=1.0 vs γ=0.3\n"
        "Clases más afectadas por sub-exposición",
        fontsize=12, y=1.005)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"  Panel por clase guardado: {output_path.name}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets_dir = Path(ultra_settings["datasets_dir"])

    baseline_dir = get_image_dir(1.0, datasets_dir)
    subexp_dir   = get_image_dir(0.3, datasets_dir)

    image_paths = sorted(baseline_dir.glob("*.jpg"))
    print(f"Cargando modelo YOLOv8n ({len(image_paths)} imágenes)...")
    model = YOLO(MODEL_PATH)

    print("Corriendo predicciones en γ=1.0...")
    baseline_preds = run_predictions(model, image_paths)

    # Para subexp usar los mismos filenames pero desde el directorio gamma=0.3
    subexp_paths = [subexp_dir / p.name for p in image_paths if (subexp_dir / p.name).exists()]
    print(f"Corriendo predicciones en γ=0.3 ({len(subexp_paths)} imágenes)...")
    subexp_preds = run_predictions(model, subexp_paths)

    # Resumen rápido
    total_base = sum(len(v) for v in baseline_preds.values())
    total_sub  = sum(len(v) for v in subexp_preds.values())
    print(f"\nDetecciones totales — γ=1.0: {total_base} | γ=0.3: {total_sub} "
          f"(−{total_base - total_sub}, {(total_base-total_sub)/total_base*100:.1f}%)")

    # Seleccionar imágenes más afectadas
    selected = select_most_affected_images(
        baseline_preds, subexp_preds, N_COMPARISON_IMAGES)
    print(f"\nImágenes seleccionadas: {selected}")

    print("\nGenerando figuras...")
    plot_comparison_grid(
        selected, baseline_dir, subexp_dir,
        baseline_preds, subexp_preds,
        OUTPUT_DIR / "comparacion_grid.png",
    )

    # Panel por clase — top 5 más afectadas
    top_classes = ["scissors", "sports ball", "baseball bat", "sink", "oven"]
    plot_single_class_panels(
        baseline_dir, subexp_dir,
        baseline_preds, subexp_preds,
        top_classes,
        OUTPUT_DIR / "fallos_por_clase.png",
    )

    print(f"\n✓ Visualizaciones en: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()