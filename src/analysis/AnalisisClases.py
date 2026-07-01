"""
per_class_analysis.py

Evalúa mAP@0.5 por clase de YOLOv8n en 5 condiciones gamma sobre COCO128.
Produce un heatmap de sensibilidad por clase y un ranking de las más afectadas.

Salida en: results/per_class_analysis/
"""

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from ultralytics import YOLO
from ultralytics import settings as ultra_settings

# ── Configuración ──────────────────────────────────────────────────────────────

GAMMAS = [0.3, 0.6, 1.0, 1.4, 1.8]
MODEL_PATH = "yolov8n.pt"
IMAGE_SIZE = 640
DEVICE = "mps"  # cambiar a "cpu" si es necesario
TOP_N_CLASSES = 15  # clases más sensibles a mostrar en el line chart
OUTPUT_DIR = Path("results/per_class_analysis")

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


# ── Setup ──────────────────────────────────────────────────────────────────────

def ensure_labels_exist(gamma_dir: Path, original_labels_dir: Path) -> None:
    """
    Copia las etiquetas de COCO128 al directorio gamma si no existen.
    Las etiquetas son idénticas para todas las condiciones gamma.
    """
    target_labels = gamma_dir / "labels" / "train2017"
    if target_labels.exists():
        return

    print(f"  Copiando etiquetas a {gamma_dir.name}...")
    shutil.copytree(original_labels_dir, target_labels)


def build_gamma_yaml(gamma: float, datasets_dir: Path) -> Path:
    """Genera un YAML temporal apuntando al directorio de imágenes gamma.
    
    Para γ=1.0 usa el directorio original coco128/ (sin modificación gamma).
    """
    gamma_str = f"{gamma:.1f}"

    # γ=1.0 es el baseline: imágenes originales sin corrección gamma
    if gamma == 1.0:
        images_subpath = "coco128/images/train2017"
    else:
        gamma_dir = datasets_dir / f"coco128_gamma{gamma_str}"
        original_labels = datasets_dir / "coco128" / "labels" / "train2017"
        ensure_labels_exist(gamma_dir, original_labels)
        images_subpath = f"coco128_gamma{gamma_str}/images/train2017"

    yaml_data = {
        "path": str(datasets_dir),
        "train": images_subpath,
        "val": images_subpath,
        "names": {i: name for i, name in enumerate(COCO_NAMES)},
    }

    yaml_path = Path(f"/tmp/coco128_gamma{gamma_str}.yaml")
    yaml_path.write_text(yaml.dump(yaml_data, allow_unicode=True))
    return yaml_path


# ── Evaluación ─────────────────────────────────────────────────────────────────

def evaluate_per_class(model: YOLO, data_yaml: Path, gamma: float) -> dict[str, float]:
    """Corre val() y retorna {nombre_clase: mAP@0.5}."""
    print(f"  Evaluando γ={gamma}...")
    results = model.val(
        data=str(data_yaml),
        imgsz=IMAGE_SIZE,
        device=DEVICE,
        verbose=False,
        plots=False,
        save_json=False,
    )

    # results.box.maps: array con mAP@0.5 por clase (indexado por clase)
    # results.box.ap_class_index: índices de clases presentes en el dataset
    per_class_map = {}
    for pos, class_idx in enumerate(results.box.ap_class_index):
        class_name = COCO_NAMES[int(class_idx)]
        per_class_map[class_name] = float(results.box.maps[pos])

    return per_class_map


def build_results_dataframe(
    all_results: dict[float, dict[str, float]]
) -> pd.DataFrame:
    """
    Construye DataFrame con filas=clases, columnas=gammas.
    Solo incluye clases presentes en al menos una condición gamma.
    """
    df = pd.DataFrame(all_results).T  # (n_gammas, n_classes)
    df = df.T  # (n_classes, n_gammas)
    df.columns = GAMMAS

    # Filtrar clases con mAP=0 en todas las condiciones (ausentes en COCO128)
    df = df[df.max(axis=1) > 0]
    return df


# ── Visualización ──────────────────────────────────────────────────────────────

def plot_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    """Heatmap de mAP@0.5 por clase, ordenado de más a menos sensible."""
    sensitivity = df.max(axis=1) - df.min(axis=1)
    df_sorted = df.loc[sensitivity.sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(9, max(8, len(df_sorted) * 0.28)))
    im = ax.imshow(df_sorted.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(len(GAMMAS)))
    ax.set_xticklabels([f"γ={g}" for g in GAMMAS], fontsize=11)
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted.index, fontsize=8)
    ax.set_title(
        "mAP@0.5 por clase y condición gamma\n(ordenado por sensibilidad γ=0.3→γ=1.8)",
        fontsize=12,
        pad=12,
    )

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("mAP@0.5", fontsize=10)

    # Línea vertical resaltando γ=0.3 (el problemático)
    ax.axvline(x=-0.5, color="red", linewidth=3, clip_on=False)
    ax.text(-0.7, -0.8, "Sub-exp.", color="red", fontsize=8, ha="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Heatmap guardado: {output_path}")


def plot_top_sensitive(df: pd.DataFrame, top_n: int, output_path: Path) -> None:
    """Line chart de las N clases más sensibles a variación de gamma."""
    sensitivity = df.max(axis=1) - df.min(axis=1)
    top_classes = sensitivity.nlargest(top_n).index

    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.get_cmap("tab20")

    for i, class_name in enumerate(top_classes):
        ax.plot(
            GAMMAS,
            df.loc[class_name],
            marker="o",
            linewidth=1.8,
            markersize=5,
            label=class_name,
            color=cmap(i / top_n),
        )

    ax.axhline(0.5, color="red", linestyle="--", linewidth=1.2, label="Umbral (0.5)")
    ax.axvline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax.set_xlabel("Gamma (γ)", fontsize=12)
    ax.set_ylabel("mAP@0.5", fontsize=12)
    ax.set_title(f"Top {top_n} clases más sensibles a la variación de gamma", fontsize=12)
    ax.set_xticks(GAMMAS)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, framealpha=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Top {top_n} clases guardado: {output_path}")


def print_summary(df: pd.DataFrame) -> None:
    """Imprime tabla resumen de clases que colapsan bajo γ=0.3."""
    summary = pd.DataFrame({
        "mAP_baseline":   df[1.0].round(3),
        "mAP_gamma03":    df[0.3].round(3),
        "caida_abs":      (df[1.0] - df[0.3]).round(3),
        "caida_pct":      ((df[1.0] - df[0.3]) / df[1.0].replace(0, np.nan) * 100).round(1),
        "colapsa_en_03":  df[0.3] < 0.5,
    })
    summary_sorted = summary.sort_values("caida_abs", ascending=False)

    print("\n─── Top 20 clases más afectadas por sub-exposición (γ=0.3) ───")
    print(summary_sorted.head(20).to_string())

    colapsos = summary_sorted[summary_sorted["colapsa_en_03"]]
    print(f"\nClases que caen bajo umbral en γ=0.3: {len(colapsos)}/{len(summary_sorted)}")
    if not colapsos.empty:
        print(colapsos.index.tolist())

    return summary_sorted


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets_dir = Path(ultra_settings["datasets_dir"])

    print("Cargando modelo YOLOv8n...")
    model = YOLO(MODEL_PATH)

    # Evaluación por gamma
    all_results: dict[float, dict[str, float]] = {}
    for gamma in GAMMAS:
        data_yaml = build_gamma_yaml(gamma, datasets_dir)
        all_results[gamma] = evaluate_per_class(model, data_yaml, gamma)

    # Construir DataFrame
    df = build_results_dataframe(all_results)
    df.to_csv(OUTPUT_DIR / "per_class_map_by_gamma.csv")
    print(f"\nResultados crudos: {OUTPUT_DIR / 'per_class_map_by_gamma.csv'}")

    # Resumen
    summary = print_summary(df)
    summary.to_csv(OUTPUT_DIR / "per_class_summary.csv")

    # Visualizaciones
    print("\nGenerando figuras...")
    plot_heatmap(df, OUTPUT_DIR / "heatmap_per_class.png")
    plot_top_sensitive(df, TOP_N_CLASSES, OUTPUT_DIR / "top_sensitive_classes.png")

    print(f"\n✓ Análisis completo. Resultados en: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()