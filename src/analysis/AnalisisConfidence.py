"""
confidence_analysis.py

Analiza la distribución de confidence scores de YOLOv8n en 5 condiciones gamma.
Corre model.predict() sobre COCO128 y recolecta scores crudos de cada detección.

Hipótesis: antes de que mAP colapse, los confidence scores caen sistemáticamente,
revelando que el modelo es "cada vez más inseguro" a medida que baja la iluminación.

Salida en: results/confidence_analysis/
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from ultralytics import YOLO
from ultralytics import settings as ultra_settings

# ── Configuración ──────────────────────────────────────────────────────────────

GAMMAS = [0.3, 0.6, 1.0, 1.4, 1.8]
MODEL_PATH = "yolov8n.pt"
IMAGE_SIZE = 640
DEVICE = "mps"
CONF_THRESHOLD = 0.001   # muy bajo para capturar detecciones débiles
OUTPUT_DIR = Path("results/confidence_analysis")

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

PALETTE = {
    0.3: "#E53935",
    0.6: "#FF7043",
    1.0: "#1E88E5",
    1.4: "#43A047",
    1.8: "#8E24AA",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


# ── Recolección de datos ───────────────────────────────────────────────────────

def get_image_dir(gamma: float, datasets_dir: Path) -> Path:
    """Retorna el directorio de imágenes para cada gamma."""
    if gamma == 1.0:
        return datasets_dir / "coco128" / "images" / "train2017"
    return datasets_dir / f"coco128_gamma{gamma:.1f}" / "images" / "train2017"


def collect_confidence_scores(
    model: YOLO, image_dir: Path, gamma: float
) -> pd.DataFrame:
    """
    Corre predict() sobre todas las imágenes y recolecta scores crudos.
    Retorna DataFrame con columnas: gamma, image, class_id, class_name, confidence.
    """
    image_paths = sorted(image_dir.glob("*.jpg"))
    records = []

    results = model.predict(
        source=str(image_dir),
        imgsz=IMAGE_SIZE,
        device=DEVICE,
        conf=CONF_THRESHOLD,
        verbose=False,
        stream=True,   # evita cargar todo en memoria
    )

    for img_path, result in zip(image_paths, results):
        if result.boxes is None or len(result.boxes) == 0:
            continue
        for box in result.boxes:
            class_id = int(box.cls.item())
            records.append({
                "gamma":      gamma,
                "image":      img_path.name,
                "class_id":   class_id,
                "class_name": COCO_NAMES[class_id] if class_id < len(COCO_NAMES) else "unknown",
                "confidence": float(box.conf.item()),
            })

    return pd.DataFrame(records)


# ── Figuras ────────────────────────────────────────────────────────────────────

def plot_distribution_overlay(df: pd.DataFrame, output_path: Path) -> None:
    """
    Histograma superpuesto de confidence scores por gamma.
    Muestra cómo la distribución se desplaza hacia valores bajos en sub-exposición.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    bins = np.linspace(0, 1, 40)
    for gamma in GAMMAS:
        scores = df[df["gamma"] == gamma]["confidence"]
        ax.hist(scores, bins=bins, alpha=0.55, label=f"γ={gamma}",
                color=PALETTE[gamma], density=True, zorder=3)

    ax.axvline(0.25, color="#666", linestyle=":", linewidth=1.2,
               label="Conf. baja (<0.25)")
    ax.set_xlabel("Confidence score", fontsize=12)
    ax.set_ylabel("Densidad", fontsize=12)
    ax.set_title("Distribución de confidence scores por condición gamma\n"
                 "Sub-exposición desplaza la distribución hacia scores bajos", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25, zorder=0)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {output_path.name}")


def plot_stats_per_gamma(df: pd.DataFrame, output_path: Path) -> None:
    """
    Box plot de confidence scores por gamma con mediana y percentiles.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: boxplot
    ax = axes[0]
    data_by_gamma = [df[df["gamma"] == g]["confidence"].values for g in GAMMAS]
    bp = ax.boxplot(data_by_gamma, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2))
    for patch, gamma in zip(bp["boxes"], GAMMAS):
        patch.set_facecolor(PALETTE[gamma])
        patch.set_alpha(0.7)

    ax.set_xticklabels([f"γ={g}" for g in GAMMAS], fontsize=10)
    ax.set_ylabel("Confidence score", fontsize=11)
    ax.set_title("Distribución de scores\npor condición gamma", fontsize=11)
    ax.axhline(0.25, color="#E53935", linestyle="--", linewidth=1.2,
               alpha=0.7, label="Conf. baja (<0.25)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)

    # Panel 2: estadísticas resumidas (mediana, media, % scores bajos)
    ax2 = axes[1]
    stats = df.groupby("gamma")["confidence"].agg(
        mediana="median", media="mean",
        pct_baja=lambda x: (x < 0.25).mean() * 100,
        n_detecciones="count",
    ).reset_index()

    x = np.arange(len(GAMMAS))
    w = 0.28
    bars1 = ax2.bar(x - w, stats["mediana"], w, label="Mediana",
                    color=[PALETTE[g] for g in GAMMAS], alpha=0.9, zorder=3)
    bars2 = ax2.bar(x,     stats["media"],   w, label="Media",
                    color=[PALETTE[g] for g in GAMMAS], alpha=0.55, zorder=3)

    ax2b = ax2.twinx()
    ax2b.plot(x + w/2, stats["pct_baja"], "s--", color="#333",
              markersize=7, linewidth=1.5, label="% scores < 0.25")
    ax2b.set_ylabel("% detecciones con conf. < 0.25", fontsize=10, color="#333")
    ax2b.yaxis.set_major_formatter(mticker.PercentFormatter())

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"γ={g}" for g in GAMMAS], fontsize=10)
    ax2.set_ylabel("Confidence score", fontsize=11)
    ax2.set_title("Estadísticas de confidence\npor condición gamma", fontsize=11)
    ax2.grid(axis="y", alpha=0.25, zorder=0)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9)

    fig.suptitle("Análisis de Confidence Scores — YOLOv8n / COCO128", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {output_path.name}")


def plot_per_class_confidence_drop(df: pd.DataFrame, output_path: Path) -> None:
    """
    Top 15 clases con mayor caída de confidence score mediana (γ=1.0 → γ=0.3).
    """
    baseline = df[df["gamma"] == 1.0].groupby("class_name")["confidence"].median()
    subexp   = df[df["gamma"] == 0.3].groupby("class_name")["confidence"].median()

    # Clases presentes en ambas condiciones
    common = baseline.index.intersection(subexp.index)
    drop = (baseline[common] - subexp[common]).sort_values(ascending=False).head(15)
    classes = drop.index.tolist()[::-1]

    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(classes))
    colors = ["#E53935" if drop[c] > 0.15 else "#FF7043" if drop[c] > 0.07
              else "#FB8C00" for c in classes]

    bars = ax.barh(y, drop[classes], color=colors, zorder=3)
    ax.axvline(0.15, color="#999", linestyle=":", linewidth=1.2,
               label="Caída severa (>0.15)")

    for bar, cls in zip(bars, classes):
        val = drop[cls]
        ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
                f"−{val:.2f}", va="center", fontsize=8.5)

    ax.set_yticks(y)
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlabel("Caída en confidence score mediano (γ=1.0 → γ=0.3)", fontsize=11)
    ax.set_title("Top 15 clases con mayor caída de confidence score\n"
                 "en sub-exposición (γ=0.3 vs baseline γ=1.0)", fontsize=11)
    ax.grid(axis="x", alpha=0.25, zorder=0)
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {output_path.name}")


def plot_confidence_vs_gamma_top_classes(
    df: pd.DataFrame, summary_path: Path, output_path: Path
) -> None:
    """
    Evolución de confidence score mediano por gamma para las 8 clases
    más afectadas (según análisis por clase previo).
    """
    summary = pd.read_csv(summary_path, index_col=0)
    top8 = summary.nlargest(8, "caida_abs").index.tolist()

    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap("tab10")

    for i, cls in enumerate(top8):
        medians = []
        for gamma in GAMMAS:
            subset = df[(df["gamma"] == gamma) & (df["class_name"] == cls)]["confidence"]
            medians.append(subset.median() if len(subset) > 0 else np.nan)

        ax.plot(GAMMAS, medians, marker="o", linewidth=1.8, markersize=6,
                label=cls, color=cmap(i / 8))

    ax.axvline(1.0, color="#999", linestyle=":", linewidth=1, alpha=0.7)
    ax.set_xlabel("Gamma (γ)", fontsize=12)
    ax.set_ylabel("Confidence score mediano", fontsize=12)
    ax.set_title("Evolución del confidence score mediano\npor gamma — Top 8 clases más afectadas",
                 fontsize=11)
    ax.set_xticks(GAMMAS)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"  Guardado: {output_path.name}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets_dir = Path(ultra_settings["datasets_dir"])
    summary_path = Path("results/per_class_analysis/per_class_summary.csv")

    print("Cargando modelo YOLOv8n...")
    model = YOLO(MODEL_PATH)

    # Recolectar scores de todas las condiciones gamma
    all_frames = []
    for gamma in GAMMAS:
        image_dir = get_image_dir(gamma, datasets_dir)
        print(f"  Recolectando scores γ={gamma} ({len(list(image_dir.glob('*.jpg')))} imágenes)...")
        frame = collect_confidence_scores(model, image_dir, gamma)
        all_frames.append(frame)
        print(f"    → {len(frame)} detecciones registradas")

    df = pd.concat(all_frames, ignore_index=True)
    df.to_csv(OUTPUT_DIR / "confidence_scores_raw.csv", index=False)
    print(f"\nScores crudos: {OUTPUT_DIR / 'confidence_scores_raw.csv'}")

    # Estadísticas rápidas
    print("\n── Estadísticas por gamma ──")
    print(df.groupby("gamma")["confidence"].agg(
        n="count", mediana="median", media="mean",
        pct_baja=lambda x: f"{(x < 0.25).mean()*100:.1f}%"
    ).to_string())

    # Figuras
    print("\nGenerando figuras...")
    plot_distribution_overlay(df,   OUTPUT_DIR / "conf_distribucion.png")
    plot_stats_per_gamma(df,        OUTPUT_DIR / "conf_estadisticas.png")
    plot_per_class_confidence_drop(df, OUTPUT_DIR / "conf_caida_por_clase.png")
    plot_confidence_vs_gamma_top_classes(df, summary_path,
                                         OUTPUT_DIR / "conf_evolucion_top8.png")

    print(f"\n✓ Análisis completo. Resultados en: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()