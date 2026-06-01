from __future__ import annotations

import csv
import html
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PHASE1 = ROOT / "srce_outputs" / "phase1_linear_probing_enumerability"
OUT_DIR = ROOT / "analysis_outputs" / "layer_disconnect"
OUT_DIR.mkdir(parents=True, exist_ok=True)


RUNS = [
    {
        "name": "Mistral-7B-Instruct zero-shot",
        "phase1_key": "mistral_7b_it",
        "phase2": ROOT
        / "srce_outputs"
        / "mistral_7b_it_chatprefill"
        / "mistral_7b_it"
        / "direction_selection_scores.csv",
    },
    {
        "name": "Mistral-7B-Instruct few-shot",
        "phase1_key": "mistral_7b_it",
        "phase2": ROOT / "srce_outputs" / "mistral_7b_it_fewshot" / "direction_selection_scores.csv",
    },
    {
        "name": "Llama-3.1-8B base",
        "phase1_key": "llama31_8b_base",
        "phase2": ROOT / "srce_outputs" / "llama31_8b_base" / "direction_selection_scores.csv",
    },
    {
        "name": "Llama-3.1-8B base few-shot",
        "phase1_key": "llama31_8b_base",
        "phase2": ROOT / "srce_outputs" / "llama31_8b_base_fewshot" / "direction_selection_scores.csv",
    },
    {
        "name": "Llama-3.1-8B-Instruct",
        "phase1_key": "llama31_8b_it",
        "phase2": ROOT / "srce_outputs" / "llama31_8b_it_chatprefill_wide" / "direction_selection_scores.csv",
    },
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_filename(name: str) -> str:
    keep = []
    for ch in name.lower():
        if ch.isalnum():
            keep.append(ch)
        elif ch in {" ", "-", "_", "."}:
            keep.append("_")
    return "".join(keep).strip("_")


def load_run(run: dict) -> list[dict] | None:
    probe_path = PHASE1 / run["phase1_key"] / "layer_probe_results.csv"
    steering_path = run["phase2"]
    if not probe_path.exists() or not steering_path.exists():
        print(f"Skipping {run['name']}: missing {probe_path if not probe_path.exists() else steering_path}")
        return None

    probe = {int(row["layer"]): row for row in read_csv(probe_path)}
    steering_best: dict[int, dict] = {}
    for row in read_csv(steering_path):
        layer = int(float(row["layer"]))
        auc = float(row["val_auc_under_addition"])
        if layer not in steering_best or auc > float(steering_best[layer]["val_auc_under_addition"]):
            steering_best[layer] = row

    merged = []
    for layer in sorted(set(probe) & set(steering_best)):
        p = probe[layer]
        s = steering_best[layer]
        merged.append(
            {
                "run": run["name"],
                "layer": layer,
                "roc_auc": float(p["roc_auc"]),
                "accuracy": float(p["accuracy"]),
                "macro_f1": float(p["macro_f1"]),
                "position": int(float(s["position"])),
                "steering_score": float(s["steering_score"]),
                "val_auc_under_addition": float(s["val_auc_under_addition"]),
            }
        )
    return merged


def points(rows: list[dict], key: str, width: int, height: int, margin: int, y_min: float, y_max: float) -> str:
    layers = [row["layer"] for row in rows]
    x_min, x_max = min(layers), max(layers)
    coords = []
    for row in rows:
        x = margin + (row["layer"] - x_min) / max(1, x_max - x_min) * (width - 2 * margin)
        y = height - margin - (row[key] - y_min) / (y_max - y_min) * (height - 2 * margin)
        coords.append(f"{x:.1f},{y:.1f}")
    return " ".join(coords)


def render_svg(rows: list[dict], run_name: str, path: Path) -> None:
    width, height, margin = 920, 500, 70
    y_min, y_max = 0.45, 0.86
    layers = [row["layer"] for row in rows]
    x_min, x_max = min(layers), max(layers)
    best_probe = max(rows, key=lambda row: row["roc_auc"])
    best_steer = max(rows, key=lambda row: row["val_auc_under_addition"])

    def x_for(layer: int) -> float:
        return margin + (layer - x_min) / max(1, x_max - x_min) * (width - 2 * margin)

    def y_for(value: float) -> float:
        return height - margin - (value - y_min) / (y_max - y_min) * (height - 2 * margin)

    grid = []
    for tick in [0.5, 0.6, 0.7, 0.8]:
        y = y_for(tick)
        grid.append(
            f'<line x1="{margin}" y1="{y:.1f}" x2="{width - margin}" y2="{y:.1f}" '
            'stroke="#d9d9d9" stroke-width="1"/>'
        )
        grid.append(f'<text x="{margin - 12}" y="{y + 4:.1f}" text-anchor="end" font-size="13">{tick:.1f}</text>')

    x_ticks = []
    step = max(1, round((x_max - x_min) / 6))
    for layer in range(x_min, x_max + 1, step):
        x = x_for(layer)
        x_ticks.append(
            f'<line x1="{x:.1f}" y1="{height - margin}" x2="{x:.1f}" y2="{height - margin + 6}" '
            'stroke="#444" stroke-width="1"/>'
        )
        x_ticks.append(f'<text x="{x:.1f}" y="{height - margin + 24}" text-anchor="middle" font-size="13">{layer}</text>')

    probe_line = points(rows, "roc_auc", width, height, margin, y_min, y_max)
    steering_line = points(rows, "val_auc_under_addition", width, height, margin, y_min, y_max)
    probe_x = x_for(best_probe["layer"])
    steer_x = x_for(best_steer["layer"])

    title = html.escape(run_name)
    note = (
        f"best probe layer={best_probe['layer']}, AUC={best_probe['roc_auc']:.3f}; "
        f"best steering layer={best_steer['layer']}, AUC={best_steer['val_auc_under_addition']:.3f}, "
        f"pos={best_steer['position']}"
    )
    note = html.escape(note)

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2}" y="34" text-anchor="middle" font-family="Arial, sans-serif" font-size="22" font-weight="700">{title}</text>
<g font-family="Arial, sans-serif" fill="#222">
{''.join(grid)}
<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#222" stroke-width="1.5"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#222" stroke-width="1.5"/>
{''.join(x_ticks)}
<text x="{width / 2}" y="{height - 18}" text-anchor="middle" font-size="15">Layer</text>
<text x="22" y="{height / 2}" text-anchor="middle" font-size="15" transform="rotate(-90 22 {height / 2})">AUC</text>
<line x1="{probe_x:.1f}" y1="{margin}" x2="{probe_x:.1f}" y2="{height - margin}" stroke="#6c63ff" stroke-width="1.3" stroke-dasharray="5 5"/>
<line x1="{steer_x:.1f}" y1="{margin}" x2="{steer_x:.1f}" y2="{height - margin}" stroke="#00a884" stroke-width="1.3" stroke-dasharray="5 5"/>
<polyline points="{probe_line}" fill="none" stroke="#6c63ff" stroke-width="3"/>
<polyline points="{steering_line}" fill="none" stroke="#00a884" stroke-width="3"/>
<g>
  <rect x="{width - 330}" y="62" width="260" height="58" rx="5" fill="#ffffff" stroke="#cccccc"/>
  <line x1="{width - 310}" y1="82" x2="{width - 270}" y2="82" stroke="#6c63ff" stroke-width="3"/>
  <text x="{width - 260}" y="87" font-size="14">Linear probe ROC-AUC</text>
  <line x1="{width - 310}" y1="106" x2="{width - 270}" y2="106" stroke="#00a884" stroke-width="3"/>
  <text x="{width - 260}" y="111" font-size="14">Direction-selection AUC</text>
</g>
<rect x="{margin + 8}" y="{height - margin - 48}" width="{width - 2 * margin - 16}" height="32" rx="5" fill="#ffffff" stroke="#cccccc" opacity="0.92"/>
<text x="{margin + 18}" y="{height - margin - 27}" font-size="13">{note}</text>
</g>
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def main() -> None:
    all_rows: list[dict] = []
    summary_rows: list[dict] = []
    for run in RUNS:
        rows = load_run(run)
        if not rows:
            continue
        render_svg(rows, run["name"], OUT_DIR / f"{safe_filename(run['name'])}.svg")
        all_rows.extend(rows)
        best_probe = max(rows, key=lambda row: row["roc_auc"])
        best_steer = max(rows, key=lambda row: row["val_auc_under_addition"])
        summary_rows.append(
            {
                "run": run["name"],
                "best_probe_layer": best_probe["layer"],
                "best_probe_roc_auc": f"{best_probe['roc_auc']:.6f}",
                "best_steering_layer": best_steer["layer"],
                "best_steering_val_auc": f"{best_steer['val_auc_under_addition']:.6f}",
                "best_steering_position": best_steer["position"],
                "layer_gap": best_steer["layer"] - best_probe["layer"],
            }
        )

    if all_rows:
        write_csv(
            OUT_DIR / "probe_vs_steering_by_layer.csv",
            all_rows,
            ["run", "layer", "roc_auc", "accuracy", "macro_f1", "position", "steering_score", "val_auc_under_addition"],
        )
    write_csv(
        OUT_DIR / "probe_vs_steering_summary.csv",
        summary_rows,
        [
            "run",
            "best_probe_layer",
            "best_probe_roc_auc",
            "best_steering_layer",
            "best_steering_val_auc",
            "best_steering_position",
            "layer_gap",
        ],
    )

    for row in summary_rows:
        print(row)
    print(f"Wrote plots and CSVs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
