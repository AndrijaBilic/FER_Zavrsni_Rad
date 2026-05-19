import csv
import math
import random
import statistics
import textwrap
from collections import defaultdict
from pathlib import Path


RESULTS_DIR = Path("gemma_judge")
OUT_DIR = Path("analysis_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {"curated": "#2563eb", "webq": "#dc2626", "ood": "#16a34a"}


def parse_bool(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def parse_float(value, default=float("nan")) -> float:
    try:
        if value == "":
            return default
        return float(value)
    except Exception:
        return default


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def bootstrap_mean_ci(values: list[float], n_boot: int = 5000, seed: int = 42) -> tuple[float, float]:
    values = [v for v in values if not math.isnan(v)]
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sample = [rng.choice(values) for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_summary(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["eval_name"], parse_float(row["alpha"]))].append(row)

    summary = []
    for (eval_name, alpha), part in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        n = len(part)
        enum = [parse_bool(row["judge_is_enumerated"]) for row in part]
        valid = [parse_bool(row["judge_valid_answer"]) for row in part]
        counts = [parse_float(row["judge_answer_count"]) for row in part]
        clean_counts = [c for c in counts if not math.isnan(c)]

        enum_low, enum_high = wilson_ci(sum(enum), n)
        valid_low, valid_high = wilson_ci(sum(valid), n)
        count_low, count_high = bootstrap_mean_ci(clean_counts)
        mean_count = sum(clean_counts) / len(clean_counts) if clean_counts else float("nan")
        std_count = statistics.stdev(clean_counts) if len(clean_counts) >= 2 else 0.0

        summary.append(
            {
                "eval_name": eval_name,
                "alpha": alpha,
                "n": n,
                "judge_enumeration_rate": sum(enum) / n,
                "judge_enumeration_ci_low": enum_low,
                "judge_enumeration_ci_high": enum_high,
                "judge_valid_rate": sum(valid) / n,
                "judge_valid_ci_low": valid_low,
                "judge_valid_ci_high": valid_high,
                "mean_judge_answer_count": mean_count,
                "mean_count_ci_low": count_low,
                "mean_count_ci_high": count_high,
                "std_judge_answer_count": std_count,
            }
        )
    return summary


def svg_line_plot(summary: list[dict], metric: str, low: str, high: str, ylabel: str, path: Path, y_max=1.0):
    width, height = 780, 460
    left, right, top, bottom = 76, 28, 30, 64
    plot_w = width - left - right
    plot_h = height - top - bottom
    alphas = sorted({float(row["alpha"]) for row in summary})
    x_min, x_max = min(alphas), max(alphas)

    def x_scale(x):
        return left + (float(x) - x_min) / (x_max - x_min) * plot_w

    def y_scale(y):
        return top + (1 - float(y) / y_max) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827"/>',
        f'<text x="{width / 2}" y="{height - 18}" text-anchor="middle" font-family="Arial" font-size="14">Steering coefficient alpha</text>',
        f'<text x="18" y="{height / 2}" text-anchor="middle" font-family="Arial" font-size="14" transform="rotate(-90 18 {height / 2})">{ylabel}</text>',
    ]

    for tick in alphas:
        x = x_scale(tick)
        parts.append(f'<line x1="{x:.1f}" y1="{top + plot_h}" x2="{x:.1f}" y2="{top + plot_h + 5}" stroke="#111827"/>')
        parts.append(f'<text x="{x:.1f}" y="{top + plot_h + 22}" text-anchor="middle" font-family="Arial" font-size="12">{tick:g}</text>')

    y_ticks = [0, y_max / 4, y_max / 2, 3 * y_max / 4, y_max]
    for tick in y_ticks:
        y = y_scale(tick)
        parts.append(f'<line x1="{left - 5}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#111827"/>')
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left - 9}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{tick:.2g}</text>')

    zero_x = x_scale(0)
    parts.append(f'<line x1="{zero_x:.1f}" y1="{top}" x2="{zero_x:.1f}" y2="{top + plot_h}" stroke="#6b7280" stroke-dasharray="4 4"/>')

    for eval_name in sorted({row["eval_name"] for row in summary}):
        color = COLORS.get(eval_name, "#111827")
        series = [row for row in summary if row["eval_name"] == eval_name]
        series.sort(key=lambda row: row["alpha"])
        points = []
        for row in series:
            x = x_scale(row["alpha"])
            y = y_scale(row[metric])
            y_low = y_scale(row[low])
            y_high = y_scale(row[high])
            points.append(f"{x:.1f},{y:.1f}")
            parts.append(f'<line x1="{x:.1f}" y1="{y_low:.1f}" x2="{x:.1f}" y2="{y_high:.1f}" stroke="{color}" stroke-width="1.5"/>')
            parts.append(f'<line x1="{x - 4:.1f}" y1="{y_low:.1f}" x2="{x + 4:.1f}" y2="{y_low:.1f}" stroke="{color}" stroke-width="1.5"/>')
            parts.append(f'<line x1="{x - 4:.1f}" y1="{y_high:.1f}" x2="{x + 4:.1f}" y2="{y_high:.1f}" stroke="{color}" stroke-width="1.5"/>')
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
        parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.4"/>')

    legend_x, legend_y = width - 150, top + 8
    for i, eval_name in enumerate(sorted({row["eval_name"] for row in summary})):
        y = legend_y + i * 22
        color = COLORS.get(eval_name, "#111827")
        parts.append(f'<circle cx="{legend_x}" cy="{y}" r="5" fill="{color}"/>')
        parts.append(f'<text x="{legend_x + 12}" y="{y + 4}" font-family="Arial" font-size="13">{eval_name}</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def compact(text: str, width: int = 190) -> str:
    return textwrap.shorten(str(text).replace("\n", " ").strip(), width=width, placeholder=" ...")


def write_examples(rows: list[dict], path: Path):
    by_alpha = defaultdict(list)
    for row in rows:
        by_alpha[parse_float(row["alpha"])].append(row)

    lines = ["# Steering Examples by Alpha", ""]
    for alpha in sorted(by_alpha):
        lines.append(f"## alpha = {alpha:g}")
        for eval_name in ["curated", "webq", "ood"]:
            candidates = [row for row in by_alpha[alpha] if row["eval_name"] == eval_name]
            if not candidates:
                continue
            valid = [row for row in candidates if parse_bool(row["judge_valid_answer"])]
            row = (valid or candidates)[0]
            lines.extend(
                [
                    f"**{eval_name}**",
                    "",
                    f"- Question: {row['question']}",
                    f"- Gold label: {row.get('gold_label_str', '')}",
                    f"- Response: {compact(row['response'])}",
                    f"- Judge count: {row['judge_answer_count']}",
                    f"- Judge enumerated: {row['judge_is_enumerated']}",
                    f"- Judge valid: {row['judge_valid_answer']}",
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    rows = load_rows(RESULTS_DIR / "judged_generations.csv")
    summary = build_summary(rows)
    write_csv(OUT_DIR / "judge_summary_with_95ci.csv", summary)

    svg_line_plot(
        summary,
        "judge_enumeration_rate",
        "judge_enumeration_ci_low",
        "judge_enumeration_ci_high",
        "Judged enumeration rate",
        OUT_DIR / "judge_enumeration_rate_95ci.svg",
        y_max=1.0,
    )
    svg_line_plot(
        summary,
        "judge_valid_rate",
        "judge_valid_ci_low",
        "judge_valid_ci_high",
        "Judged valid-answer rate",
        OUT_DIR / "judge_valid_rate_95ci.svg",
        y_max=1.0,
    )
    max_count = max(float(row["mean_count_ci_high"]) for row in summary)
    svg_line_plot(
        summary,
        "mean_judge_answer_count",
        "mean_count_ci_low",
        "mean_count_ci_high",
        "Mean judged answer count",
        OUT_DIR / "judge_mean_answer_count_95ci.svg",
        y_max=math.ceil(max_count),
    )
    write_examples(rows, OUT_DIR / "examples_by_alpha.md")

    example_rows = []
    seen = set()
    for row in rows:
        key = (row["eval_name"], row["alpha"])
        if key not in seen:
            example_rows.append(row)
            seen.add(key)
    write_csv(OUT_DIR / "examples_by_alpha.csv", example_rows)
    print("Wrote report assets to", OUT_DIR)


if __name__ == "__main__":
    main()
