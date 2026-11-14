"""Render documentation plots from profile_wheat_data.py's aggregate JSON."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator, PercentFormatter, StrMethodFormatter

BACKGROUND = "#f8fafc"
FOREGROUND = "#182638"
GRID = "#d9e2eb"
TRAIN = "#3159a9"
VALIDATION = "#a85429"
FONT_CANDIDATES = ("Hiragino Sans", "Hiragino Kaku Gothic ProN", "Noto Sans CJK JP", "Noto Sans JP",
                   "Yu Gothic", "IPAGothic")


def sources(profile):
    for partition, label, color in (("train", "学習", TRAIN), ("validation", "検証", VALIDATION)):
        group = profile["partitions"][partition]
        for source, stats in group["sources"].items():
            yield label, source, color, stats


def make_axis(title, labels, xlabel, height):
    fig, ax = plt.subplots(figsize=(8.6, height), layout="constrained")
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    ax.set_title(title, loc="left", pad=14, color=FOREGROUND, fontweight="bold", fontsize=16)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, color=FOREGROUND)
    ax.set_axisbelow(True)
    ax.xaxis.grid(color=GRID)
    ax.tick_params(axis="both", colors=FOREGROUND, length=0, pad=7)
    ax.spines[:].set_visible(False)
    return fig, ax


def save(fig, path):
    fig.savefig(path, dpi=155, facecolor=BACKGROUND, pad_inches=0.18, bbox_inches="tight")
    plt.close(fig)


def plot_sources(profile, output):
    rows = list(sources(profile))
    labels = [f"{label}  {source}" for label, source, _, _ in rows]
    counts = [stats["images"] for _, _, _, stats in rows]
    fig, ax = make_axis("撮影元ごとの画像枚数", labels, "画像枚数", 3.0)
    bars = ax.barh(range(len(rows)), counts, height=0.55, color=[color for _, _, color, _ in rows])
    ax.bar_label(bars, labels=[f"{count:,}" for count in counts], padding=6, color=FOREGROUND)
    ax.set_xlim(0, max(counts) * 1.16)
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    save(fig, output / "wheat-data-sources.png")


def plot_distribution(profile, output, field, total, title, xlabel, bins, colors, name):
    rows = [("学習 全体", profile["partitions"]["train"]["all"]),
            ("検証 全体", profile["partitions"]["validation"]["all"])]
    rows.extend((f"{label}  {source}", stats) for label, source, _, stats in sources(profile))
    labels = [label for label, _ in rows]
    fig, ax = make_axis(title, labels, xlabel, 3.35)
    left = [0.0] * len(rows)
    for index, (label, color) in enumerate(zip(bins, colors, strict=True)):
        width = [stats[field][index] / stats[total] for _, stats in rows]
        ax.barh(range(len(rows)), width, left=left, height=0.55, color=color, label=label)
        left = [offset + value for offset, value in zip(left, width)]
    ax.set_xlim(0, 1)
    ax.axhline(1.5, color=GRID, linewidth=1)
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.26), ncols=3, frameon=False,
              labelcolor=FOREGROUND, borderaxespad=0, fontsize=10)
    save(fig, output / f"wheat-data-{name}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    font = next((name for name in FONT_CANDIDATES if name in font_manager.get_font_names()), None)
    if font is None:
        parser.error("A Japanese sans-serif font is required, such as Noto Sans CJK JP.")
    plt.rcParams.update({"font.family": font, "font.size": 10.5})
    profile = json.loads(args.profile.read_text())
    if profile["count_edges"] != [10, 25, 50, 75, 100] or profile["area_edges"] != [0.002, 0.01]:
        raise ValueError("Profile bin boundaries do not match the chart labels.")
    args.output.mkdir(parents=True, exist_ok=True)
    plot_sources(profile, args.output)
    plot_distribution(profile, args.output, "count_bins", "images", "1枚あたりの穂数の分布", "画像の割合",
                      ("1–9本", "10–24本", "25–49本", "50–74本", "75–99本", "100本以上"),
                      ("#b75824", "#e5a263", "#adc6e8", "#628fc8", "#345895", "#203964"), "head-counts")
    plot_distribution(profile, args.output, "area_bins", "heads", "正解枠の面積の分布", "注釈の割合",
                      ("0.2%未満", "0.2%以上・1%未満", "1%以上"),
                      ("#b75824", "#8fb3dd", "#345895"), "box-areas")


if __name__ == "__main__":
    main()
