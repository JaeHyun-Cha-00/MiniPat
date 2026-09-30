# scores.json (from score.py) -> comparison figures for the README.
# Runs in the scoring environment (requirements/score.txt).
#
#   .venv-score/bin/python evaluation/plot.py [--scores evaluation/results/scores.json]
#
#   size_vs_quality.png   model size vs. quality, base and fine-tuned, one panel per direction;
#                         API systems drawn as reference lines (they have no size to plot)
#   cost_vs_quality.png   USD per 1,000 translations vs. quality, every system with a price
#
# Quality is COMET when score.py computed it, otherwise chrF++.

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")  # file output only; no display needed on a server
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training"))
from config import load_config, model_config_path  # noqa: E402

# Colors from a colorblind-checked categorical palette (slots 1-3, validated all-pairs).
# Marker shape repeats the category, so color never carries identity alone.
STYLE = {
    "finetuned": {"color": "#2a78d6", "marker": "o", "fill": True, "label": "Fine-tuned (LoRA)"},
    "base": {"color": "#eb6834", "marker": "o", "fill": False, "label": "Base (before fine-tuning)"},
    "api": {"color": "#1baf7a", "marker": "D", "fill": True, "label": "Commercial API"},
}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
API_DASHES = [(0, (5, 3)), (0, (1.5, 2)), (0, (7, 2, 1.5, 2))]  # API reference lines, told apart in the legend
DIRECTIONS = {"ko-en": "Korean → English", "en-ko": "English → Korean"}


def parse_system(system):
    """'qwen35_2b-finetuned' -> ('local', 'qwen35_2b', 'finetuned'); API systems -> ('api', name, None)."""
    for kind in ("finetuned", "base"):
        if system.endswith(f"-{kind}"):
            run_name = system[: -len(kind) - 1]
            if os.path.exists(model_config_path(run_name)):
                return "local", run_name, kind
    return "api", system, None


def quality_metric(systems):
    any_q = next(iter(systems.values()))["quality"]["all"]
    return ("comet", "COMET") if "comet" in any_q else ("chrf++", "chrF++")


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)


def point(ax, x, y, kind, **kw):
    s = STYLE[kind]
    ax.plot(x, y, linestyle="none", marker=s["marker"], markersize=8, markeredgewidth=2,
            color=s["color"], markerfacecolor=s["color"] if s["fill"] else SURFACE,
            markeredgecolor=s["color"], zorder=3, **kw)


# Label spots tried in order around a point: (dx, dy) in points, horizontal and vertical alignment.
LABEL_SPOTS = [(7, 3, "left", "bottom"), (7, -3, "left", "top"), (-7, 3, "right", "bottom"),
               (-7, -3, "right", "top"), (0, 8, "center", "bottom"), (0, -8, "center", "top")]


def place_labels(ax, items, fontsize=8):
    """Annotate (x, y, text) points so no label covers another label or any point.

    Each label takes the first free spot in LABEL_SPOTS; if all six collide it keeps the first,
    which only happens when points sit almost on top of each other.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Every marker is an obstacle: a ~10 px box around its center.
    obstacles = []
    for x, y, _ in items:
        px, py = ax.transData.transform((x, y))
        obstacles.append(matplotlib.transforms.Bbox([[px - 6, py - 6], [px + 6, py + 6]]))
    for x, y, text in sorted(items, key=lambda t: -t[1]):  # top to bottom
        for i, (dx, dy, ha, va) in enumerate(LABEL_SPOTS):
            ann = ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy), ha=ha, va=va,
                              fontsize=fontsize, color=INK_2, zorder=4,
                              # surface-colored backing so reference lines don't strike through the text
                              bbox={"boxstyle": "round,pad=0.15", "fc": SURFACE, "ec": "none", "alpha": 0.85})
            box = ann.get_window_extent(renderer).expanded(1.02, 1.1)
            if not any(box.overlaps(b) for b in obstacles) or i == len(LABEL_SPOTS) - 1:
                break
            ann.remove()
        if any(box.overlaps(b) for b in obstacles):  # all spots taken: fall back to the first one
            ann.remove()
            dx, dy, ha, va = LABEL_SPOTS[0]
            ann = ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy), ha=ha, va=va,
                              fontsize=fontsize, color=INK_2, zorder=4)
            box = ann.get_window_extent(renderer)
        obstacles.append(box)


def size_vs_quality(systems, key, label, out_path):
    local, api = {}, {}
    for name, r in systems.items():
        kind, run_name, variant = parse_system(name)
        if kind == "local":
            local.setdefault(run_name, {})[variant] = r
        else:
            api[name] = r
    if not local:
        print("size_vs_quality: no local-model scores, skipped")
        return

    sizes = {run: load_config(model_config_path(run))["params_b"] for run in local}
    # Models of equal size (e.g. qwen35_2b, gemma4_e2b) get a small log-scale offset so they don't overlap.
    xpos, by_size = {}, {}
    for run in sorted(local):
        by_size.setdefault(sizes[run], []).append(run)
    for size, runs in by_size.items():
        for i, run in enumerate(runs):
            xpos[run] = size * 1.07 ** (i - (len(runs) - 1) / 2)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True, facecolor=SURFACE)
    for ax, (direction, title) in zip(axes, DIRECTIONS.items()):
        style_axes(ax)
        labels = []
        for run, variants in sorted(local.items(), key=lambda kv: sizes[kv[0]]):
            x = xpos[run]
            ys = {v: r["quality"][direction][key] for v, r in variants.items() if direction in r["quality"]}
            if len(ys) == 2:  # thin connector shows the fine-tuning gain for this model
                ax.plot([x, x], [ys["base"], ys["finetuned"]], color=GRID, linewidth=2, zorder=2)
            for variant, y in ys.items():
                point(ax, x, y, variant)
            if "finetuned" in ys:
                labels.append((x, ys["finetuned"], run))
        for i, (name, r) in enumerate(sorted(api.items())):  # reference lines: an API has no parameter count
            ax.axhline(r["quality"][direction][key], color=INK_2, linewidth=1, linestyle=API_DASHES[i % 3], zorder=1)
        ax.set_xscale("log")
        ax.margins(x=0.12)
        ax.set_xticks(sorted(set(sizes.values())))
        ax.set_xticklabels([f"{s:g}B" for s in sorted(set(sizes.values()))])
        ax.minorticks_off()
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.set_xlabel("Model size (parameters, log scale)", fontsize=9, color=INK_2)
        ax.label_outer()
        place_labels(ax, labels)
    axes[0].set_ylabel(label, fontsize=9, color=INK_2)

    handles = [plt.Line2D([], [], linestyle="none", marker="o", markersize=8, markeredgewidth=2,
                          color=STYLE[k]["color"], markerfacecolor=STYLE[k]["color"] if STYLE[k]["fill"] else SURFACE,
                          label=STYLE[k]["label"]) for k in ("finetuned", "base")]
    handles += [plt.Line2D([], [], color=INK_2, linewidth=1, linestyle=API_DASHES[i % 3], label=name)
                for i, name in enumerate(sorted(api))]
    fig.legend(handles=handles, loc="upper left", ncol=len(handles), frameon=False, fontsize=9,
               labelcolor=INK, bbox_to_anchor=(0.01, 1.0))
    fig.suptitle(f"Translation quality by model size ({label}, test split)", x=0.01, y=1.07,
                 ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out_path}")


def cost_vs_quality(systems, key, label, out_path):
    rows = [(name, r) for name, r in systems.items() if r["cost_usd_per_1k"] is not None]
    if not rows:
        print("cost_vs_quality: no system has a price in evaluation/config.yaml, skipped")
        return
    fig, ax = plt.subplots(figsize=(7.5, 5), facecolor=SURFACE)
    style_axes(ax)
    seen, labels = set(), []
    for name, r in rows:
        kind, _, variant = parse_system(name)
        k = variant if kind == "local" else "api"
        x, y = r["cost_usd_per_1k"], r["quality"]["all"][key]
        point(ax, x, y, k)
        seen.add(k)
        labels.append((x, y, parse_system(name)[1]))  # model name only; color + shape give base/fine-tuned
    ax.set_xscale("log")
    ax.margins(x=0.15)
    ax.set_xlabel("Cost per 1,000 translations (USD, log scale)", fontsize=9, color=INK_2)
    ax.set_ylabel(f"{label} (both directions)", fontsize=9, color=INK_2)
    handles = [plt.Line2D([], [], linestyle="none", marker=STYLE[k]["marker"], markersize=8, markeredgewidth=2,
                          color=STYLE[k]["color"], markerfacecolor=STYLE[k]["color"] if STYLE[k]["fill"] else SURFACE,
                          label=STYLE[k]["label"]) for k in ("finetuned", "base", "api") if k in seen]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9, labelcolor=INK)
    ax.set_title(f"Cost vs. quality ({label}, test split)", fontsize=13, color=INK, loc="left")
    fig.tight_layout()
    # Every point is labeled (a palette slot is below 3:1 contrast, so labels carry identity too).
    place_labels(ax, labels)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="evaluation/results/scores.json")
    ap.add_argument("--out-dir", default="evaluation/results/figures")
    args = ap.parse_args()

    with open(args.scores) as f:
        systems = json.load(f)["systems"]
    key, label = quality_metric(systems)
    os.makedirs(args.out_dir, exist_ok=True)
    size_vs_quality(systems, key, label, os.path.join(args.out_dir, "size_vs_quality.png"))
    cost_vs_quality(systems, key, label, os.path.join(args.out_dir, "cost_vs_quality.png"))


if __name__ == "__main__":
    main()
