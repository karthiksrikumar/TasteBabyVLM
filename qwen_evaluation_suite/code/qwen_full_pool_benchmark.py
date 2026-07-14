#!/usr/bin/env python3
"""
Benchmark 1 — full clip pool + inter-annotator agreement science.

Runs Qwen3-VL-30B on ALL 506 taste clips (reusing the 101 already annotated
for annotator 5's benchmark), then analyzes Qwen as if it were a 16th
annotator alongside the 11 collected human annotators:

  * Pairwise percent agreement + Cohen's kappa per field, human<->human vs
    Qwen<->human. The scientific question: does Qwen sit inside the human
    agreement envelope?
  * Accuracy against majority-vote consensus labels (clips with >=2 humans).
  * Difficulty transfer: is Qwen more wrong on clips humans disagreed on?
  * Noun semantic agreement (MiniLM cosine): human-human vs Qwen-human.
  * Agreement heatmap with Qwen embedded in the annotator matrix.

Usage (GPU session):
    source /usr4/spclpgm/srikumar/envs/qwen/bin/activate
    python3 qwen_full_pool_benchmark.py all | run | analyze
"""

import glob
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen_taste_benchmark as base

HOME = base.HOME
LIMIT = int(os.environ.get("QWEN_POOL_LIMIT", "0"))
SHARD_GLOB = os.path.join(HOME, "qwen_full_pool_shard*.jsonl")
OLD_SHARD_GLOB = os.path.join(HOME, "qwen_taste_predictions_shard*.jsonl")
PREDICTIONS_FILE = os.path.join(HOME, "qwen_full_pool_predictions.json")
HUMAN_GLOB = os.path.join(HOME, "annotationAnalysis", "unzipped", "**", "annotationTaste_*.json")
RESULTS_DIR = os.path.join(HOME, "qwen_full_pool_results")

FIELDS = ["scenario", "verb", "additional_condition", "flavor_profile", "noun"]
CONDITIONAL_FIELDS = {"verb", "additional_condition", "flavor_profile", "noun"}

ALL_CLIPS = list(base.APP.CLIP_FILENAMES)


# ---------------------------------------------------------------------------
# Run — same prompt/model as the annotator-5 benchmark, over the whole pool
# ---------------------------------------------------------------------------

def previously_annotated():
    """Raw rows from the annotator-5 benchmark run (same prompt), by file."""
    rows = {}
    for path in sorted(glob.glob(OLD_SHARD_GLOB)):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    rows[r["file_name"]] = r
    return rows


def run_shard(shard, num_shards):
    tag = f"[pool shard {shard}]"
    old = previously_annotated()
    todo = [f for i, f in enumerate(ALL_CLIPS) if i % num_shards == shard]
    if LIMIT:
        todo = todo[:LIMIT]

    out_path = os.path.join(HOME, f"qwen_full_pool_shard{shard}.jsonl")
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(l)["file_name"] for l in f if l.strip()}
    remaining = [f for f in todo if f not in done and f not in old]
    print(f"{tag} {len(todo)} clips assigned, {len(done)} done, "
          f"{len(todo) - len(done) - len(remaining)} reused from annotator-5 run, "
          f"{len(remaining)} to annotate", flush=True)
    if not remaining:
        return

    t0 = time.time()
    model, processor = base.load_model_and_processor()
    print(f"{tag} model loaded in {time.time() - t0:.1f}s", flush=True)

    with open(out_path, "a", encoding="utf-8") as out:
        for n, fname in enumerate(remaining, start=1):
            t0 = time.time()
            try:
                response = base.annotate_clip(model, processor, fname)
            except Exception as e:
                print(f"{tag} ERROR {fname}: {e}", flush=True)
                response = ""
            raw = base.parse_model_json(response)
            row = {"file_name": fname, "parse_ok": raw is not None,
                   "raw_response": response}
            row.update(base.normalize_prediction(raw or {}))
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(f"{tag} {n}/{len(remaining)} {fname} -> {row['scenario']} | "
                  f"{row['noun']} ({time.time() - t0:.1f}s)", flush=True)


def merge():
    rows = previously_annotated()
    for path in sorted(glob.glob(SHARD_GLOB)):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    rows[r["file_name"]] = r

    ordered = []
    for i, fname in enumerate(ALL_CLIPS):
        if fname in rows:
            r = rows[fname]
            ordered.append({
                "clip_id": i + 1, "file_name": fname,
                "scenario": r["scenario"], "verb": r["verb"], "noun": r["noun"],
                "additional_condition": r["additional_condition"],
                "flavor_profile": r["flavor_profile"],
            })
    payload = {"model": base.MODEL_ID, "clip_dir": base.APP.CLIP_DIR,
               "clip_count": len(ALL_CLIPS), "completed_count": len(ordered),
               "annotations": ordered}
    with open(PREDICTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Merged {len(ordered)}/{len(ALL_CLIPS)} predictions -> {PREDICTIONS_FILE}")


def run_all_gpus():
    import torch
    n_gpus = max(torch.cuda.device_count(), 1)
    print(f"Full pool: {len(ALL_CLIPS)} clips on {n_gpus} GPU(s)")
    procs = []
    for shard in range(n_gpus):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard), HF_HUB_OFFLINE="1")
        procs.append(subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "run-shard",
             str(shard), str(n_gpus)], env=env))
        time.sleep(15)
    codes = [p.wait() for p in procs]
    if any(codes):
        print(f"WARNING: shard exit codes {codes}")
    merge()


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def load_raters():
    """{rater_name: {filename: row}} for 11 humans + QWEN."""
    raters = {}
    for p in sorted(glob.glob(HUMAN_GLOB, recursive=True)):
        d = json.load(open(p, encoding="utf-8"))
        raters[f"H{d['annotator_id']:02d}"] = {
            r["filename"]: r for r in d["annotations"]}
    qwen = json.load(open(PREDICTIONS_FILE, encoding="utf-8"))
    raters["QWEN"] = {r["file_name"]: r for r in qwen["annotations"]}
    return raters


def field_pair_values(row_a, row_b, field):
    """Comparable label pair for one clip+field, or None.

    Conditional fields are hidden by the UI after 'not-eating', so a blank
    means 'not asked' — only compare when both raters answered."""
    a = str(row_a.get(field, "")).strip().lower()
    b = str(row_b.get(field, "")).strip().lower()
    if field in CONDITIONAL_FIELDS and (not a or not b):
        return None
    return a or "(blank)", b or "(blank)"


def pooled_pairs(raters, field, kind):
    """kind='hh' -> all human-human pairs; 'qh' -> qwen-human pairs."""
    names = sorted(raters)
    out = []
    for i, na in enumerate(names):
        for nb in names[i + 1:]:
            if kind == "hh" and (na == "QWEN" or nb == "QWEN"):
                continue
            if kind == "qh" and "QWEN" not in (na, nb):
                continue
            shared = set(raters[na]) & set(raters[nb])
            for f in shared:
                pair = field_pair_values(raters[na][f], raters[nb][f], field)
                if pair:
                    out.append(pair)
    return out


def kappa(pairs):
    from sklearn.metrics import cohen_kappa_score
    if len(pairs) < 10:
        return float("nan")
    a, b = zip(*pairs)
    try:
        return cohen_kappa_score(a, b)
    except ValueError:
        return float("nan")


def pct(pairs):
    return sum(a == b for a, b in pairs) / len(pairs) if pairs else float("nan")


def consensus_labels(raters, field):
    """{filename: majority label} over human raters, ties skipped."""
    votes = defaultdict(list)
    for name, rows in raters.items():
        if name == "QWEN":
            continue
        for f, row in rows.items():
            v = str(row.get(field, "")).strip().lower()
            if field in CONDITIONAL_FIELDS and not v:
                continue
            votes[f].append(v)
    out = {}
    for f, vs in votes.items():
        if len(vs) < 2:
            continue
        (top, n), *rest = Counter(vs).most_common()
        if not rest or n > rest[0][1]:
            out[f] = top
    return out


def analyze():
    import numpy as np
    plt = base.setup_matplotlib()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    raters = load_raters()
    humans = [n for n in sorted(raters) if n != "QWEN"]
    qwen = raters["QWEN"]
    summary = {}

    # --- 1. kappa per field: human-human vs qwen-human ----------------------
    field_stats = {}
    for field in FIELDS:
        hh, qh = pooled_pairs(raters, field, "hh"), pooled_pairs(raters, field, "qh")
        field_stats[field] = {
            "hh_kappa": kappa(hh), "qh_kappa": kappa(qh),
            "hh_pct": pct(hh), "qh_pct": pct(qh),
            "hh_n": len(hh), "qh_n": len(qh),
        }
    summary["field_stats"] = field_stats

    fig, ax = plt.subplots(figsize=(9, 4.8))
    xs = np.arange(len(FIELDS))
    hh_vals = [field_stats[f]["hh_kappa"] for f in FIELDS]
    qh_vals = [field_stats[f]["qh_kappa"] for f in FIELDS]
    b1 = ax.bar(xs - 0.19, hh_vals, 0.36, color=base.BLUE, label="human ↔ human")
    b2 = ax.bar(xs + 0.19, qh_vals, 0.36, color=base.AQUA, label="Qwen ↔ human")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, max(bar.get_height(), 0) + 0.02,
                    f"{bar.get_height():.2f}", ha="center", fontsize=9, color=base.INK)
    ax.set_xticks(xs, ["scenario", "verb", "condition", "flavor", "noun (exact)"])
    ax.axhline(0, color=base.INK_2, linewidth=0.8)
    ax.set_ylabel("Cohen's κ (pooled pairs)")
    ax.set_title("Is Qwen inside the human agreement envelope?  Cohen's κ by field")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "1_kappa_envelope.png"))
    plt.close(fig)

    # --- 2. agreement heatmap with QWEN as a rater (scenario) ---------------
    names = humans + ["QWEN"]
    n = len(names)
    mat = np.full((n, n), np.nan)
    for i, na in enumerate(names):
        for j, nb in enumerate(names):
            if i == j:
                continue
            shared = set(raters[na]) & set(raters[nb])
            pairs = [field_pair_values(raters[na][f], raters[nb][f], "scenario")
                     for f in shared]
            pairs = [p for p in pairs if p]
            if len(pairs) >= 5:
                mat[i, j] = pct(pairs)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seqblue", base.SEQ_RAMP)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    ax.grid(False)
    im = ax.imshow(mat, cmap=cmap, vmin=0.3, vmax=1.0)
    ax.set_xticks(range(n), names, rotation=45, ha="right")
    ax.set_yticks(range(n), names)
    for i in range(n):
        for j in range(n):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]*100:.0f}", ha="center", va="center",
                        fontsize=8,
                        color="#ffffff" if mat[i, j] > 0.75 else base.INK)
    ax.set_title("Scenario % agreement between every rater pair (QWEN = 12th rater)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="% agreement")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "2_rater_agreement_heatmap.png"))
    plt.close(fig)

    # --- 3. accuracy vs consensus -------------------------------------------
    cons_acc = {}
    for field in FIELDS:
        cons = consensus_labels(raters, field)
        pairs = []
        for f, label in cons.items():
            if f in qwen:
                q = str(qwen[f].get(field, "")).strip().lower()
                if field in CONDITIONAL_FIELDS and not q:
                    q = "(blank)"
                pairs.append((label, q))
        cons_acc[field] = (pct(pairs), len(pairs))
    summary["consensus_accuracy"] = {f: {"acc": a, "n": n} for f, (a, n) in cons_acc.items()}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = ["scenario", "verb", "condition", "flavor", "noun (exact)"]
    vals = [cons_acc[f][0] for f in FIELDS]
    ns = [cons_acc[f][1] for f in FIELDS]
    bars = ax.bar(np.arange(len(FIELDS)), vals, 0.6, color=base.BLUE)
    for bar, v, m in zip(bars, vals, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.0%}\nn={m}",
                ha="center", fontsize=9, color=base.INK)
    ax.set_xticks(np.arange(len(FIELDS)), labels)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_title("Qwen accuracy vs majority-vote consensus (clips with ≥2 human labels)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "3_consensus_accuracy.png"))
    plt.close(fig)

    # --- 4. difficulty transfer: unanimous vs split clips (scenario) --------
    votes = defaultdict(list)
    for name in humans:
        for f, row in raters[name].items():
            votes[f].append(str(row.get("scenario", "")).strip().lower())
    groups = {"unanimous (3/3)": [], "majority (2/3)": [], "2 raters agree": [],
              "2 raters split": []}
    for f, vs in votes.items():
        if f not in qwen:
            continue
        q = qwen[f]["scenario"]
        c = Counter(vs).most_common()
        if len(vs) >= 3:
            key = "unanimous (3/3)" if c[0][1] == len(vs) else "majority (2/3)"
            groups[key].append(q == c[0][0])
        elif len(vs) == 2:
            if c[0][1] == 2:
                groups["2 raters agree"].append(q == c[0][0])
            else:
                groups["2 raters split"].append(q in vs)  # matches either rater
    fig, ax = plt.subplots(figsize=(8, 4.5))
    keys = [k for k in groups if groups[k]]
    vals = [np.mean(groups[k]) for k in keys]
    ns = [len(groups[k]) for k in keys]
    bars = ax.bar(np.arange(len(keys)), vals, 0.6, color=base.BLUE)
    for bar, v, m in zip(bars, vals, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.0%}\nn={m}",
                ha="center", fontsize=9, color=base.INK)
    ax.set_xticks(np.arange(len(keys)), keys)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_title("Difficulty transfer — Qwen scenario accuracy by human agreement level")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "4_difficulty_transfer.png"))
    plt.close(fig)
    summary["difficulty"] = {k: {"acc": float(np.mean(groups[k])), "n": len(groups[k])}
                             for k in keys}

    # --- 5. per-annotator agreement with Qwen (scenario) --------------------
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    rows = []
    for name in humans:
        shared = set(raters[name]) & set(qwen)
        pairs = [field_pair_values(raters[name][f], qwen[f], "scenario") for f in shared]
        pairs = [p for p in pairs if p]
        rows.append((name, pct(pairs), len(pairs)))
    rows.sort(key=lambda r: -r[1])
    xs = np.arange(len(rows))
    bars = ax.bar(xs, [r[1] for r in rows], 0.6, color=base.BLUE)
    hh_mean = field_stats["scenario"]["hh_pct"]
    ax.axhline(hh_mean, color=base.INK, linestyle="--", linewidth=1.2)
    ax.text(len(rows) - 0.4, hh_mean + 0.015, f"human↔human mean {hh_mean:.0%}",
            ha="right", color=base.INK, fontsize=9)
    for bar, (_, v, m) in zip(bars, rows):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.0%}",
                ha="center", fontsize=9, color=base.INK)
    ax.set_xticks(xs, [r[0] for r in rows])
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_title("Qwen ↔ each human, scenario agreement (dashed = human↔human mean)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "5_per_annotator_agreement.png"))
    plt.close(fig)

    # --- 6. noun semantic agreement: human-human vs qwen-human --------------
    from sentence_transformers import SentenceTransformer
    def noun_pairs(kind):
        pairs = pooled_pairs(raters, "noun", kind)
        return [(a, b) for a, b in pairs if a not in ("", "(blank)") and b not in ("", "(blank)")]
    hh_np, qh_np = noun_pairs("hh"), noun_pairs("qh")
    st = SentenceTransformer(base.EMBED_MODEL_ID)
    vocab = sorted({w for p in hh_np + qh_np for w in p})
    emb = dict(zip(vocab, st.encode(vocab, normalize_embeddings=True)))
    hh_sims = [float(emb[a] @ emb[b]) for a, b in hh_np]
    qh_sims = [float(emb[a] @ emb[b]) for a, b in qh_np]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    bins = np.arange(0, 1.05, 0.05)
    ax.hist(hh_sims, bins=bins, density=True, alpha=0.85, color=base.BLUE,
            label=f"human ↔ human (n={len(hh_sims)}, mean {np.mean(hh_sims):.2f})")
    ax.hist(qh_sims, bins=bins, density=True, alpha=0.6, color=base.AQUA,
            label=f"Qwen ↔ human (n={len(qh_sims)}, mean {np.mean(qh_sims):.2f})")
    ax.set_xlabel("noun cosine similarity (MiniLM)")
    ax.set_ylabel("density")
    ax.set_title("Noun semantic agreement — is Qwen's food vocabulary human-like?")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "6_noun_semantic_agreement.png"))
    plt.close(fig)
    summary["noun_cosine"] = {"hh_mean": float(np.mean(hh_sims)),
                              "qh_mean": float(np.mean(qh_sims))}

    # --- 7. scenario confusion vs consensus ---------------------------------
    cons = consensus_labels(raters, "scenario")
    y_true = [cons[f] for f in cons if f in qwen]
    y_pred = [qwen[f]["scenario"] for f in cons if f in qwen]
    base.plot_confusion(plt, y_true, y_pred,
                        "Scenario confusion vs consensus (full pool)",
                        os.path.join(RESULTS_DIR, "7_scenario_confusion_consensus.png"),
                        ylabel="Human consensus")

    # --- report ---------------------------------------------------------------
    lines = ["Full-pool benchmark — Qwen as a 12th annotator", ""]
    lines.append(f"{'field':<12} {'κ h↔h':>8} {'κ q↔h':>8} {'% h↔h':>8} {'% q↔h':>8}")
    for f in FIELDS:
        s = field_stats[f]
        lines.append(f"{f:<12} {s['hh_kappa']:8.2f} {s['qh_kappa']:8.2f} "
                     f"{s['hh_pct']:8.1%} {s['qh_pct']:8.1%}")
    lines.append("")
    for f, (a, m) in cons_acc.items():
        lines.append(f"consensus acc {f:<22} {a:6.1%} (n={m})")
    lines.append("")
    for k, v in summary["difficulty"].items():
        lines.append(f"difficulty [{k:<16}] {v['acc']:6.1%} (n={v['n']})")
    report = "\n".join(lines)
    with open(os.path.join(RESULTS_DIR, "report.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(report)
    print(f"\nGraphs + report written to {RESULTS_DIR}/")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "run-shard":
        run_shard(int(sys.argv[2]), int(sys.argv[3]))
    elif mode == "run":
        run_all_gpus()
    elif mode == "merge":
        merge()
    elif mode == "analyze":
        analyze()
    elif mode == "all":
        run_all_gpus()
        analyze()
    else:
        sys.exit(f"Unknown mode {mode!r}")


if __name__ == "__main__":
    main()
