#!/usr/bin/env python3
"""
Benchmark 3 — the scientific test: rater reliability + few-shot transfer.

Part A, TEST-RETEST RELIABILITY (intra-rater).
    Qwen annotates annotator 5's 101 clips three more times with sampling
    (temperature 0.7, different seeds). A human annotator re-labeling the
    same clip weeks later would not agree with themselves 100% either — the
    scientific question is whether Qwen's self-agreement is above the
    human INTER-rater envelope. A model can only be a useful annotator if
    it is at least a *stable* one; self-agreement is also the ceiling on
    its achievable human-agreement.

Part B, FEW-SHOT TRANSFER ("how to transfer annotation quality into Qwen").
    The same clips annotated with 4 and 8 in-context video examples whose
    labels come from multi-annotator consensus on clips OUTSIDE annotator
    5's set (no leakage). Examples are chosen to cover the known failure
    mode (human-fed feeding). Comparing 0-shot (existing benchmark run) ->
    4-shot -> 8-shot accuracy per field is the in-context learning curve:
    how much annotation quality can be moved into the model without any
    fine-tuning, and which fields respond.

Usage (GPU session):
    source /usr4/spclpgm/srikumar/envs/qwen/bin/activate
    python3 qwen_reliability_fewshot.py all | run | analyze
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
LIMIT = int(os.environ.get("QWEN_RELFS_LIMIT", "0"))
SHARD_GLOB = os.path.join(HOME, "qwen_relfs_shard*.jsonl")
HUMAN_GLOB = os.path.join(HOME, "annotationAnalysis", "unzipped", "**", "annotationTaste_*.json")
FEWSHOT_FILE = os.path.join(HOME, "qwen_fewshot_examples.json")
RESULTS_DIR = os.path.join(HOME, "qwen_reliability_fewshot_results")

RELIABILITY_RUNS = 3          # sampled re-annotation passes
TEMPERATURE = 0.7
FIELDS = ["scenario", "verb", "additional_condition", "flavor_profile", "noun"]
CONDITIONAL_FIELDS = {"verb", "additional_condition", "flavor_profile", "noun"}

EVAL_CLIPS = list(base.ASSIGNED_FILENAMES)      # annotator 5's 101 clips

CONDITIONS = ([f"rel{r}" for r in range(RELIABILITY_RUNS)] + ["fs4", "fs8"])


# ---------------------------------------------------------------------------
# Few-shot example selection (consensus clips outside the eval set)
# ---------------------------------------------------------------------------

def load_humans():
    raters = {}
    for p in sorted(glob.glob(HUMAN_GLOB, recursive=True)):
        d = json.load(open(p, encoding="utf-8"))
        raters[d["annotator_id"]] = {r["filename"]: r for r in d["annotations"]}
    return raters

# Target verb coverage: the 8-shot set must include human-fed examples,
# since human-fed is the documented blind spot. fs4 uses the first 4.
TARGET_SLOTS = [
    ("not-eating", None),
    ("eating", "self-fed, hand"),
    ("eating", "human-fed, utensil"),
    ("possibly-eating", None),
    ("eating", "human-fed, hand"),
    ("eating", "self-fed, utensil"),
    ("not-eating", None),
    ("eating", "self-fed, drink"),
]


def build_fewshot_examples():
    if os.path.exists(FEWSHOT_FILE):
        return json.load(open(FEWSHOT_FILE, encoding="utf-8"))

    raters = load_humans()
    eval_set = set(EVAL_CLIPS)
    by_clip = defaultdict(list)
    for rows in raters.values():
        for f, row in rows.items():
            if f not in eval_set:
                by_clip[f].append(row)

    def consensus_row(fname):
        """Majority label per field among this clip's raters; None w/o
        a >=2-rater scenario majority."""
        rows = by_clip[fname]
        scen = Counter(str(r["scenario"]).strip().lower() for r in rows).most_common(1)[0]
        if scen[1] < 2:
            return None
        out = {"filename": fname, "scenario": scen[0]}
        agreeing = [r for r in rows
                    if str(r["scenario"]).strip().lower() == scen[0]]
        for field in ["verb", "additional_condition", "flavor_profile", "noun"]:
            vals = [str(r.get(field, "")).strip().lower() for r in agreeing]
            vals = [v for v in vals if v]
            out[field] = Counter(vals).most_common(1)[0][0] if vals else ""
        out["n_raters"] = len(rows)
        out["n_agree_scenario"] = scen[1]
        return out

    consensus = [c for c in (consensus_row(f) for f in by_clip) if c]
    chosen, used = [], set()
    for scenario, verb in TARGET_SLOTS:
        pool = [c for c in consensus if c["filename"] not in used
                and c["scenario"] == scenario
                and (verb is None or c["verb"] == verb)]
        # strongest consensus first; deterministic tiebreak by filename
        pool.sort(key=lambda c: (-c["n_agree_scenario"], c["filename"]))
        if not pool and verb is not None:      # fallback: same agency, any instrument
            agency = verb.split(",")[0]
            pool = [c for c in consensus if c["filename"] not in used
                    and c["scenario"] == scenario and c["verb"].startswith(agency)]
            pool.sort(key=lambda c: (-c["n_agree_scenario"], c["filename"]))
        if pool:
            chosen.append(pool[0])
            used.add(pool[0]["filename"])
    with open(FEWSHOT_FILE, "w", encoding="utf-8") as f:
        json.dump(chosen, f, indent=2)
    print(f"Selected {len(chosen)} few-shot examples -> {FEWSHOT_FILE}")
    for c in chosen:
        print(f"  {c['scenario']:<16} {c['verb']:<22} {c['noun']:<14} {c['filename']}")
    return chosen


def example_label_json(ex):
    return json.dumps({
        "scenario": ex["scenario"], "verb": ex["verb"],
        "additional_condition": ex["additional_condition"],
        "noun": ex["noun"], "flavor_profile": ex["flavor_profile"],
    })


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def build_messages(condition, clip_file, examples):
    content = []
    if condition.startswith("fs"):
        k = int(condition[2:])
        content.append({"type": "text", "text": (
            f"You will first see {k} example clips, each followed by its CORRECT "
            "annotation produced by trained human annotators. Study how the schema "
            "is applied, then annotate the final clip yourself.")})
        for ex in examples[:k]:
            content.append({"type": "video",
                            "video": os.path.join(base.APP.CLIP_DIR, ex["filename"])})
            content.append({"type": "text",
                            "text": "Correct annotation: " + example_label_json(ex)})
        content.append({"type": "text", "text": "Now annotate this new clip."})
    content.append({"type": "video",
                    "video": os.path.join(base.APP.CLIP_DIR, clip_file)})
    content.append({"type": "text", "text": base.ANNOTATION_PROMPT})
    return [{"role": "user", "content": content}]


def run_shard(shard, num_shards):
    import torch
    tag = f"[relfs shard {shard}]"
    examples = json.load(open(FEWSHOT_FILE, encoding="utf-8"))
    clips = EVAL_CLIPS[shard::num_shards]
    if LIMIT:
        clips = clips[:LIMIT]

    out_path = os.path.join(HOME, f"qwen_relfs_shard{shard}.jsonl")
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            done = {(json.loads(l)["file_name"], json.loads(l)["condition"])
                    for l in f if l.strip()}
    total = len(clips) * len(CONDITIONS) - len(done)
    print(f"{tag} {total} generations to go", flush=True)
    if total <= 0:
        return

    model, processor = base.load_model_and_processor()
    n = 0
    with open(out_path, "a", encoding="utf-8") as out:
        for ci, fname in enumerate(clips):
            for cond in CONDITIONS:
                if (fname, cond) in done:
                    continue
                t0 = time.time()
                try:
                    messages = build_messages(cond, fname, examples)
                    inputs = processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=True,
                        return_dict=True, return_tensors="pt",
                        processor_kwargs={"num_frames": base.NUM_FRAMES, "fps": None},
                    ).to(model.device)
                    if cond.startswith("rel"):
                        torch.manual_seed(hash((cond, fname)) % (2**31))
                        gen_kwargs = dict(do_sample=True, temperature=TEMPERATURE,
                                          top_p=0.9)
                    else:
                        gen_kwargs = dict(do_sample=False)
                    ids = model.generate(**inputs, max_new_tokens=160, **gen_kwargs)
                    response = processor.batch_decode(
                        ids[:, inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True)[0].strip()
                except Exception as e:
                    print(f"{tag} ERROR {fname} {cond}: {e}", flush=True)
                    response = ""
                raw = base.parse_model_json(response)
                row = {"file_name": fname, "condition": cond,
                       "parse_ok": raw is not None, "raw_response": response}
                row.update(base.normalize_prediction(raw or {}))
                out.write(json.dumps(row) + "\n")
                out.flush()
                n += 1
                print(f"{tag} {n}/{total} {cond} {fname} -> {row['scenario']} | "
                      f"{row['verb']} ({time.time()-t0:.1f}s)", flush=True)


def run_all_gpus():
    import torch
    build_fewshot_examples()
    n_gpus = max(torch.cuda.device_count(), 1)
    print(f"Reliability+fewshot: {len(EVAL_CLIPS)} clips x {len(CONDITIONS)} "
          f"conditions on {n_gpus} GPU(s)")
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


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def field_pair(row_a, row_b, field):
    a = str(row_a.get(field, "")).strip().lower()
    b = str(row_b.get(field, "")).strip().lower()
    if field in CONDITIONAL_FIELDS and (not a or not b):
        return None
    return a or "(blank)", b or "(blank)"


def pct(pairs):
    return sum(a == b for a, b in pairs) / len(pairs) if pairs else float("nan")


def analyze():
    import numpy as np
    plt = base.setup_matplotlib()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    rows = defaultdict(dict)      # condition -> {file: row}
    for path in sorted(glob.glob(SHARD_GLOB)):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    rows[r["condition"]][r["file_name"]] = r

    # 0-shot greedy = the original benchmark predictions
    zero = {r["file_name"]: r for r in json.load(
        open(base.PREDICTIONS_FILE, encoding="utf-8"))["annotations"]}
    gt = {r["filename"]: r for r in json.load(
        open(base.GROUND_TRUTH_FILE, encoding="utf-8"))["annotations"]}

    # human-human envelope from the full-pool analysis
    hh = json.load(open(os.path.join(HOME, "qwen_full_pool_results", "summary.json"),
                        encoding="utf-8"))["field_stats"]

    summary = {}

    # --- Part A: test-retest self-agreement ---------------------------------
    rel = [rows[f"rel{r}"] for r in range(RELIABILITY_RUNS)]
    self_agree, human_env = {}, {}
    for field in FIELDS:
        pairs = []
        for i in range(len(rel)):
            for j in range(i + 1, len(rel)):
                for f in set(rel[i]) & set(rel[j]):
                    p = field_pair(rel[i][f], rel[j][f], field)
                    if p:
                        pairs.append(p)
        self_agree[field] = pct(pairs)
        human_env[field] = hh[field]["hh_pct"]
    summary["self_agreement"] = self_agree

    fig, ax = plt.subplots(figsize=(9, 4.8))
    xs = np.arange(len(FIELDS))
    b1 = ax.bar(xs - 0.19, [self_agree[f] for f in FIELDS], 0.36, color=base.BLUE,
                label="Qwen ↔ Qwen (test-retest, T=0.7)")
    b2 = ax.bar(xs + 0.19, [human_env[f] for f in FIELDS], 0.36, color=base.AQUA,
                label="human ↔ human (different people)")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                    f"{bar.get_height():.0%}", ha="center", fontsize=9, color=base.INK)
    ax.set_xticks(xs, ["scenario", "verb", "condition", "flavor", "noun"])
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_title("Reliability — Qwen's agreement with itself vs humans with each other")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "1_test_retest_reliability.png"))
    plt.close(fig)

    # --- Part B: few-shot learning curve ------------------------------------
    def acc_vs_gt(pred_rows, field, subset=None):
        pairs = []
        for f, g in gt.items():
            p = pred_rows.get(f)
            if not p:
                continue
            if subset == "eating" and g["scenario"] in ("not-eating", ""):
                continue
            if subset == "eating" and str(g["scenario"]).lower() == "not-eating":
                continue
            gv = str(g.get(field, "")).strip().lower()
            pv = str(p.get(field, "")).strip().lower()
            if field in CONDITIONAL_FIELDS:
                if not gv:
                    continue
                pv = pv or "(blank)"
            pairs.append((gv, pv))
        return pct(pairs), len(pairs)

    shot_rows = {0: zero, 4: rows["fs4"], 8: rows["fs8"]}
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    colors = {"scenario": base.BLUE, "verb": base.AQUA, "flavor_profile": "#4a3aa7",
              "noun": "#e34948", "additional_condition": "#eda100"}
    curve = {}
    for field in FIELDS:
        ys = [acc_vs_gt(shot_rows[k], field)[0] for k in (0, 4, 8)]
        curve[field] = ys
        ax.plot([0, 4, 8], ys, marker="o", markersize=7, linewidth=2,
                color=colors[field])
        ax.annotate(field.replace("additional_condition", "condition")
                    .replace("flavor_profile", "flavor"),
                    (8, ys[-1]), xytext=(8, 0), textcoords="offset points",
                    color=colors[field], fontweight="bold", fontsize=9, va="center")
    ax.set_xticks([0, 4, 8])
    ax.set_xlim(-0.5, 10.5)
    ax.set_xlabel("labeled video examples in context (consensus clips, no leakage)")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_title("Few-shot transfer — accuracy vs annotator 5 by in-context examples")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "2_fewshot_curve.png"))
    plt.close(fig)
    summary["fewshot_curve"] = curve

    # --- Part B2: does few-shot fix the human-fed blind spot? ----------------
    def agency_acc(pred_rows, want):
        hits = tot = 0
        for f, g in gt.items():
            gv = str(g.get("verb", "")).strip().lower()
            if not gv or gv == "unclear":
                continue
            g_agency = "self" if "self" in gv else "other"
            if g_agency != want:
                continue
            p = pred_rows.get(f)
            if not p:
                continue
            pv = str(p.get("verb", "")).strip().lower()
            p_agency = "self" if "self" in pv else "other" if "human" in pv else "?"
            tot += 1
            hits += p_agency == g_agency
        return hits / tot if tot else float("nan"), tot

    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = np.arange(3)
    self_vals = [agency_acc(shot_rows[k], "self")[0] for k in (0, 4, 8)]
    other_vals = [agency_acc(shot_rows[k], "other")[0] for k in (0, 4, 8)]
    b1 = ax.bar(xs - 0.19, self_vals, 0.36, color=base.BLUE, label="true self-fed")
    b2 = ax.bar(xs + 0.19, other_vals, 0.36, color=base.AQUA, label="true human-fed")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{bar.get_height():.0%}", ha="center", fontsize=9, color=base.INK)
    ax.set_xticks(xs, ["0-shot", "4-shot", "8-shot"])
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_title("Does in-context evidence fix the human-fed blind spot?")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "3_fewshot_agency.png"))
    plt.close(fig)
    summary["fewshot_agency"] = {"self": self_vals, "other": other_vals}

    # --- report ---------------------------------------------------------------
    lines = ["Reliability + few-shot transfer", "",
             f"{'field':<12} {'self-agree':>11} {'human env':>11}"]
    for f in FIELDS:
        lines.append(f"{f:<12} {self_agree[f]:11.1%} {human_env[f]:11.1%}")
    lines.append("")
    lines.append(f"{'field':<12} {'0-shot':>8} {'4-shot':>8} {'8-shot':>8}")
    for f in FIELDS:
        ys = curve[f]
        lines.append(f"{f:<12} {ys[0]:8.1%} {ys[1]:8.1%} {ys[2]:8.1%}")
    lines.append("")
    lines.append("agency (human-fed clips): " + "  ".join(
        f"{k}-shot={v:.0%}" for k, v in zip((0, 4, 8), other_vals)))
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
    elif mode == "examples":
        build_fewshot_examples()
    elif mode == "run":
        run_all_gpus()
    elif mode == "analyze":
        analyze()
    elif mode == "all":
        run_all_gpus()
        analyze()
    else:
        sys.exit(f"Unknown mode {mode!r}")


if __name__ == "__main__":
    main()
