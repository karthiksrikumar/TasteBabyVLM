#!/usr/bin/env python3
"""
Benchmark 2 — schema transfer to unseen clips (touch dataset).

Samples clips from the ivc-ml touch corpus (~2.2M clips, same home-video
style as the taste pool but never annotated with our schema), then has
Qwen3-VL-30B produce (a) a full taste-schema annotation and (b) a one-to-two
sentence free-text description per clip.

What this measures, without any human labels:
  * Domain shift: does the label distribution on new-domain clips stay
    plausible (vs the taste-pool distribution), or does the schema collapse
    (everything "unclear"/"not-eating")?
  * Pre-annotation value: the output JSON doubles as machine pre-annotations
    a human annotator can verify instead of labeling from scratch — the
    standard model-in-the-loop annotation pipeline.
  * A stratified human-audit sample (default 20 clips) is written out for
    quick manual verification.

Usage (GPU session):
    source /usr4/spclpgm/srikumar/envs/qwen/bin/activate
    python3 qwen_touch_transfer.py all | sample | run | analyze
    QWEN_TOUCH_N=240 controls sample size.
"""

import glob
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen_taste_benchmark as base

HOME = base.HOME
TOUCH_DIR = "/projectnb/ivc-ml/maxwh/code/senses/touch/public_repo/data/clips"
SAMPLE_N = int(os.environ.get("QWEN_TOUCH_N", "240"))
SAMPLE_SEED = 20260714
LIMIT = int(os.environ.get("QWEN_TOUCH_LIMIT", "0"))

SAMPLE_FILE = os.path.join(HOME, "qwen_touch_sample.json")
SHARD_GLOB = os.path.join(HOME, "qwen_touch_shard*.jsonl")
PREDICTIONS_FILE = os.path.join(HOME, "qwen_touch_annotations.json")
RESULTS_DIR = os.path.join(HOME, "qwen_touch_transfer_results")
AUDIT_FILE = os.path.join(RESULTS_DIR, "human_audit_sample.json")

TRANSFER_PROMPT = base.ANNOTATION_PROMPT.replace(
    'in exactly this format:\n{"scenario"',
    'in exactly this format (description = 1-2 sentences of what happens in '
    'the clip: who is visible, what they do, what objects are involved):\n'
    '{"description": "...", "scenario"')


def build_sample():
    """Reservoir-sample SAMPLE_N clip filenames from the touch corpus."""
    if os.path.exists(SAMPLE_FILE):
        sample = json.load(open(SAMPLE_FILE, encoding="utf-8"))
        if len(sample) >= SAMPLE_N:
            print(f"Sample already exists: {len(sample)} clips in {SAMPLE_FILE}")
            return sample
    rng = random.Random(SAMPLE_SEED)
    reservoir, seen = [], 0
    t0 = time.time()
    with os.scandir(TOUCH_DIR) as it:
        for entry in it:
            if not entry.name.endswith(".mp4"):
                continue
            seen += 1
            if len(reservoir) < SAMPLE_N:
                reservoir.append(entry.name)
            else:
                j = rng.randrange(seen)
                if j < SAMPLE_N:
                    reservoir[j] = entry.name
    reservoir.sort()
    with open(SAMPLE_FILE, "w", encoding="utf-8") as f:
        json.dump(reservoir, f, indent=1)
    print(f"Sampled {len(reservoir)} of {seen} clips in {time.time()-t0:.0f}s "
          f"-> {SAMPLE_FILE}")
    return reservoir


def annotate(model, processor, path):
    messages = [{
        "role": "user",
        "content": [{"type": "video", "video": path},
                    {"type": "text", "text": TRANSFER_PROMPT}],
    }]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True,
        return_tensors="pt",
        processor_kwargs={"num_frames": base.NUM_FRAMES, "fps": None},
    ).to(model.device)
    out = model.generate(**inputs, max_new_tokens=260, do_sample=False)
    gen = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(gen, skip_special_tokens=True)[0].strip()


def run_shard(shard, num_shards):
    tag = f"[touch shard {shard}]"
    sample = json.load(open(SAMPLE_FILE, encoding="utf-8"))
    todo = sample[shard::num_shards]
    if LIMIT:
        todo = todo[:LIMIT]
    out_path = os.path.join(HOME, f"qwen_touch_shard{shard}.jsonl")
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(l)["file_name"] for l in f if l.strip()}
    remaining = [f for f in todo if f not in done]
    print(f"{tag} {len(remaining)} clips to annotate", flush=True)
    if not remaining:
        return
    model, processor = base.load_model_and_processor()
    with open(out_path, "a", encoding="utf-8") as out:
        for n, fname in enumerate(remaining, start=1):
            t0 = time.time()
            try:
                response = annotate(model, processor, os.path.join(TOUCH_DIR, fname))
            except Exception as e:
                print(f"{tag} ERROR {fname}: {e}", flush=True)
                response = ""
            raw = base.parse_model_json(response) or {}
            row = {"file_name": fname, "parse_ok": bool(raw),
                   "description": str(raw.get("description", "")).strip(),
                   "raw_response": response}
            row.update(base.normalize_prediction(raw))
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(f"{tag} {n}/{len(remaining)} {fname} -> {row['scenario']} | "
                  f"{row['noun']} ({time.time()-t0:.1f}s)", flush=True)


def merge():
    rows = {}
    for path in sorted(glob.glob(SHARD_GLOB)):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    rows[r["file_name"]] = r
    sample = json.load(open(SAMPLE_FILE, encoding="utf-8"))
    ordered = [dict(rows[f], clip_id=i + 1) for i, f in enumerate(sample) if f in rows]
    payload = {"model": base.MODEL_ID, "clip_dir": TOUCH_DIR,
               "sample_seed": SAMPLE_SEED, "sample_size": len(sample),
               "completed_count": len(ordered), "annotations": ordered}
    with open(PREDICTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Merged {len(ordered)} touch annotations -> {PREDICTIONS_FILE}")


def run_all_gpus():
    import torch
    build_sample()
    n_gpus = max(torch.cuda.device_count(), 1)
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


def analyze():
    import numpy as np
    plt = base.setup_matplotlib()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    touch = json.load(open(PREDICTIONS_FILE, encoding="utf-8"))["annotations"]
    taste = json.load(open(base.__dict__["HOME"] + "/qwen_full_pool_predictions.json",
                           encoding="utf-8"))["annotations"]
    print(f"{len(touch)} touch clips, {len(taste)} taste clips (Qwen labels both)")

    # --- 1. scenario distribution: taste pool vs touch pool -----------------
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    labels = ["eating", "possibly-eating", "not-eating"]
    xs = np.arange(len(labels))
    t1 = [sum(r["scenario"] == l for r in taste) / len(taste) for l in labels]
    t2 = [sum(r["scenario"] == l for r in touch) / len(touch) for l in labels]
    b1 = ax.bar(xs - 0.19, t1, 0.36, color=base.BLUE, label=f"taste pool (n={len(taste)})")
    b2 = ax.bar(xs + 0.19, t2, 0.36, color=base.AQUA, label=f"touch pool (n={len(touch)})")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.0%}", ha="center", fontsize=9, color=base.INK)
    ax.set_xticks(xs, labels)
    ax.set_ylabel("share of clips")
    ax.set_title("Domain shift — Qwen scenario distribution, taste vs touch corpus")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "1_scenario_domain_shift.png"))
    plt.close(fig)

    # --- 2. schema-collapse check: unclear/blank rates ----------------------
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    fields = ["verb", "additional_condition", "noun", "flavor_profile"]
    def unclear_rate(rows, field):
        eating = [r for r in rows if r["scenario"] != "not-eating"]
        if not eating:
            return 0
        return sum(r.get(field, "") in ("unclear", "n/a", "") for r in eating) / len(eating)
    xs = np.arange(len(fields))
    u1 = [unclear_rate(taste, f) for f in fields]
    u2 = [unclear_rate(touch, f) for f in fields]
    b1 = ax.bar(xs - 0.19, u1, 0.36, color=base.BLUE, label="taste pool")
    b2 = ax.bar(xs + 0.19, u2, 0.36, color=base.AQUA, label="touch pool")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.0%}", ha="center", fontsize=9, color=base.INK)
    ax.set_xticks(xs, ["verb", "condition", "noun", "flavor"])
    ax.set_ylabel('"unclear" / "n/a" rate on eating clips')
    ax.set_title("Schema collapse check — does Qwen give up more on the new domain?")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "2_schema_collapse.png"))
    plt.close(fig)

    # --- 3. top nouns on the touch pool --------------------------------------
    nouns = Counter(r["noun"] for r in touch
                    if r["scenario"] != "not-eating" and r.get("noun"))
    top = nouns.most_common(15)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ys = np.arange(len(top))
    ax.barh(ys, [c for _, c in top], 0.6, color=base.BLUE)
    for y, (_, c) in zip(ys, top):
        ax.text(c + 0.15, y, str(c), va="center", fontsize=9, color=base.INK)
    ax.set_yticks(ys, [n for n, _ in top])
    ax.set_xlabel("clip count")
    ax.set_title("Top nouns Qwen assigns on the touch corpus (eating clips)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "3_touch_top_nouns.png"))
    plt.close(fig)

    # --- 4. description quality proxies --------------------------------------
    lens = [len(r.get("description", "").split()) for r in touch]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(lens, bins=np.arange(0, max(lens) + 5, 5), color=base.BLUE,
            edgecolor=base.SURFACE)
    ax.axvline(np.mean(lens), color=base.INK, linestyle="--", linewidth=1.3)
    ax.text(np.mean(lens) + 1, ax.get_ylim()[1] * 0.9,
            f"mean {np.mean(lens):.0f} words", color=base.INK)
    ax.set_xlabel("description length (words)")
    ax.set_ylabel("clip count")
    empty = sum(1 for l in lens if l == 0)
    ax.set_title(f"Free-text description length (empty: {empty}/{len(touch)})")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "4_description_length.png"))
    plt.close(fig)

    # --- human audit sample: stratified by scenario ---------------------------
    rng = random.Random(SAMPLE_SEED)
    audit = []
    for label in ["eating", "possibly-eating", "not-eating"]:
        pool = [r for r in touch if r["scenario"] == label]
        rng.shuffle(pool)
        for r in pool[:7]:
            audit.append({k: r[k] for k in
                          ("file_name", "description", "scenario", "verb", "noun",
                           "additional_condition", "flavor_profile")}
                         | {"clip_path": os.path.join(TOUCH_DIR, r["file_name"]),
                            "human_verdict": ""})
    with open(AUDIT_FILE, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    parse_fail = sum(1 for r in touch if not r.get("parse_ok"))
    lines = [
        f"Touch-corpus transfer — {len(touch)} clips, parse failures {parse_fail}",
        f"scenario: " + ", ".join(f"{l}={sum(r['scenario']==l for r in touch)}"
                                  for l in ["eating", "possibly-eating", "not-eating"]),
        f"mean description length: {np.mean(lens):.1f} words",
        f"distinct nouns on eating clips: {len(nouns)}",
        f"human audit sample ({len(audit)} clips): {AUDIT_FILE}",
    ]
    report = "\n".join(lines)
    with open(os.path.join(RESULTS_DIR, "report.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(report)
    print(f"\nGraphs written to {RESULTS_DIR}/")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "run-shard":
        run_shard(int(sys.argv[2]), int(sys.argv[3]))
    elif mode == "sample":
        build_sample()
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
