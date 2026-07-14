#!/usr/bin/env python3
"""
Evaluation benchmark: Qwen3-VL-30B-A3B-Instruct vs. human annotator #5
(Karthik Srikumar) on the taste-annotation clip set.

Pipeline
--------
1.  Recovers annotator 5's 101 assigned clips using the exact same
    deterministic assignment + shuffle logic as annotation_app.py (verified:
    the recovered list matches annotationTaste_5.json set AND order).
2.  Runs Qwen3-VL-30B on every clip with an annotation prompt built from the
    Taste Annotation Tutorial PDF and the option vocabulary in
    annotation_app.py, forcing a strict JSON answer per clip.
    The run is sharded across all visible GPUs (one model copy per GPU,
    each shard handles clips[i::n]); each shard appends to its own JSONL
    so the run is resumable.
3.  Compares Qwen's predictions to the human ground truth
    (annotationTaste_5.json), including a sentence-embedding cosine
    similarity model (all-MiniLM-L6-v2) for the free-text noun field, and
    writes 7 graphs + a summary report to qwen_taste_benchmark_results/.

NOTE on ground truth: /usr4/spclpgm/srikumar/annotation_05_karthik_srikumar.json
is a SMELL-task file (scenarios "Smelling"/"None", 59 clips, zero filename
overlap with taste_annotation_clips), so it cannot score a taste run. The
taste ground truth for annotator 5 is annotationTaste_5.json (101/101 clips,
taste schema) and that is what this benchmark compares against. Override with
QWEN_BENCH_GT=/path/to/file.json if a different taste file should be used.

Usage (from a GPU session):
    source /usr4/spclpgm/srikumar/envs/qwen/bin/activate
    python3 qwen_taste_benchmark.py all            # run model on all clips, then analyze
    python3 qwen_taste_benchmark.py run            # inference only (all GPUs)
    python3 qwen_taste_benchmark.py analyze        # graphs + report from saved predictions
    QWEN_BENCH_LIMIT=2 python3 qwen_taste_benchmark.py run   # smoke test
"""

import glob
import importlib.util
import json
import os
import re
import subprocess
import sys
import time

HOME = os.path.dirname(os.path.abspath(__file__))

os.environ.setdefault("HF_HOME", "/projectnb/rise-ivc/srikumar/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MODEL_ID = os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen3-VL-30B-A3B-Instruct")
ANNOTATOR_ID = int(os.environ.get("QWEN_BENCH_ANNOTATOR", "5"))
NUM_FRAMES = int(os.environ.get("QWEN_NUM_FRAMES", "8"))
LIMIT = int(os.environ.get("QWEN_BENCH_LIMIT", "0"))  # 0 = all clips

GROUND_TRUTH_FILE = os.environ.get("QWEN_BENCH_GT", os.path.join(HOME, "annotationTaste_5.json"))
PREDICTIONS_FILE = os.path.join(HOME, "qwen_taste_predictions.json")
SHARD_GLOB = os.path.join(HOME, "qwen_taste_predictions_shard*.jsonl")
RESULTS_DIR = os.path.join(HOME, "qwen_taste_benchmark_results")

EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def load_annotation_app():
    spec = importlib.util.spec_from_file_location(
        "annotation_app", os.path.join(HOME, "annotation_app.py")
    )
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    return app


APP = load_annotation_app()

# Annotator 5's clips, in the annotator's own (seed-shuffled) order — the
# same order annotationTaste_5.json rows are written in, so clip_number
# (1-based position) lines up between prediction and ground truth files.
ASSIGNED_FILENAMES = [
    APP.CLIP_FILENAMES[i] for i in APP.assigned_global_indices(ANNOTATOR_ID)
]

VERB_OPTIONS = sorted(APP.VERBS)
CONDITION_OPTIONS = sorted(APP.ADDITIONAL_CONDITIONS)
FLAVOR_OPTIONS = list(APP.FLAVOR_PROFILE_OPTIONS)

# ---------------------------------------------------------------------------
# Prompt (from the Taste Annotation Tutorial PDF + annotation_app.py vocab)
# ---------------------------------------------------------------------------

ANNOTATION_PROMPT = f"""You are a careful video annotator for the Gong Lab (Boston University) taste-annotation project. The clips are short home-video recordings, usually of a young child. Watch the clip and annotate it by answering five questions, exactly the way a trained human annotator would.

STEP 1 — SCENARIO: Is the person eating, possibly eating, or clearly not eating in this clip?
Allowed values: "eating", "possibly-eating", "not-eating".
This is the first choice for every clip. If the answer is "not-eating", every remaining field must be the empty string "".

STEP 2 — VERB: How is the food getting to their mouth? Self-fed (by hand, utensil, or drink), human-fed (fed to them by another person), or unclear if you can't tell from the clip.
Allowed values: {json.dumps(VERB_OPTIONS)}.
Skipped (empty string) when scenario is "not-eating".

STEP 3 — CONDITION: Any additional context around the moment in the clip (e.g. appears to be at a kitchen table = eating setting).
Allowed values: {json.dumps(CONDITION_OPTIONS)}.
"actively-taking-a-bite" means the bite is happening on camera. Skipped (empty string) when scenario is "not-eating".

STEP 4 — NOUN / OBJECT: The specific food or object involved in the clip, e.g. "apple", "sandwich", "coffee cup", "cereal", "pasta". Use a short lowercase noun phrase. If something looks like mush, "baby food" is a viable option. Write "unclear" if you genuinely can't tell what the object is.
Skipped (empty string) when scenario is "not-eating".

STEP 5 — FLAVOR PROFILE: Your best judgment of the food's flavor based on appearance only — not something you can taste directly. Use visual cues only (color, texture, garnish, packaging) to estimate the flavor. Use the food around the image as an indicator, and your own opinion for discrepancies (e.g. is pasta neutral or salty?).
Allowed values: {json.dumps(FLAVOR_OPTIONS)}.
Skipped (empty string) when scenario is "not-eating".

Respond with ONLY a single JSON object on one line, no markdown, no extra text, in exactly this format:
{{"scenario": "...", "verb": "...", "additional_condition": "...", "noun": "...", "flavor_profile": "..."}}"""


# ---------------------------------------------------------------------------
# Prediction normalization — snap free-form model text onto the app vocab
# ---------------------------------------------------------------------------

def _slug(value):
    return re.sub(r"[\s_]+", "-", str(value).strip().lower())


def normalize_scenario(value):
    v = _slug(value)
    if v in APP.SCENARIOS:
        return v
    if "not" in v or v in ("none", ""):
        return "not-eating"
    if "possibl" in v or "maybe" in v:
        return "possibly-eating"
    if "eat" in v or "drink" in v:
        return "eating"
    return "possibly-eating"


def normalize_verb(value):
    v = str(value).strip().lower()
    if v in APP.VERBS:
        return v
    who = "human-fed" if ("human" in v or "fed by" in v or "another" in v) else \
          "self-fed" if "self" in v else ""
    how = "drink" if ("drink" in v or "sip" in v or "cup" in v or "bottle" in v) else \
          "utensil" if ("utensil" in v or "spoon" in v or "fork" in v) else \
          "hand" if "hand" in v else ""
    if who and how and f"{who}, {how}" in APP.VERBS:
        return f"{who}, {how}"
    if who:
        return f"{who}, other"
    return "unclear"


def normalize_condition(value):
    v = _slug(value)
    if v in APP.ADDITIONAL_CONDITIONS:
        return v
    if "bite" in v:
        return "actively-taking-a-bite"
    if "non-eating" in v or "non-food" in v:
        return "non-eating-setting"
    if "eating" in v or "meal" in v or "kitchen" in v:
        return "eating-setting"
    return "n/a"


def normalize_flavor(value):
    v = _slug(value)
    if v in APP.FLAVOR_PROFILES:
        return v
    for opt in APP.FLAVOR_PROFILES:
        if opt != "n/a" and opt in v:
            return opt
    return "unclear"


def normalize_prediction(raw):
    scenario = normalize_scenario(raw.get("scenario", ""))
    if scenario == "not-eating":
        return {"scenario": scenario, "verb": "", "additional_condition": "",
                "noun": "", "flavor_profile": ""}
    return {
        "scenario": scenario,
        "verb": normalize_verb(raw.get("verb", "")),
        "additional_condition": normalize_condition(raw.get("additional_condition", "")),
        "noun": str(raw.get("noun", "")).strip().lower() or "unclear",
        "flavor_profile": normalize_flavor(raw.get("flavor_profile", "")),
    }


def parse_model_json(response):
    match = re.search(r"\{.*?\}", response, re.DOTALL)
    if not match:
        return None
    text = match.group(0)
    for candidate in (text, text.replace("'", '"')):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Inference (one shard = one GPU = one model copy)
# ---------------------------------------------------------------------------

def load_model_and_processor():
    # This environment has no torchcodec and torchvision no longer exposes
    # read_video, so route transformers' "torchvision" decoder slot to PyAV
    # (same fix as qwen_taste_classifier.py).
    import transformers.video_utils as video_utils
    if hasattr(video_utils, "VIDEO_DECODERS") and hasattr(video_utils, "read_video_pyav"):
        video_utils.VIDEO_DECODERS["torchvision"] = video_utils.read_video_pyav

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    common_kwargs = dict(dtype="auto", device_map={"": 0}, trust_remote_code=True)
    try:
        from transformers import Qwen3VLMoeForConditionalGeneration
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(MODEL_ID, **common_kwargs)
    except ImportError:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, **common_kwargs)
    return model, processor


def annotate_clip(model, processor, clip_file):
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": os.path.join(APP.CLIP_DIR, clip_file)},
            {"type": "text", "text": ANNOTATION_PROMPT},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs={"num_frames": NUM_FRAMES, "fps": None},
    ).to(model.device)

    output_ids = model.generate(**inputs, max_new_tokens=160, do_sample=False)
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def shard_output_path(shard):
    return os.path.join(HOME, f"qwen_taste_predictions_shard{shard}.jsonl")


def run_shard(shard, num_shards):
    tag = f"[shard {shard}]"
    todo = [
        (i + 1, fname) for i, fname in enumerate(ASSIGNED_FILENAMES)
        if i % num_shards == shard
    ]
    if LIMIT:
        todo = todo[:LIMIT]

    out_path = shard_output_path(shard)
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(line)["file_name"] for line in f if line.strip()}
        print(f"{tag} resuming: {len(done)} clips already annotated", flush=True)

    remaining = [(n, f) for n, f in todo if f not in done]
    if not remaining:
        print(f"{tag} nothing to do", flush=True)
        return

    print(f"{tag} loading {MODEL_ID} ...", flush=True)
    t0 = time.time()
    model, processor = load_model_and_processor()
    print(f"{tag} loaded in {time.time() - t0:.1f}s, {len(remaining)} clips to annotate", flush=True)

    with open(out_path, "a", encoding="utf-8") as out:
        for count, (clip_number, fname) in enumerate(remaining, start=1):
            t0 = time.time()
            try:
                response = annotate_clip(model, processor, fname)
            except Exception as e:  # keep the run alive; a bad clip scores as a miss
                print(f"{tag} ERROR on {fname}: {e}", flush=True)
                response = ""
            raw = parse_model_json(response)
            row = {"clip_id": clip_number, "file_name": fname,
                   "parse_ok": raw is not None, "raw_response": response}
            row.update(normalize_prediction(raw or {}))
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(f"{tag} {count}/{len(remaining)} #{clip_number} {fname} -> "
                  f"{row['scenario']} | {row['verb']} | {row['noun']} | "
                  f"{row['additional_condition']} | {row['flavor_profile']} "
                  f"({time.time() - t0:.1f}s)", flush=True)


def merge_shards():
    rows = {}
    for path in sorted(glob.glob(SHARD_GLOB)):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    rows[row["file_name"]] = row

    ordered = []
    for i, fname in enumerate(ASSIGNED_FILENAMES):
        if fname in rows:
            row = rows[fname]
            ordered.append({
                "clip_id": i + 1,
                "file_name": fname,
                "scenario": row["scenario"],
                "verb": row["verb"],
                "noun": row["noun"],
                "additional_condition": row["additional_condition"],
                "flavor_profile": row["flavor_profile"],
                "parse_ok": row.get("parse_ok", True),
            })

    payload = {
        "model": MODEL_ID,
        "annotator_id": ANNOTATOR_ID,
        "annotator_name": APP.ANNOTATORS.get(ANNOTATOR_ID, "?"),
        "clip_dir": APP.CLIP_DIR,
        "num_frames": NUM_FRAMES,
        "assigned_clip_count": len(ASSIGNED_FILENAMES),
        "completed_count": len(ordered),
        "annotations": ordered,
    }
    with open(PREDICTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Merged {len(ordered)} predictions -> {PREDICTIONS_FILE}")
    return payload


def run_all_gpus():
    import torch
    n_gpus = max(torch.cuda.device_count(), 1)
    print(f"Launching {n_gpus} shard worker(s), {len(ASSIGNED_FILENAMES)} clips total")

    procs = []
    for shard in range(n_gpus):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard))
        procs.append(subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "run-shard",
             str(shard), str(n_gpus)],
            env=env,
        ))
        # Concurrent unauthenticated hub-metadata lookups against the shared
        # HF cache can race and crash one worker; stagger the starts.
        time.sleep(15)
    failed = [p.wait() for p in procs]
    if any(failed):
        print(f"WARNING: shard exit codes {failed}; merging whatever finished")
    merge_shards()


# ---------------------------------------------------------------------------
# Analysis: accuracy metrics, noun cosine similarity, 7 graphs
# ---------------------------------------------------------------------------

# Data-viz reference palette (validated) — light surface
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e8e7e4"
BLUE = "#2a78d6"   # categorical slot 1 / sequential hue
AQUA = "#1baf7a"   # categorical slot 2
SEQ_RAMP = ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#0d366b"]

EATING_SCENARIOS = ("eating", "possibly-eating")


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "text.color": INK,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
        "figure.dpi": 150,
    })
    return plt


def load_pairs():
    gt_data = json.load(open(GROUND_TRUTH_FILE, encoding="utf-8"))
    pred_data = json.load(open(PREDICTIONS_FILE, encoding="utf-8"))
    gt_by_file = {}
    for row in gt_data["annotations"]:
        gt_by_file[row.get("filename") or row.get("file_name")] = row

    pairs = []
    for pred in pred_data["annotations"]:
        gt = gt_by_file.get(pred["file_name"])
        if gt:
            pairs.append((gt, pred))
    print(f"Comparing {len(pairs)} clips "
          f"({len(gt_by_file)} ground truth, {len(pred_data['annotations'])} predicted)")
    return pairs


def field_values(pairs, field, subset="all"):
    """(gt_value, pred_value) lists for one field; empty -> '(none)'."""
    y_true, y_pred = [], []
    for gt, pred in pairs:
        if subset == "eating" and gt["scenario"] not in EATING_SCENARIOS:
            continue
        y_true.append(str(gt.get(field, "")).strip().lower() or "(none)")
        y_pred.append(str(pred.get(field, "")).strip().lower() or "(none)")
    return y_true, y_pred


def accuracy(y_true, y_pred):
    if not y_true:
        return float("nan")
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def plot_confusion(plt, y_true, y_pred, title, path, xlabel="Qwen prediction",
                   ylabel="Human (annotator 5)"):
    from matplotlib.colors import LinearSegmentedColormap
    from sklearn.metrics import confusion_matrix

    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cmap = LinearSegmentedColormap.from_list("seqblue", SEQ_RAMP)

    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(labels)), max(5, 0.9 * len(labels))))
    ax.grid(False)
    im = ax.imshow(cm, cmap=cmap, vmin=0)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    thresh = cm.max() * 0.55 if cm.max() else 1
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="#ffffff" if cm[i, j] > thresh else INK,
                        fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    acc = accuracy(y_true, y_pred)
    ax.set_title(f"{title}\naccuracy {acc:.1%}  (n={len(y_true)})")
    fig.colorbar(im, ax=ax, shrink=0.8, label="clip count")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return acc


def noun_cosine_similarities(pairs):
    """Cosine similarity between GT and predicted noun, on clips where the
    human wrote a noun (GT scenario is eating/possibly-eating and noun set)."""
    from sentence_transformers import SentenceTransformer

    rows = []
    for gt, pred in pairs:
        gt_noun = str(gt.get("noun", "")).strip().lower()
        pred_noun = str(pred.get("noun", "")).strip().lower()
        if gt["scenario"] in EATING_SCENARIOS and gt_noun:
            rows.append((gt_noun, pred_noun or "(none)"))

    model = SentenceTransformer(EMBED_MODEL_ID)
    gt_emb = model.encode([r[0] for r in rows], normalize_embeddings=True)
    pred_emb = model.encode([r[1] for r in rows], normalize_embeddings=True)
    sims = (gt_emb * pred_emb).sum(axis=1)
    return [(g, p, float(s)) for (g, p), s in zip(rows, sims)]


def analyze():
    import numpy as np
    plt = setup_matplotlib()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pairs = load_pairs()

    summary = {"model": MODEL_ID, "n_clips": len(pairs)}

    # --- field accuracies -------------------------------------------------
    scen_t, scen_p = field_values(pairs, "scenario")
    binary_t = [("eating" if s in EATING_SCENARIOS else "not-eating") for s in scen_t]
    binary_p = [("eating" if s in EATING_SCENARIOS else "not-eating") for s in scen_p]
    verb_t, verb_p = field_values(pairs, "verb", subset="eating")
    cond_t, cond_p = field_values(pairs, "additional_condition", subset="eating")
    flav_t, flav_p = field_values(pairs, "flavor_profile", subset="eating")
    noun_t, noun_p = field_values(pairs, "noun", subset="eating")

    sims = noun_cosine_similarities(pairs)
    sim_values = [s for _, _, s in sims]
    mean_sim = float(np.mean(sim_values)) if sim_values else float("nan")

    metrics = [
        ("Scenario (3-way)", accuracy(scen_t, scen_p), len(scen_t)),
        ("Eating vs not (binary)", accuracy(binary_t, binary_p), len(binary_t)),
        ("Verb*", accuracy(verb_t, verb_p), len(verb_t)),
        ("Condition*", accuracy(cond_t, cond_p), len(cond_t)),
        ("Flavor profile*", accuracy(flav_t, flav_p), len(flav_t)),
        ("Noun exact match*", accuracy(noun_t, noun_p), len(noun_t)),
        ("Noun mean cosine sim*", mean_sim, len(sim_values)),
    ]
    summary["metrics"] = {name: {"value": val, "n": n} for name, val, n in metrics}

    # --- graph 1: overall per-field scores --------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = [m[0] for m in metrics][::-1]
    vals = [m[1] for m in metrics][::-1]
    bars = ax.barh(names, vals, height=0.6, color=BLUE)
    for bar, val in zip(bars, vals):
        ax.text(val + 0.015, bar.get_y() + bar.get_height() / 2, f"{val:.1%}",
                va="center", color=INK, fontweight="bold", fontsize=10)
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_title("Qwen3-VL-30B vs annotator 5 — per-field agreement\n"
                 "* scored on human eating / possibly-eating clips only",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "1_overall_accuracy.png"))
    plt.close(fig)

    # --- graphs 2-5: confusion matrices ------------------------------------
    plot_confusion(plt, scen_t, scen_p, "Scenario confusion",
                   os.path.join(RESULTS_DIR, "2_scenario_confusion.png"))
    plot_confusion(plt, verb_t, verb_p, "Verb confusion (human-eating clips)",
                   os.path.join(RESULTS_DIR, "3_verb_confusion.png"))
    plot_confusion(plt, cond_t, cond_p, "Additional-condition confusion (human-eating clips)",
                   os.path.join(RESULTS_DIR, "4_condition_confusion.png"))
    plot_confusion(plt, flav_t, flav_p, "Flavor-profile confusion (human-eating clips)",
                   os.path.join(RESULTS_DIR, "5_flavor_confusion.png"))

    # --- graph 6: noun semantic similarity ---------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(sim_values, bins=np.arange(0, 1.05, 0.05), color=BLUE, edgecolor=SURFACE)
    ax.axvline(mean_sim, color=INK, linestyle="--", linewidth=1.5)
    ax.text(mean_sim + 0.01, ax.get_ylim()[1] * 0.92, f"mean {mean_sim:.2f}", color=INK)
    exact = sum(1 for g, p, _ in sims if g == p)
    ax.set_xlabel(f"cosine similarity, {EMBED_MODEL_ID.split('/')[-1]} embeddings")
    ax.set_ylabel("clip count")
    ax.set_title(f"Noun semantic similarity, Qwen vs human "
                 f"(n={len(sims)}, exact match {exact})")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "6_noun_cosine_similarity.png"))
    plt.close(fig)

    # --- graph 7: scenario label distribution, human vs Qwen ---------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = ["eating", "possibly-eating", "not-eating"]
    x = np.arange(len(labels))
    gt_counts = [scen_t.count(l) for l in labels]
    pred_counts = [scen_p.count(l) for l in labels]
    b1 = ax.bar(x - 0.19, gt_counts, width=0.36, color=BLUE, label="Human (annotator 5)")
    b2 = ax.bar(x + 0.19, pred_counts, width=0.36, color=AQUA, label="Qwen3-VL-30B")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6,
                    str(int(bar.get_height())), ha="center", color=INK, fontsize=10)
    ax.set_xticks(x, labels)
    ax.set_ylabel("clip count")
    ax.set_title("Scenario label distribution — human vs Qwen")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "7_scenario_distribution.png"))
    plt.close(fig)

    # --- report -------------------------------------------------------------
    lines = [f"Qwen taste-annotation benchmark — {MODEL_ID}",
             f"Ground truth: {GROUND_TRUTH_FILE}",
             f"Clips compared: {len(pairs)}", ""]
    for name, val, n in metrics:
        lines.append(f"  {name:<28s} {val:6.1%}  (n={n})")
    lines.append("")
    lines.append("Noun pairs (human -> qwen, cosine):")
    for g, p, s in sorted(sims, key=lambda r: r[2]):
        lines.append(f"  {s:5.2f}  {g!r:<18s} -> {p!r}")
    report = "\n".join(lines)
    with open(os.path.join(RESULTS_DIR, "summary_report.txt"), "w", encoding="utf-8") as f:
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
        merge_shards()
    elif mode == "analyze":
        analyze()
    elif mode == "all":
        run_all_gpus()
        analyze()
    else:
        sys.exit(f"Unknown mode {mode!r}. Use: run | analyze | merge | all")


if __name__ == "__main__":
    main()
