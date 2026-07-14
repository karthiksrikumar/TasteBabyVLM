#!/usr/bin/env python3
"""
Diagnostic benchmark: WHY does Qwen3-VL-30B fail the VERB field?

The main benchmark (qwen_taste_benchmark.py) showed verb accuracy of 33%
with a very structured error: instrument (hand/utensil/drink) mostly right,
agency (self-fed vs human-fed) mostly wrong, and always in the same
direction (human-fed clips called self-fed). This script separates the
competing explanations by running the same 48 human-verified eating clips
through 9 controlled conditions:

  H1 "agency binding"   — VLMs can't attribute a hand/arm to a person.
       -> conditions: agency_8f (agency-only question), whose_hand_8f
          (pure-perception probe: "whose hand holds the food?")
  H2 "instrument is the easy part" — control for H1.
       -> condition: instrument_8f
  H3 "temporal undersampling" — 8 frames miss the feeder's arm entering.
       -> conditions: composite_4f / 8f / 16f / 32f frame sweep
  H4 "attention scaffolding fixes it" — model can see it but doesn't look.
       -> condition: scaffold_8f (describe people+hands first, then decide)
  H5 "label wording" — 'human-fed' is ambiguous (the eater is human too).
       -> condition: reworded_8f ('fed-by-another-person' vocabulary)

Ground truth for agency/instrument is derived from annotator 5's verb
("self-fed, hand" -> agency=self, instrument=hand).

Usage (GPU session):
    source /usr4/spclpgm/srikumar/envs/qwen/bin/activate
    python3 qwen_verb_diagnostic.py all        # run 9 conditions x 48 clips, then analyze
    python3 qwen_verb_diagnostic.py analyze    # re-analyze saved responses
    QWEN_VERB_LIMIT=1 python3 qwen_verb_diagnostic.py run   # smoke test
"""

import glob
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen_taste_benchmark as base

HOME = base.HOME
LIMIT = int(os.environ.get("QWEN_VERB_LIMIT", "0"))
RESPONSES_GLOB = os.path.join(HOME, "qwen_verb_diagnostic_shard*.jsonl")
RESULTS_DIR = os.path.join(HOME, "qwen_verb_diagnostic_results")

VERB_VOCAB = json.dumps(sorted(base.APP.VERBS))

COMPOSITE_PROMPT = f"""Watch this short home-video clip. A person (usually a young child) is eating or possibly eating. Question: how is the food getting to their mouth? Choose exactly one option from this list: {VERB_VOCAB}. "self-fed" means the person eating moves the food to their own mouth; "human-fed" means ANOTHER person (e.g. a parent) delivers the food to the eater's mouth; "unclear" if you can't tell from the clip.
Respond with ONLY one-line JSON: {{"verb": "..."}}"""

AGENCY_PROMPT = """Watch this short home-video clip. A person (usually a young child) is eating or possibly eating. Question: WHO delivers the food to the eater's mouth?
- "self": the eater moves the food to their own mouth with their own hand/arm.
- "other": a different person (e.g. a parent or caregiver) moves the food to the eater's mouth.
- "unclear": you cannot tell from the clip.
Respond with ONLY one-line JSON: {"agency": "..."}"""

INSTRUMENT_PROMPT = """Watch this short home-video clip. A person (usually a young child) is eating or possibly eating. Question: WHAT carries the food to the mouth?
- "hand": fingers/hand holding the food directly.
- "utensil": spoon, fork, or similar tool.
- "drink": cup, bottle, sippy cup, or other drinking container.
- "other": something else.
- "unclear": you cannot tell from the clip.
Respond with ONLY one-line JSON: {"instrument": "..."}"""

WHOSE_HAND_PROMPT = """Watch this short home-video clip carefully. It shows a person (usually a young child) with food. Answer two perception questions:
1. How many distinct people are visible or partially visible in the clip? Count anyone whose body part (a hand, an arm, a torso) appears, even briefly from off-screen.
2. The hand or arm that is holding/moving the food: does it belong to the person eating, or to someone else? Answer "eater", "someone-else", "no-hand-visible", or "unclear".
Respond with ONLY one-line JSON: {"people_visible": <number>, "hand_owner": "..."}"""

SCAFFOLD_PROMPT = f"""Watch this short home-video clip. A person (usually a young child) is eating or possibly eating.
Step 1 — look closely and describe: how many people are visible (including partial arms/hands entering from off-screen), where the food is, and whose hand or arm is holding or moving the food (follow the arm: does it belong to the eater's body, or does it enter the frame from another person's position?).
Step 2 — based ONLY on your step-1 observations, classify how the food gets to the eater's mouth. Choose one option from: {VERB_VOCAB}. Remember: "human-fed" means another person delivers the food to the eater's mouth.
Respond with ONLY one-line JSON: {{"description": "...", "verb": "..."}}"""

REWORD_VOCAB = ["feeds-themselves, hand", "feeds-themselves, utensil",
                "feeds-themselves, drink", "feeds-themselves, other",
                "fed-by-another-person, hand", "fed-by-another-person, utensil",
                "fed-by-another-person, other", "unclear"]
REWORDED_PROMPT = f"""Watch this short home-video clip. A person (usually a young child) is eating or possibly eating. Question: how is the food getting to their mouth? Choose exactly one option from this list: {json.dumps(REWORD_VOCAB)}. "feeds-themselves" means the eater moves the food to their own mouth; "fed-by-another-person" means a parent/caregiver/other person delivers the food to the eater's mouth; "unclear" if you can't tell.
Respond with ONLY one-line JSON: {{"verb": "..."}}"""

# (name, num_frames, prompt, max_new_tokens)
CONDITIONS = [
    ("composite_4f", 4, COMPOSITE_PROMPT, 60),
    ("composite_8f", 8, COMPOSITE_PROMPT, 60),
    ("composite_16f", 16, COMPOSITE_PROMPT, 60),
    ("composite_32f", 32, COMPOSITE_PROMPT, 60),
    ("agency_8f", 8, AGENCY_PROMPT, 60),
    ("instrument_8f", 8, INSTRUMENT_PROMPT, 60),
    ("whose_hand_8f", 8, WHOSE_HAND_PROMPT, 80),
    ("scaffold_8f", 8, SCAFFOLD_PROMPT, 350),
    ("reworded_8f", 8, REWORDED_PROMPT, 60),
]


def eval_clips():
    """(clip_number, file_name, gt_verb) for clips the human gave a verb."""
    gt = json.load(open(base.GROUND_TRUTH_FILE, encoding="utf-8"))
    rows = []
    for i, row in enumerate(gt["annotations"], start=1):
        if str(row.get("verb", "")).strip():
            rows.append((i, row.get("filename") or row.get("file_name"), row["verb"]))
    return rows


def split_verb(verb):
    """'self-fed, hand' -> ('self', 'hand'); 'unclear' -> ('unclear','unclear')."""
    v = str(verb).strip().lower()
    if v in ("", "unclear"):
        return "unclear", "unclear"
    who, _, how = v.partition(",")
    agency = "self" if "self" in who else "other" if "human" in who else "unclear"
    return agency, how.strip() or "unclear"


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def annotate(model, processor, clip_file, prompt, num_frames, max_new_tokens):
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": os.path.join(base.APP.CLIP_DIR, clip_file)},
            {"type": "text", "text": prompt},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True,
        return_tensors="pt", processor_kwargs={"num_frames": num_frames, "fps": None},
    ).to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def run_shard(shard, num_shards):
    tag = f"[shard {shard}]"
    clips = eval_clips()[shard::num_shards]
    if LIMIT:
        clips = clips[:LIMIT]

    out_path = os.path.join(HOME, f"qwen_verb_diagnostic_shard{shard}.jsonl")
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            done = {(json.loads(l)["file_name"], json.loads(l)["condition"])
                    for l in f if l.strip()}
        print(f"{tag} resuming: {len(done)} responses already saved", flush=True)

    total = len(clips) * len(CONDITIONS) - len(done)
    print(f"{tag} loading model, {total} generations to go ...", flush=True)
    model, processor = base.load_model_and_processor()

    n = 0
    with open(out_path, "a", encoding="utf-8") as out:
        for clip_number, fname, gt_verb in clips:
            for cond, frames, prompt, max_tok in CONDITIONS:
                if (fname, cond) in done:
                    continue
                t0 = time.time()
                try:
                    response = annotate(model, processor, fname, prompt, frames, max_tok)
                except Exception as e:
                    print(f"{tag} ERROR {fname} {cond}: {e}", flush=True)
                    response = ""
                out.write(json.dumps({
                    "clip_number": clip_number, "file_name": fname,
                    "gt_verb": gt_verb, "condition": cond, "response": response,
                }) + "\n")
                out.flush()
                n += 1
                print(f"{tag} {n}/{total} #{clip_number} {cond} "
                      f"({time.time() - t0:.1f}s): {response[:90]!r}", flush=True)


def run_all_gpus():
    import torch
    n_gpus = max(torch.cuda.device_count(), 1)
    n_clips = len(eval_clips())
    print(f"Verb diagnostic: {n_clips} clips x {len(CONDITIONS)} conditions "
          f"on {n_gpus} GPU(s)")
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

def parse_response(cond, response):
    """-> dict with any of: verb, agency, instrument, people_visible, hand_owner."""
    raw = base.parse_model_json(response) or {}
    out = {}
    if cond.startswith(("composite", "scaffold", "reworded")):
        verb = str(raw.get("verb", "")).strip().lower()
        if cond.startswith("reworded"):
            verb = (verb.replace("feeds-themselves", "self-fed")
                        .replace("fed-by-another-person", "human-fed"))
        out["verb"] = base.normalize_verb(verb)
        out["agency"], out["instrument"] = split_verb(out["verb"])
    elif cond.startswith("agency"):
        a = str(raw.get("agency", "")).strip().lower()
        out["agency"] = a if a in ("self", "other", "unclear") else \
            "other" if ("other" in a or "another" in a or "parent" in a) else \
            "self" if "self" in a else "unclear"
    elif cond.startswith("instrument"):
        i = str(raw.get("instrument", "")).strip().lower()
        out["instrument"] = i if i in ("hand", "utensil", "drink", "other", "unclear") \
            else base.normalize_verb(f"self-fed, {i}").split(", ")[-1]
    elif cond.startswith("whose_hand"):
        try:
            out["people_visible"] = int(raw.get("people_visible", 0))
        except (TypeError, ValueError):
            out["people_visible"] = None
        h = str(raw.get("hand_owner", "")).strip().lower()
        out["hand_owner"] = h
        out["agency"] = "self" if "eater" in h else \
            "other" if ("else" in h or "other" in h) else "unclear"
    return out


def load_responses():
    rows = {}
    for path in sorted(glob.glob(RESPONSES_GLOB)):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    rows[(r["file_name"], r["condition"])] = r
    return rows


def analyze():
    import numpy as np
    plt = base.setup_matplotlib()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    clips = eval_clips()
    responses = load_responses()
    print(f"{len(clips)} clips, {len(responses)} saved responses")

    # per-condition parsed predictions aligned to clips
    parsed = {}   # cond -> {file_name: {...}}
    for cond, *_ in CONDITIONS:
        parsed[cond] = {}
        for _, fname, _ in clips:
            r = responses.get((fname, cond))
            if r:
                parsed[cond][fname] = parse_response(cond, r["response"])

    gt = {fname: dict(zip(("agency", "instrument"), split_verb(verb)), verb=verb.lower())
          for _, fname, verb in clips}

    def acc(cond, field, subset=None):
        pairs = [(gt[f][field], p[field]) for f, p in parsed[cond].items()
                 if field in p and (subset is None or gt[f]["agency"] == subset)]
        return (sum(t == pr for t, pr in pairs) / len(pairs) if pairs else float("nan"),
                len(pairs))

    summary = {}

    # --- graph 1: sub-skill decomposition ----------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars_spec = [
        ("Full verb\n(composite)", acc("composite_8f", "verb")[0]),
        ("Full verb\n(scaffolded)", acc("scaffold_8f", "verb")[0]),
        ("Full verb\n(reworded)", acc("reworded_8f", "verb")[0]),
        ("Agency\n(from composite)", acc("composite_8f", "agency")[0]),
        ("Agency\n(asked alone)", acc("agency_8f", "agency")[0]),
        ("Agency\n(whose-hand probe)", acc("whose_hand_8f", "agency")[0]),
        ("Instrument\n(from composite)", acc("composite_8f", "instrument")[0]),
        ("Instrument\n(asked alone)", acc("instrument_8f", "instrument")[0]),
    ]
    xs = np.arange(len(bars_spec))
    vals = [v for _, v in bars_spec]
    ax.bar(xs, vals, width=0.62, color=base.BLUE)
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.02, f"{v:.0%}", ha="center", color=base.INK, fontweight="bold")
    ax.set_xticks(xs, [n for n, _ in bars_spec], fontsize=8.2, rotation=20, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_title("Verb sub-skill decomposition — where the verb score is lost (n=48)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "1_subskill_decomposition.png"))
    plt.close(fig)
    summary["subskills"] = {n.replace("\n", " "): v for n, v in bars_spec}

    # --- graph 2: frame-count sweep -----------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    frames = [4, 8, 16, 32]
    for field, color, label in [("verb", base.BLUE, "full verb"),
                                ("agency", base.AQUA, "agency part"),
                                ("instrument", "#4a3aa7", "instrument part")]:
        ys = [acc(f"composite_{n}f", field)[0] for n in frames]
        ax.plot(frames, ys, marker="o", markersize=7, linewidth=2, color=color, label=label)
        ax.annotate(label, (frames[-1], ys[-1]), xytext=(6, 0),
                    textcoords="offset points", color=color, fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_xticks(frames, [str(n) for n in frames])
    ax.set_xlabel("frames sampled from clip")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_title("H3 temporal undersampling — accuracy vs frames sampled")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "2_frame_sweep.png"))
    plt.close(fig)
    summary["frame_sweep"] = {f"{n}f": acc(f"composite_{n}f", "verb")[0] for n in frames}

    # --- graph 3: agency accuracy by ground-truth class (bias probe) --------
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    conds = [("composite_8f", "composite"), ("agency_8f", "agency-only"),
             ("whose_hand_8f", "whose-hand probe"), ("scaffold_8f", "scaffolded"),
             ("reworded_8f", "reworded")]
    xs = np.arange(len(conds))
    self_vals = [acc(c, "agency", subset="self")[0] for c, _ in conds]
    other_vals = [acc(c, "agency", subset="other")[0] for c, _ in conds]
    b1 = ax.bar(xs - 0.19, self_vals, width=0.36, color=base.BLUE,
                label="human said self-fed (n=31)")
    b2 = ax.bar(xs + 0.19, other_vals, width=0.36, color=base.AQUA,
                label="human said human-fed (n=17)")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{bar.get_height():.0%}", ha="center", color=base.INK, fontsize=9)
    ax.set_xticks(xs, [n for _, n in conds])
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_title("H1/H5 self-fed bias — agency accuracy split by true class")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "3_agency_bias_by_class.png"))
    plt.close(fig)

    # --- graph 4: whose-hand perception probe confusion ---------------------
    y_true = [gt[f]["agency"] for f in parsed["whose_hand_8f"]]
    y_pred = [p["agency"] for p in parsed["whose_hand_8f"].values()]
    base.plot_confusion(plt, y_true, y_pred,
                        "H1 perception probe — whose hand holds the food?",
                        os.path.join(RESULTS_DIR, "4_whose_hand_confusion.png"),
                        xlabel="Qwen (eater=self / someone-else=other)",
                        ylabel="Human verb (agency part)")

    # --- graph 5: people-visible count vs agency correctness ----------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    counts = {}
    for f, p in parsed["whose_hand_8f"].items():
        pv = p.get("people_visible")
        pv = "3+" if isinstance(pv, int) and pv >= 3 else str(pv) if pv else "?"
        ok = parsed["composite_8f"].get(f, {}).get("agency") == gt[f]["agency"]
        counts.setdefault(pv, [0, 0])[0 if ok else 1] += 1
    keys = sorted(counts, key=lambda k: (k == "?", k))
    xs = np.arange(len(keys))
    right = [counts[k][0] for k in keys]
    wrong = [counts[k][1] for k in keys]
    ax.bar(xs, right, width=0.55, color=base.BLUE, label="agency correct")
    ax.bar(xs, wrong, width=0.55, bottom=right, color="#e34948", label="agency wrong")
    for x, (r, w) in zip(xs, zip(right, wrong)):
        ax.text(x, r + w + 0.3, str(r + w), ha="center", color=base.INK, fontsize=10)
    ax.set_xticks(xs, keys)
    ax.set_xlabel("people Qwen says are visible in the clip")
    ax.set_ylabel("clip count")
    ax.set_title("Does Qwen even see the second person?", fontsize=12)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "5_people_visible_vs_agency.png"))
    plt.close(fig)

    # --- report --------------------------------------------------------------
    lines = ["Qwen verb diagnostic — sub-skill and ablation results", ""]
    for name, v in summary["subskills"].items():
        lines.append(f"  {name:<32s} {v:6.1%}")
    lines.append("")
    lines.append("Frame sweep (full verb): " + "  ".join(
        f"{k}={v:.1%}" for k, v in summary["frame_sweep"].items()))
    lines.append("")
    for c, n in conds:
        s, _ = acc(c, "agency", subset="self")
        o, _ = acc(c, "agency", subset="other")
        lines.append(f"  agency acc [{n:<18s}] self-fed clips {s:6.1%} | human-fed clips {o:6.1%}")
    report = "\n".join(lines)
    with open(os.path.join(RESULTS_DIR, "diagnostic_report.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
    with open(os.path.join(RESULTS_DIR, "diagnostic_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(report)
    print(f"\nGraphs + report written to {RESULTS_DIR}/")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode == "run-shard":
        run_shard(int(sys.argv[2]), int(sys.argv[3]))
    elif mode == "run":
        run_all_gpus()
    elif mode == "analyze":
        analyze()
    elif mode == "all":
        run_all_gpus()
        analyze()
    else:
        sys.exit(f"Unknown mode {mode!r}. Use: run | analyze | all")


if __name__ == "__main__":
    main()
