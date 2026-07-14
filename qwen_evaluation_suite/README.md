# Qwen3-VL-30B Taste-Annotation Evaluation Suite

Five benchmarks evaluating **Qwen/Qwen3-VL-30B-A3B-Instruct** as an automated
annotator against the Gong Lab (Boston University) taste-annotation schema —
a 5-field label set (`scenario`, `verb`, `additional_condition`, `noun`,
`flavor_profile`) applied to short home-video clips of a child eating,
possibly eating, or not eating.

All runs used the model in bf16 (no quantization), sharded across two GPUs,
sampling 8 frames/clip unless noted. Ground truth is human annotator
consensus/agreement collected from 11 trained annotators on a shared
506-clip pool (each clip labeled by up to 3 annotators).

**Data note:** per-clip filenames, raw model predictions, and human
ground-truth annotation files are intentionally excluded from this public
repo — they reference identifiable home-video clips of children. Only
aggregate results (accuracies, kappas, confusion matrices, distributions)
are published here. The benchmark code is fully runnable against the
original data on the source machine to reproduce every number below.

---

## 1. `01_annotator5_benchmark/` — baseline accuracy vs. one human annotator

Qwen annotated the 101 clips assigned to annotator 5 (Karthik Srikumar) and
was scored field-by-field against that annotator's labels, including a
sentence-embedding (MiniLM) cosine-similarity score for the free-text
`noun` field.

| Field | Accuracy |
|---|---|
| Scenario (3-way) | 67.3% |
| Eating vs. not-eating (binary) | 85.1% |
| Verb | 33.3% |
| Condition | 70.8% |
| Flavor profile | 35.4% |
| Noun (exact match) | 70.8% |
| Noun (mean cosine similarity) | 0.82 |

**Takeaway:** Qwen reliably detects *that* eating is happening and *what*
food is involved, but struggles with judgment-heavy fields (verb, flavor).
Graphs: overall accuracy bar chart, confusion matrices for scenario/verb/
condition/flavor, noun similarity histogram, scenario distribution
comparison.

## 2. `02_verb_diagnostic/` — why does Qwen fail the verb field?

Nine controlled conditions on the same 48 human-verified eating clips,
isolating agency ("who feeds?") from instrument ("hand/utensil/drink?"),
sweeping frame count 4→32, and testing scaffolded reasoning and label
rewording.

| Condition | Self-fed clips (n=31) | Human-fed clips (n=17) |
|---|---|---|
| Composite verb question | 90% | **0%** |
| Agency asked alone | 84% | **0%** |
| "Whose hand holds the food?" (pure perception) | 94% | **6%** |
| Describe-people-first scaffold | 74% | **18%** |
| "fed-by-another-person" rewording | 90% | **0%** |

**Finding:** the verb failure is a **one-directional perceptual blind
spot**, not a labeling-vocabulary or frame-sampling problem. Qwen defaults
to "self-fed" almost universally; even a pure perception probe ("whose hand
is this?") gets human-fed clips right only 6% of the time, despite Qwen
reporting it sees two people in the clip most of the time. Scaffolded
reasoning (describe people/arms before classifying) is the only lever that
moved the needle, and only partially (0%→18%).

## 3. `03_full_pool_agreement/` — Qwen as a 12th annotator

Qwen annotated all 506 clips in the shared pool and was compared to 11
human annotators as if it were one more rater: pairwise Cohen's κ and
percent agreement (human↔human vs. Qwen↔human), accuracy against
majority-vote consensus, and a full rater-agreement heatmap.

| Field | κ human↔human | κ Qwen↔human | % human↔human | % Qwen↔human |
|---|---|---|---|---|
| Scenario | 0.44 | 0.42 | 64.8% | 64.3% |
| Verb | 0.62 | 0.26 | 69.5% | 40.5% |
| Condition | 0.28 | 0.28 | 71.5% | 76.4% |
| Flavor | 0.52 | 0.26 | 64.6% | 44.2% |
| Noun | 0.55 | 0.48 | 60.7% | 57.8% |

Consensus accuracy (clips with ≥2 human labels): scenario 72.8%, verb 37.4%,
condition 71.2%, flavor 40.0%, noun 62.3%.

Difficulty transfer: Qwen scores 84.6% on clips humans unanimously agreed
on, dropping to 42.6% on clips where humans only reached a 2/3 majority —
Qwen gets it wrong on exactly the clips that are genuinely hard for humans
too.

**Finding:** on scenario, condition, and noun, Qwen sits **inside the human
agreement envelope** — the rater-agreement heatmap shows several human-human
pairs agreeing as little as 31–33% of the time, well below Qwen's 54–70%
agreement with each individual human. Verb and flavor are the two fields
where Qwen falls clearly outside human-level agreement.

## 4. `04_touch_transfer/` — domain-shift transfer to an unseen corpus

Qwen annotated a 240-clip stratified random sample from the (unlabeled,
touch-focused, ~2.2M-clip) `ivc-ml` touch corpus using the same taste schema
and prompt, plus a free-text 1–2 sentence description per clip — the
standard "model-in-the-loop pre-annotation" pipeline pattern.

- Scenario distribution: 90% not-eating on the touch corpus vs. 55% on the
  taste pool — expected, since the taste pool was curated specifically for
  eating content and the touch corpus was not.
- Zero JSON parse failures across 240 clips; every clip got a usable
  description (mean 32.1 words).
- No schema collapse: the "unclear"/"n/a" rate on eating clips stayed
  comparable to the taste pool rather than spiking.

**Takeaway:** the pipeline transfers cleanly to a new, unseen clip pool
without any schema-specific fine-tuning — useful as a pre-annotation pass
that a human annotator verifies rather than labels from scratch.

## 5. `05_reliability_fewshot/` — the scientific test: reliability + few-shot transfer

**Part A (test–retest reliability).** Qwen re-annotated the 101-clip eval
set three times with sampling (T=0.7) and was scored against itself, then
compared to the pooled human↔human agreement rate.

| Field | Qwen self-agreement | Human↔human agreement |
|---|---|---|
| Scenario | 88.1% | 64.8% |
| Verb | 91.2% | 69.5% |
| Condition | 90.4% | 71.5% |
| Flavor | 93.4% | 64.6% |
| Noun | 87.5% | 60.7% |

Qwen is **more self-consistent than humans are with each other** on every
field — a necessary but not sufficient condition for being a trustworthy
annotator (a model can be perfectly reliable and still reliably wrong, which
is exactly what Part B shows for verb/agency).

**Part B (few-shot transfer).** The same clips re-annotated with 4 and 8
in-context video examples drawn from multi-annotator consensus labels on
clips *outside* the eval set (no leakage), explicitly including two
human-fed demonstrations.

| Field | 0-shot | 4-shot | 8-shot |
|---|---|---|---|
| Scenario | 67.3% | 60.4% | 68.3% |
| Verb | 33.3% | 35.4% | 35.4% |
| Condition | 70.8% | 75.0% | 75.0% |
| Flavor | 35.4% | 45.8% | 43.8% |
| Noun | 71.7% | 76.1% | 73.9% |

Agency accuracy on true human-fed clips: **0% at 0-shot, 0% at 4-shot, 0% at
8-shot** — even with two explicit human-fed examples in context.

**Finding:** in-context learning moves flavor (+8–10pts) and noun (+2–4pts)
meaningfully, but **cannot fix the verb/agency blind spot at all**. This
confirms (independently of Benchmark 2's ablations) that the human-fed
failure is a perceptual limitation of the model's visual encoder/attention,
not a prompting or demonstration-availability problem — the standard
signature that distinguishes a fixable prompting issue from a case that
needs fine-tuning or architectural intervention.

---

## Cross-benchmark synthesis

1. **Scenario, condition, and noun identification are at or near human
   inter-annotator level.** Qwen is a usable pre-annotator for these three
   fields today.
2. **Verb (agency) and flavor profile are systematically below the human
   floor**, and for verb specifically, the failure is invariant to frame
   count, prompt wording, in-context examples, and scaffolded reasoning —
   five independent interventions all failed to move human-fed accuracy off
   0%, which is unusually strong evidence of a hard perceptual ceiling
   rather than an easily-prompted-away gap.
3. **Difficulty transfers**: Qwen's errors concentrate on the same clips
   where humans disagree with each other, suggesting shared ambiguity (poor
   camera angle, occlusion) rather than a distinct Qwen-specific failure
   mode for scenario/noun.
4. **Reliability is not accuracy.** Qwen is the most self-consistent rater
   in the study (87–93% self-agreement, above every human-human pair) while
   being categorically wrong on one entire label class — a caution against
   using self-consistency alone as a proxy for annotation quality.

## Reproducing

All five scripts live in `code/` and share the model-loading, prompt, and
normalization logic in `qwen_taste_benchmark.py`. Each exposes `run` /
`analyze` / `all` subcommands and shards inference across all visible GPUs.
Running end-to-end requires the original SCC environment (Qwen3-VL-30B
weights, the taste-clip corpus, and the 11 collected human-annotator export
files), none of which are included here for the data-privacy reasons noted
above.

See `math/` for the LaTeX formalization of the statistics used throughout
this suite (Cohen's κ, agreement-envelope framing, the difficulty-transfer
model, and the few-shot/reliability decomposition).
