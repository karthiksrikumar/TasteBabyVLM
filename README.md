# TasteBabyVLM

Evaluating **Qwen3-VL-30B-A3B-Instruct** as an automated annotator on the taste-annotation schema for infant/child
home-video clips — how well it detects eating, identifies the food, judges
flavor, and (crucially) attributes *who* is feeding whom.

- **[`qwen_evaluation_suite/`](qwen_evaluation_suite/)** — five benchmarks
  (baseline accuracy, a verb-failure diagnostic, full-pool inter-annotator
  agreement, domain transfer to an unseen clip corpus, and a
  reliability/few-shot ablation), each with graphs, reports, and runnable
  code. Start with its own `README.md` for the full write-up and results.
- **[`math/`](math/)** — a LaTeX formalization of the statistics used
  throughout the suite (Cohen's κ and the agreement-envelope framing,
  consensus/confusion under structurally missing fields, a cosine-similarity
  soft-accuracy relaxation, classical test theory explaining why
  self-consistency isn't validity, and a significance test for the
  human-fed blind spot). Compiled PDF at `math/main.pdf`.

**Headline finding:** Qwen matches human-level agreement on *scenario*,
*condition*, and *noun* identification, but has a near-total,
prompting-resistant blind spot on *who* feeds whom (self-fed vs.
human-fed) — five independent interventions, including in-context
demonstrations, all fail to move accuracy on human-fed clips off ~0%.
