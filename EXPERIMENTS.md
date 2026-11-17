# Experiment checklist: Lost at the Switch

Ordered so each stage validates the next. Tick off as you go. Things marked
**DECIDE** are open design choices the proposal doesn't pin down.

## 0. Setup and correctness (before any real numbers)
- [ ] `pytest`: all 31 unit tests pass (CPU, no downloads)
- [ ] `make_toy_setup.py` + toy smoke run of every script (see README)
- [ ] HF access: Qwen3.5 0.8B/9B, Llama 3.2 1B/3B (gated) — **confirm exact IDs and base vs instruct match Paudel et al.**
- [ ] `sanity_check.py` on each real pair, on the GPU you will use:
  - [ ] shared tokenizer
  - [ ] crop vs recompute logits match → set `cache_mode` (Qwen3.5 may have non-croppable layers)
  - [ ] T=0 spec decoding == target greedy for γ = 2, 3, 4
  - [ ] record `c` (drafter/target forward time) per pair and GPU
- [ ] T>0 losslessness on the real pair: 5 seeds, compare output token distributions vs target-only sampling (proposal "Controls")

## 1. Data
- [ ] LinCE LID: spa-eng, hin-eng, nep-eng → `data/raw/lince/` (test labels are withheld, so evaluate on dev)
- [ ] `data_stats.py` on each: switches per sentence, tag counts (sanity vs LinCE paper)
- [ ] SentiMix (Hinglish, Spanglish) — check tag names, extend `TAG_MAP` in `lid.py`
- [ ] PHINC, MixMT 2022 (generation prompts), COMI-LINGUA (native-script Hinglish), Nepali-English sentiment test set
- [ ] FLORES-200 en/es/hi/ne devtest (monolingual controls)
- [ ] Transliteration: IndicXlit (separate env if deps clash); round-trip CER on 200 sentences/language; check against Dakshina
- [ ] Build native-script twins (mixed + fully native) for hin-eng and nep-eng
- [ ] Compare twins vs COMI-LINGUA Devanagari Hinglish ("likeness" check: **DECIDE** the metric, e.g. char n-gram overlap / tokenizer fertility)
- [ ] **DECIDE** generation tasks/prompts (continuation vs chat reply vs translate-into-Hinglish); `make_prompts.py` only does continuation

## 2. Baseline reproduction (Paudel et al. 2026) — do this before anything new
- [ ] B1 on Paudel's Nepali translation + story-generation splits, both pairs, γ ∈ {2,3,4}, top-k 100, top-p 0.9, 128 tokens → match their acceptance rates
  - check whether they report accepted/verified (`alpha`) or accepted/drafted (`draft_efficiency`); both are logged
- [ ] B2 n-gram drafter, n ∈ {2,3,4}, reproduce their bigram-is-fastest finding
- [ ] B3 distilled drafter, both variants (task-specific on target greedy outputs; general on existing text): `train_distill.py --lam 0 --mu 0 --beta-tv 0`
- [ ] Wall-clock tokens/sec at batch size 1 with vLLM speculative decoding (not in repo yet; vLLM's spec-decode config API changes between versions)

## 3. Measurement study (the core contribution)
- [ ] Exp 1 `run_positional.py`: expected acceptance (1−TV) by category (switch / near_switch / inside), per pair, per model pair
- [ ] Same, split by word_start (first subtoken vs continuation) → tokenization vs word-choice
- [ ] Exp 2 `run_specdec.py`: real α, TPT, expected speedup f on code-switched prompts, γ ∈ {2,3,4}, B1/B2/B3
- [ ] LID-tag generated outputs (MuRIL or trained tagger) → α at switch points in *generated* text
- [ ] Switching penalty: α on monolingual FLORES (en, es, hi, ne) vs code-switched
- [ ] Script × switching 2×2 (Table 2) for hin-eng and nep-eng; spa-eng as same-script control
- [ ] Resource level: plot α vs FineWeb2 size per language

## 4. Methods
- [ ] Approach A: sweep λ, μ (and β_TV); eval with `run_positional.py --drafter-ckpt` and `run_specdec.py --drafter-ckpt`
  - success = α at switch ↑ without α inside ↓ (check generalisation on held-out SentiMix/MixMT)
- [ ] Approach B: switch/rejection predictor. Repo has only the hook (`stop_fn`) and a toy entropy policy
  - [ ] train predictor (features: drafter entropy, LID-tagger confidence) → P/R/F1 on "switch within next k tokens"
  - [ ] plug in as `stop_fn`, compare speedup vs fixed γ
- [ ] Approach C: n-gram tables built separately from romanised vs Devanagari text; route by script; compare to single table

## 5. Analysis
- [ ] Error analysis: 200 rejected switch tokens per pair → wrong language / right language-wrong word / spelling variant / NE
- [ ] Insertional vs alternational switches
- [ ] Token-by-token accept/reject case studies per pair
- [ ] Sub-component metrics: LID tagger F1/acc (LinCE dev, COMI-LINGUA), transliteration CER, KL/TV before vs after distillation
- [ ] All numbers with 95% CIs (sentence-clustered bootstrap is in `metrics.py`), fixed seeds, batch size 1, single GPU
