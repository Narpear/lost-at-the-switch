# Lost at the Switch: Speculative Decoding on Code-Switched Text

11-711 ANLP (Fall 2026) project: Prerana Kulkarni, Manjari Joshi.

See `EXPERIMENTS.md` for the full checklist of experiments and setup.

## Layout
```
src/switchspec/
  specdec.py       instrumented speculative sampling loop (per-token accept/TV logging), KV-cache rollback
  ngram.py         n-gram drafter (B2, Approach C)
  lid.py           LinCE reader, switch / near_switch / inside labelling
  align.py         word labels -> tokenizer tokens (offset mapping)
  positional.py    teacher-forced 1-TV per position on gold LID text (Exp 1)
  metrics.py       expected speedup (Eq. 2), clustered bootstrap CIs, TV/KL
  distill.py       switch-weighted distillation loss (Approach A, Eq. 3)
  transliterate.py native-script twins, CER, IndicXlit wrapper
  models.py        load drafter/target pairs
scripts/           sanity_check, data_stats, make_prompts, run_positional, run_specdec,
                   build_ngram, train_distill, make_toy_setup
configs/           models.yaml (real pairs), toy.yaml (smoke tests)
tests/             31 CPU tests with tiny random models
```

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                                   # ~40 s on CPU
python scripts/make_toy_setup.py         # tiny local models + toy LinCE file
python scripts/sanity_check.py --config configs/toy.yaml --device cpu
```

## Two ways to measure acceptance
1. **Teacher-forced** (`run_positional.py`): both models read gold code-switched text;
   at every token we log TV(p, q). 1 − TV is the exact expected acceptance at that
   position, with gold LID labels. Low variance, no tagger needed. Best for
   "does acceptance drop at switch points?".
2. **Generation** (`run_specdec.py`): real speculative sampling with every verified
   draft token logged. Gives real α, tokens per target call, expected speedup.
   Switch positions in generated text need the LID tagger afterwards.

Wall-clock speed should come from vLLM, not this loop.

## Data
Raw data goes in `data/raw/` and is git-ignored (check each dataset's license
before redistributing). LinCE LID files are expected as
`data/raw/lince/lid_{hineng,spaeng,nepeng}/{train,dev}.conll`.
