"""Regenerates the run notebooks. python notebooks/_build_notebooks.py"""
import json
import re

def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s}
def code(s):
    # join backslash continuations: IPython "!" commands are safest on one line
    s = re.sub(r"[ \t]*\\\n[ \t]*", " ", s)
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}

SETUP = [
md("""### S. Setup (run every time you open a new runtime)
Runtime -> Change runtime type -> **A100**, else **L4** (T4 is too small).
Secrets (key icon, notebook access ON): `GITHUB_TOKEN`, `HF_TOKEN`."""),
code("""!nvidia-smi --query-gpu=name,memory.total --format=csv
import torch, os
assert torch.cuda.is_available(), "No GPU: Runtime -> Change runtime type"
print(torch.cuda.get_device_name(0))"""),
code("""from google.colab import drive, userdata
drive.mount('/content/drive')
DRIVE_DIR = '/content/drive/MyDrive/lost-at-the-switch'
if not os.path.isdir(DRIVE_DIR):
    raise SystemExit("Shared folder not found. In Google Drive: 'Shared with me' -> right-click "
                     "'lost-at-the-switch' -> Organize -> Add shortcut -> My Drive. Then rerun this cell.")
for sub in ['results', 'data/raw', 'data/processed']:
    os.makedirs(f'{DRIVE_DIR}/{sub}', exist_ok=True)
GITHUB_TOKEN = userdata.get('GITHUB_TOKEN')
os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
os.environ['HF_HOME'] = '/content/hf_cache'
REPO = '/content/lost-at-the-switch'
if os.path.isdir(f'{REPO}/.git'):
    !git -C {REPO} pull --ff-only
else:
    !git clone -q https://{GITHUB_TOKEN}@github.com/Narpear/lost-at-the-switch.git {REPO}
%cd {REPO}
!git log --oneline -1
!pip install -q -e ".[dev]"
# data/ and results/ live in the shared Drive folder, so both of us see everything
!rm -rf results data/raw data/processed
!ln -s {DRIVE_DIR}/results results && ln -s {DRIVE_DIR}/data/raw data/raw && ln -s {DRIVE_DIR}/data/processed data/processed
!ls -la results data/raw data/processed | head -20"""),
]

KERNELS = code("""# Fast kernels for Qwen3.5's linear-attention layers (results are identical, just faster).
# flash-linear-attention is quick. causal-conv1d may compile for 5-15 min; if it fails, that's OK.
!pip install -q flash-linear-attention 2>&1 | tail -2
!timeout 1200 pip install -q causal-conv1d --no-build-isolation 2>&1 | tail -2
!python -c "import fla; print('fla OK')" ; python -c "import causal_conv1d; print('causal_conv1d OK')"
# If the sanity check below crashes on import, undo with:
#   !pip uninstall -y flash-linear-attention causal-conv1d""")

PAIRS = "['lid_hineng', 'lid_spaeng', 'lid_nepeng']"

PROMPTS = code(f"""# Code-switched continuation prompts from LinCE dev (first half of each code-switched sentence).
# Same files for both of us; skipped if they already exist.
for cfg in {PAIRS}:
    out = f'data/processed/{{cfg}}_dev_prompts.jsonl'
    if not os.path.exists(out):
        !python scripts/make_prompts.py data/raw/lince/{{cfg}}/dev.conll {{out}}
    else:
        print(out, 'exists')""")

REPORT = code("""!python scripts/make_report.py
from IPython.display import Image, display
if os.path.exists('results/report/positional.png'): display(Image('results/report/positional.png'))""")

prerana = [md("""# Prerana: models and runs (Thu Nov 19)
Track status in the tracker sheet. Each **P** cell = one row in your tab.
Outputs go to the shared Drive folder `lost-at-the-switch/results/`.""")] + SETUP + [
md("""## P1. Fast kernels + re-run the sanity check
Then copy the new `c` from line [4] into the `C = ...` line in P5. Monday's c=0.69 was with slow fallback kernels."""),
KERNELS,
code("""!python scripts/sanity_check.py --pair qwen 2>&1 | grep -v -E "falling back|Loading|Fetching|it/s" | tee results/sanity_qwen_fastkernels.txt"""),
md("""## P2. Llama pair sanity check (only if HF access to Llama-3.2-1B and 3B is approved)"""),
code("""!python scripts/sanity_check.py --pair llama 2>&1 | grep -v -E "Loading|Fetching|it/s" | tee results/sanity_llama.txt"""),
md("""## P3. Exp 1: teacher-forced acceptance at switch points (needs Manjari's M1 LinCE data)
~10-30 min per pair. Up to 800 code-switched dev sentences each. Skips pairs already done."""),
code(f"""for cfg in {PAIRS}:
    pair = cfg.replace('lid_', '')
    print('=' * 20, pair)
    !python scripts/run_positional.py --pair qwen --data data/raw/lince/{{cfg}}/dev.conll \\
        --out results/positional/qwen_{{pair}}_dev.jsonl --only-code-switched --limit 800 --skip-existing \\
        2>&1 | grep -v -E "falling back|Loading|it/s" | tail -25"""),
md("""## P4. Make code-switched prompts (skipped if Manjari already made them)"""),
PROMPTS,
md("""## P5. B1 on code-switched text: Qwen 0.8B drafter, gamma 2/3/4, 50 prompts per pair
Raw continuation prompts (no chat template). Resumes automatically if Colab disconnects: just rerun the cell."""),
code(f"""C = None   # <- paste c from P1, e.g. C = 0.12
for cfg in {PAIRS}:
    pair = cfg.replace('lid_', '')
    print('=' * 20, pair)
    !python scripts/run_specdec.py --pair qwen --prompts data/processed/{{cfg}}_dev_prompts.jsonl \\
        --out results/specdec/qwen_b1_{{pair}}.jsonl --limit 50 --gammas 2 3 4 --skip-existing \\
        {{'--c ' + str(C) if C else ''}} 2>&1 | grep -v -E "falling back|Loading|it/s" | tail -30"""),
md("""## P6. B1 reproduction on Nepali (overnight): FLORES English->Nepali translation, chat template
Needs Manjari's M2 FLORES prompts. Paudel's own Nepali test set isn't released, so FLORES is our stand-in."""),
code("""!python scripts/run_specdec.py --pair qwen --prompts data/processed/flores_translate_npi_Deva.jsonl \\
    --out results/specdec/qwen_b1_flores_en2ne.jsonl --limit 100 --gammas 2 3 4 --chat-template --skip-existing \\
    2>&1 | grep -v -E "falling back|Loading|it/s" | tail -30"""),
md("## P7. Tables + figure for Friday"), REPORT,
]

manjari = [md("""# Manjari: data and baselines (Thu Nov 19)
Track status in the tracker sheet. Each **M** cell = one row in your tab.
Outputs go to the shared Drive folder `lost-at-the-switch/` (shared by Prerana)."""),
md("""**One-time setup before cell S:**
1. Accept the GitHub invite to `Narpear/lost-at-the-switch`, then make a **classic** token (github.com -> Settings -> Developer settings -> Tokens (classic), scope `repo`).
2. Make a Hugging Face **Read** token, and click "Agree and access" on https://huggingface.co/datasets/openlanguagedata/flores_plus
3. In Colab's key icon add `GITHUB_TOKEN` and `HF_TOKEN`, notebook access ON.
4. In Google Drive: Shared with me -> `lost-at-the-switch` -> Organize -> Add shortcut -> My Drive.""")] + SETUP + [
md("""## M1. Get LinCE (Hindi-English, Spanish-English, Nepali-English)
Tries the official server first. If it times out (it did on Monday): download LinCE yourself
(https://ritual.uh.edu/lince, or a classmate's copy), upload the zip/folder to the shared Drive folder,
and run the second cell with its path."""),
code("""!python scripts/fetch_lince.py --timeout 120"""),
code("""# Only if the cell above failed: point --from at your downloaded zip or folder
# !python scripts/fetch_lince.py --from /content/drive/MyDrive/lost-at-the-switch/lince_download.zip"""),
code("""!python scripts/data_stats.py data/raw/lince/lid_hineng/dev.conll data/raw/lince/lid_spaeng/dev.conll data/raw/lince/lid_nepeng/dev.conll | tee results/data_stats_lince_dev.txt"""),
md("""## M2. FLORES-200 monolingual controls + English->X translation prompts
Needs the FLORES access click (setup step 2). Untested from our side: if it errors, paste the error to Claude."""),
code("""!python scripts/fetch_flores.py --split devtest"""),
md("""## M3. Switching-penalty control: teacher-forced acceptance on monolingual FLORES
Same measurement as Prerana's P3 but on text with no switches. 500 sentences per language."""),
code("""for lang in ['eng_Latn', 'spa_Latn', 'hin_Deva', 'npi_Deva']:
    print('=' * 20, lang)
    !python scripts/run_positional.py --pair qwen --data data/raw/flores/{lang}.conll \\
        --out results/positional/qwen_flores_{lang}.jsonl --limit 500 --skip-existing \\
        2>&1 | grep -v -E "falling back|Loading|it/s" | tail -12"""),
md("## M4. Make code-switched prompts (shared with Prerana's P4; skipped if they exist)"), PROMPTS,
md("""## M5. Bigram drafter tables (B2), built from LinCE train with the Qwen tokenizer"""),
code(f"""for cfg in {PAIRS}:
    pair = cfg.replace('lid_', '')
    !python scripts/build_ngram.py --pair qwen --n 2 --inputs data/raw/lince/{{cfg}}/train.conll \\
        --out results/ngram/qwen_{{pair}}_n2.pkl"""),
md("""## M6. B2 on code-switched text: bigram drafter, gamma 2/3/4, same 50 prompts as Prerana's B1
Resumes automatically if Colab disconnects."""),
code(f"""for cfg in {PAIRS}:
    pair = cfg.replace('lid_', '')
    print('=' * 20, pair)
    !python scripts/run_specdec.py --pair qwen --ngram results/ngram/qwen_{{pair}}_n2.pkl \\
        --prompts data/processed/{{cfg}}_dev_prompts.jsonl --out results/specdec/qwen_b2_{{pair}}.jsonl \\
        --limit 50 --gammas 2 3 4 --skip-existing 2>&1 | grep -v -E "falling back|Loading|it/s" | tail -30"""),
md("## M7. Tables + figure for Friday"), REPORT,
]

meta = {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"}}
for name, cells in [("01_prerana_runs.ipynb", prerana), ("02_manjari_runs.ipynb", manjari)]:
    json.dump({"cells": cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 0},
              open(f"notebooks/{name}", "w"), indent=1)
    print("wrote", name, len(cells), "cells")
