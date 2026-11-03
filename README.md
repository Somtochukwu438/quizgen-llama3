# Automated Domain-Specific Quiz Generation with Llama 3.2

**CSCI 390: Natural Language Processing and Applications (Option 5: Text Generation)**
Instructor: Dr. Qi Li · Author: Somtochukwu Augustus Nwanno · Fisk University, Fall 2026

This project fine-tunes **Llama 3.2 3B Instruct** with **LoRA** to turn a passage of course text into a multiple-choice question (one correct answer plus three distractors) in a fixed JSON format. It compares the fine-tuned model against the same model used zero-shot, adds **retrieval-augmented generation (RAG)** over biological psychology course notes, and includes an interactive **Gradio** demo.

```json
{"question": "What type of receptors signal touch, temperature, body position, and tissue threat?",
 "answer": "somatosensory",
 "distractors": ["visual", "kinetic", "kinesthetic"]}
```

---

## Contents

1. [Results](#results)
2. [Repository structure](#repository-structure)
3. [Before you start: Hugging Face access](#before-you-start-hugging-face-access)
4. [Option A: Run in Google Colab (recommended)](#option-a-run-in-google-colab-recommended)
5. [Option B: Run locally in VS Code (macOS, Linux, Windows)](#option-b-run-locally-in-vs-code-macos-linux-windows)
6. [Reproducing the results](#reproducing-the-results)
7. [Method details](#method-details)
8. [Findings and limitations](#findings-and-limitations)
9. [Troubleshooting](#troubleshooting)
10. [Acknowledgments](#acknowledgments)

---

## Results

Evaluation on **300 held-out SciQ test passages** (fixed sample, `seed=42`). Both models use the same prompt, the same 4-bit base model and greedy decoding; the only difference is the LoRA adapter.

### Raw outputs

| Metric | Baseline (zero-shot) | Fine-tuned (LoRA) |
|---|---|---|
| Format valid (correct JSON, 1 answer, 3 distractors) | 292/300 (97.3%) | **298/300 (99.3%)** |
| Answer repeated as a distractor | **5/300 (1.7%)** | 60/300 (20.0%) |
| Duplicate distractors | **0/300 (0.0%)** | 83/300 (27.7%) |
| Fully clean (valid, no repeats, no duplicates) | **287/300 (95.7%)** | 187/300 (62.3%) |
| ROUGE-L vs. SciQ reference question | 0.294 | **0.452** |
| BERTScore F1 (rescaled) | 0.387 | **0.482** |
| Answerable from the passage | 0.648 | **0.824** |

ROUGE-L, BERTScore and answerability are computed on the 290 passages where *both* models produced valid JSON. For reference, SciQ's own human-written questions score **0.897** on answerability, which is the practical ceiling of the checker.

- **Significance:** on passages where the two models disagree on answerability, 75 favour the fine-tuned model and 24 the baseline (exact McNemar test, p = 2.8 × 10⁻⁷).
- **Length control:** fine-tuned answers are shorter (1.61 vs. 4.40 words). Restricted to the 109 passages where both answers have ≤ 2 words, answerability is still 0.798 vs. 0.670.

### With validation and regeneration

Outputs that fail validation are regenerated with light sampling (temperature 0.7, top-p 0.9, seed 42), up to three retries.

| Metric (300 passages) | Baseline + regen | Fine-tuned + regen |
|---|---|---|
| Fully clean | 300/300 (100%) | 298/300 (99.3%) |
| ROUGE-L | 0.292 | **0.443** |
| BERTScore F1 (rescaled) | 0.385 | **0.474** |
| Answerable | 0.65 | **0.82** |
| Extra generations needed | 15 | 144 |

**Summary:** fine-tuning makes questions closer to SciQ's style and much better grounded in the passage, but it introduces a distractor problem (repeating the answer or duplicating distractors). Validation and regeneration remove almost all of those failures, at the cost of more generation. The fine-tuned model with regeneration is the best overall system and is the one the demo uses.

---

## Repository structure

```
quizgen-llama3/
├── adapters/                  Trained LoRA adapter (fp16, ~47 MB)
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── app/
│   └── app.py                 Gradio demo interface
├── data/
│   └── notes/                 Biological psychology notes (Weeks 4, 7, 9; author's own)
├── notebooks/
│   ├── 01_baseline.ipynb      Zero-shot baseline on 300 SciQ test passages
│   ├── 02_finetune.ipynb      LoRA fine-tuning on 2,000 SciQ training examples
│   ├── 03_eval.ipynb          All metrics, significance tests, regeneration
│   └── 04_demo.ipynb          RAG test, regression check, demo launch
├── src/
│   ├── prompts.py             Shared prompt (identical for both models)
│   ├── parse.py               JSON parsing, format validity and quality flags
│   ├── generate.py            Model loading, generation, validation and regeneration
│   ├── notes.py               Extracts lecture passages from the HTML notes
│   ├── rag.py                 Sentence embeddings + Chroma retrieval
│   └── answerability.py       Extractive-QA answerability check
├── run_demo.py                Launches the demo locally (NVIDIA GPU required)
├── check_setup.py             Quick no-GPU check: notes load, parser works
└── requirements.txt           Exact library versions used for the results
```

---

## Before you start: Hugging Face access

Llama 3.2 is a **gated model**, so you need approval from Meta before you can download it. This is required for every option below.

1. Create an account at [huggingface.co](https://huggingface.co) and verify your email.
2. Open [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct), fill in the form, accept the license and click **Submit**. Approval usually takes from a few minutes to a few hours; check the status under **Settings → Gated Repositories**.
3. Create a token: **Settings → Access Tokens → Create new token**, choose type **Read**, and copy it (it starts with `hf_`). Never put this token in code or commit it to GitHub.

---

## Option A: Run in Google Colab (recommended)

Colab provides the NVIDIA GPU the model needs, so this works from any computer, including Macs and Chromebooks.

### A1. One-time setup

1. Go to [colab.research.google.com](https://colab.research.google.com) and sign in.
2. Store your Hugging Face token as a secret: click the **key icon** in the left sidebar → **Add new secret** → name it exactly `HF_TOKEN`, paste the token as the value, and switch **Notebook access** on.

### A2. Run the demo (about 5 minutes)

1. **File → New notebook**, then **Runtime → Change runtime type → L4 GPU** (T4 also works, more slowly) → **Save**.
2. Paste each block below into its own cell and run them in order (**Shift+Enter**).

**Cell 1: install and log in**
```python
!pip -q install transformers peft bitsandbytes accelerate chromadb sentence-transformers gradio beautifulsoup4
from google.colab import userdata
from huggingface_hub import login
login(token=userdata.get("HF_TOKEN"))
```

**Cell 2: get the code**
```python
import os, sys
REPO = "/content/quizgen-llama3"
if os.path.exists(REPO):
    !git -C {REPO} pull
else:
    !git clone https://github.com/Somtochukwu438/quizgen-llama3.git {REPO}
sys.path.append(REPO)
from src.generate import load_model
from src.notes import load_notes
from src.rag import NotesIndex
from app.app import build_app
```

**Cell 3: load the model and notes (about 1 minute)**
```python
model, tok = load_model(f"{REPO}/adapters")
index = NotesIndex(load_notes(f"{REPO}/data/notes"))
```

**Cell 4: launch**
```python
demo = build_app(model, tok, index)
demo.launch(share=True, debug=True)
```

Open the `https://….gradio.live` link printed under Cell 4. The cell keeps running while the app is live; stop it with the ■ button when you are done. Each launch creates a new link, so always use the latest one.

> For exact reproducibility, you can pin the tested versions in Cell 1 with
> `!pip -q install -r /content/quizgen-llama3/requirements.txt` after cloning (Cell 2).

### A3. Using the demo

- **Topic from my notes:** type a topic (e.g. `sleep stages`, `pain`, `brain imaging`) and click **Make quiz**. The most relevant passages are retrieved from the notes and one question is generated per passage. Each click gives a new set of questions.
- **Paste my own text:** paste any text (at least a few sentences). It is split into passages of roughly 120 words, with one question per passage.
- **Answering:** select an option to see ✅ or ❌ with the correct answer.
- **Compare baseline vs fine-tuned:** runs the same passage through both models and shows them side by side under **Model comparison**.

### A4. Running the notebooks

Each notebook in `notebooks/` has an **Open in Colab** badge at the top. They save their outputs to Google Drive under `MyDrive/quizgen/` (you will be asked to allow Drive access). Run them in order, because later notebooks read earlier outputs:

| Notebook | Needs | Produces | Runtime on L4 |
|---|---|---|---|
| `01_baseline` | HF access | `baseline_outputs.jsonl` | ~5 min |
| `02_finetune` | HF access | LoRA adapter, `train_log.json` | ~15 min |
| `03_eval` | outputs of 01 | `finetuned_outputs.jsonl`, results tables (CSV), regeneration outputs | ~15 min |
| `04_demo` | `finetuned_outputs.jsonl` (for the regression check) | demo launch | ~5 min |

You do not need to run `02_finetune` to use the demo or `03_eval`: the trained adapter is already in `adapters/`.

---

## Option B: Run locally in VS Code (macOS, Linux, Windows)

### What runs where

The model is loaded in 4-bit with `bitsandbytes`, which requires an **NVIDIA GPU with CUDA** (at least ~8 GB of GPU memory recommended).

| Your computer | What you can run locally |
|---|---|
| **Linux or Windows with an NVIDIA GPU** | Everything: the demo, generation and evaluation |
| **macOS (any Mac)**, or a PC without an NVIDIA GPU | Editing code and the lightweight checks below. Run the model in Colab (Option A). |

### B1. Install the tools (all systems)

- **Python 3.13** (tested with 3.13.16): [python.org/downloads](https://www.python.org/downloads/). On Windows, tick **"Add python.exe to PATH"** in the installer.
- **Git:** [git-scm.com](https://git-scm.com/downloads). On macOS, running `git --version` in Terminal offers to install it.
- **VS Code:** [code.visualstudio.com](https://code.visualstudio.com), plus the **Python** extension.

### B2. Get the code

Open a terminal (macOS: **Terminal**; Windows: **PowerShell**; Linux: your terminal), then:

```bash
git clone https://github.com/Somtochukwu438/quizgen-llama3.git
cd quizgen-llama3
code .
```

(`code .` opens the folder in VS Code. If the command is not found, open VS Code and use **File → Open Folder…** instead.) In VS Code, open a terminal with **Terminal → New Terminal**; it starts in the project folder.

### B3. Create a virtual environment

**macOS and Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```
If PowerShell refuses to run the script, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then try again.

Your prompt now starts with `(.venv)`. In VS Code, also pick this environment with **Cmd/Ctrl+Shift+P → Python: Select Interpreter → .venv**.

### B4a. Linux or Windows with an NVIDIA GPU: full install

1. Install PyTorch with CUDA support. Use the selector at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) (choose your OS, **Pip**, **Python** and a **CUDA** version supported by your driver) and run the command it gives you. The results were produced with `torch 2.11.0+cu130`.
2. Install the remaining packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Log in to Hugging Face (paste your token when asked):
   ```bash
   hf auth login
   ```
   (Older versions of `huggingface_hub` use `huggingface-cli login`.)
4. Launch the demo:
   ```bash
   python run_demo.py
   ```
   Open the local link it prints (usually `http://127.0.0.1:7860`). Add `--share` for a public `gradio.live` link.

### B4b. macOS (or no NVIDIA GPU): lightweight checks

```bash
pip install beautifulsoup4
```

Run the quick check (the same command on macOS, Linux and Windows):
```bash
python check_setup.py
```
Expected output: `12 passages (expected 12)`, a parsed question, then `(None, 'no_json')`.

To run the model itself, use Colab (Option A). You can also edit in VS Code and run on a Colab GPU with Google's official **Colab extension for VS Code** (search "Colab" in the Extensions panel and select a Colab runtime as the notebook kernel).

---

## Reproducing the results

| Result | Where |
|---|---|
| Baseline format validity and failure types | `01_baseline.ipynb` |
| Training loss curve, adapter | `02_finetune.ipynb` (log saved as `train_log.json`) |
| Format, quality flags, ROUGE-L, BERTScore, answerability | `03_eval.ipynb` (saved as `results_table.csv`) |
| McNemar test and length-controlled answerability | `03_eval.ipynb` |
| Regeneration tables | `03_eval.ipynb` (saved as `results_table_regen.csv`) |
| Regression check of the shared generation code | `04_demo.ipynb` |

**A note on exact reproducibility.** Greedy generation is deterministic within a GPU session but can differ slightly across sessions or GPUs because of floating-point differences. In a later session, 7 of 8 regression outputs were identical; the eighth diverged at a near-tied word (the first distractor) and continued differently from there. Reported metrics come from the saved outputs of the evaluation run.

---

## Method details

| | |
|---|---|
| Base model | `meta-llama/Llama-3.2-3B-Instruct`, loaded in 4-bit NF4 with double quantization |
| Dataset | SciQ (`allenai/sciq`); examples without a supporting passage removed (10,481 / 887 / 884 train / val / test) |
| Training data | 2,000 training examples, 100 validation examples (`seed=42`) |
| Evaluation data | 300 test passages (`seed=42`), never seen in training |
| LoRA | r = 16, alpha = 32, dropout = 0.05, on all attention and MLP projections |
| Training | 1 epoch, 125 steps, effective batch 16 (4 × 4 accumulation), lr 2e-4 cosine, 4 warmup steps, paged AdamW 8-bit, bf16 |
| Loss masking | Only the answer JSON is learned; prompt tokens are masked (e.g. 56 of 289 tokens in the first example) |
| Training time | ~14 minutes on one L4 GPU; validation loss 0.775 → 0.724 |
| Generation | Greedy, max 200 new tokens; regeneration uses temperature 0.7, top-p 0.9 |
| Retrieval | `all-MiniLM-L6-v2` embeddings in an in-memory Chroma collection, 12 lecture passages |
| Answerability | `deepset/roberta-base-squad2` answers the question from the passage only; counted as answerable if token F1 with the answer key ≥ 0.5 |

---

## Findings and limitations

- **Distractor collapse.** The fine-tuned model often repeats the answer as a distractor or repeats a distractor. Only 2 of the 2,000 training examples contain that pattern, so it was not copied from the data. The most likely explanation is that fine-tuning taught SciQ's style of distractors close to the answer, and greedy decoding then sometimes selects the answer itself. Whether it happens can hinge on a near-tie between candidate words.
- **Near-duplicate and synonym distractors.** In the demo, exact-match checks miss options such as "apnea" vs. "sleep apnea" or "shortage" vs. "shortages". The demo adds a word-overlap check with a simple plural rule. Synonyms ("price cap" vs. "price ceiling") are still missed and would need a meaning-based comparison.
- **Copied sentences.** Fine-tuned questions often rephrase a passage sentence, a style inherited from SciQ. They are well grounded but test recall more than comprehension.
- **Wrong answer keys.** Occasionally the answer key is wrong (in one demo run, a question describing REM sleep was keyed "nrem sleep"). The demo shows a disclaimer.
- **Answerability as a filter did not work.** Used as a live filter, the QA checker rejected 2 of 3 correct demo questions and could not separate a wrong key from right ones (all scored 0.667), because it returns shorter spans and avoids text copied from the question. It is used only as an evaluation metric. The same behaviour means the metric can overcount when an answer differs by one critical word (e.g. "nrem sleep" vs. "NREM" scores above 0.5); both models were scored identically, so the comparison holds.
- **Similarity metrics measure SciQ style.** ROUGE-L and BERTScore rise partly by design, since the model was trained on SciQ. Some SciQ reference questions are noisy.
- **Small notes corpus.** With 12 passages, retrieval is easy; the RAG component demonstrates the pipeline rather than retrieval at scale.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `403` or "gated repo" when loading the model | Your access request is still pending, or the token is missing. See [Hugging Face access](#before-you-start-hugging-face-access). |
| `KeyError: 'HF_TOKEN'` or login fails in Colab | Check the secret is named exactly `HF_TOKEN` and **Notebook access** is on. |
| `ModuleNotFoundError: No module named 'src'` in Colab | The session restarted. Run all cells again from Cell 1. |
| "Could not parse server response" in the demo | You are using an old link. Open the latest `gradio.live` link from the cell output in a new tab. Chrome is more reliable than Safari for share links. |
| `CUDA out of memory` | Use an L4 GPU, or reduce `batch_size` in `src/generate.py`. |
| `unexpected keyword argument` errors | A different library version is installed. Install the pinned versions: `pip install -r requirements.txt`. |
| `bitsandbytes` errors on macOS | Expected: the model needs an NVIDIA GPU. Use Colab. |

---

## Acknowledgments

- **SciQ:** Welbl, Liu and Gardner (2017), *Crowdsourcing Multiple Choice Science Questions*.
- **Llama 3.2** by Meta, used under the Llama 3.2 Community License.
- **Hugging Face** Transformers, PEFT, Datasets and Evaluate; **bitsandbytes**; **Chroma**; **Sentence-Transformers**; **Gradio**.
- QA model `deepset/roberta-base-squad2`; embedding model `sentence-transformers/all-MiniLM-L6-v2`.
- Course notes in `data/notes/` are the author's own.
