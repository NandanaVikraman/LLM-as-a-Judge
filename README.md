# LLM-as-a-Judge: Robustness and Evaluation for Coding Tasks

Team **NeuroNauts** | Arizona State University
Mentor: Dr. Neeraj Varshney

This project investigates how reliably large language models (LLMs) can act as automated judges for coding tasks, focusing on **robustness** and **bias**, not just raw accuracy. We construct a controlled A/B benchmark for **code summarization** and **code translation** by generating perturbed problem statements and buggy code variants, ensuring every pair has exactly one correct option. We then evaluate several open-source LLMs as judges under multiple prompting strategies and adversarial conditions designed to surface positional and conformity bias.

Full writeup and poster: [`docs/Final_Report.pdf`](docs/Final_Report.pdf), [`docs/Poster.pdf`](docs/Poster.pdf)

> **Note:** This codebase was built jointly by two teams sharing the same underlying framework. One team built the core pipeline and infrastructure — the dataset-generation entry point (`main.py`), the execution-tracing task, and the partial-code-completion task. The Chain-of-Thought, Best-of-N, and self-verification evaluation methods were a combined effort of both teams. The **NeuroNauts** team (this report's authors) built on top of that shared framework to run the **code summarization** and **code translation** benchmarking documented in [`docs/Final_Report.pdf`](docs/Final_Report.pdf) and [`docs/Poster.pdf`](docs/Poster.pdf). See [Team](#team) below for full contributor attribution.

## Problem Statement

LLMs are increasingly used as automated judges in software engineering workflows — assessing code quality, evaluating translations, and comparing candidate solutions. Prior work shows these judges can exhibit systematic vulnerabilities: positional bias (favoring whichever candidate is placed first/second), bandwagon bias (being swayed by a stated "crowd preference"), and instability under reasoning-based prompting. Existing benchmarks mostly check whether the final answer is correct, but rarely test whether the judgment is robust to these non-semantic factors. This project builds a benchmark and evaluation pipeline specifically to measure that robustness.

## Approach

The pipeline has four stages:

1. **Construct task variants (P, C, Pm, Cm).** For each seed coding problem `P` with a known-correct solution `C`, we generate:
   - `Pm` — a modified problem statement (via Gemma-3B), using controlled transformations: minor constraint changes, edge-case extensions, intent-preserving parameter renames, and objective swaps.
   - `Cm` — an incorrect code variant (via Qwen2.5), using controlled semantic bugs: conditional-logic changes, logical inversion, off-by-one errors, partial implementation, or removed edge-case handling.
2. **Build pairwise A/B judgment tasks.** Each example pairs one correct candidate against one incorrect/perturbed candidate for code summarization or code translation, and the judge model must pick which is correct.
3. **Benchmark judge models under diverse settings.** Four open-source models (Qwen2.5-Coder-7B-Instruct, Qwen3-8B, LLaMA-3.1-8B-Instruct, Gemma-3-4B-it) are evaluated under four prompting styles: Baseline, Chain-of-Thought, Self-Verification, and Best-of-N.
4. **Validate perturbations and analyze bias susceptibility.** Each example is also run with candidates swapped (position-bias analysis) and with an injected synthetic "crowd preference" signal (bandwagon-bias analysis).

See [`docs/Final_Report.pdf`](docs/Final_Report.pdf) for the full methodology, results tables, and figures.

## Key Findings

- **Qwen3-8B** was the most reliable judge overall; **LLaMA-8B** was consistently the weakest.
- **Self-Verification** hurt summarization accuracy but improved translation accuracy — extra reasoning steps can introduce noise on more open-ended tasks.
- **Best-of-N** sampling mainly benefited weaker models (LLaMA, Gemma); stronger models saw marginal or negative gains.
- **All models showed measurable position bias and bandwagon bias** — accuracy dropped noticeably for every model when a synthetic "crowd preference" cue was injected, even on deterministic translation tasks.
- These results motivate the project's next phase: fine-tuning a specialized, bias-resistant LLM judge (see `docs/Final_Report.pdf`, Appendix B).

## Repository Structure

```
├── main.py                    # Entry point for dataset generation tasks (pm, cm, code_s, exec_t, pc_comp)
├── settings.py                 # CLI argument definitions and global config
├── utils.py                    # Shared helpers (code/problem extraction, cleaning)
├── pool_task.py                 # Generates Pm (modified problems) / Cm (buggy code) variants
├── code_summarization.py        # Builds code summarization A/B dataset
├── partial_code.py              # Builds partial-code-completion A/B dataset
├── exec_tracing.py              # Builds execution-tracing A/B dataset (traces code execution via pysnooper)
│
├── eval/
│   ├── basic_inference.py       # Baseline + bandwagon-variant judge evaluation
│   ├── cot_eval.py              # Chain-of-Thought judge evaluation
│   ├── self_verification.py     # Self-verification (two-pass) judge evaluation
│   └── bestofn_eval.py          # Aggregates multiple baseline runs into a Best-of-N majority vote
│
├── prompts/
│   ├── pm_prompts/               # Problem-modification prompt templates
│   ├── cm_prompts/               # Code-mutation prompt templates
│   ├── pc_comp_prompts/          # Partial-code-completion prompt templates
│   ├── cot_prompts/              # Chain-of-Thought judge prompt templates
│   └── inference_prompts/        # Baseline/self-verification judge prompt templates
│
├── run_*.sh                     # Example shell scripts wiring the above scripts together
├── docs/                        # Final report and poster (PDF)
└── results/                     # Evaluation output logs (JSONL, git-ignored)
```

## Setup

**Requirements:** Python 3.10+ (required by `main.py`'s `match` statement), a CUDA-capable GPU is strongly recommended (models are 3B–8B parameters).

```bash
git clone https://github.com/NandanaVikraman/LLM-as-a-Judge.git
cd LLM-as-a-Judge
pip install -r requirements.txt
```

Create a `.env` file (see [`.env.example`](.env.example)) if you need to authenticate with HuggingFace Hub to pull gated models:

```
HF_TOKEN=your_huggingface_token_here
```

## Usage

### 1. Generate dataset variants

`main.py` drives dataset construction. The task determines which variant is generated:

| Task | Description |
|---|---|
| `pm` | Generate modified problem statements (Pm) |
| `cm` | Generate buggy code variants (Cm) |
| `code_s` | Build the code summarization A/B dataset |
| `exec_t` | Build the execution-tracing A/B dataset |
| `pc_comp` | Build the partial-code-completion A/B dataset |

```bash
python main.py \
  --model google/gemma-3-4b-it --dataset greengerong/leetcode --hf-login \
  --task pm --prompt-task edge_case \
  --path dataset/pm_output.jsonl --start-sample 0 --num-samples 100
```

Run `python main.py --help` for the full list of options. Example invocations for each task are in the `run_*.sh` scripts.

### 2. Evaluate judge models

Each script in `eval/` implements one prompting strategy and takes a JSONL dataset (produced above), a prompt template, and a HuggingFace model name:

```bash
# Baseline (and bandwagon-variant) evaluation
python eval/basic_inference.py \
  --dataset data/code_summarisation.jsonl \
  --prompt-file prompts/inference_prompts/summarisation.txt \
  --model-name Qwen/Qwen2.5-Coder-7B-Instruct \
  --output results/baseline_summarisation.jsonl

# Chain-of-Thought evaluation
python eval/cot_eval.py \
  --dataset data/exec_trace.jsonl \
  --prompt-file prompts/cot_prompts/exec_trace_prompt.txt \
  --model-name Qwen/Qwen2.5-Coder-7B-Instruct \
  --task-type exec_trace \
  --output results/cot_exec_trace.jsonl

# Self-verification evaluation
python eval/self_verification.py \
  --dataset data/partial_code.jsonl \
  --prompt-file prompts/inference_prompts/partial_code.txt \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --output results/self_verification_partial_code.jsonl

# Best-of-N: first run basic_inference.py with --best-of-n to produce run0..run4 files, then aggregate:
python eval/bestofn_eval.py --dataset results/baseline_summarisation
```

Use `--position-mode random` to test position-bias robustness and `--variant bandwagon` to inject the synthetic crowd-preference cue. Ready-to-run examples for each configuration are in `run_pool.sh`, `run_summarization.sh`, `run_exec.sh`, `run_partial.sh`, `run_cot_exec.sh`, `run_cot_partial.sh`, and `run_self_verification.sh`.

## Team

This repository's codebase combines work from two teams that shared the same underlying framework.

**Team NeuroNauts** (Arizona State University, mentored by Dr. Neeraj Varshney) — built the code summarization and code translation benchmarking documented in [`docs/Final_Report.pdf`](docs/Final_Report.pdf):

| Name | Contribution |
|---|---|
| [Megha Suresh](https://github.com/amm01u) | Unified dataset-creation pipeline; benchmarking framework (prompt construction, inference variants, evaluation scripts); baseline + bandwagon-bias evaluation; large-scale experiment runs |
| [Nandana Vikraman](https://github.com/NandanaVikraman) | Code summarization task and dataset generation; self-verification evaluation updates; prompting-strategy testing |
| [Avantika Tiwari](https://github.com/Avantika-27) | Code summarization dataset preparation and evaluation |
| [Amulya Nekkanti](https://github.com/AmulyaNekkanti03) | Code translation benchmarking; translation dataset verification and cleaning |
| [Jahnavi Krishna Kovvuri](https://github.com/Jahnavik2002) | Code translation A/B pair generation and judge-performance analysis |

**Second team** — built the shared pipeline infrastructure and additional judgment tasks this benchmark runs on:

| Name | Contribution |
|---|---|
| [Sagar6250](https://github.com/Sagar6250) | Core pipeline architecture; initial Best-of-N evaluation; partial-code-completion task; prompting-structure design and bug fixes |
| [kmathi-creator](https://github.com/kmathi-creator) | Execution-tracing task: prompts, wrapper, and refinements |
| [Abhig2002](https://github.com/Abhig2002) | Initial Chain-of-Thought evaluation implementation; initial self-verification script; LiveCodeBench dataset integration |

## Ongoing Work

Detailed error analysis on model predictions is in progress, feeding into Supervised/Instruction Fine-Tuning of a specialized, bias-resistant judge model on the curated A/B dataset (see `docs/Final_Report.pdf`, Appendix A & B).

## References

- H. Jiang, Y. Chen, Y. Cao, H.-Y. Lee, R. T. Tan. "CodeJudgeBench: Benchmarking LLM-as-a-Judge for Coding Tasks." arXiv:2502.10533, 2025.
- Z. Wang, Y. Zhou, X. Zhou, D. Lu, T. Chen. "CODE-EDITING: A Reasoning Benchmark for Functional Alignment in Code." arXiv:2502.11022, 2025.
- L. Bishay, B. Shrestha, M. Abdelnabi, V. Sharma, C. Wauthier, R. A. Popa, T. Brunschwiler. "Phoenix: A Framework for Evaluating LLMs as Judges." ICLR, 2024.
- X. Zou, K. Kim, T. Zhang, M. Weyssow, L. F. Gomes, G. Yang, D. Lo. "An Automatic Metric for Bridging the Gap with Human Evaluation in SE Tasks." ACM TOSEM, 2025.
- T.-H. Tsai, Y. Zhuo, Z. Teddu, J. Sun, Z. Xing, X. Zhu, D. Lo. "From Code to Contracts: LLMs as the New Software Judges." arXiv:2503.02246, 2025.

## Acknowledgements

We thank Dr. Neeraj Varshney for his guidance and feedback throughout this project.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
