# eval/self_verification.py

import argparse
import json
import os
import random
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import login
from datasets import load_dataset

from basic_inference import (
    JudgeModel,
    load_prompt_template,
    build_prompt,
    add_bandwagon_signal,
)

import re


# Default max new tokens for self-verification step
MAX_NEW_TOKENS = 128


def _extract_label_from_suffix(suffix: str) -> str:
    
    if not suffix or not suffix.strip():
        raise ValueError("Empty model output, cannot extract label.")

    lines = [ln.strip() for ln in suffix.strip().splitlines() if ln.strip()]

    # 1) Look for "Correct: A/B" anywhere in the text (case-insensitive).
    correct_pattern = r"Correct\s*[:\-]?\s*([AB])\b"
    m = re.findall(correct_pattern, suffix, flags=re.IGNORECASE)
    if m:
        # re.findall returns a list of matches; take the last one
        return m[-1].upper()

    # 2) Otherwise, check the last non-empty line specifically.
    if not lines:
        raise ValueError("No non-empty lines in model output.")

    last_line = lines[-1]

    # If last line itself is "Correct: A" / "Correct: B"
    m = re.fullmatch(correct_pattern, last_line, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 3) If we get here, we truly couldn't parse a label.
    raise ValueError(f"Could not extract label from suffix: {repr(suffix[:200])}...")


def _maybe_wrap_qwen_chat_prompt(model: JudgeModel, prompt: str) -> str:
    """
    If the tokenizer supports apply_chat_template (e.g., Qwen3),
    wrap the plain prompt into a chat format with enable_thinking=False
    to force non-thinking mode. Otherwise, just return the original prompt.
    """
    tok = getattr(model, "tokenizer", None)
    if tok is None:
        return prompt

    apply_chat = getattr(tok, "apply_chat_template", None)
    if apply_chat is None:
        # Not a chat-template tokenizer, just use raw prompt
        return prompt

    # Heuristic: if tokenizer has a chat template (typical for Qwen3 and other chat models),
    # build a single-turn conversation and disable thinking.
    try:
        messages = [
            {"role": "user", "content": prompt}
        ]
        chat_text = apply_chat(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            # This is the key for Qwen3 non-thinking mode:
            # thinking is disabled in the template.
            enable_thinking=False,
        )
        return chat_text
    except TypeError:
        # Some tokenizers may not support enable_thinking; fall back to raw prompt.
        return prompt
    except Exception:
        # Be conservative: don't break the flow if something unexpected happens.
        return prompt


def run_model_with_suffix(
    model: JudgeModel,
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    do_sample: bool | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    min_p: float | None = None,
):
    """
    Run the underlying model once, return (label, suffix).

    Sampling params:
      - If a value is None, we don't pass it to the pipeline and it uses
        whatever defaults the JudgeModel / pipeline was created with.
      - If a value is provided, we pass it directly (good for Qwen, etc.).
    """
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": model.tokenizer.eos_token_id,
    }

    if do_sample is not None:
        gen_kwargs["do_sample"] = do_sample
    if temperature is not None:
        gen_kwargs["temperature"] = temperature
    if top_p is not None:
        gen_kwargs["top_p"] = top_p
    if top_k is not None:
        gen_kwargs["top_k"] = top_k
    if min_p is not None:
        gen_kwargs["min_p"] = min_p

    # IMPORTANT: for Qwen3, wrap the prompt into a chat template
    # with enable_thinking=False to force non-thinking mode.
    model_input_text = _maybe_wrap_qwen_chat_prompt(model, prompt)

    out = model.pipe(
        model_input_text,
        **gen_kwargs,
    )[0]["generated_text"]

    # The suffix is everything generated after the model_input_text
    suffix = out[len(model_input_text):]
    label = _extract_label_from_suffix(suffix)
    return label, suffix


def build_self_verification_prompt_simple(
    original_prompt: str,
    first_label: str,
    first_suffix: str | None = None,
) -> str:
    """
    Simple self-verification: remind the model of its previous final answer,
    then ask it to double-check and end with Correct: A / Correct: B.
    """
    verification_prompt = original_prompt

    verification_prompt += "\n\n---\n"
    verification_prompt += "You previously answered this question as follows.\n"
    verification_prompt += f"Previous final answer: Correct: {first_label}\n"

    verification_prompt += (
        "\nNow, double-check your decision carefully.\n"
        "You may keep or change your answer, but you MUST finish by writing\n"
        "exactly one line of the form:\n"
        "Correct: A\n"
        "or\n"
        "Correct: B\n"
        "Do not add anything after that line.\n"
    )

    return verification_prompt


def build_self_verification_prompt_reasoning(
    original_prompt: str,
    first_label: str,
    first_suffix: str | None = None,
) -> str:
    """
    Reasoning self-verification: ask model to explain briefly before giving
    the final Correct: A / Correct: B line.
    """
    verification_prompt = original_prompt

    verification_prompt += "\n\n---\n"
    verification_prompt += "You previously answered this question as follows.\n"
    verification_prompt += f"Previous final answer: Correct: {first_label}\n"

    verification_prompt += (
        "\nNow, verify your previous answer in 2-3 sentences. "
        "If your verification contradicts the previous answer, explain the contradiction in 1-2 sentences. "
        "Then on a new line at the very end of your response, write your final decision in the format:\n"
        "Correct: A\n"
        "or\n"
        "Correct: B\n"
        "Do NOT add anything after that final line.\n"
    )

    return verification_prompt


def evaluate_self_verification(
    dataset,
    model: JudgeModel,
    base_prompt: str,
    seed: int,
    output_path: str | None,
    position_mode: str = "random",
    variant: str = "baseline",
    max_samples: int | None = None,
    sv_mode: str = "simple",
    initial_max_new_tokens: int = 10,
    sv_max_new_tokens: int = MAX_NEW_TOKENS,
    do_sample: bool | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    min_p: float | None = None,
):
    """
    Evaluate self-verification:
      - initial_max_new_tokens: token budget for baseline (first pass)
      - sv_max_new_tokens: token budget for self-verification (second pass)
      - sampling params: for Qwen/Llama/Gemma etc. (optional)
    """
    random.seed(seed)

    total = 0
    correct_initial = 0
    correct_verified = 0
    flips = 0
    flips_helped = 0
    flips_hurt = 0

    logs = []

    for idx, ex in enumerate(dataset):
        if max_samples is not None and idx >= max_samples:
            print(f"[INFO] Reached max_samples={max_samples}, stopping at idx={idx}")
            break

        pid = ex.get("pid")
        context = ex["input"]
        chosen = ex["chosen"]
        rejected = ex["rejected"]

        if isinstance(context, dict) and "pm" in context:
            del context["pm"]

        if position_mode == "random":
            if random.random() < 0.5:
                cand_a, cand_b = chosen, rejected
                gold_label = "A"
                src_a, src_b = "chosen", "rejected"
            else:
                cand_a, cand_b = rejected, chosen
                gold_label = "B"
                src_a, src_b = "rejected", "chosen"
        else:
            cand_a, cand_b = chosen, rejected
            gold_label = "A"
            src_a, src_b = "chosen", "rejected"

        for source in ["A", "B"]:
            if variant == "baseline":
                if source == "A":
                    cand_a, cand_b = chosen, rejected
                    src_a, src_b = "chosen", "rejected"
                else:
                    cand_a, cand_b = rejected, chosen
                    src_a, src_b = "rejected", "chosen"
                gold_label = source

            prompt = build_prompt(base_prompt, context, cand_a, cand_b)

            if variant == "bandwagon":
                prompt = add_bandwagon_signal(prompt, preferred_option=source)

            # First pass: initial judgment (baseline-style, small token budget)
            try:
                label_initial, suffix_initial = run_model_with_suffix(
                    model,
                    prompt,
                    max_new_tokens=initial_max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                )
            except ValueError as e:
                print(f"[WARN] Could not extract INITIAL label for pid={pid}, source={source}: {e}")
                logs.append(
                    {
                        "pid": pid,
                        "gold_label": gold_label,
                        "initial_label": None,
                        "verified_label": None,
                        "is_correct_initial": 0,
                        "is_correct_verified": 0,
                        "flipped": False,
                        "variant": variant,
                        "bandwagon_preferred": source,
                        "sv_mode": sv_mode,
                        "input": {
                            "context": context,
                            "candidate_A_source": src_a,
                            "candidate_A_text": cand_a,
                            "candidate_B_source": src_b,
                            "candidate_B_text": cand_b,
                        },
                        "raw": {
                            "initial_suffix": suffix_initial if "suffix_initial" in locals() else "",
                            "verified_suffix": "",
                            "initial_prompt": prompt,
                            "verification_prompt": "",
                        },
                    }
                )
                total += 1
                continue

            is_correct_initial = int(label_initial == gold_label)

            # Build self-verification prompt
            if sv_mode == "simple":
                sv_prompt = build_self_verification_prompt_simple(
                    original_prompt=prompt,
                    first_label=label_initial,
                    first_suffix=suffix_initial,
                )
            else:
                sv_prompt = build_self_verification_prompt_reasoning(
                    original_prompt=prompt,
                    first_label=label_initial,
                    first_suffix=suffix_initial,
                )

            # Second pass: verification judgment (larger token budget)
            try:
                label_verified, suffix_verified = run_model_with_suffix(
                    model,
                    sv_prompt,
                    max_new_tokens=sv_max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                )
            except ValueError as e:
                print(f"[WARN] Could not extract VERIFIED label for pid={pid}, source={source}: {e}")
                label_verified = None
                suffix_verified = ""
                is_correct_verified = 0
                flipped = False
            else:
                is_correct_verified = int(label_verified == gold_label)
                flipped = (label_verified != label_initial)

            total += 1
            correct_initial += is_correct_initial
            correct_verified += is_correct_verified

            if flipped and label_verified is not None:
                flips += 1
                if is_correct_initial == 0 and is_correct_verified == 1:
                    flips_helped += 1
                elif is_correct_initial == 1 and is_correct_verified == 0:
                    flips_hurt += 1

            logs.append(
                {
                    "pid": pid,
                    "gold_label": gold_label,
                    "initial_label": label_initial,
                    "verified_label": label_verified,
                    "is_correct_initial": is_correct_initial,
                    "is_correct_verified": is_correct_verified,
                    "flipped": flipped,
                    "variant": variant,
                    "bandwagon_preferred": source,
                    "sv_mode": sv_mode,
                    "input": {
                        "context": context,
                        "candidate_A_source": src_a,
                        "candidate_A_text": cand_a,
                        "candidate_B_source": src_b,
                        "candidate_B_text": cand_b,
                    },
                    "raw": {
                        "initial_suffix": suffix_initial,
                        "verified_suffix": suffix_verified,
                        "initial_prompt": prompt,
                        "verification_prompt": sv_prompt,
                    },
                }
            )

            if total % 10 == 0:
                print(f"[INFO] Processed {total} judgments (dataset idx={idx+1})")

    acc_initial = correct_initial / total if total > 0 else 0.0
    acc_verified = correct_verified / total if total > 0 else 0.0
    flip_rate = flips / total if total > 0 else 0.0
    flip_help_rate = flips_helped / flips if flips > 0 else 0.0
    flip_hurt_rate = flips_hurt / flips if flips > 0 else 0.0

    print(f"Position mode: {position_mode}, Variant: {variant}, SV mode: {sv_mode}")
    print(
        f"[RESULT] Initial accuracy  = {acc_initial:.4f} "
        f"({correct_initial}/{total})"
    )
    print(
        f"[RESULT] Verified accuracy = {acc_verified:.4f} "
        f"({correct_verified}/{total})"
    )
    print(
        f"[RESULT] Flip rate = {flip_rate:.4f} "
        f"({flips}/{total}), flips helped = {flip_help_rate:.4f}, "
        f"flips hurt = {flip_hurt_rate:.4f}"
    )

    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for row in logs:
                f.write(json.dumps(row) + "\n")
        print(f"[INFO] Saved self-verification predictions to {out_path}")

    return {
        "acc_initial": acc_initial,
        "acc_verified": acc_verified,
        "flip_rate": flip_rate,
        "flip_help_rate": flip_help_rate,
        "flip_hurt_rate": flip_hurt_rate,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Self-verification LLM-as-Judge evaluation for a code dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to JSONL dataset (same format as basic_inference).",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        required=True,
        help="Path to prompt template (e.g. prompts/summarisation.txt)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="HF model name (e.g. Qwen/Qwen2.5-Coder-7B-Instruct)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="GPU id (e.g. 0) or -1 for CPU",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for A/B shuffling",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Where to save prediction logs (JSONL). If omitted, nothing is saved.",
    )
    parser.add_argument(
        "--hf-login",
        action="store_true",
        help="If set, will login to HuggingFace using HF_TOKEN from .env",
    )
    parser.add_argument(
        "--position-mode",
        type=str,
        default="random",
        choices=["random", "fixed"],
        help="A/B assignment mode.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="baseline",
        choices=["baseline", "bandwagon"],
        help="Evaluation variant.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of dataset examples for quick testing",
    )
    parser.add_argument(
        "--sv-mode",
        type=str,
        default="simple",
        choices=["simple", "reasoning"],
        help="Self-verification mode: 'simple' (no explicit reasoning) or "
             "'reasoning' (explain first, then output Correct: A/B).",
    )
    parser.add_argument(
        "--initial-max-new-tokens",
        type=int,
        default=10,
        help="Max new tokens for the INITIAL (baseline-style) judgment step.",
    )
    parser.add_argument(
        "--sv-max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help="Max new tokens for the SELF-VERIFICATION step.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (e.g., 0.7 for Qwen non-thinking; None = model default).",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling (recommended for Qwen non-thinking).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p nucleus sampling (e.g., 0.8 for Qwen; None = model default).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling (e.g., 20 for Qwen; None = model default).",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=None,
        help="Min-p sampling (e.g., 0.0 for Qwen; None = model default).",
    )
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    if args.hf_login:
        token = os.getenv("HF_TOKEN")
        if token:
            print("[INFO] Logging into HuggingFace Hub")
            login(token=token)
        else:
            print("[WARN] --hf-login set but HF_TOKEN not found in environment")

    dataset = load_dataset("json", data_files=args.dataset)["train"]
    print(f"[INFO] Loaded {len(dataset)} examples from {args.dataset}")

    base_prompt = load_prompt_template(args.prompt_file)

    # For the underlying pipeline we give it the larger SV budget
    judge = JudgeModel(
        args.model_name,
        device=args.device,
        max_new_tokens=args.sv_max_new_tokens,
    )

    stats = evaluate_self_verification(
        dataset=dataset,
        model=judge,
        base_prompt=base_prompt,
        seed=args.seed,
        output_path=args.output,
        position_mode=args.position_mode,
        variant=args.variant,
        max_samples=args.max_samples,
        sv_mode=args.sv_mode,
        initial_max_new_tokens=args.initial_max_new_tokens,
        sv_max_new_tokens=args.sv_max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
    )

    print("[INFO] Final stats:", stats)


if __name__ == "__main__":
    main()
