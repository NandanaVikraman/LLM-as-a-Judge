import os
import re
import json
from tqdm import tqdm
from datasets import load_dataset
from settings import *
from utils import *

# -----------------------------
# Helpers
# -----------------------------
def _normalize(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"(?i)\[/?inst\]", "", text)
    text = text.replace("```", " ").replace("`", " ")
    text = " ".join(text.split()).strip()
    if text and text[-1] not in ".!?":
        text += "."
    return text

def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\\_", "_")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _read_prompt_messages(args, prompt_variables):
    """
    Try: prompts/{task}_prompts/{args.prompt_task}.txt   (same pattern as partial_code.py)
    Supports optional "system ----- user" split.
    Falls back to a built-in summarization prompt if file missing.
    """
    task = args.task.lower() if args.task is not None else None
    path = os.path.join(os.getcwd(), f"prompts/{task}_prompts/{args.prompt_task}.txt") if task and args.prompt_task else None

    content = None
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            print(f"[WARN] Could not read prompt file: {e}")

    if content:
        try:
            return [
                {"role": "system", "content": content.split("-----", 1)[0].strip().format(**prompt_variables)},
                {"role": "user",   "content": content.split("-----", 1)[1].strip().format(**prompt_variables)},
            ]
        except Exception:
            # single-block template fallback
            return [{"role": "user", "content": content.format(**prompt_variables)}]

    # Built-in fallback (correct summary prompt)
    code = prompt_variables.get("code", "")
    user = (
        "You are a helpful AI that writes clear, accurate explanations for code. "
        "Write a concise paragraph (3–5 sentences) explaining what the code is doing. "
        "Do not use bullet points or quotes.\n\n"
        f"Code:\n{code}"
    )
    return [{"role": "user", "content": user}]

def _incorrect_messages_from_correct(correct_summary: str):
    user = (
        "Here is a correct explanation of a piece of code:\n\n"
        f"{correct_summary}\n\n"
        "Now write another explanation of the same code that sounds confident and professional "
        "but contains one or two subtle factual or logical mistakes. Keep it similar in length "
        "and tone (3–5 sentences). Do not state that it is incorrect—make it sound fully confident."
    )
    return [{"role": "user", "content": user}]

def _generate(model, tokenizer, messages, max_new_tokens=256, temperature=0.7, do_sample=True):
    # Try chat template first (matches partial_code.py style)
    try:
        text = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)]
    except Exception:
        text = [messages[-1]["content"]]

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        top_p=0.9,
        pad_token_id=getattr(tokenizer, "eos_token_id", None),
        eos_token_id=getattr(tokenizer, "eos_token_id", None),
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # mirror partial_code.py trick to strip any leading role labels
    decoded = re.split(r"(model|assistant)", decoded)[-1]
    return _normalize(decoded)

# -----------------------------
# Entry point (repo-compatible)
# -----------------------------
def code_summarization_dataset_generation(model, tokenizer, args):
    """
    Expects dataset items with:
      - pid
      - c  (code string)
      - metadata (dict, optional)
    Writes JSONL to args.path with:
      pid, input (code), chosen (correct summary), rejected (incorrect summary), metadata
    """
    dataset = load_dataset("json", data_files=args.dataset)["train"]

    # track already processed pids (append-safe)
    processed_pids = set()
    if os.path.exists(args.path):
        with open(args.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    processed_pids.add(json.loads(line)["pid"])
                except Exception:
                    continue

    with open(args.path, "a", encoding="utf-8") as fout:
        for sample in tqdm(dataset):
            pid = sample.get("pid")
            if pid in processed_pids:
                if args.debug_mode:
                    print(f"[SKIP] already processed pid={pid}")
                continue

            code_c = sample.get("c", "")
            if not code_c:
                if args.debug_mode:
                    print(f"[WARN] pid={pid}: missing 'c'; skipping.")
                continue

            prompt_vars = {"code": code_c}

            if args.debug_mode:
                print("\nDEBUG ----------------")
                print("PID:", pid)
                print("Code (truncated):\n", code_c[:800], "...\n")

            # deterministic "chosen"
            correct_msgs = _read_prompt_messages(args, prompt_vars)
            correct_summary = _generate(
                model, tokenizer, correct_msgs,
                max_new_tokens=getattr(args, "max_new_tokens", 256),
                temperature=0.0, do_sample=False
            )

            # creative "rejected"
            incorrect_msgs = _incorrect_messages_from_correct(correct_summary)
            incorrect_summary = _generate(
                model, tokenizer, incorrect_msgs,
                max_new_tokens=getattr(args, "max_new_tokens", 256),
                temperature=0.65, do_sample=True
            )

            metadata = (sample.get("metadata") or {}).copy()
            metadata["model"] = args.model

            record = {
                "pid": pid,
                "input": code_c,
                "chosen": _clean_text(correct_summary),
                "rejected": _clean_text(incorrect_summary),
                "metadata": metadata,
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            if args.debug_mode:
                print(f"[WROTE] pid={pid}")
