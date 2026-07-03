import io
import os
import re
import json
from tqdm import tqdm

import pysnooper
import tempfile
import textwrap
import traceback
import subprocess

from datasets import load_dataset
from utils import get_clean_inputs_from_problem


def trace_function_from_code(code_str, inputs, max_lines=200, timeout=5):
    trace_text = ""
    with tempfile.NamedTemporaryFile(mode="r+", suffix=".log", delete=False) as trace_file:
        trace_path = trace_file.name
    # --- Write code + input assignments + function call to temp file ---
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
        tmp_path = tmp_file.name
        input_code = "\n".join(inputs)
        # print(input_code)
        input_vars = []
        for i in inputs:
          input_vars.append(i.split('=')[0].strip())
        # print(input_vars)
        tmp_file.write(textwrap.dedent(f"""
import pysnooper
from typing import Optional, List, Any, Dict, Tuple
import types
{code_str}

# Input setup
{input_code}

# Detect first function
func_candidates = [
    (name, obj)
    for name, obj in globals().items()
    if callable(obj)
    and isinstance(obj, (types.FunctionType, types.BuiltinFunctionType))
]
func_name, function = func_candidates[-1]
func = globals()[func_name]

# Trace function call to stdout
traced_func = pysnooper.snoop("{trace_path}", color=False, normalize=True, depth=2)(func)

# Extract argument names
import inspect
arg_names = inspect.getfullargspec(func).args
arg_names = {input_vars}
args = [globals()[a] for a in arg_names]

traced_func(*args)
"""))

    try:
        # Run subprocess and capture stdout/stderr
        result = subprocess.run(
            ["python3", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        # trace_text = result.stdout + result.stderr

    except subprocess.TimeoutExpired:
        # Process took too long — we still read what was logged
        trace_text += f"[Warning] Execution timed out after {timeout} seconds.\n"
        raise TimeoutError("Time out")

    # --- Always read whatever was written to the log file ---
    try:
        with open(trace_path, "r") as f:
            trace_text += f.read()
    except Exception as e:
        trace_text += f"[Error reading trace log: {e}]\n"

    finally:
        os.remove(tmp_path)
        os.remove(trace_path)
    
    ignore_prefixes = ("Source path:",)
    trace_text = [
        line for line in trace_text.splitlines()
        if not any(line.strip().startswith(p) for p in ignore_prefixes)
    ]

    if not trace_text:
      raise ValueError("Empty trace")

    for i, line in enumerate(reversed(trace_text[-3:])):  # check last ~3 lines for safety
      if "Call ended by exception" in line:
          raise ValueError("Trace ended with exception")
    # Truncate if too long
    # trace_text = trace_text.splitlines()
    # if len(trace_text) > max_lines:
    #     return "\n".join(trace_text[:max_lines]) + "\n...[truncated]..."
    return "\n".join(trace_text[:max_lines])

def run_execution_tracing(args):
    dataset = load_dataset("json", data_files=args.dataset)['train']
    leetcode = load_dataset("greengerong/leetcode")["train"]
    leetcode_map = {s["slug"]: s["content"] for s in leetcode}
    
    processed_pids = set()
    if os.path.exists(args.path):
        with open(args.path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                pid = json.loads(line)["pid"]
                if args.debug_mode:
                    print(f"Already processed pid: {pid}\n")
                processed_pids.add(pid)

    for i, sample in enumerate(tqdm(dataset)):
        # if i == 50:
        #   break
        if sample['pid'] in processed_pids:
            continue
        slug = sample.get("metadata", {}).get("slug", "")
        P = sample.get("p", "")
        C = sample.get("c", "")
        Cm = sample.get("cm", "")
        # tqdm.write(sample.get("pid"))
        # ---- Detect and skip non-Python samples ----
        language = sample.get("metadata", {}).get("language: ")
        # tqdm.write(language)
        if language != "python":
            if args.debug_mode:
                tqdm.write(f"[exec_t] Skipping {slug}: non-Python language ({language})")
            continue

        # ---- Skip class-based problems ----
        if re.search(r"^\s*class\s+\w+", C, re.MULTILINE) or re.search(r"^\s*class\s+\w+", Cm, re.MULTILINE):
            if args.debug_mode:
                tqdm.write(f"[exec_t] Skipping {slug}: contains class definitions.")
            continue

        # ---- Extract inputs and keep ONLY the first example ----
        full_problem = leetcode_map.get(slug, "")
        extracted_inputs = get_clean_inputs_from_problem(full_problem)
      
        if extracted_inputs:
            c_input = extracted_inputs[0]
            code_input = re.split(r",\s*(?=[A-Za-z_]\w*\s*=)", c_input)  # take only the first one
            if args.debug_mode:
                tqdm.write(f"[exec_t] Using real inputs for {slug}: {code_input}")
        else:
            if args.debug_mode:
                tqdm.write(f"No input found")
            continue
        # tqdm.write("Input: ", code_input)
        trace_A_path = f"_trace_A_{slug}.log"
        trace_B_path = f"_trace_B_{slug}.log"

        try:
            trace_A = trace_function_from_code(C, code_input, 300)
            # tqdm.write(result)
        except Exception as e:
            if args.debug_mode:
                tqdm.write(f"[exec_t] Failed tracing correct code ({slug}): {e}")
            continue

        try:
            trace_B = trace_function_from_code(Cm, code_input, 300)
            # tqdm.write(result)
        except Exception as e:
            if args.debug_mode:
                tqdm.write(f"[exec_t] Failed tracing buggy code ({slug}): {e}")
            continue

        # metadata = sample["metadata"]

        data =[{
            "pid": sample['pid'],
            "input": {
                "problem": P,
                "c": C,
                "cm": Cm,
                "input": code_input,
            },
            "chosen": trace_A,
            "rejected": trace_B,
            "metadata": sample["metadata"],
        }]

        path = args.path
        # df = pd.DataFrame(data)
        # df.to_csv(path, mode='a', index=False, header=False)
        with open(path, 'a', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')