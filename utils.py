import re
import math
import random

# Pool Stage Utilities
def get_codeblock(text):
    matches = re.findall(r"```([\w\+\#\-]*)\n([\s\S]*?)```", text)
    if matches:
      for lang, code in matches:
          lang = lang.strip().lower() if lang else "unknown"
          return ({
              "language": lang,
              "code": code.strip()
          })
    return ""

def get_problem(text):   
    text = text.strip()
    text = re.sub(r"```[\s\S]*?```", "", text)  # remove fenced code
    text = re.sub(r"`([^`]*)`", r"\1", text)     # remove inline backticks

    split_keywords = ["Example", "Examples", "Constraints", "**Example", "**Examples", "**Constraints"]
    idx = len(text)  # default to full text
    for kw in split_keywords:
        pos = text.find(kw)
        if pos != -1:
            idx = min(idx, pos)
    explanation = text[:idx].strip()
    return explanation

# Model Output Cleaning Utility
def clean_code_block(output: str, partial_prefix: str = None) -> str:
    """
    Cleans and normalizes a multi-line code block.
    - Removes markdown code fences (```python, ```java, etc.)
    - Removes comments (#, //, /* */)
    - Strips extra spaces
    - Optionally removes a given partial prefix if the code starts with it
    """
    if not output:
        return ""

    # 1. Remove code block markers
    output = re.sub(r"```[a-zA-Z0-9+]*", "", output)
    output = output.replace("```", "")

    # 2. Normalize escaped newlines and tabs
    output = output.replace("\\n", "\n").replace("\\t", "\t")

    # 3. Split into lines and strip whitespace
    lines = [l.rstrip() for l in output.split("\n") if l.strip()]

    # 4. Remove comments
    clean_lines = []
    for line in lines:
        stripped = line.lstrip()
        # remove single-line comments
        line = re.sub(r"//.*", "", line)
        if stripped.startswith("#") and re.match(r"#[A-Za-z!]", stripped):
            clean_lines.append(line)
            continue
        line = re.sub(r"#.*", "", line)
        # remove block comment markers
        line = line.replace("/*", "").replace("*/", "")
        if line.strip():
            clean_lines.append(line)

    if not clean_lines:
        return ""

    # 5. Join the cleaned lines back into a block
    cleaned_code = "\n".join(clean_lines).strip()

    # 6. Optionally remove a given partial prefix
    if partial_prefix:
        prefix_pattern = re.escape(partial_prefix.strip())
        cleaned_code = re.sub(rf"^{prefix_pattern}", "", cleaned_code).lstrip()

    return cleaned_code

# Partial Code Completion Task Utilities
def is_logic_line(line):
    s = line.strip()
    if not s:
        return False
    # Skip comments
    if s.startswith(("#", "//", "/*", "*", "*/")):
        return False
    # Skip imports/includes/usings
    if re.match(r"^(import|from|#include|using)\b", s):
        return False
    # Skip lines that are only braces or parentheses
    if re.fullmatch(r"[\{\}\(\)\[\];]+", s):
        return False
    # Skip pure declarations like class or function headers with only braces
    if re.match(r"^(class|interface|public|private|protected|def|function)\b.*[\{\}]?$", s) and not re.search(r"[=;]", s):
        return False
    # Skip decorator or annotation lines
    if s.startswith("@"):
        return False
    return True

def get_code_cA(full_code):
    lines = full_code.strip().split('\n')
    candidates = [i for i, line in enumerate(lines) if is_logic_line(line)]
    if not candidates:
        return None
    
    n = len(candidates)
    center = n / 2
    sigma = n / 4
    weights = [math.exp(-((i - center) ** 2) / (2 * sigma ** 2)) for i in range(n)]

    # weights = [math.exp(-0.2 * i) for i in candidates]
    k = random.choices(candidates, weights=weights, k=1)[0]
    # k = random.choice(candidates)

    prompt = '\n'.join(lines[:k]) + "\n# <FILL HERE>\n" # + '\n'.join(lines[k+1:])
    candidate_A = '\n'.join(lines[k:])

    return [prompt, candidate_A]

# Execution Tracing Task Utilities
def extract_examples_from_content(content: str):
    examples = []
    pattern = re.compile(r"Input:(.*?)Output:(.*?)(?:Example|\Z)", re.S | re.IGNORECASE)
    for m in pattern.finditer(content):
        inp = m.group(1).strip()
        out = m.group(2).strip()
        examples.append({"input": inp, "output": out})
    return examples


def clean_example_text(text: str):
    text = re.sub(r"[*`]+", "", text)
    text = text.replace("\\[", "[").replace("\\]", "]")
    text = re.sub(r"\bnull\b", "None", text)
    text = re.sub(r"\s+", " ", text).strip()

    assignments = []
    for part in re.split(r",\s*(?=[A-Za-z_]\w*\s*=)", text):
        # Remove stray spaces around = and inside string values
        cleaned = re.sub(r"\s*=\s*", " = ", part.strip())
        cleaned = re.sub(r"'([^']*)\s+'", lambda m: f"'{m.group(1)}'", cleaned)
        cleaned = re.sub(r'"([^"]*)\s+"', lambda m: f'"{m.group(1)}"', cleaned)
        assignments.append(cleaned)
    return ", ".join(assignments)
    return text


def get_clean_inputs_from_problem(content: str):
    examples = extract_examples_from_content(content)
    cleaned_inputs = []
    for ex in examples:
        if ex.get("input"):
            cleaned_inputs.append(clean_example_text(ex["input"]))
    return cleaned_inputs
# Code Summarization Task Utilities

# Code Translation Task Utilities