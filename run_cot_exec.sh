#!/bin/bash

# Run Chain-of-Thought evaluation for Execution Trace task

python eval/cot_eval.py \
    --dataset "data/exec_trace.jsonl" \
    --prompt-file "prompts/cot_prompts/exec_trace_prompt.txt" \
    --model-name "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --task-type exec_trace \
    --output "results/cot_exec_trace_baseline.jsonl" \
    --max-tokens 512 \
    --temperature 0.7 \
    --seed 42 \
    --position-mode fixed \
    --variant baseline \
    # --hf-login \
    # --do-sample \
    # --position-mode random \
    # --variant bandwagon \

