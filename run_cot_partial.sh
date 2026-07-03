#!/bin/bash

# Run Chain-of-Thought evaluation for Partial Code Completion task

python eval/cot_eval.py \
    --dataset "partial_code_random_cleaned_no_pm.jsonl" \
    --prompt-file "prompts/cot_prompts/partial_code_prompt.txt" \
    --model-name "meta-llama/Llama-3.1-8B-Instruct" \
    --task-type partial_code \
    --output "results/cot_partial_code_baseline.jsonl" \
    --max-tokens 512 \
    --temperature 0.7 \
    --seed 42 \
    --position-mode fixed \
    --variant baseline \
    # --hf-login \
    # --do-sample \
    # --position-mode random \
    # --variant bandwagon \

