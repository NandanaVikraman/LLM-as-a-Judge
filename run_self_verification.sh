#!/bin/bash

# Run Self-Verification evaluation for Partial Code Completion task

python eval/self_verification.py \
    --dataset "partial_code_random_cleaned_no_pm.jsonl" \
    --prompt-file "prompts/inference_prompts/partial_code.txt" \
    --model-name "meta-llama/Llama-3.1-8B-Instruct" \
    --output "results/self_verification_partial_code_baseline.jsonl" \
    --seed 42 \
    --position-mode fixed \
    --variant baseline \
    # --hf-login \
    # --max-samples 50 \
    # --position-mode random \
    # --variant bandwagon \




