python ./main.py \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --dataset dataset/cm_output.jsonl \
    --task code_s \
    --prompt-task test_prompt \
    --path dataset/code_summarization_test.jsonl \
    # --max-new-tokens 512 \
    # --debug-mode \
