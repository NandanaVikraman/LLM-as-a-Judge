python ./main.py \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --dataset dataset/pm_output.jsonl \
    --task pc_comp \
    --prompt-task test_prompt \
    --path dataset/partial_code_test.jsonl \
    # --debug-mode \