python ./main.py \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --dataset /content/drive/MyDrive/LLM-as-a-judge-Outputs/pm_output_0_101.jsonl \
    --task pc_comp \
    --prompt-task test_prompt \
    --path /content/drive/MyDrive/partial_code_test.jsonl \
    # --debug-mode \