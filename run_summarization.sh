python ./main.py \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --dataset /content/drive/MyDrive/LLM-as-a-judge-Outputs/cm_output_final1_100.jsonl \
    --task code_s \
    --prompt-task test_prompt \
    --path /content/drive/MyDrive/code_summarization_test.jsonl \
    # --max-new-tokens 512 \
    # --debug-mode \
