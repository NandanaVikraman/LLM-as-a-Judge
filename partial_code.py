import re
import os
import json
import pandas as pd

from tqdm import tqdm
from settings import *
from utils import *

from datasets import load_dataset


def partial_code_dataset_generation(model, tokenizer, args):
    dataset = load_dataset("json", data_files=args.dataset)['train']
    
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
        if sample['pid'] in processed_pids:
            continue
        partial_code, candidate_a = get_code_cA(sample['c'])

        prompt_variables = {
            "problem": sample['pm'], 
            "code": partial_code
        }
        
        if args.debug_mode:
            print("DEBUG STATEMENTS---------->\n")
            print("Problem Statement: \n")
            print(prompt_variables['problem'])
            print("-------------")
            print("Code Block: \n")
            print(prompt_variables['code'])
            print("-------------")
        
        task = args.task.lower() if args.task is not None else None
        file_path = os.path.join(os.getcwd(), f"prompts/{task}_prompts/{args.prompt_task}.txt")
        
        try:
            with open(file_path, 'r') as file:
                content = file.read()
        except FileNotFoundError:
            print(f"Error: The file '{file_path}' was not found.")
        except Exception as e:
            print(f"An error occurred: {e}")

        if args.debug_mode:
            print("Prompt Template: \n")
            print(content.format(**prompt_variables))
            print("-------------")
        
        try:
            messages = [
                {"role": "system", "content": content.split("-----",1)[0].strip().format(**prompt_variables)},
                {"role": "user", "content": content.split("-----",1)[1].strip().format(**prompt_variables)}
            ]
            text = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)]

        except Exception as se:
            try:
                messages = [{"role": "user", "content": content.format(**prompt_variables)}]               
                text = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)]

            except Exception as ke:
                text = content.format(**prompt_variables)

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.95
        )

        metadata = sample["metadata"].copy()
        metadata["model"] = args.model
        # metadata["processed_timestamp"] = "2025-10-30"

        data = [{
            "pid": sample['pid'],
            # "partial-c": partial_code,
            "input": {
                "p": sample['p'],
                "pm": sample['pm'],
                "partial-c": partial_code,
                },
            "chosen": candidate_a,
            "rejected": clean_code_block(re.split('model|assistant', tokenizer.decode(outputs[0], skip_special_tokens=True))[-1]),
            "metadata": metadata
        }]
   
        path = args.path
        # df = pd.DataFrame(data)
        # df.to_csv(path, mode='a', index=False, header=False)
        with open(path, 'a', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')