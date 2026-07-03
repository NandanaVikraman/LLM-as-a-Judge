import re
import os
import torch
import json
import pandas as pd
from tqdm import tqdm

from settings import *
from utils import *

from datasets import load_dataset


def generate_pool_dataset(model, tokenizer, args):
    dataset = load_dataset(args.dataset)
    if args.dataset_split == "train":
        dataset = dataset['train']
    elif args.dataset_split == "test":
        dataset = dataset['test']
    else:
        dataset = dataset['validation']

    for i in tqdm(range(args.start_sample, args.start_sample+args.num_samples)):
        # i = random.randint(0, len(dataset))
        language_list = ["python", "java", "c++", "javascript"]
        # language = random.choice(language_list)
        for language in language_list:
            prompt_variables = {
                "problem": get_problem(dataset[i]['content']), 
                "code": get_codeblock(dataset[i][language])['code']
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

            lang_map = {"python": "py", "java": "j", "c++": "cpp", "javascript": "js"}
            prompt_map = {
                "edge_case": "ec",
                "intent_preservation": "ip",
                "minor_constraint": "mc",
                "objective_swap": "os",
                "conditional_logic": "cl",
                "logical_inversion": "li",
                "off_by_one": "obo",
                "remove_edge": "re"
            }

            match task:
                case "pm":
                    pool_data = [{
                        "pid": f"{dataset[i]['slug']}_{lang_map.get(language, language)}_{prompt_map.get(args.prompt_task, args.prompt_task)}",
                        "p": prompt_variables['problem'],
                        "c": prompt_variables['code'],
                        # tokenizer.decode(outputs[0], skip_special_tokens=True).split("model")[-1]
                        "pm": re.split('model|assistant', tokenizer.decode(outputs[0], skip_special_tokens=True))[-1],
                        "metadata": {
                            "prompt": f"{task}_{args.prompt_task}",
                            "language: ": language,
                            "model": args.model,
                            "slug": dataset[i]['slug'],
                            }
                    }]
                case "cm":
                    pool_data = [{
                        "pid": f"{dataset[i]['slug']}_{lang_map.get(language, language)}_{prompt_map.get(args.prompt_task, args.prompt_task)}",
                        "p": prompt_variables['problem'],
                        "c": prompt_variables['code'],
                        "cm": clean_code_block(re.split('model|assistant', tokenizer.decode(outputs[0], skip_special_tokens=True))[-1]),
                        "metadata": {
                            "prompt": f"{task}_{args.prompt_task}",
                            "language: ": language,
                            "model": args.model,
                            "slug": dataset[i]['slug'],
                            }
                    }]
                case _:
                    print("Invalid task specified. Please choose 'pm' or 'cm'.")
                    return
            
                
                
            path = args.path
            # df = pd.DataFrame(pool_data)
            # df.to_csv(path, mode='a', index=False, header=False)
            with open(path, 'a', encoding='utf-8') as f:
                for item in pool_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')