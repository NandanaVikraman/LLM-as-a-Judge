import re
import os
import math
import random
import torch
import json
import pandas as pd
from dotenv import load_dotenv

from tqdm import tqdm
from partial_code import partial_code_dataset_generation
from exec_tracing import run_execution_tracing
from code_summarization import code_summarization_dataset_generation
from pool_task import generate_pool_dataset
from settings import *
from utils import *

from huggingface_hub import login
from datasets import load_dataset
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

def main(args):
    if args.hf_login:
        login(os.getenv("HF_TOKEN"))
    
    if not args.no_model:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto", device_map="auto")

    if not os.path.exists(args.path):
        with open(args.path, 'w', encoding='utf-8') as f:
            pass
    
    match args.task:
        case "pm" | "cm":
            generate_pool_dataset(model, tokenizer, args)
            print(f"Pool dataset created, Saved to {args.path}\n")
            return

        case "code_s":
            code_summarization_dataset_generation(model, tokenizer, args)
            print(f"Code Summarization Completed, Saved to {args.path}\n")
            return
        
        case "code_t":
            return        
        
        case "exec_t":
            run_execution_tracing(args)
            print(f"Execution Tracing Completed, Saved to {args.path}\n")
            return

        case "pc_comp":
            partial_code_dataset_generation(model, tokenizer, args)
            print(f"Partial Code Completed, Saved to {args.path}")
            return


if __name__ == '__main__':
    load_dotenv()
    main(args)
