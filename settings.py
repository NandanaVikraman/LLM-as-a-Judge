import torch
import argparse

#----CommandLineArguements------------
parser = argparse.ArgumentParser(description='Settings for LLM as Judge', formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('--no-model', default=False, action='store_true',
                    help='Use if task needs model')

parser.add_argument('--model', default='google/gemma-3-4b-it', type=str, metavar='HF MODEL NAME',
                    help='Assign HuggingFace Model Name (default: google/gemma-3-4b-it)')

parser.add_argument('--dataset', default='greengerong/leetcode', type=str, metavar='HF dataset',
                    help='Assign HuggingFace Dataset Name (default: greengerong/leetcode)')

parser.add_argument('--hf-login', default=False, action='store_true',
                    help='Use if you want to login to HuggingFace Hub (default: False)')

parser.add_argument('--task', default=None, type=str, metavar='CODE TASK',
                    help='Specify the code-related task to perform\n pm: Problem modification\n cm: Code modification \n code_s : code summarization\n code_t : code translation\n exec_t : execution tracing\n pc_comp: partial code completion \n(default: None)')

parser.add_argument('--debug-mode', default=False, action='store_true',
                    help='Use to enable debug')

parser.add_argument('--prompt-task', default='edge_case', type=str, metavar='HF dataset',
                    help='Assign Prompt task file name (default: edge_case)')

parser.add_argument('--max-new-tokens', '--mnt', type=int, default = 512, metavar='N',
                    help='max new tokens to generate (default: 250)')

parser.add_argument('--dataset-split', default='train', type=str, metavar='HF dataset split',
                    help='Assign HuggingFace Dataset Split (default: train)')

parser.add_argument('--path', default="dataset/partial_code_switch.csv", type=str, metavar='Save path',
                    help='Path to save the results (default: dataset/partial_code_switch.csv)')

parser.add_argument('--num-samples', type=int, default = 10, metavar='N',
                    help='Number of samples to evaluate (default: 10)')

parser.add_argument('--start-sample', type=int, default = 0, metavar='N',
                    help='Starting sample index (default: 0)')

#old args for training script

# parser.add_argument('--batch_size','--b',type=int,default=4,metavar="batch",
#                     help='input batch size for training(default: 32) ')

# parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
#                         help='number of Data loading workers (default: 16)')

# parser.add_argument("--lr",'--learning-rate',type=float,default=1e-4,metavar='learning rate',
#                     help="initial learning rate (default 1e-4)")

# parser.add_argument("--weight_decay",'--wd',type=float,default=1e-4,metavar="wd",
#                     help="weight decay(default: 1e-4)")

# parser.add_argument('--resume', default='', type=str, metavar='PATH',
#                     help='path to latest checkpoint (default: none)')

# parser.add_argument('-e', '--eval', type=str, default='',
#                     help='evaluate models on validation set')

args = parser.parse_args()


#----GlobalVariables------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"