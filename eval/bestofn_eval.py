import argparse
import json
import glob

def parse_args():
    parser = argparse.ArgumentParser(
        description="Basic LLM-as-Judge accuracy evaluation for a code dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to JSONL dataset (e.g. data/code_summarisation.jsonl)",
    )
    return parser.parse_args()

import json
from collections import defaultdict, Counter
import glob

def run_best_of_n(run_files):
    """
    Compute best-of-N predictions from multiple JSONL runs and calculate accuracy
    """
    # Step 1: Collect all labels and store metadata by pid
    pid_data = defaultdict(list)  # pid -> list of {"model_label": ..., "gold_label": ..., "input": ...}
    
    for file in run_files:
        with open(file, "r") as f:
            for line in f:
                data = json.loads(line)
                key = (data["pid"], data["gold_label"], data['bandwagon_preferred'])
                pid_data[key].append(data)
    
    # Step 2: Compute majority vote for each pid
    final_results = {}
    correct_count = 0
    total = 0
    
    for (pid, gold_label, source), runs in pid_data.items():
        labels = [run["model_label"] for run in runs]
        counts = Counter(labels)
        
        # Majority vote
        if counts["A"] > counts["B"]:
            final_label = "A"
        elif counts["B"] > counts["A"]:
            final_label = "B"
        else:
            final_label = "A"  # tie-breaker
        
        # Gold label (all runs have same gold)
        gold_label = runs[0]["gold_label"]
        is_correct = final_label == gold_label
        
        # Update stats
        total += 1
        if is_correct:
            correct_count += 1
        
        # Save final result with metadata
        final_results[(pid, gold_label, source)] = {
            "pid": pid,
            "final_label": final_label,
            "gold_label": gold_label,
            "is_correct": is_correct,
            "vote_counts": dict(counts),
            "input": runs[0]["input"]  # keep original input context and candidates
        }
    
    accuracy = correct_count / total if total > 0 else 0.0
    return final_results, accuracy



if __name__ == "__main__":
    args = parse_args()
    run_files = sorted(glob.glob(f"{args.dataset}_*.jsonl"))  # all your run01.jsonl, run02.jsonl...
    final_results, accuracy = run_best_of_n(run_files)

    print(f"Accuracy over {len(final_results)} samples: {accuracy:.2%}")

    # Optional: save final results to JSONL
    with open(f"{args.dataset}_best_of_n.jsonl", "w") as f:
        for res in final_results.values():
            f.write(json.dumps(res) + "\n")

    