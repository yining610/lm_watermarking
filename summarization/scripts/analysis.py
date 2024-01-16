import json
import math
from typing import List, Text, Any

def compute_avg(data: List[Text, Any]):
    
    avg_num_tokens_scored = [x['num_tokens_scored'] for x in data]
    avg_num_green_tokens = [x['num_green_tokens'] for x in data]
    avg_green_fraction = [x['green_fraction'] for x in data]
    avg_z_score = [x['z_score'] for x in data]
    avg_p_value = [x['p_value'] for x in data]
    avg_prediction = [int(x['prediction']) for x in data]

    print(f"Total number of test examples: {len(data)}")
    print(f"Average number of tokens scored: {sum(avg_num_tokens_scored) / len(avg_num_tokens_scored)}")
    print(f"Average number of green tokens: {sum(avg_num_green_tokens) / len(avg_num_green_tokens)}")
    print(f"Average green fraction: {sum(avg_green_fraction) / len(avg_green_fraction)}")
    print(f"Average z-score: {sum(avg_z_score) / len(avg_z_score)}")
    print(f"Average p-value: {sum(avg_p_value) / len(avg_p_value)}")
    print(f"Average prediction: {sum(avg_prediction) / len(avg_prediction)}")

def compute_confusion_matrix(data: List[Text, Any]):
   
    for example in data:
        example['label'] = 1

    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    for example in data:
        if example['prediction'] == 1 and example['label'] == 1:
            true_positives += 1
        elif example['prediction'] == 0 and example['label'] == 0:
            true_negatives += 1
        elif example['prediction'] == 1 and example['label'] == 0:
            false_positives += 1
        elif example['prediction'] == 0 and example['label'] == 1:
            false_negatives += 1
        else:
            raise ValueError(f"Invalid example: {example}")

    print(f"True positive rate: {true_positives / (true_positives + false_negatives)}")
    print(f"True negative rate: 0.0")  # always positive (watermarked)
    print(f"False positive rate: 1.0") # always positive (watermarked)
    print(f"False negative rate: {false_negatives / (false_negatives + true_positives)}")

def compute_min_expected_green_tokens(delta: float, gamma: float, min_spike_entropy: float):
    """Compute the minimum expcted ratio of green tokens
    """
    print(f"Minimum ratio of green tokens: {gamma * math.exp(delta) * min_spike_entropy / (1 + (math.exp(delta) - 1) * gamma)}")
