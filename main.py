import math
from collections import Counter


def calculate_entropy(text: str) -> float:
    char_counts = Counter(text)
    total_chars = len(text)
    entropy = 0.0
    for count in char_counts.values():
        probability = count / total_chars
        entropy -= probability * math.log2(probability)
    return entropy


sample_string = "what is text entropy?"
entropy_value = calculate_entropy(sample_string)
print(f"Entropy of the sample string: {entropy_value:.2f}")
