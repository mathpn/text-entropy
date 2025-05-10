import math
from collections import Counter


def character_entropy(text: str) -> float:
    char_counts = Counter(text)
    total_chars = len(text)
    entropy = 0.0
    for count in char_counts.values():
        probability = count / total_chars
        entropy -= probability * math.log2(probability)
    return entropy


def word_entropy(text: str) -> float:
    word_counts = Counter(text.split())
    total_chars = len(text.split())
    entropy = 0.0
    for count in word_counts.values():
        probability = count / total_chars
        entropy -= probability * math.log2(probability)
    return entropy


def type_token_ratio(text: str) -> float:
    words = text.split()
    return len(set(words)) / len(words)


sample_string = "what is entropy after all? entropy is something mathy"
char_e = character_entropy(sample_string)
word_e = word_entropy(sample_string)
ttr = type_token_ratio(sample_string)
print(f"Character entropy: {char_e:.2f}")
print(f"Word entropy: {word_e:.2f}")
print(f"TTR: {ttr:.2f}")
