import math
from collections import Counter, defaultdict

from nltk.util import ngrams


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


def brunet_index(text: str) -> float:
    words = text.split()
    return len(words) ** len(set(words)) ** -0.165


def honore_statistic(text: str) -> float:
    words = text.split()
    word_counts = Counter(words)
    v1 = len([word for word, count in word_counts.items() if count == 1])
    v = len(word_counts)
    return 100 * math.log(len(words)) / (1 - v1 / v)


def ngram_entropy(text, n=2, mode="char"):
    if mode == "char":
        tokens = list(text)
    elif mode == "word":
        tokens = text.split()
    else:
        raise ValueError("Invalid mode. Use 'char' or 'word'")

    sequence = list(ngrams(tokens, n))
    if len(sequence) < 1:
        return 0.0

    counts = defaultdict(int)
    for gram in sequence:
        counts[gram] += 1

    total = len(sequence)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)

    return entropy


sample_string = "what is entropy after all? entropy is something mathy and weird. entropy entropy entropy"
char_e = character_entropy(sample_string)
word_e = word_entropy(sample_string)
ttr = type_token_ratio(sample_string)
brunet = brunet_index(sample_string)
honore = honore_statistic(sample_string)
print(f"Character entropy: {char_e:.2f}")
print(f"Word entropy: {word_e:.2f}")
print(f"TTR: {ttr:.2f}")
print(f"Brunet: {brunet:.2f}")
print(f"Honoré: {honore:.2f}")
print(ngram_entropy(sample_string, n=3))
