import argparse
import math
import zlib
from collections import Counter, defaultdict

import torch
from nltk.util import ngrams
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer


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
    eps = 1e-6
    words = text.split()
    word_counts = Counter(words)
    v1 = len([word for word, count in word_counts.items() if count == 1])
    v = len(word_counts)
    return 100 * math.log(len(words)) / (1 - v1 / v + eps)


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


def perplexity(text: str) -> float:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base")
    inputs = tokenizer(text, return_tensors="pt")
    loss = model(**inputs, labels=inputs["input_ids"]).loss
    perplexity = torch.exp(loss).item()
    return perplexity


def compression_ratio(text: str) -> float:
    original_bytes = text.encode("utf-8")
    compressed_bytes = zlib.compress(original_bytes, level=9)
    return len(original_bytes) / len(compressed_bytes)


def main():
    parser = argparse.ArgumentParser(description="Calculate text complexity metrics")
    parser.add_argument("input_file", help="Path to the input text file")
    args = parser.parse_args()

    with open(args.input_file, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Character Entropy: {character_entropy(text):.4f}")
    print(f"Word Entropy: {word_entropy(text):.4f}")
    print(f"Type-Token Ratio: {type_token_ratio(text):.4f}")
    print(f"Brunet Index: {brunet_index(text):.4f}")
    print(f"Honore Statistic: {honore_statistic(text):.4f}")
    print(f"Character Bigram Entropy: {ngram_entropy(text, n=2, mode='char'):.4f}")
    print(f"Word Bigram Entropy: {ngram_entropy(text, n=2, mode='word'):.4f}")
    print(f"Compression Ratio: {compression_ratio(text):.4f}")
    print(f"Perplexity: {perplexity(text):.4f}")


if __name__ == "__main__":
    main()
