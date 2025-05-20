from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def chunked_token_perplexity(
    text: str,
    model_name: str = "Qwen/Qwen3-0.6B-Base",
    chunk_size: int = 512,
    overlap: int = 50,
) -> Tuple[List[str], List[float]]:
    """
    Calculate token-level perplexity using overlapping token chunks for better performance.

    Args:
        text: Input text
        model_name: HuggingFace model name
        chunk_size: Number of tokens per chunk
        overlap: Number of tokens to overlap between chunks

    Returns:
        Tuple of (tokens, perplexities) where both are aligned lists
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    input_ids = encoded["input_ids"].squeeze()

    if len(input_ids.shape) == 0:
        input_ids = input_ids.unsqueeze(0)

    all_tokens = []
    all_perplexities = []

    stride = chunk_size - overlap
    processed_positions = set()

    for start_idx in range(0, len(input_ids), stride):
        end_idx = min(start_idx + chunk_size, len(input_ids))
        chunk_ids = input_ids[start_idx:end_idx]

        with torch.no_grad():
            chunk_tensor = chunk_ids.unsqueeze(0).to(device)
            outputs = model(chunk_tensor)
            logits = outputs.logits.squeeze()

            if len(logits.shape) == 1:  # Single token case
                logits = logits.unsqueeze(0)

            loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

            for i in range(1, len(chunk_ids)):
                global_pos = start_idx + i

                if global_pos in processed_positions:
                    continue

                token_id = chunk_ids[i]
                prev_logits = logits[i - 1]

                cross_entropy = loss_fn(
                    prev_logits.unsqueeze(0), torch.tensor([token_id]).to(device)
                )

                token_perplexity = torch.exp(cross_entropy).item()

                token = tokenizer.decode([token_id])

                all_tokens.append(token)
                all_perplexities.append(token_perplexity)
                processed_positions.add(global_pos)

        if end_idx >= len(input_ids):
            break

    return all_tokens, all_perplexities


def visualize_perplexity(
    tokens: List[str],
    perplexities: List[float],
    title: str = "Token-Level Perplexity",
    max_tokens: int = 100,
):
    """
    Visualize token-level perplexity.

    Args:
        tokens: List of tokens
        perplexities: List of perplexity values
        title: Plot title
        max_tokens: Maximum number of tokens to display
    """
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        perplexities = perplexities[:max_tokens]

    finite_perplexities = [
        min(p, 1000) for p in perplexities
    ]  # Cap at 1000 for display

    plt.figure(figsize=(15, 8))

    bars = plt.bar(range(len(tokens)), finite_perplexities, alpha=0.7)

    colors = [
        "green" if p < 10 else "yellow" if p < 50 else "red"
        for p in finite_perplexities
    ]
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    plt.xlabel("Token Position")
    plt.ylabel("Perplexity")
    plt.title(title)
    plt.xticks(range(len(tokens)), tokens, rotation=45, ha="right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="green", label="Low perplexity (< 10)"),
        Patch(facecolor="yellow", label="Medium perplexity (10-50)"),
        Patch(facecolor="red", label="High perplexity (> 50)"),
    ]
    plt.legend(handles=legend_elements)

    plt.show()


if __name__ == "__main__":
    sample_text = """The quick brown fox jumps over the lazy dog. This is a common sentence used in typography and testing. Language models might find some words more surprising than others."""

    print("Calculating token-level perplexity using sliding window...")
    tokens_sw, perplexities_sw = token_level_perplexity(sample_text, context_window=64)

    print("Calculating token-level perplexity using chunks...")
    tokens_chunk, perplexities_chunk = chunked_token_perplexity(
        sample_text, chunk_size=64, overlap=10
    )

    # Visualize results
    visualize_perplexity(tokens_sw, perplexities_sw, "Sliding Window Approach")
    visualize_perplexity(tokens_chunk, perplexities_chunk, "Chunked Approach")

    # Print some statistics
    print(
        f"\nSliding Window - Tokens: {len(tokens_sw)}, Avg Perplexity: {np.mean(perplexities_sw):.2f}"
    )
    print(
        f"Chunked Approach - Tokens: {len(tokens_chunk)}, Avg Perplexity: {np.mean(perplexities_chunk):.2f}"
    )
