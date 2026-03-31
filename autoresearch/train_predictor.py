#!/usr/bin/env python3
"""
Train a per-layer expert predictor from routing samples.

Input:  autoresearch/routing_samples.bin
Output: autoresearch/predictor_weights.npz

Architecture: one small linear layer per MoE layer
  W[l]: (hidden_size, num_experts)  — maps attention output → expert logits
  Loss: balanced BCE (equal weight on positive/negative examples)
  Training: ~2 epochs over collected samples
"""

from __future__ import annotations

import struct
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SAMPLES_FILE = REPO / "autoresearch/routing_samples.bin"
OUTPUT_FILE = REPO / "autoresearch/predictor_weights.npz"

NUM_EXPERTS = 256
HIDDEN_SIZE = 2048      # Qwen3.5-35B hidden size
NUM_LAYERS = 40
LEARNING_RATE = 1e-3
EPOCHS = 2
BATCH_SIZE = 256


def load_samples(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Load routing samples, grouped by layer index.
    Returns {layer_idx: (hidden_states [N, H], expert_ids [N, K])}
    """
    print(f"Loading samples from {path} ({path.stat().st_size/1e6:.0f} MB)...")
    t0 = time.time()

    layer_hidden: dict[int, list[np.ndarray]] = {}
    layer_experts: dict[int, list[np.ndarray]] = {}
    total = 0

    with open(path, "rb") as f:
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            layer_idx, k = struct.unpack("<ii", hdr)
            if k <= 0 or k > 256:
                print(f"WARNING: invalid k={k} at layer {layer_idx}, stopping read")
                break
            hidden = np.frombuffer(f.read(HIDDEN_SIZE * 4), dtype=np.float32).copy()
            experts = np.frombuffer(f.read(k * 4), dtype=np.int32).copy()

            if layer_idx not in layer_hidden:
                layer_hidden[layer_idx] = []
                layer_experts[layer_idx] = []
            layer_hidden[layer_idx].append(hidden)
            layer_experts[layer_idx].append(experts)
            total += 1

    print(f"Loaded {total} samples across {len(layer_hidden)} layers in {time.time()-t0:.1f}s")

    result = {}
    for l in sorted(layer_hidden.keys()):
        H = np.stack(layer_hidden[l])   # [N, hidden_size]
        E = np.stack(layer_experts[l])  # [N, k]
        result[l] = (H, E)
    return result


def make_multihot(expert_ids: np.ndarray, num_experts: int) -> np.ndarray:
    """Convert [N, k] expert indices to [N, num_experts] multi-hot."""
    N = expert_ids.shape[0]
    labels = np.zeros((N, num_experts), dtype=np.float32)
    for i in range(N):
        labels[i, expert_ids[i]] = 1.0
    return labels


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def balanced_bce_loss_and_grad(
    W: np.ndarray,          # [H, E]
    X: np.ndarray,          # [B, H]
    Y: np.ndarray,          # [B, E] multi-hot
) -> tuple[float, np.ndarray]:
    """Balanced BCE: equal weight on positive (chosen) and negative (unchosen) experts."""
    logits = X @ W          # [B, E]
    probs = sigmoid(logits) # [B, E]

    pos_mask = Y            # 1 where expert chosen
    neg_mask = 1.0 - Y      # 1 where expert not chosen

    pos_count = pos_mask.sum(axis=1, keepdims=True).clip(min=1)
    neg_count = neg_mask.sum(axis=1, keepdims=True).clip(min=1)

    # Per-sample balanced weights
    pos_w = pos_mask / pos_count
    neg_w = neg_mask / neg_count

    # BCE loss with balanced weighting
    eps = 1e-7
    loss_pos = -pos_w * np.log(probs + eps)
    loss_neg = -neg_w * np.log(1.0 - probs + eps)
    loss = (loss_pos + loss_neg).mean()

    # Gradient
    # d(BCE)/d(logit) = prob - label (for standard BCE)
    # with balanced weights: multiply by the per-class weight
    d_logits = (pos_w * (probs - 1.0) + neg_w * probs)  # [B, E]
    d_logits /= X.shape[0]
    grad_W = X.T @ d_logits   # [H, E]

    return float(loss), grad_W


def train_layer(
    layer_idx: int,
    hidden: np.ndarray,     # [N, H]
    experts: np.ndarray,    # [N, k]
) -> np.ndarray:
    """Train a single linear predictor for one layer. Returns W [H, E]."""
    N = hidden.shape[0]
    labels = make_multihot(experts, NUM_EXPERTS)  # [N, E]

    # Normalize hidden states for stable training
    std = hidden.std(axis=0, keepdims=True) + 1e-6
    hidden_norm = hidden / std

    # Initialize weights small
    H = hidden.shape[1]
    rng = np.random.default_rng(42 + layer_idx)
    W = rng.normal(0, 0.01, (H, NUM_EXPERTS)).astype(np.float32)

    # Adam optimizer state
    m = np.zeros_like(W)
    v = np.zeros_like(W)
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
    step = 0

    indices = np.arange(N)
    t0 = time.time()

    for epoch in range(EPOCHS):
        rng.shuffle(indices)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, N, BATCH_SIZE):
            batch_idx = indices[start:start + BATCH_SIZE]
            X_b = hidden_norm[batch_idx]
            Y_b = labels[batch_idx]

            loss, grad = balanced_bce_loss_and_grad(W, X_b, Y_b)
            epoch_loss += loss
            n_batches += 1

            # Adam update
            step += 1
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            m_hat = m / (1 - beta1 ** step)
            v_hat = v / (1 - beta2 ** step)
            W -= LEARNING_RATE * m_hat / (np.sqrt(v_hat) + eps_adam)

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"  Layer {layer_idx:02d} epoch {epoch+1}/{EPOCHS}: loss={avg_loss:.4f} ({time.time()-t0:.1f}s)")

    # Store normalization std alongside weights
    return W, std.reshape(-1).astype(np.float32)


def evaluate_predictor(W: np.ndarray, std: np.ndarray, hidden: np.ndarray, experts: np.ndarray, threshold: float = 0.5) -> dict:
    """Measure hit rate: fraction of true experts that appear in top-K predictions."""
    hidden_norm = hidden / (std + 1e-6)
    logits = hidden_norm @ W        # [N, E]
    k = experts.shape[1]

    # Top-K predicted experts
    top_k_pred = np.argpartition(logits, -k, axis=1)[:, -k:]

    hits = 0
    total = 0
    for i in range(len(experts)):
        true_set = set(experts[i].tolist())
        pred_set = set(top_k_pred[i].tolist())
        hits += len(true_set & pred_set)
        total += len(true_set)

    return {"hit_rate": hits / total if total > 0 else 0.0, "n_samples": len(experts)}


def main() -> None:
    samples = load_samples(SAMPLES_FILE)

    weights = {}
    stds = {}

    for layer_idx in sorted(samples.keys()):
        hidden, experts = samples[layer_idx]
        N = hidden.shape[0]
        print(f"\nLayer {layer_idx:02d}: {N} samples, hidden={hidden.shape[1]}, k={experts.shape[1]}")

        W, std = train_layer(layer_idx, hidden, experts)

        # Evaluate on training data (proxy for quality)
        eval_result = evaluate_predictor(W, std, hidden, experts)
        print(f"  Layer {layer_idx:02d} train hit_rate={eval_result['hit_rate']:.3f}")

        weights[f"W_{layer_idx:02d}"] = W
        stds[f"std_{layer_idx:02d}"] = std

    # Save all weights
    np.savez(OUTPUT_FILE, **weights, **stds)
    size_mb = OUTPUT_FILE.stat().st_size / 1e6
    print(f"\nSaved predictor weights to {OUTPUT_FILE} ({size_mb:.1f} MB)")
    print(f"Layers trained: {len(weights)}")


if __name__ == "__main__":
    main()
