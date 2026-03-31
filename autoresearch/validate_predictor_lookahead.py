#!/usr/bin/env python3
"""
Exp01 logging-only validation: N→N+1 lookahead predictor.

Reads routing samples from a single continuous generation, reconstructs
per-token per-layer (hidden, experts) sequences, simulates LRU bank=176,
and checks whether the one-step-ahead predictor catches would-be misses.

Metric: incremental_hit_rate = misses_predicted / total_misses
Gate:   incremental_hit_rate > 0.50

Does NOT touch model.py or the prefetch path.
"""

from __future__ import annotations

import struct
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from flash_moe_mlx.predictor import ExpertPredictor

SAMPLES_FILE = REPO / "autoresearch/routing_samples_500.bin"
WEIGHTS_FILE = REPO / "autoresearch/predictor_weights.npz"
BANK_SIZE = 176
HIDDEN_SIZE = 2048
TOP_K = 4


# ---------------------------------------------------------------------------
# Binary format loader
# ---------------------------------------------------------------------------

def load_samples_ordered(path: Path):
    """
    Load routing samples in emission order.

    Records are written layer-by-layer within each token:
        token_0_layer_A, token_0_layer_B, ..., token_1_layer_A, ...

    Returns:
        tokens: list of dicts  {layer_idx: (hidden_np, experts_np)}
                one dict per decoded token
    """
    records: list[tuple[int, np.ndarray, np.ndarray]] = []
    with open(path, "rb") as f:
        while True:
            hdr = f.read(8)
            if len(hdr) < 8:
                break
            layer_idx, k = struct.unpack("<ii", hdr)
            if k <= 0 or k > 256:
                print(f"WARNING: invalid k={k} at layer {layer_idx}, stopping", file=sys.stderr)
                break
            hidden = np.frombuffer(f.read(HIDDEN_SIZE * 4), dtype=np.float32).copy()
            experts = np.frombuffer(f.read(k * 4), dtype=np.int32).copy()
            records.append((layer_idx, hidden, experts))

    if not records:
        raise RuntimeError(f"No records read from {path}")

    # Discover the layer indices used (may not be 0-based consecutive)
    first_token_layers: list[int] = []
    seen: set[int] = set()
    for (li, _, _) in records:
        if li in seen:
            break
        seen.add(li)
        first_token_layers.append(li)

    num_layers = len(first_token_layers)
    print(f"  Detected {num_layers} MoE layers per token: {first_token_layers[:5]}...{first_token_layers[-3:]}")

    # Group into tokens (chunks of num_layers)
    tokens: list[dict[int, tuple[np.ndarray, np.ndarray]]] = []
    for i in range(0, len(records), num_layers):
        chunk = records[i : i + num_layers]
        if len(chunk) < num_layers:
            break
        token_dict = {li: (h, e) for (li, h, e) in chunk}
        tokens.append(token_dict)

    return tokens, first_token_layers


# ---------------------------------------------------------------------------
# LRU bank simulation
# ---------------------------------------------------------------------------

class LRUBank:
    def __init__(self, size: int) -> None:
        self.size = size
        self._cache: OrderedDict[int, None] = OrderedDict()

    def query_and_update(self, expert_ids: list[int]) -> set[int]:
        """Return set of misses, then update LRU state."""
        misses: set[int] = set()
        for eid in expert_ids:
            if eid not in self._cache:
                misses.add(eid)

        for eid in expert_ids:
            if eid in self._cache:
                self._cache.move_to_end(eid)
            else:
                while len(self._cache) >= self.size:
                    self._cache.popitem(last=False)
                self._cache[eid] = None

        return misses


# ---------------------------------------------------------------------------
# Main validation loop
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Loading samples from {SAMPLES_FILE} ...")
    tokens, layer_ids = load_samples_ordered(SAMPLES_FILE)
    print(f"  Loaded {len(tokens)} tokens")

    print(f"Loading predictor from {WEIGHTS_FILE} ...")
    predictor = ExpertPredictor(WEIGHTS_FILE, top_k=TOP_K)
    print(f"  Predictor covers {predictor.num_layers} layers")

    banks = {li: LRUBank(BANK_SIZE) for li in layer_ids}

    total_requests = 0
    total_misses = 0
    predicted_misses = 0

    # Stats for the first token (no previous hidden available) are counted
    # in total_requests/total_misses but not eligible for prediction.
    for t, token_dict in enumerate(tokens):
        for li in layer_ids:
            if li not in token_dict:
                continue
            hidden, experts = token_dict[li]
            expert_ids = experts.tolist()

            misses = banks[li].query_and_update(expert_ids)
            total_requests += len(expert_ids)
            total_misses += len(misses)

            if misses and t > 0:
                # Predict using previous token's hidden state (N → N+1)
                prev_hidden = tokens[t - 1][li][0]
                pred_set = set(predictor.predict(li, prev_hidden))
                predicted_misses += len(misses & pred_set)

    hit_rate = 1.0 - total_misses / total_requests if total_requests else 0.0
    incr_hit_rate = predicted_misses / total_misses if total_misses else 0.0

    print()
    print("=== Exp01 N→N+1 Lookahead Validation ===")
    print(f"  Tokens evaluated:      {len(tokens)}")
    print(f"  Bank size:             {BANK_SIZE}")
    print(f"  Total expert requests: {total_requests}")
    print(f"  LRU misses:            {total_misses}")
    print(f"  LRU hit rate:          {hit_rate:.4f}  ({hit_rate*100:.2f}%)")
    print(f"  Misses caught (N→N+1): {predicted_misses}")
    print(f"  Incremental hit rate:  {incr_hit_rate:.4f}  ({incr_hit_rate*100:.2f}%)")
    print()
    gate = incr_hit_rate > 0.50
    print(f"  Gate (>50%): {'PASS ✓' if gate else 'FAIL ✗'}")
    if gate:
        print("  → Proceed to Step 2: prefetch integration")
    else:
        print("  → Do not integrate; predictor does not identify misses reliably")


if __name__ == "__main__":
    main()
