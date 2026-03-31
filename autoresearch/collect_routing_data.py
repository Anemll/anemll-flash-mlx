#!/usr/bin/env python3
"""
Collect routing samples for predictor training.

Runs the model in --resident mode (no SSD I/O) on diverse prompts,
capturing (hidden_state, expert_ids) pairs via --collect-routing.

Output: autoresearch/routing_samples.bin
Format per sample: int32(layer_idx), int32(k), float32[hidden_size], int32[k]
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MLX_MODEL = Path.home() / "Models/mlx-community-Qwen3.5-35B-A3B-4bit"
OUTPUT = REPO / "autoresearch/routing_samples.bin"

# Diverse prompts covering different domains and styles
PROMPTS = [
    # Science & Tech
    "Explain quantum computing in one concise paragraph.",
    "What is the difference between a transformer and an RNN?",
    "How does CRISPR gene editing work?",
    "Explain the theory of general relativity simply.",
    "What is a Fourier transform and why is it useful?",
    "How do neural networks learn from data?",
    "Explain how GPS satellites determine your location.",
    "What is the difference between fusion and fission?",
    "How does the immune system recognize pathogens?",
    "Explain the concept of entropy in thermodynamics.",
    # Math
    "What is the Riemann hypothesis?",
    "Explain Bayes' theorem with a simple example.",
    "What is the traveling salesman problem?",
    "How does gradient descent work in optimization?",
    "Explain the P vs NP problem.",
    # History & Culture
    "What caused the fall of the Roman Empire?",
    "Summarize the key events of World War II.",
    "What was the significance of the Silk Road?",
    "Explain the French Revolution in one paragraph.",
    "What were the main causes of the Cold War?",
    # Philosophy & Society
    "What is utilitarianism?",
    "Explain the trolley problem in ethics.",
    "What is the difference between democracy and republic?",
    "What is consciousness according to philosophy of mind?",
    "Explain the concept of free will.",
    # Practical / Everyday
    "How does a refrigerator work?",
    "Explain how the stock market works.",
    "What causes inflation?",
    "How do vaccines work?",
    "Explain how the internet routes data.",
    # Code / Logic
    "Write a Python function to reverse a linked list.",
    "Explain the difference between TCP and UDP.",
    "What is a hash table and how does it work?",
    "Explain binary search and its time complexity.",
    "What is the difference between processes and threads?",
    # Creative / Open-ended
    "Write a haiku about artificial intelligence.",
    "Describe the color blue to someone who has never seen it.",
    "What would happen if humans could photosynthesize?",
    "Write a one-paragraph story about a robot discovering music.",
    "Imagine you are explaining the internet to someone from 1800.",
    # Medicine / Biology
    "How does DNA replication work?",
    "Explain the difference between viruses and bacteria.",
    "What is the blood-brain barrier?",
    "How do antibiotics work?",
    "Explain how memory is formed in the brain.",
    # Economics
    "What is game theory?",
    "Explain supply and demand with an example.",
    "What is quantitative easing?",
    "What caused the 2008 financial crisis?",
    "Explain the concept of opportunity cost.",
]

QWEN_TEMPLATE = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def run_collection(prompt: str, output: Path, append: bool) -> bool:
    formatted = QWEN_TEMPLATE.format(prompt=prompt)
    cmd = [
        sys.executable, str(REPO / "scripts/run_qwen35.py"),
        "--mlx", str(MLX_MODEL),
        "--resident",
        "--prompt", formatted,
        "--max-tokens", "80",
        "--temperature", "0",
        "--collect-routing", str(output),
    ]
    if append:
        cmd.append("--append-routing")

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    total = len(PROMPTS)
    print(f"Collecting routing samples from {total} prompts...")
    print(f"Output: {OUTPUT}")
    print()

    ok = 0
    fail = 0
    t0 = time.time()

    for i, prompt in enumerate(PROMPTS):
        append = (i > 0)
        t1 = time.time()
        success = run_collection(prompt, OUTPUT, append)
        elapsed = time.time() - t1

        if success:
            ok += 1
            status = f"ok ({elapsed:.1f}s)"
        else:
            fail += 1
            status = f"FAILED ({elapsed:.1f}s)"

        print(f"[{i+1:02d}/{total}] {status} — {prompt[:60]}")

    total_elapsed = time.time() - t0
    size_mb = OUTPUT.stat().st_size / 1e6 if OUTPUT.exists() else 0
    print()
    print(f"Done: {ok} ok, {fail} failed in {total_elapsed/60:.1f}m")
    print(f"Output size: {size_mb:.1f} MB")
    print(f"File: {OUTPUT}")


if __name__ == "__main__":
    main()
