# anemll-flash-mlx Autoresearch

## Project
Optimizing `anemll-flash-mlx` — a Python + MLX inference engine for large MoE models on Apple
Silicon. Dense compute stays in MLX. Sparse expert I/O is handled by `csrc/expert_io.c` via a
stable slot-bank with hit/miss separation.

**Target model:** Qwen3.5-35B-A3B-4bit (256 experts/layer, k=4)

## Hardware
- Machine: Apple MacBook Pro M5 Max, 128GB unified memory
- Storage: 2TB NVMe (~17.5 GB/s sequential read)
- Stack: Python 3.10+, MLX, clang

## CAMPAIGN COMPLETE — Final Results

**55.74 tok/s at 500 tokens — 96.5% of oracle ceiling. No further experiments warranted.**

| Metric | Value |
|---|---|
| Final decode speed | **55.74 tok/s** (avg runs 2+3, 500 tokens) |
| Hit rate | **92.11%** |
| Oracle ceiling (bank=176, perfect prediction) | **57.74 tok/s** |
| Gap to ceiling | **2.00 tok/s (3.5%)** |
| Max unique experts per layer | 234 (across 500 tokens) |
| Irreducible misses | ~6,400 per 500-token generation (cold-start floor) |

**Why the campaign is closed:** At 500 tokens, the slot-bank is already at 96.5% of the
theoretical oracle ceiling. The remaining 2 tok/s gap is cold-start misses for experts that
appear for the first time — no eviction policy, predictor, or prefetch strategy can eliminate
these without perfect future knowledge. All five optimization levers were evaluated:

| Lever | Result |
|---|---|
| LFU eviction (Lever 2) | Discard — 170 unique experts fit in bank=176; LRU and LFU make identical decisions |
| Routing predictor N→N+1 (Lever 1) | Discard — 3.35% incremental hit rate (gate=50%); token N hidden state has no signal for token N+1 experts |
| I/O fanout (Lever 3) | Exhausted — cache-io-split=4 is optimal for 35B expert size |
| Async prefetch (Lever 4) | Not viable — MLX threading constraints prevent background compute |
| Python→C migration (Lever 5) | Not attempted — ceiling analysis shows <2 tok/s upside; not worth the complexity |

## Key files
- `flash_moe_mlx/model.py` — MLX runtime, routing, generation loop
- `flash_moe_mlx/expert_io.py` — expert geometry, slot-bank management
- `csrc/expert_io.c` — native pread I/O with thread pool, LRU victim selection
- `scripts/run_qwen35.py` — main inference entrypoint
- `tools/diagnostics/bench_slot_bank_oracle_hits.py` — oracle ceiling tool
- `autoresearch/predictor_weights.npz` — trained predictor weights (84.2 MB, 40 layers; not integrated)
- `autoresearch_results.tsv` — full experiment log

## Standard benchmark command
Run 3x, discard run 1 (cold-start), average runs 2+3. Always report both tok/s and hit rate.

```bash
python scripts/run_qwen35.py \
  --mlx ~/Models/mlx-community-Qwen3.5-35B-A3B-4bit \
  --experts ~/Models/flash_mlx_35b_4bit/packed_experts \
  --slot-bank 176 --cache-io-split 4 --stream \
  --prompt "Explain the differences between transformer attention variants in detail." \
  --max-tokens 500 --k 4 --temperature 0
```

## Autoresearch rules (for any future campaign on a different model)
- **Measure first.** Run 3x, discard run 1 (cold-start), average runs 2+3. Never report a
  single run or include the cold-start run in the average.
- **Always use --max-tokens 500.** The 120-token number is cold-start-dominated and
  misleading. All benchmarks use 500 tokens so the bank is warm and hit rate is stable.
- **Check oracle ceiling first.** If actual ≥ 95% of oracle, the campaign is over.
- **Commit on success.** If tok/s improves AND hit_rate_pct does not decrease AND output is
  coherent, log as `keep` and commit.
- **Revert on failure.** If performance drops or output degrades, log as `discard` with a
  one-line explanation, then `git reset --hard`.
- **Do not trade hit rate for speed.** A change that raises tok/s by dropping experts silently
  is a bug, not an optimization.

## Oracle ceiling reference command
```bash
python tools/diagnostics/bench_slot_bank_oracle_hits.py \
  --mlx ~/Models/mlx-community-Qwen3.5-35B-A3B-4bit \
  --experts ~/Models/flash_mlx_35b_4bit/packed_experts \
  --slot-bank 176 --max-tokens 500 --k 4 \
  --prompt "Explain the differences between transformer attention variants in detail."
```
