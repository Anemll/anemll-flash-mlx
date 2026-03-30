# anemll-flash-mlx

Flash-MoE inference for large Mixture-of-Experts models on Apple Silicon, using MLX.


## Purpose

Flash-MoE is not mainly about SSD reads.
It is about changing the boundary between dense execution, expert storage, expert selection, and expert consumption.

The dense path should stay inside the backend that is already good at it.
The sparse path should be reshaped around a stable bank or slot-bank plus ids-as-data.

By "banking" we mean this: instead of treating every routed expert as a fresh tensor object that has to be materialized on demand, the runtime keeps a small resident execution surface per layer, made of stable slots. A routed `expert_id` is first resolved to a resident `slot_id`; if the expert is already there, execution is just an indexed hit, and if not, the runtime loads the expert bytes and commits them into a victim slot before use. The important part is that the execution shape stays stable while expert identity changes, which is much cheaper than rebuilding a tiny K-expert bank every token.

## The big lesson from the MLX port

A stable bank with changing ids was fast.
Per-token K-expert materialization was slow.

The expensive part was not dynamic routing — it was asking the framework to keep rebuilding the selected experts in the hot path. Once that became clear, the rest of the architecture started to make sense. A streamed slot-bank became the first realistic external-expert shape. Miss path and hit path had to be treated as different problems. And oracle ceilings became mandatory, because they tell you whether the next problem is consume, commit, or prefetch.

## The reusable workflow

This is the workflow we would reuse in another backend.

1. Measure the normal resident baseline.
2. Build a resident packed-bank path.
3. Prove that changing ids is cheap on that good path.
4. Add one intentionally bad K-materialization diagnostic.
5. Build a real streamed slot-bank.
6. Measure locality before designing fancy cache policy.
7. Build oracle all-hit replay and one-step oracle prefetch replay.
8. Decide whether the next work belongs in the hit path or the miss path.
9. Only then move the boundary to native consume if framework-level slot-bank work stalls.

## What matters most

Stable bank/slot + ids is the first shape to try.
Per-token K-expert materialization is useful as a diagnostic, but usually the wrong product shape.
Hit path and miss path are different bottlenecks.
Early slot-bank numbers are rungs, not ceilings.
And native code only matters if the boundary really moves.

The repo is kept intentionally small — a runtime, a native I/O helper, a few export scripts, and diagnostics. No experiment archive, no framework abstractions. The goal is a codebase you can read in one sitting and modify without tracing through layers of indirection. Measure first, change one thing, measure again.

Currently targets **Qwen3.5-35B-A3B** and **Qwen3.5-397B-A17B** via [mlx-community](https://huggingface.co/mlx-community) 4-bit checkpoints. Also supports dynamic quantization checkpoints such as [Qwen3.5-35B-A3B-UD-Q2_K_XL-mlx](https://huggingface.co/Brooooooklyn/Qwen3.5-35B-A3B-UD-Q2_K_XL-mlx) (Unsloth-style mixed 2/3/4/5/8-bit).

## Quick start

Requirements: Python 3.10+, Apple Silicon, `clang` (for native I/O helper).

```bash
# install with uv (recommended)
uv sync
source .venv/bin/activate

# install in editable mode
uv pip install -e .

# optional: export dependencies (torch + safetensors)
uv sync --extra export
# or: uv pip install -e ".[export]"

# verify
make smoke
```

## Exporting packed experts

Before running streamed inference you need to export the expert sidecar from the MLX checkpoint.

### Mixed-precision sidecar (preserves original quantization)

```bash
python3 scripts/export_mixed_sidecar.py \
  --model ~/Models/mlx-Qwen3.5-35B-A3B-4bit \
  --output ~/Models/packed_experts
```


## Running inference

The runtime takes two separate paths — that split is the whole point:

- `--mlx` points to the dense MLX model directory
- `--experts` points to the packed expert sidecar (created above)

### Resident mode (all experts in MLX memory — model must fit in RAM; skip for large models)

```bash
python3 scripts/run_qwen35.py \
  --mlx ~/Models/mlx-Qwen3.5-35B-A3B-4bit \
  --resident \
  --prompt "What is Apple Neural Engine?" \
  --max-tokens 120 --temperature 0 --stream
```

### Streamed experts from SSD

```bash
python3 scripts/run_qwen35.py \
  --mlx ~/Models/mlx-Qwen3.5-35B-A3B-4bit \
  --experts ~/Models/packed_experts \
  --prompt "What is Apple Neural Engine?" \
  --max-tokens 120 --k 4 --temperature 0 \
  --slot-bank 64 --slot-bank-native --prefetch-temporal \
  --cache-io-split 4 --stream
```

### Benchmarks — M5 Max 128 GB

**Qwen3.5-35B-A3B-4bit** (256 experts/layer, k=4, 120 tokens)

| Mode | Decode tok/s | Hit rate | Notes |
|------|-------------|----------|-------|
| `--resident-pread-mlx` | **101.6** | 100% | Packed-bank ceiling |
| `--resident` | **94.5** | 100% | All experts in MLX memory |
| `--slot-bank 128` | **47.1** | 82.7% | Best SSD config |
| `--slot-bank 64` | **42.8** | 80.8% | Good SSD default |
| `--slot-bank 16` | **17.1** | 59.5% | Small bank, many misses |

**Qwen3.5-35B-A3B-UD-Q2_K_XL-mlx** (dynamic quant, 256 experts/layer, k=4, 120 tokens)

| Mode | Decode tok/s | Hit rate | Expert size | Notes |
|------|-------------|----------|-------------|-------|
| `--resident-pread-mlx` | **91.1** | 100% | 0.94 MB | Packed-bank ceiling |
| `--resident` | **86.8** | 100% | 0.94 MB | All experts in MLX memory |
| `--slot-bank 128` | **42.2** | 82.5% | 0.94 MB | Best SSD config |
| `--slot-bank 64` | **20.9** | 80.7% | 0.94 MB | Cold-cache first run |

Expert size: 4-bit = 1.69 MB, UD-Q2 = 0.94 MB (44% smaller, 37% less SSD I/O)

## How it works

The repo has a small number of moving parts:

| File | Purpose |
|------|---------|
| `flash_moe_mlx/model.py` | MLX runtime: loading, routing, generation |
| `flash_moe_mlx/expert_io.py` | Expert geometry, packed expert parsing, slot-bank |
| `csrc/expert_io.c` | Native pread I/O with thread pool |
| `scripts/run_qwen35.py` | Main inference entrypoint |
| `scripts/export_mixed_sidecar.py` | Export experts preserving quantization |
| `scripts/export_tiered_35b_2bit.py` | Convert tiered experts to uniform 2-bit |

The native disk-I/O helper (`csrc/expert_io.c`) is built automatically on first use via `clang`.

## Runtime modes

- **`--resident`** — Keep all experts in MLX memory. Ceiling check, no SSD needed.
- **`--slot-bank N`** — Stable per-layer slot bank of size N, reload only misses from SSD.
- **`--slot-bank-native`** — Use native C slot ownership and LRU victim selection.
- **`--resident-pread-mlx`** — Packed-bank ceiling: pread all experts at startup, index by id.
- **`--prefetch-temporal`** — After each token, prefetch same experts for next step.
- **`--compiled-tail`** — Compile the final sparse-block shared-expert tail.
- **`--2-bit`** — Use 2-bit affine experts (smaller, faster, lower quality).

## Diagnostics

```bash
# Oracle all-hit and one-step oracle prefetch replay
python3 tools/diagnostics/bench_slot_bank_oracle_hits.py --help

# Miss-service timing
python3 tools/diagnostics/bench_slot_commit.py --help

# Repeatable capture/regression runs
python3 tools/diagnostics/bench_neo_capture.py --help
```

## Project structure

```
flash_moe_mlx/          runtime package
scripts/                main entrypoints
csrc/                   native disk-I/O helper
tools/diagnostics/      diagnostics
examples/               copy-paste shell examples
ci/                     smoke checks
tests/                  import tests
docs/                   notes
```

## Acknowledgements

- Apple's [LLM in a Flash](https://machinelearning.apple.com/research/efficient-large-language) and [M2R2](https://machinelearning.apple.com/research/multi-rate-residuals-transformers) papers
- [danveloper/flash-moe](https://github.com/danveloper/flash-moe) — the original Flash-MoE implementation
- [@alexintosh](https://github.com/alexintosh) and `@danpacary` (https://github.com/ncdrone/)  for ideas and improvements

## Contact

[@anemll](https://x.com/anemll) on X

## License

MIT. See [LICENSE](LICENSE).
