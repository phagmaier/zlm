# Zig-LLM 🤖

A small Large Language Model (LLM) implementation built from scratch in the Zig programming language. The full pipeline — tensor library, autograd engine, BPE tokenizer, transformer model, training loop, and sampler — is hand-written, with no external ML dependencies beyond OpenBLAS for matmul.

---

## ✨ What's in here

- **Tensor library** (`src/tensor.zig`) — N-dimensional tensors with strided views, broadcasting, reshapes, transposes, and BLAS-accelerated matmul via OpenBLAS `sgemm`. Reference-counted storage so views are cheap.
- **Reverse-mode autograd** (`src/autograd.zig`) — Builds a dynamic graph during the forward pass, then walks it in reverse-topological order on `backward`. Supported ops: add, mul, matmul (with transpose flags), embedding, RMS-norm primitives (rsqrt, sum-keepdim, mul-scalar), SiLU, softmax, log-softmax, cross-entropy, RoPE, reshape/transpose/contiguous.
- **Transformer layers** — `Embedding`, `Linear`, `RMSNorm`, `SwiGLU` MLP, multi-head `Attention` with causal mask + RoPE, full `TransformerBlock` (pre-norm + residual), and a stack-of-blocks `Model`.
- **Optimizer** — `AdamW` with bias correction and weight decay.
- **BPE tokenizer** (`src/tokenizer.zig`) — Linked-list + pair-count training, priority-queue-based encoding for $O(N \log N)$ tokenization, four reserved special tokens (`<|system|>`, `<|user|>`, `<|assistant|>`, `<|endoftext|>`), and streaming corpus encoding.
- **Training loop** (`src/main.zig`) — Random-window sampling from an mmap'd encoded corpus, gradient clipping, cosine LR schedule with linear warmup, and checkpointed AdamW state for resumable multi-day runs.
- **Sampler** — Temperature + top-K decoding via the same `Model.forward` used in training.

77 unit tests cover the tensor library, autograd ops, layer forward/backward shapes, tokenizer round-trip, and weight save/load.

---

## 🧩 Model defaults

Set as constants at the top of `src/main.zig`:

| | |
|---|---|
| `VOCAB_SIZE` | 4096 |
| `D_MODEL` | 256 |
| `N_HEADS` | 4 (head dim 64) |
| `D_FFN` | 1024 |
| `N_LAYERS` | 6 |
| `SEQ_LEN` | 512 |
| `BATCH_SIZE` | 1 |
| `LR` | 3e-4 (cosine to `LR*0.1`) |
| `WARMUP_STEPS` | 1000 |
| `N_STEPS` | 300000 |
| `CHECKPOINT_EVERY` | 1000 |
| `GRAD_CLIP` | 1.0 |
| `BETA1 / BETA2 / EPS / WEIGHT_DECAY` | 0.9 / 0.999 / 1e-8 / 0.01 |

These are sized for a multi-day CPU run on a modest desktop. Edit and rebuild to change.

---

## 🚀 Quick start

### 1. Prerequisites

- **Zig 0.16.0**
- **OpenBLAS**
    - Arch: `sudo pacman -S openblas pkgconf`
    - Ubuntu/Debian: `sudo apt-get install libopenblas-dev pkg-config`
    - macOS: `brew install openblas pkg-config`
- **binutils** (`objcopy`) — only needed on Linux to regenerate `crt/` (see workaround section).

### 2. Provide training data

Put your raw text corpus at `Data/training_data.txt`. UTF-8, plain text. The tokenizer is trained on the first 50 MB; the entire file is then encoded into `Data/encoded.bin`. For best results with the default 4096-vocab BPE, target at least a few hundred MB of text.

If your data is conversational and you want sampling to start from a turn boundary, format it with the reserved special tokens:

```
<|user|>Hello, who are you?<|assistant|>I'm a tiny transformer.<|endoftext|>
<|user|>...
```

### 3. Build

```bash
zig build -Doptimize=ReleaseFast
```

The binary lands at `zig-out/bin/Ai`.

### 4. Tokenize the corpus (one-time)

```bash
zig build run -Doptimize=ReleaseFast -- tokenize
```

This trains the BPE on the first 50 MB of `Data/training_data.txt`, writes the merge table to `Data/saved.bin`, and streams the full corpus into `Data/encoded.bin`.

### 5. Train

```bash
zig build run -Doptimize=ReleaseFast -- train
```

Logs every 100 steps look like:

```
step 1000/300000: loss 6.2143 | lr 3.00e-04 | 1.23 steps/s | ETA 67h32m18s
```

Checkpoints (`Data/weights.bin` + `Data/checkpoint_opt.bin`) are written every 1000 steps. Re-running `train` automatically resumes from the latest checkpoint. If the checkpoints are missing or fail to load (e.g. dimensions changed), training starts from scratch.

### 6. Sample

After (or during, between training sessions) training:

```bash
zig build run -Doptimize=ReleaseFast -- sample
zig build run -Doptimize=ReleaseFast -- sample "<|user|>Tell me a story.<|assistant|>"
```

Generates 200 tokens with temperature 0.8 and top-K 40. The default prompt is `<|user|>`.

### Other subcommands

```bash
zig build run -- save        # write fresh random weights to Data/weights.bin
zig build run -- load        # initialize a Model and load Data/weights.bin (smoke test)
```

### Run tests

```bash
zig build test --summary all
```

### `run.sh`

Tiny convenience wrapper:

```bash
./run.sh test
./run.sh release   # zig build run -Doptimize=ReleaseFast
```

---

## 🖥️ Running a multi-day training session on a fresh machine

Here's the exact path from a clean clone to a model you can sample from. Assumes you have a copy of `training_data.txt` on a USB stick, in cloud storage, or wherever you keep it.

```bash
# 0. Prereqs (Arch example; substitute for your distro)
sudo pacman -S zig openblas pkgconf  # or install Zig 0.16.0 from ziglang.org

# 1. Clone
git clone <your-repo-url> Ai
cd Ai

# 2. Drop the training data in place
mkdir -p Data
cp /path/to/training_data.txt Data/training_data.txt

# 3. (Linux only, if you hit the .sframe linker error — see workaround section below)
./crt/make-crt.sh

# 4. Build optimized
zig build -Doptimize=ReleaseFast

# 5. Sanity-check the build (optional but recommended)
zig build test --summary all

# 6. Tokenize the corpus. This produces Data/saved.bin (merges) and Data/encoded.bin.
#    Takes a few minutes to an hour depending on corpus size.
zig build run -Doptimize=ReleaseFast -- tokenize

# 7. Kick off training. Logs every 100 steps. Checkpoints every 1000.
#    Use tmux/screen/nohup so it survives disconnects.
tmux new -s train
zig build run -Doptimize=ReleaseFast -- train | tee Data/train.log
# Detach with Ctrl+B, then D. Reattach with: tmux attach -t train

# 8. If the machine reboots or training stops, just re-run step 7 — it picks
#    up from the last checkpoint automatically.

# 9. When you want to peek at progress, sample from another shell:
zig build run -Doptimize=ReleaseFast -- sample "<|user|>What is the meaning of life?<|assistant|>"
```

The first ~1000 steps go through linear warmup (LR ramps up to 3e-4). After that the cosine schedule slowly decays toward `LR * 0.1`. With 300K steps at `SEQ_LEN=512`, you're training on roughly 150M tokens of context if `BATCH_SIZE=1`. At ~1 step/s on a modest CPU, expect ~3-4 days of wall time. Watch the ETA in the log line.

---

## 📁 Files written / read

| Path | Owner | What |
|------|-------|------|
| `Data/training_data.txt` | you | Source corpus (UTF-8, gitignored) |
| `Data/saved.bin` | `tokenize` | BPE merge table |
| `Data/encoded.bin` | `tokenize` | u16-token-stream of the corpus, with an 8-byte length header (gitignored) |
| `Data/weights.bin` | `train` / `save` | Model parameters, written every `CHECKPOINT_EVERY` steps |
| `Data/checkpoint_opt.bin` | `train` | AdamW state (step, t, m, v buffers) — paired with `weights.bin` |

---

## 🐧 Linux `.sframe` workaround

If your distro is modern enough that binutils emits `.sframe` unwind info in glibc's CRT objects (Arch with GCC ≥ 16.1, recent Debian/Fedora, etc.), Zig 0.16's LLD will refuse to link with:

```
error: fatal linker error: unhandled relocation type R_X86_64_PC64 at offset 0x1c
    note: in /usr/lib/.../crt1.o:.sframe
```

**Why we link libc at all.** OpenBLAS on most Linux distros is built against OpenMP (`libgomp`). When a matmul is big enough to take the parallel path, libgomp calls `malloc`, which reads glibc's thread-local storage at offset `0xf0` (`thread_arena`). `libc.so` is marked `STATIC_TLS`, so that slot only exists if the executable was linked against libc at startup. Without it, the first non-trivial matmul (i.e. anywhere in `train`) segfaults. Small matmuls in the test suite stay on the serial path and never hit this.

**The fix.** This repo ships `crt/` with stripped copies of `crt1.o`, `Scrt1.o`, `crti.o`, `crtn.o` (the `.sframe` and `.rela.sframe` sections removed via `objcopy`), plus a `libc.txt` telling Zig to use them. `build.zig` wires it in with `setLibCFile`. Functionally identical CRTs, just missing the unwind metadata LLD chokes on.

Regenerate `crt/` when:

- You clone on a new Linux box for the first time.
- Your system glibc gets upgraded.

```bash
./crt/make-crt.sh
```

**Cleanup someday.** Once LLD handles `R_X86_64_PC64` against `.sframe` (or distros stop emitting it), delete `crt/` and the `setLibCFile` calls in `build.zig` and use a vanilla `link_libc = true` build.

macOS users: ignore this section — Apple's CRT setup is different and Zig links libc cleanly there.

---

## 🗂️ Project layout

- `src/tensor.zig` — Tensor data structures, views, broadcasting, CBLAS-backed matmul.
- `src/autograd.zig` — Reverse-mode autograd, layers (Embedding, Linear, RMSNorm, SwiGLU, Attention, TransformerBlock, Model), AdamW, weight save/load.
- `src/tokenizer.zig` — BPE training, encoding, save/load, special-token handling.
- `src/main.zig` — CLI (`tokenize` / `train` / `sample` / `save` / `load`), training loop with checkpointing, sampler.
- `src/root.zig` — Library root re-exports for the test runner.
- `src/cblas_wrapper.h` — Single-line include consumed by `addTranslateC`.
- `Data/` — Training corpus and checkpoints (gitignored, mostly).
- `crt/` — Stripped Linux CRT objects + `libc.txt`. See workaround section.
- `build.zig` / `build.zig.zon` — Build config.

---

## 📜 License

Educational project. Use it, learn from it, fork it, break it.
