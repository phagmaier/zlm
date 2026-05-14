# Zig-LLM (Ai) 🤖

A lightweight, hand-written Large Language Model (LLM) implementation in Zig, featuring a custom tensor library, reverse-mode autograd engine, BPE tokenizer, and transformer architecture.

## 🏗️ Project Overview

- **Core Technology:** Zig 0.16.0, OpenBLAS (via CBLAS).
- **Architecture:** Hand-written transformer with components like Embedding, Linear, RMSNorm, SwiGLU, and Attention (Causal + RoPE).
- **Key Modules:**
    - `src/tensor.zig`: N-dimensional tensor library with strided views and BLAS-accelerated matmul.
    - `src/autograd.zig`: Reverse-mode autograd, transformer layers, and AdamW optimizer.
    - `src/tokenizer.zig`: BPE tokenizer (training, encoding, decoding).
    - `src/main.zig`: CLI application orchestrating tokenization, training, and sampling.

## 🚀 Building and Running

### Prerequisites
- **Zig 0.16.0**
- **OpenBLAS** (Development headers required)
- **Linux Workaround:** Run `./crt/make-crt.sh` once after cloning or upgrading glibc to handle `.sframe` linker issues.

### Key Commands
- **Build (Optimized):** `zig build -Doptimize=ReleaseFast`
- **Run Tests:** `zig build test --summary all`
- **Tokenize Corpus:** `zig build run -Doptimize=ReleaseFast -- tokenize`
- **Train Model:** `zig build run -Doptimize=ReleaseFast -- train`
- **Sample/Generate:** `zig build run -Doptimize=ReleaseFast -- sample [prompt]`
- **Interactive Chat:** `zig build run -Doptimize=ReleaseFast -- chat`

### Data Paths
- Training data: `Data/training_data.txt`
- Tokenizer merges: `Data/saved.bin`
- Encoded corpus: `Data/encoded.bin`
- Model weights: `Data/weights.bin`
- Optimizer state: `Data/checkpoint_opt.bin`

## 🛠️ Development Conventions

- **Memory Management:** Manual memory management via Zig's `std.mem.Allocator`. The tensor library uses reference-counted storage for efficient views.
- **Testing:** Unit tests are extensively used (77+ tests). Run them with `zig build test`.
- **Formatting:** Use standard `zig fmt`.
- **Math Performance:** Heavy lifting for matrix multiplication is delegated to OpenBLAS `sgemm`.

## 🧩 Model Defaults (configurable in `src/main.zig`)
- **Vocab Size:** 4096
- **Model Dimension (D_MODEL):** 256
- **Heads:** 4
- **Layers:** 6
- **Sequence Length:** 512
- **Optimizer:** AdamW (LR: 3e-4, Warmup: 1000 steps)
