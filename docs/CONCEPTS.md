# Zlm: Concepts and Implementation Reference

A walkthrough of every major concept used to build this LLM, with notes on how each is implemented in this codebase. Read top to bottom and you have a complete picture of how the model trains and generates text.

---

## Part 1: Foundations

### 1.1 BPE Tokenizer

**Theory.** Byte-Pair Encoding starts with the raw byte alphabet (256 symbols) and repeatedly merges the most frequent adjacent pair of tokens into a new token. After `N` merges, the vocabulary has `256 + N` symbols. The merge list is the model. Encoding new text means greedily applying merges in learned order; decoding reverses them.

Why BPE: it's a middle ground between character-level (vocab tiny, sequences long) and word-level (vocab huge, OOV problems). It lets common substrings ("the ", "ing", "tion") become single tokens, and rare words gracefully fall back to constituent pieces.

**This codebase.** `src/tokenizer.zig`. Three entry points:

- `train` — count adjacent pair frequencies across the corpus, pick the most frequent, emit a merge rule, repeat until vocab size hits target.
- `encode` — apply the merge list to a byte string to produce token IDs.
- `decode` — invert the merges, produce bytes.

Merges are stored in `Data/saved.bin`, the encoded corpus in `Data/encoded.bin` (one `u16` per token since vocab fits in 16 bits).

### 1.2 Tensors with Strided Views

**Theory.** A tensor is a flat buffer of floats plus metadata: shape, strides, offset. Strides let you walk dimensions independently — element `(i, j)` lives at `offset + i*stride[0] + j*stride[1]`. This is what makes transposes, slices, and reshapes free: change the metadata, share the buffer.

A view is **contiguous** if its strides match what you'd get from a fresh allocation (rightmost stride = 1, each stride to the left = next stride × next dimension's size). Contiguous data can be `memcpy`'d and walked with a tight inner loop; non-contiguous data has to be navigated by strides.

**This codebase.** `src/tensor.zig`. `Tensor` owns a refcounted `Storage` (the actual `[]f32`) plus its own `shape`, `strides`, `offset`. Helpers:

- `transpose(perm)` — returns a new `Tensor` view with permuted shape and strides, same storage.
- `contiguous()` — if already contiguous, returns self; otherwise materializes a fresh contiguous copy.
- `contiguousSlice()` — fast path for hot loops: returns `?[]const f32` only if the view is contiguous, so callers can bypass the strided indexer.

Why this matters: every transpose in attention used to call `.contiguous()`, allocating and copying. We now pass transpose flags to BLAS instead (next section), so transposes are truly free.

### 1.3 Matrix Multiplication via BLAS (sgemm)

**Theory.** `sgemm` is the most heavily optimized routine in numerical computing. It computes `C = alpha * op(A) * op(B) + beta * C`, where `op` is optional transpose. Modern BLAS implementations (OpenBLAS, MKL) use blocked algorithms, SIMD intrinsics, and threading; you cannot beat them with hand-written code.

Key parameters: `M`, `N`, `K` (output shape and contraction dim), `lda`/`ldb`/`ldc` (leading dimensions — the stride between rows in row-major), and the transpose flags `CblasTrans`/`CblasNoTrans`.

**This codebase.** `src/tensor.zig: matmul`. Two key design decisions:

1. **Pass transpose flags, don't materialize transposes.** If the user asks for `A.T @ B`, we set `CblasTrans` for `op(A)` and pass `A`'s raw storage. The math is the same, allocation is zero. The 4 combinations of `(trans_left, trans_right)` are all handled in `matmulT`.

2. **1D-promoted right operand special case.** When you multiply `(M, K) @ (K,)` (matrix times vector), the right operand is logically `(K, 1)` and its row stride is 1 — not `r_phys_cols`. We detect this and set `ldb = 1`. This was a real bug we hit during the transpose-flag refactor.

### 1.4 The Storage / Tensor / Val Layers

This codebase has three layers, easy to confuse:

| Layer | Owns | Lifetime | Purpose |
|---|---|---|---|
| `Storage` | `[]f32` data | refcounted | The actual numbers |
| `Tensor` | shape, strides, offset | manual | A view into a Storage |
| `Val` | a `Tensor` + grad + parents + backward fn | arena-allocated | A node in the autograd graph |

`Tensor` is the math object. `Val` wraps a `Tensor` and records what operation produced it, so backward can replay the graph in reverse.

---

## Part 2: The Model

### 2.1 Embedding Layer

**Theory.** An embedding is just a lookup table: a learned matrix of shape `(vocab_size, d_model)` where row `t` is the dense vector representation of token `t`. Forward is `output[i] = E[input_ids[i]]`. Gradient is sparse: only the rows for tokens that appeared in the batch get a nonzero gradient.

**This codebase.** `src/autograd.zig: Embedding`. Forward gathers rows. Backward does sparse row scatter-add into the gradient buffer.

### 2.2 RMSNorm

**Theory.** LayerNorm normalizes activations to zero mean, unit variance, then applies a learned scale and bias: `y = gamma * (x - mean) / sqrt(var + eps) + beta`. RMSNorm drops the mean-centering and bias: `y = gamma * x / sqrt(mean(x^2) + eps)`.

Why simpler? Empirically the mean-centering term in LayerNorm contributes very little to model quality, and removing it cuts about half the FLOPs of the normalization. Modern transformers (LLaMA, Mistral) use RMSNorm universally. Also: it's numerically stabler because you can't accidentally subtract two near-equal large numbers.

**This codebase.** `src/autograd.zig: RMSNorm`. Forward: compute `rms = sqrt(mean(x^2) + eps)`, scale `x` by `gamma / rms`. Backward derives w.r.t. `x` and `gamma`; the `x` gradient has two terms (direct and through `rms`) which fold together.

### 2.3 RoPE (Rotary Position Embedding)

**Theory.** Traditional positional encoding is additive — you add a position vector to the token embedding. RoPE is multiplicative and applies inside attention: it rotates pairs of dimensions in Q and K by an angle proportional to position.

For each pair `(x_{2i}, x_{2i+1})` at position `p`, rotate by `theta_i * p` where `theta_i = base^(-2i/d_head)` and `base = 10000`. The rotation matrix is the standard 2D one: `(cos*a - sin*b, sin*a + cos*b)`.

The reason this works for attention: the dot product `Q_p · K_q` after rotation only depends on the relative position `p - q`, not on the absolute positions. So the model gets relative-position information for free, with no learned parameters and excellent extrapolation behavior.

**This codebase.** `src/autograd.zig: applyRope` (and friends). Precomputes `cos` and `sin` tables of shape `(seq_len, d_head/2)`, then for each `(token_position, head_pair)` does the 2D rotation in place on Q and K (V is not rotated — only the query/key match needs position info).

### 2.4 Causal Mask

**Theory.** In a decoder LM, when predicting token `t`, the model must not see tokens at positions `> t` — that would be cheating. The mask is a `(seq_len, seq_len)` matrix with `0` below and on the diagonal, `-inf` above. It's added to the pre-softmax attention scores; the `-inf` entries become 0 after softmax.

**This codebase.** `src/autograd.zig: causalMask`. Constructed once per batch element (it's the same for every layer and every head, so it could even be precomputed once globally — minor optimization for later).

### 2.5 Multi-Head Self-Attention

**Theory.** Self-attention: given input `X` of shape `(T, d_model)`, project to three matrices `Q, K, V` each `(T, d_model)`. Reshape into `H` heads of dim `d_head = d_model / H`, giving `(H, T, d_head)`. For each head:

```
scores = Q @ K.T / sqrt(d_head)        # (T, T)
scores += causal_mask                  # mask out future
weights = softmax(scores, axis=-1)     # row-stochastic
output  = weights @ V                  # (T, d_head)
```

Concatenate heads back to `(T, d_model)`, apply output projection.

Intuition: each query token looks across all earlier tokens, computes a similarity (dot product), normalizes (softmax), and takes a weighted average of values. Multi-head means doing this with several different `Q/K/V` subspaces in parallel — different heads can attend to different patterns.

The `sqrt(d_head)` scaling is to keep the dot-product variance from blowing up with dimension, which would push softmax into one-hot territory.

**This codebase.** `src/autograd.zig: Attention.forward`. Four linear projections (`Wq`, `Wk`, `Wv`, `Wo`), reshape into heads, apply RoPE to Q and K, dot-product with `matmulT(Q_rope, K_rope, trans_l=false, trans_r=true)` so we never materialize the transpose of K, add causal mask, softmax, multiply by V, concatenate, project.

### 2.6 SwiGLU FFN

**Theory.** A transformer FFN has historically been `Linear → ReLU → Linear` with a hidden dim of `4 * d_model`. SwiGLU replaces this with a *gated* variant:

```
FFN(x) = (swish(x @ W_gate) * (x @ W_up)) @ W_down
swish(x) = x * sigmoid(x)
```

There are three matrices instead of two: a gate path that gets passed through swish, an up path that doesn't, and the elementwise product of the two is then projected down. Empirically this is consistently better than ReLU FFNs at the same parameter count, which is why every modern open-weight model uses it.

**This codebase.** `src/autograd.zig: SwiGLU`. Three `Linear` layers: `w_gate`, `w_up`, `w_down`. Forward implements the formula directly. Backward differentiates through the swish (which needs the gate pre-activation) and the elementwise multiply.

### 2.7 Transformer Block (Pre-Norm)

**Theory.** The "block" is the repeating unit. Modern transformers use pre-norm:

```
x = x + Attention(RMSNorm(x))
x = x + FFN(RMSNorm(x))
```

Two residual connections, two RMSNorms (one before attention, one before FFN). The pre-norm variant (norm before sublayer, not after) is more stable to train than the post-norm version from the original Transformer paper.

**This codebase.** `src/autograd.zig: Block.forward`. Stacks `N_LAYERS` of these. The final output goes through one more RMSNorm and then a linear projection to vocab size (the "LM head").

### 2.8 Cross-Entropy Loss for Next-Token Prediction

**Theory.** The model outputs logits of shape `(T, vocab_size)`. The target at position `t` is `input_ids[t+1]` (predict the next token). Cross-entropy loss for one position:

```
loss_t = -log(softmax(logits_t)[target_t])
       = -logits_t[target_t] + logsumexp(logits_t)
```

Averaged over all positions. The `logsumexp` formulation is what you compute in practice — it's numerically stable (subtract max before exp).

**This codebase.** `src/autograd.zig: crossEntropy`. Forward returns scalar loss. Backward analytically computes `d_loss / d_logits = softmax(logits) - one_hot(target)`, which is the elegant gradient that makes cross-entropy + softmax cheap.

---

## Part 3: Training

### 3.1 Reverse-Mode Autograd

**Theory.** Reverse-mode automatic differentiation computes gradients by running the chain rule backwards through a computation graph. Build the graph during forward: every operation produces a new node that remembers its inputs and a `backward` closure that, given an upstream gradient, computes downstream gradients.

To get `d_loss/d_param` for every parameter: start with `d_loss/d_loss = 1`, walk the graph in reverse topological order, accumulating gradients into each node's `grad` field. At the leaves (the parameters), you have what AdamW needs.

The trick that makes this efficient: each node only needs to handle its *local* derivative. The chain rule is applied automatically by the graph walk.

**This codebase.** `src/autograd.zig`. `Val` is the node type — it owns its forward output `Tensor`, a `grad` `Tensor` of the same shape, a list of parents, and a `backward` function. Every operation (`add`, `matmul`, `softmax`, `rmsnorm`, ...) is a constructor that builds a new `Val` wired to its inputs.

`Val.backward()` does topological sort + reverse walk. Gradients accumulate via `accumulateGrad` (because the same tensor can be used in multiple downstream ops). The whole graph is allocated on a per-batch-element arena so it can be freed in one shot at the end of the step — no individual frees in the hot path.

### 3.2 AdamW Optimizer

**Theory.** Adam maintains a running average of gradients (`m`, first moment) and gradient squares (`v`, second moment) for each parameter. The update is:

```
m = beta1 * m + (1 - beta1) * g
v = beta2 * v + (1 - beta2) * g^2
m_hat = m / (1 - beta1^t)             # bias correction
v_hat = v / (1 - beta2^t)
p -= lr * m_hat / (sqrt(v_hat) + eps)
```

AdamW differs from Adam in one detail: weight decay is applied as `p -= lr * weight_decay * p`, *separately* from the adaptive step. In vanilla Adam, weight decay got tangled up with the adaptive denominator; AdamW decouples it, which both works better and matches what people meant by "weight decay" all along.

**This codebase.** `src/autograd.zig: AdamW`. Maintains `m` and `v` buffers of the same shape as each parameter. `step` walks all parameters and applies the update with a fast path for contiguous tensors. The bias-correction terms `1 - beta^t` are computed once per step, not per parameter element.

### 3.3 Gradient Clipping

**Theory.** Compute the global L2 norm of all gradients (one number across every parameter), and if it exceeds a threshold (typically 1.0), scale every gradient down so the norm becomes the threshold. This caps catastrophic step sizes when something goes weird (rare-token gradient spike, numerical edge case, bad batch).

It's especially important early in training when activations and gradients haven't settled into a stable regime.

**This codebase.** `src/autograd.zig: gradClip`. Sums squared grads across all parameters, computes the global norm, scales if needed.

### 3.4 Warmup + Cosine Learning Rate Schedule

**Theory.** Two stages:

- **Warmup** (first `W` steps): linearly ramp LR from 0 (or near 0) to `max_lr`. Why: at initialization, the activation distribution is far from where Adam's running averages expect it to be. Big steps early can destabilize. Slow start lets Adam build up reasonable estimates of the moments.

- **Cosine decay** (remaining steps): smoothly decay LR from `max_lr` to `min_lr` following `min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress))`. Cosine schedules empirically work better than linear or step decays — they spend more time at high LR (exploration) and ease into low LR (refinement).

The floor `min_lr` (not 0) matters: at the end of training, you still want *some* progress. Going to 0 just wastes the tail of the run.

**This codebase.** `src/main.zig: getLR`. Warmup uses `(step + 1) / warmup_steps` so step 0 gets a nonzero LR rather than literal zero. Cosine uses the formula above with `MIN_LR = 0.2 * LR`.

---

## Part 4: Putting It Together — A Training Step

For one batch element:

1. Pick a random window of `SEQ_LEN+1` tokens from the encoded corpus. First `SEQ_LEN` = input, last `SEQ_LEN` (shifted by 1) = target.
2. Build the autograd graph by running forward: embedding → `N_LAYERS` × (RMSNorm + Attention + residual + RMSNorm + SwiGLU + residual) → final RMSNorm → LM head → logits → cross-entropy loss.
3. Call `backward()` on the loss `Val`. Gradients accumulate into every parameter's `grad`.
4. Drop the arena. The forward graph is gone, but the parameter gradients live in their own buffers and are kept.

Repeat for `BATCH_SIZE` elements, accumulating gradients. Then:

5. Divide accumulated gradients by `BATCH_SIZE` (so the effective gradient is the *mean* over the batch).
6. Global gradient clip.
7. Update LR for this step.
8. `AdamW.step` — update every parameter in place.
9. Zero gradients.
10. Repeat from step 1.

Every `CHECKPOINT_EVERY` steps: save weights and optimizer state.

---

## Part 5: Performance Notes

Things that are not concepts but were load-bearing for making this train in human-relevant time:

- **OpenBLAS for matmul.** Hand-written matmul in Zig is roughly 50–100× slower than `sgemm` on a modern CPU. The whole library is built around that delegation.

- **Strided views + transpose flags.** Attention shuffles tensors a lot. Naively each shuffle materializes a copy. We instead carry stride metadata and pass `CblasTrans` flags into `sgemm` so the transpose costs nothing.

- **Arena allocator per batch element.** Forward and backward allocate hundreds of intermediate tensors. With a general-purpose allocator, the `free` calls were a real cost. An arena lets us bulk-free everything at the end of each batch element in O(1).

- **Contiguous fast paths in hot loops.** `accumulateGrad`, `AdamW.step`, `sumKeepdim`, `reduceToShape` — anything called per element of a large tensor — check `contiguousSlice()` first and use a tight inner loop when possible. Strided indexing pays `div/mod` per element, which is expensive.

- **`matmulT` instead of `transpose` + `matmul`.** Used in attention's `Q @ K.T`. Saves an explicit transpose materialization.

These collectively took the bench from ~0.07 sps to ~0.30 sps (4×).

---

## Glossary of Symbols

| Symbol | Meaning | This codebase |
|---|---|---|
| `B` | batch size | `BATCH_SIZE = 4` |
| `T` | sequence length | `SEQ_LEN = 512` |
| `V` | vocabulary size | `VOCAB_SIZE = 4096` |
| `D` | model dimension | `D_MODEL = 256` |
| `H` | number of attention heads | `N_HEADS = 4` |
| `d_h` | per-head dim, = `D / H` | `64` |
| `D_ffn` | FFN hidden dim | `D_FFN = 1024` |
| `L` | number of transformer blocks | `N_LAYERS = 6` |

---

## Where to Read the Code

Read in this order for maximum sense-making:

1. `src/tokenizer.zig` — start here, it's the most standalone.
2. `src/tensor.zig` — `Tensor`, `Storage`, `matmul`, the view helpers.
3. `src/autograd.zig` — top to bottom. `Val` first, then ops, then layers (`Linear`, `Embedding`, `RMSNorm`, `Attention`, `SwiGLU`, `Block`), then `Model`, then `AdamW` at the bottom.
4. `src/main.zig` — the CLI dispatch + the training loop + bench.
