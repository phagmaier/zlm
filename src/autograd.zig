const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;
const TESTSEED: u64 = 42;
const N_HEADS: usize = 4;

pub const LinearForward = struct {
    out: *Val,
    weight: *Val,
};

pub const EmbeddingForward = struct {
    out: *Val,
    weight: *Val,
};

pub const RMSNormForward = struct {
    out: *Val,
    gamma: *Val,
};

pub const AttentionForward = struct {
    out: *Val,
    w_q: LinearForward,
    w_k: LinearForward,
    w_v: LinearForward,
    w_o: LinearForward,
};

pub const SwiGLUForward = struct {
    out: *Val,
    w_gate: LinearForward,
    w_up: LinearForward,
    w_down: LinearForward,
};

pub const TransformerBlockForward = struct {
    out: *Val,
    norm1: RMSNormForward,
    attn: AttentionForward,
    norm2: RMSNormForward,
    ffn: SwiGLUForward,
};

pub const ModelForward = struct {
    out: *Val,
    embedding: EmbeddingForward,
    blocks: []TransformerBlockForward,
    norm: RMSNormForward,
    lm_head: LinearForward,
};

pub const Attention = struct {
    w_q: Linear,
    w_k: Linear,
    w_v: Linear,
    w_o: Linear,
    n_heads: usize,
    cos: Tensor,
    sin: Tensor,

    pub fn init(allocator: std.mem.Allocator, d_model: usize, n_heads: usize, seq_len: usize, seed: u64) !Attention {
        const d_head = d_model / n_heads;
        const tables = try ropeTable(allocator, seq_len, d_head);
        var cos = tables.cos;
        errdefer cos.deinit(allocator);
        var sin = tables.sin;
        errdefer sin.deinit(allocator);

        var wq = try Linear.init(allocator, d_model, d_model, seed);
        errdefer wq.deinit(allocator);
        var wk = try Linear.init(allocator, d_model, d_model, seed +% 1);
        errdefer wk.deinit(allocator);
        var wv = try Linear.init(allocator, d_model, d_model, seed +% 2);
        errdefer wv.deinit(allocator);
        var wo = try Linear.init(allocator, d_model, d_model, seed +% 3);
        errdefer wo.deinit(allocator);

        return .{ .w_q = wq, .w_k = wk, .w_v = wv, .w_o = wo, .n_heads = n_heads, .cos = cos, .sin = sin };
    }

    pub fn deinit(self: *Attention, allocator: std.mem.Allocator) void {
        self.w_q.deinit(allocator);
        self.w_k.deinit(allocator);
        self.w_v.deinit(allocator);
        self.w_o.deinit(allocator);
        self.cos.deinit(allocator);
        self.sin.deinit(allocator);
    }

    pub fn forward(self: *const Attention, allocator: std.mem.Allocator, x: *Val, mask: *Val) !AttentionForward {
        const q_proj = try self.w_q.forward(allocator, x);
        const k_proj = try self.w_k.forward(allocator, x);
        const v_proj = try self.w_v.forward(allocator, x);

        const seq = x.value.shape[0];
        const d_model = x.value.shape[1];
        const d_head = d_model / self.n_heads;
        const inv_sqrt: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(d_head)));

        const Q_r = try Val.reshape(allocator, q_proj.out, &.{ seq, self.n_heads, d_head });
        const K_r = try Val.reshape(allocator, k_proj.out, &.{ seq, self.n_heads, d_head });
        const V_r = try Val.reshape(allocator, v_proj.out, &.{ seq, self.n_heads, d_head });

        const Q_t = try Val.transpose(allocator, Q_r, &.{ 1, 0, 2 });
        const K_t = try Val.transpose(allocator, K_r, &.{ 1, 0, 2 });
        const V_t = try Val.transpose(allocator, V_r, &.{ 1, 0, 2 });

        const Q_rope = try Val.rope(allocator, Q_t, self.cos, self.sin);
        const K_rope = try Val.rope(allocator, K_t, self.cos, self.sin);

        const K_tt = try Val.transpose(allocator, K_rope, &.{ 0, 2, 1 });
        const qkt = try Val.matmul(allocator, Q_rope, K_tt);
        const scaled = try Val.mulScalar(allocator, qkt, inv_sqrt);
        const masked = try Val.add(allocator, scaled, mask);
        const weights = try Val.softmax(allocator, masked);
        const attn = try Val.matmul(allocator, weights, V_t);

        const attn_t = try Val.transpose(allocator, attn, &.{ 1, 0, 2 });
        const attn_c = try Val.contiguous(allocator, attn_t);
        const attn_r = try Val.reshape(allocator, attn_c, &.{ seq, d_model });
        const out_proj = try self.w_o.forward(allocator, attn_r);

        return .{ .out = out_proj.out, .w_q = q_proj, .w_k = k_proj, .w_v = v_proj, .w_o = out_proj };
    }
};

pub fn attention(allocator: std.mem.Allocator, x: *Val, w_q: *Val, w_k: *Val, w_v: *Val, w_o: *Val, mask: *Val, n_heads: usize, cos: Tensor, sin: Tensor) !*Val {
    const seq = x.value.shape[0];
    const d_model = x.value.shape[1];
    const d_head = d_model / n_heads;
    const inv_sqrt: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(d_head)));

    const Q = try Val.matmul(allocator, x, w_q);
    const K = try Val.matmul(allocator, x, w_k);
    const V = try Val.matmul(allocator, x, w_v);

    const Q_r = try Val.reshape(allocator, Q, &.{ seq, n_heads, d_head });
    const K_r = try Val.reshape(allocator, K, &.{ seq, n_heads, d_head });
    const V_r = try Val.reshape(allocator, V, &.{ seq, n_heads, d_head });

    const Q_t = try Val.transpose(allocator, Q_r, &.{ 1, 0, 2 });
    const K_t = try Val.transpose(allocator, K_r, &.{ 1, 0, 2 });
    const V_t = try Val.transpose(allocator, V_r, &.{ 1, 0, 2 });

    const Q_rope = try Val.rope(allocator, Q_t, cos, sin);
    const K_rope = try Val.rope(allocator, K_t, cos, sin);

    const K_tt = try Val.transpose(allocator, K_rope, &.{ 0, 2, 1 });
    const qkt = try Val.matmul(allocator, Q_rope, K_tt);
    const scaled = try Val.mulScalar(allocator, qkt, inv_sqrt);
    const masked = try Val.add(allocator, scaled, mask);
    const weights = try Val.softmax(allocator, masked);
    const attn = try Val.matmul(allocator, weights, V_t);

    const attn_t = try Val.transpose(allocator, attn, &.{ 1, 0, 2 });
    const attn_c = try Val.contiguous(allocator, attn_t);
    const attn_r = try Val.reshape(allocator, attn_c, &.{ seq, d_model });
    const out = try Val.matmul(allocator, attn_r, w_o);

    return out;
}

const MatmulFlags = struct { trans_l: bool, trans_r: bool };

const Op = enum {
    leaf,
    add,
    mul,
    embedding,
    sum_keepdim,
    rsqrt,
    add_scalar,
    mul_scalar,
    silu,
    softmax,
    log_softmax,
    matmul_t,
    reshape,
    transpose,
    contiguous,
    rope,
    cross_entropy,
};

const OpData = union(Op) {
    leaf: void,
    add: void,
    mul: void,
    embedding: []const u16,
    sum_keepdim: usize,
    rsqrt: void,
    add_scalar: f32,
    mul_scalar: f32,
    silu: void,
    softmax: void,
    log_softmax: void,
    matmul_t: MatmulFlags,
    reshape: void,
    transpose: []const usize,
    contiguous: void,
    rope: struct { cos: Tensor, sin: Tensor },
    cross_entropy: []const u16,
};

pub const Embedding = struct {
    weight: Tensor,
    grad: ?Tensor,

    pub fn init(allocator: std.mem.Allocator, vocab_size: usize, d_model: usize) !Embedding {
        var weight = try Tensor.init(allocator, &.{ vocab_size, d_model });
        errdefer weight.deinit(allocator);

        var prng = std.Random.DefaultPrng.init(42);
        const rng = prng.random();
        for (weight.storage.data) |*v| {
            v.* = rng.floatNorm(f32) * 0.02;
        }

        return .{ .weight = weight, .grad = null };
    }

    pub fn deinit(self: *Embedding, allocator: std.mem.Allocator) void {
        if (self.grad) |*g| g.deinit(allocator);
        self.weight.deinit(allocator);
    }

    pub fn forward(self: *const Embedding, allocator: std.mem.Allocator, ids: []const u16) !EmbeddingForward {
        const weight = try Val.makeLeaf(allocator, &self.weight);
        const out = try Val.embedding(allocator, weight, ids);
        return .{ .out = out, .weight = weight };
    }
};

pub const Linear = struct {
    weight: Tensor,
    grad: ?Tensor,

    pub fn init(allocator: std.mem.Allocator, in_dim: usize, out_dim: usize, seed: u64) !Linear {
        var weight = try Tensor.init(allocator, &.{ in_dim, out_dim });
        errdefer weight.deinit(allocator);

        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (weight.storage.data) |*v| {
            v.* = rng.floatNorm(f32) * 0.02;
        }

        return .{ .weight = weight, .grad = null };
    }

    pub fn deinit(self: *Linear, allocator: std.mem.Allocator) void {
        if (self.grad) |*g| g.deinit(allocator);
        self.weight.deinit(allocator);
    }

    pub fn forward(self: *const Linear, allocator: std.mem.Allocator, x: *Val) !LinearForward {
        const weight = try Val.makeLeaf(allocator, &self.weight);
        const out = try Val.matmul(allocator, x, weight);
        return .{ .out = out, .weight = weight };
    }
};

pub fn causalMask(allocator: std.mem.Allocator, seq_len: usize) !Tensor {
    var tensor = try Tensor.init(allocator, &.{ seq_len, seq_len });
    errdefer tensor.deinit(allocator);

    const data = tensor.storage.data;
    for (0..seq_len) |i| {
        for (0..seq_len) |j| {
            // j is the 'key' position, i is the 'query' position
            // We can only see keys where j <= i
            data[i * seq_len + j] = if (j <= i) 0.0 else -1e9;
        }
    }
    return tensor;
}

pub const RMSNorm = struct {
    gamma: Tensor,
    gamma_grad: ?Tensor,
    eps: f32,

    pub fn init(allocator: std.mem.Allocator, d_model: usize, eps: f32) !RMSNorm {
        var gamma = try Tensor.init(allocator, &.{d_model});
        errdefer gamma.deinit(allocator);
        @memset(gamma.storage.data, 1.0);
        return .{ .gamma = gamma, .gamma_grad = null, .eps = eps };
    }

    pub fn deinit(self: *RMSNorm, allocator: std.mem.Allocator) void {
        if (self.gamma_grad) |*g| g.deinit(allocator);
        self.gamma.deinit(allocator);
    }

    pub fn forward(self: *const RMSNorm, allocator: std.mem.Allocator, x: *Val) !RMSNormForward {
        const last_axis = x.value.shape.len - 1;
        const d_model = x.value.shape[last_axis];
        const inv_d: f32 = 1.0 / @as(f32, @floatFromInt(d_model));

        const sq = try Val.mul(allocator, x, x);
        const summed = try Val.sumKeepdim(allocator, sq, last_axis);

        const mean_sq = try Val.mulScalar(allocator, summed, inv_d);
        const shifted = try Val.addScalar(allocator, mean_sq, self.eps);

        const inv = try Val.rsqrt(allocator, shifted);
        const normed = try Val.mul(allocator, x, inv);

        const gamma_val = try Val.makeLeaf(allocator, &self.gamma);
        const out = try Val.mul(allocator, normed, gamma_val);

        return .{ .out = out, .gamma = gamma_val };
    }
};

pub const TransformerBlock = struct {
    attn: Attention,
    ffn: SwiGLU,
    norm1: RMSNorm,
    norm2: RMSNorm,

    pub fn init(allocator: std.mem.Allocator, d_model: usize, n_heads: usize, d_ffn: usize, seq_len: usize, seed: u64) !TransformerBlock {
        var norm1 = try RMSNorm.init(allocator, d_model, 1e-5);
        errdefer norm1.deinit(allocator);
        var norm2 = try RMSNorm.init(allocator, d_model, 1e-5);
        errdefer norm2.deinit(allocator);
        var attn = try Attention.init(allocator, d_model, n_heads, seq_len, seed);
        errdefer attn.deinit(allocator);
        const ffn = try SwiGLU.init(allocator, d_model, d_ffn, seed +% 4);
        return .{ .attn = attn, .ffn = ffn, .norm1 = norm1, .norm2 = norm2 };
    }

    pub fn deinit(self: *TransformerBlock, allocator: std.mem.Allocator) void {
        self.attn.deinit(allocator);
        self.ffn.deinit(allocator);
        self.norm1.deinit(allocator);
        self.norm2.deinit(allocator);
    }

    pub fn forward(self: *const TransformerBlock, allocator: std.mem.Allocator, x: *Val, mask: *Val) !TransformerBlockForward {
        const normed1 = try self.norm1.forward(allocator, x);
        const attn_out = try self.attn.forward(allocator, normed1.out, mask);
        const residual1 = try Val.add(allocator, x, attn_out.out);

        const normed2 = try self.norm2.forward(allocator, residual1);
        const ffn_out = try self.ffn.forward(allocator, normed2.out);
        const residual2 = try Val.add(allocator, residual1, ffn_out.out);

        return .{ .out = residual2, .norm1 = normed1, .attn = attn_out, .norm2 = normed2, .ffn = ffn_out };
    }
};

pub const Model = struct {
    embedding: Embedding,
    blocks: []TransformerBlock,
    norm: RMSNorm,
    lm_head: Linear,

    pub fn init(allocator: std.mem.Allocator, vocab_size: usize, d_model: usize, n_heads: usize, d_ffn: usize, n_layers: usize, seq_len: usize, seed: u64) !Model {
        var emb = try Embedding.init(allocator, vocab_size, d_model);
        errdefer emb.deinit(allocator);

        var norm = try RMSNorm.init(allocator, d_model, 1e-5);
        errdefer norm.deinit(allocator);

        var lm_head = try Linear.init(allocator, d_model, vocab_size, seed +% 100);
        errdefer lm_head.deinit(allocator);

        const blocks = try allocator.alloc(TransformerBlock, n_layers);
        errdefer allocator.free(blocks);

        var initialized: usize = 0;
        errdefer for (0..initialized) |i| blocks[i].deinit(allocator);

        for (0..n_layers) |i| {
            blocks[i] = try TransformerBlock.init(allocator, d_model, n_heads, d_ffn, seq_len, seed +% (i * 10));
            initialized += 1;
        }

        return .{ .embedding = emb, .blocks = blocks, .norm = norm, .lm_head = lm_head };
    }

    pub fn deinit(self: *Model, allocator: std.mem.Allocator) void {
        self.embedding.deinit(allocator);
        for (self.blocks) |*block| block.deinit(allocator);
        allocator.free(self.blocks);
        self.norm.deinit(allocator);
        self.lm_head.deinit(allocator);
    }

    pub fn forward(self: *const Model, allocator: std.mem.Allocator, ids: []const u16, mask: *Val) !ModelForward {
        const emb = try self.embedding.forward(allocator, ids);
        var x = emb.out;

        const block_results = try allocator.alloc(TransformerBlockForward, self.blocks.len);
        errdefer allocator.free(block_results);

        for (self.blocks, 0..) |*block, i| {
            block_results[i] = try block.forward(allocator, x, mask);
            x = block_results[i].out;
        }

        const normed = try self.norm.forward(allocator, x);

        const logits = try self.lm_head.forward(allocator, normed.out);

        return .{ .out = logits.out, .embedding = emb, .blocks = block_results, .norm = normed, .lm_head = logits };
    }

    fn zeroGrad(allocator: std.mem.Allocator, grad_slot: *?Tensor) void {
        if (grad_slot.*) |*grad| {
            grad.deinit(allocator);
            grad_slot.* = null;
        }
    }

    pub fn accumulateGrads(self: *Model, allocator: std.mem.Allocator, result: *const ModelForward) !void {
        if (result.blocks.len != self.blocks.len) return error.SizeMisMatch;

        try result.embedding.weight.accumulateToParam(&self.embedding.grad, allocator);

        for (self.blocks, result.blocks) |*block, block_result| {
            try block_result.norm1.gamma.accumulateToParam(&block.norm1.gamma_grad, allocator);

            try block_result.attn.w_q.weight.accumulateToParam(&block.attn.w_q.grad, allocator);
            try block_result.attn.w_k.weight.accumulateToParam(&block.attn.w_k.grad, allocator);
            try block_result.attn.w_v.weight.accumulateToParam(&block.attn.w_v.grad, allocator);
            try block_result.attn.w_o.weight.accumulateToParam(&block.attn.w_o.grad, allocator);

            try block_result.norm2.gamma.accumulateToParam(&block.norm2.gamma_grad, allocator);

            try block_result.ffn.w_gate.weight.accumulateToParam(&block.ffn.w_gate.grad, allocator);
            try block_result.ffn.w_up.weight.accumulateToParam(&block.ffn.w_up.grad, allocator);
            try block_result.ffn.w_down.weight.accumulateToParam(&block.ffn.w_down.grad, allocator);
        }

        try result.norm.gamma.accumulateToParam(&self.norm.gamma_grad, allocator);
        try result.lm_head.weight.accumulateToParam(&self.lm_head.grad, allocator);
    }

    pub fn zeroGrads(self: *Model, allocator: std.mem.Allocator) void {
        zeroGrad(allocator, &self.embedding.grad);

        for (self.blocks) |*block| {
            zeroGrad(allocator, &block.norm1.gamma_grad);
            zeroGrad(allocator, &block.attn.w_q.grad);
            zeroGrad(allocator, &block.attn.w_k.grad);
            zeroGrad(allocator, &block.attn.w_v.grad);
            zeroGrad(allocator, &block.attn.w_o.grad);
            zeroGrad(allocator, &block.norm2.gamma_grad);
            zeroGrad(allocator, &block.ffn.w_gate.grad);
            zeroGrad(allocator, &block.ffn.w_up.grad);
            zeroGrad(allocator, &block.ffn.w_down.grad);
        }

        zeroGrad(allocator, &self.norm.gamma_grad);
        zeroGrad(allocator, &self.lm_head.grad);
    }

    pub fn collectParams(self: *Model, allocator: std.mem.Allocator, params: *std.ArrayList(*Tensor), grads: *std.ArrayList(*?Tensor)) !void {
        try params.append(allocator, &self.embedding.weight);
        try grads.append(allocator, &self.embedding.grad);

        for (self.blocks) |*block| {
            try params.append(allocator, &block.norm1.gamma);
            try grads.append(allocator, &block.norm1.gamma_grad);

            try params.append(allocator, &block.attn.w_q.weight);
            try grads.append(allocator, &block.attn.w_q.grad);
            try params.append(allocator, &block.attn.w_k.weight);
            try grads.append(allocator, &block.attn.w_k.grad);
            try params.append(allocator, &block.attn.w_v.weight);
            try grads.append(allocator, &block.attn.w_v.grad);
            try params.append(allocator, &block.attn.w_o.weight);
            try grads.append(allocator, &block.attn.w_o.grad);

            try params.append(allocator, &block.norm2.gamma);
            try grads.append(allocator, &block.norm2.gamma_grad);

            try params.append(allocator, &block.ffn.w_gate.weight);
            try grads.append(allocator, &block.ffn.w_gate.grad);
            try params.append(allocator, &block.ffn.w_up.weight);
            try grads.append(allocator, &block.ffn.w_up.grad);
            try params.append(allocator, &block.ffn.w_down.weight);
            try grads.append(allocator, &block.ffn.w_down.grad);
        }

        try params.append(allocator, &self.norm.gamma);
        try grads.append(allocator, &self.norm.gamma_grad);
        try params.append(allocator, &self.lm_head.weight);
        try grads.append(allocator, &self.lm_head.grad);
    }
};

pub fn swiglu(allocator: std.mem.Allocator, x: *Val, w_gate: *Val, w_up: *Val, w_down: *Val) !*Val {
    const gate = try Val.matmul(allocator, x, w_gate);
    const up = try Val.matmul(allocator, x, w_up);
    const activated = try Val.silu(allocator, gate);
    const gated = try Val.mul(allocator, activated, up);
    const out = try Val.matmul(allocator, gated, w_down);
    return out;
}

pub const SwiGLU = struct {
    w_gate: Linear,
    w_up: Linear,
    w_down: Linear,

    pub fn init(allocator: std.mem.Allocator, d_model: usize, d_ffn: usize, seed: u64) !SwiGLU {
        var w_gate = try Linear.init(allocator, d_model, d_ffn, seed);
        errdefer w_gate.deinit(allocator);
        var w_up = try Linear.init(allocator, d_model, d_ffn, seed +% 1);
        errdefer w_up.deinit(allocator);
        const w_down = try Linear.init(allocator, d_ffn, d_model, seed +% 2);
        return .{ .w_gate = w_gate, .w_up = w_up, .w_down = w_down };
    }

    pub fn deinit(self: *SwiGLU, allocator: std.mem.Allocator) void {
        self.w_gate.deinit(allocator);
        self.w_up.deinit(allocator);
        self.w_down.deinit(allocator);
    }

    pub fn forward(self: *const SwiGLU, allocator: std.mem.Allocator, x: *Val) !SwiGLUForward {
        const gate = try self.w_gate.forward(allocator, x);
        const up = try self.w_up.forward(allocator, x);
        const activated = try Val.silu(allocator, gate.out);
        const gated = try Val.mul(allocator, activated, up.out);
        const down = try self.w_down.forward(allocator, gated);
        return .{ .out = down.out, .w_gate = gate, .w_up = up, .w_down = down };
    }
};

pub const Val = struct {
    value: Tensor,
    grad: ?Tensor,
    op: OpData,
    parents: []const *Val,

    pub fn makeLeaf(allocator: std.mem.Allocator, tensor: *const Tensor) !*Val {
        var value = try tensor.clone(allocator);
        errdefer value.deinit(allocator);

        const self = try allocator.create(Val);
        self.* = .{
            .value = value,
            .grad = null,
            .op = .leaf,
            .parents = &.{},
        };
        return self;
    }

    pub fn rope(allocator: std.mem.Allocator, x: *Val, cos: Tensor, sin: Tensor) !*Val {
        const n_heads = x.value.shape[0];
        const seq = x.value.shape[1];
        const d_head = x.value.shape[2];

        var value = try Tensor.init(allocator, x.value.shape);
        errdefer value.deinit(allocator);

        for (0..n_heads) |h| {
            for (0..seq) |pos| {
                for (0..d_head / 2) |i| {
                    const base = h * seq * d_head + pos * d_head + 2 * i;
                    const x0 = x.value.storage.data[base];
                    const x1 = x.value.storage.data[base + 1];
                    const c = cos.storage.data[pos * d_head + 2 * i];
                    const s = sin.storage.data[pos * d_head + 2 * i];
                    value.storage.data[base] = x0 * c - x1 * s;
                    value.storage.data[base + 1] = x1 * c + x0 * s;
                }
            }
        }

        const parents = try allocator.alloc(*Val, 1);
        errdefer allocator.free(parents);
        parents[0] = x;

        const self = try allocator.create(Val);
        self.* = .{
            .value = value,
            .grad = null,
            .op = .{ .rope = .{ .cos = cos, .sin = sin } },
            .parents = parents,
        };
        return self;
    }
    pub fn contiguous(allocator: std.mem.Allocator, x: *Val) !*Val {
        var tensor = try x.value.contiguous(allocator);
        errdefer tensor.deinit(allocator);
        const parents = try allocator.alloc(*Val, 1);
        errdefer allocator.free(parents);
        parents[0] = x;
        const self = try allocator.create(Val);
        self.* = .{ .value = tensor, .grad = null, .op = .contiguous, .parents = parents };
        return self;
    }

    pub fn transpose(allocator: std.mem.Allocator, x: *Val, permutation: []const usize) !*Val {
        var tensor = try Tensor.transpose(allocator, &x.value, permutation);
        errdefer tensor.deinit(allocator);

        const cpy = try allocator.alloc(usize, permutation.len);
        errdefer allocator.free(cpy);
        @memcpy(cpy, permutation);

        const parents = try allocator.alloc(*Val, 1);
        errdefer allocator.free(parents);
        parents[0] = x;

        const val = try allocator.create(Val);
        val.* = .{
            .value = tensor,
            .grad = null,
            .op = .{ .transpose = cpy },
            .parents = parents,
        };
        return val;
    }

    pub fn add(allocator: std.mem.Allocator, l: *Val, r: *Val) !*Val {
        return makeBinaryOp(allocator, l, r, .add, try Tensor.add(allocator, &l.value, &r.value));
    }

    pub fn addScalar(allocator: std.mem.Allocator, x: *Val, scalar: f32) !*Val {
        var value = try Tensor.addScalar(allocator, &x.value, scalar);
        errdefer value.deinit(allocator);

        const parents = try allocator.alloc(*Val, 1);
        errdefer allocator.free(parents);
        parents[0] = x;

        const self = try allocator.create(Val);
        self.* = .{
            .value = value,
            .grad = null,
            .op = .{ .add_scalar = scalar },
            .parents = parents,
        };
        return self;
    }

    pub fn mul(allocator: std.mem.Allocator, l: *Val, r: *Val) !*Val {
        return makeBinaryOp(allocator, l, r, .mul, try Tensor.mul(allocator, &l.value, &r.value));
    }

    pub fn mulScalar(allocator: std.mem.Allocator, x: *Val, scalar: f32) !*Val {
        var value = try Tensor.scale(allocator, &x.value, scalar);
        errdefer value.deinit(allocator);

        const parents = try allocator.alloc(*Val, 1);
        errdefer allocator.free(parents);
        parents[0] = x;

        const self = try allocator.create(Val);
        self.* = .{
            .value = value,
            .grad = null,
            .op = .{ .mul_scalar = scalar },
            .parents = parents,
        };
        return self;
    }

    pub fn rsqrt(allocator: std.mem.Allocator, x: *Val) !*Val {
        var value = try Tensor.init(allocator, x.value.shape);
        errdefer value.deinit(allocator);

        for (x.value.storage.data, 0..) |v, i| {
            value.storage.data[i] = 1.0 / @sqrt(v);
        }

        const parents = try allocator.alloc(*Val, 1);
        errdefer allocator.free(parents);
        parents[0] = x;

        const self = try allocator.create(Val);
        self.* = .{
            .value = value,
            .grad = null,
            .op = .rsqrt,
            .parents = parents,
        };
        return self;
    }

    pub fn silu(allocator: std.mem.Allocator, x: *Val) !*Val {
        var value = try Tensor.init(allocator, x.value.shape);
        errdefer value.deinit(allocator);

        for (x.value.storage.data, 0..) |v, i| {
            const sig = 1.0 / (1.0 + @exp(-v));
            value.storage.data[i] = v * sig;
        }

        const parents = try allocator.alloc(*Val, 1);
        errdefer allocator.free(parents);
        parents[0] = x;

        const self = try allocator.create(Val);
        self.* = .{
            .value = value,
            .grad = null,
            .op = .silu,
            .parents = parents,
        };
        return self;
    }

    pub fn matmul(allocator: std.mem.Allocator, l: *Val, r: *Val) !*Val {
        return matmulT(allocator, l, r, false, false);
    }

    fn validMatMul(lShape: []usize, rShape: []usize) bool {
        const lRank = lShape.len;
        const rRank = rShape.len;
        if (rRank < 2 or rRank > 3 or lRank < 2 or lRank > 3) return false;
        if (lRank == 3 and rRank == 3 and lShape[0] != rShape[0]) return false;
        return true;
    }

    pub fn matmulT(allocator: std.mem.Allocator, l: *Val, r: *Val, trans_l: bool, trans_r: bool) !*Val {
        if (!validMatMul(l.value.shape, r.value.shape)) return error.UnsupportedRank;
        const value = try Tensor.matmul(allocator, &l.value, &r.value, trans_l, trans_r);
        return makeBinaryOp(allocator, l, r, .{ .matmul_t = .{ .trans_l = trans_l, .trans_r = trans_r } }, value);
    }

    pub fn softmax(allocator: std.mem.Allocator, x: *Val) !*Val {
        var value = try Tensor.init(allocator, x.value.shape);
        errdefer value.deinit(allocator);

        if (x.value.shape.len == 0) return error.UnsupportedRank;

        const last_dim = x.value.shape[x.value.shape.len - 1];
        const rows = x.value.numel() / last_dim;

        for (0..rows) |r| {
            const start = r * last_dim;
            const end = start + last_dim;
            const row_data = x.value.storage.data[start..end];
            const out_data = value.storage.data[start..end];

            var max = row_data[0];
            for (row_data[1..]) |v| {
                if (v > max) max = v;
            }

            var sum: f32 = 0;
            for (0..last_dim) |i| {
                out_data[i] = @exp(row_data[i] - max);
                sum += out_data[i];
            }

            for (0..last_dim) |i| {
                out_data[i] /= sum;
            }
        }

        const parents = try allocator.alloc(*Val, 1);
        errdefer allocator.free(parents);
        parents[0] = x;

        const self = try allocator.create(Val);
        self.* = .{
            .value = value,
            .grad = null,
            .op = .softmax,
            .parents = parents,
        };
        return self;
    }

    pub fn logSoftmax(allocator: std.mem.Allocator, x: *Val) !*Val {
        var value = try Tensor.init(allocator, x.value.shape);
        errdefer value.deinit(allocator);

        const last_dim = x.value.shape[x.value.shape.len - 1];
        const rows = x.value.numel() / last_dim;

        for (0..rows) |r| {
            const start = r * last_dim;
            const end = start + last_dim;
            const in_row = x.value.storage.data[start..end];
            const out_row = value.storage.data[start..end];

            var max: f32 = in_row[0];
            for (in_row[1..]) |v| {
                if (v > max) max = v;
            }

            var sum: f32 = 0;
            for (in_row) |v| {
                sum += @exp(v - max);
            }
            const log_sum = @log(sum);

            for (0..last_dim) |i| {
                out_row[i] = in_row[i] - max - log_sum;
            }
        }

        const parents = try allocator.alloc(*Val, 1);
        errdefer allocator.free(parents);
        parents[0] = x;

        const self = try allocator.create(Val);
        self.* = .{
            .value = value,
            .grad = null,
            .op = .log_softmax,
            .parents = parents,
        };
        return self;
    }

    pub fn crossEntropy(allocator: std.mem.Allocator, logits: *Val, ids: []const u16) !*Val {
        const log_probs = try Val.logSoftmax(allocator, logits);

        const seq = logits.value.shape[0];
        const vocab_size = logits.value.shape[1];

        var loss: f32 = 0;
        for (0..seq) |s| {
            const correct = ids[s];
            loss += log_probs.value.storage.data[s * vocab_size + correct];
        }
        loss = -loss / @as(f32, @floatFromInt(seq));

        var value = try Tensor.zeroes(allocator, &.{});
        errdefer value.deinit(allocator);
        value.storage.data[0] = loss;

        const ids_copy = try allocator.dupe(u16, ids);
        errdefer allocator.free(ids_copy);

        const parents = try allocator.alloc(*Val, 1);
        errdefer allocator.free(parents);
        parents[0] = log_probs;

        const self = try allocator.create(Val);
        self.* = .{
            .value = value,
            .grad = null,
            .op = .{ .cross_entropy = ids_copy },
            .parents = parents,
        };
        return self;
    }

    fn makeBinaryOp(allocator: std.mem.Allocator, l: *Val, r: *Val, op: OpData, tensor: Tensor) !*Val {
        var value = tensor;
        errdefer value.deinit(allocator);

        const parents = try allocator.alloc(*Val, 2);
        errdefer allocator.free(parents);
        parents[0] = l;
        parents[1] = r;

        const self = try allocator.create(Val);
        self.* = .{
            .value = value,
            .grad = null,
            .op = op,
            .parents = parents,
        };
        return self;
    }

    fn topoSort(allocator: std.mem.Allocator, root: *Val) !std.ArrayList(*Val) {
        var visited = std.AutoHashMap(*Val, void).init(allocator);
        defer visited.deinit();

        var ordered: std.ArrayList(*Val) = .empty;
        try visit(allocator, root, &visited, &ordered);
        return ordered;
    }

    fn visit(
        allocator: std.mem.Allocator,
        node: *Val,
        visited: *std.AutoHashMap(*Val, void),
        ordered: *std.ArrayList(*Val),
    ) !void {
        if (visited.contains(node)) return;
        try visited.put(node, {});
        for (node.parents) |parent| {
            try visit(allocator, parent, visited, ordered);
        }
        try ordered.append(allocator, node);
    }

    fn logicalStrides(allocator: std.mem.Allocator, shape: []const usize) ![]usize {
        const strides = try allocator.alloc(usize, shape.len);
        if (shape.len == 0) return strides;

        var stride: usize = 1;
        var i = shape.len;
        while (i > 0) {
            i -= 1;
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }

    fn reduceToShape(allocator: std.mem.Allocator, grad: *const Tensor, shape: []const usize) !Tensor {
        if (std.mem.eql(usize, grad.shape, shape)) {
            return grad.clone(allocator);
        }
        if (shape.len > grad.shape.len) return error.IncompatibleShapes;

        const axis_offset = grad.shape.len - shape.len;
        for (shape, 0..) |dim, axis| {
            const grad_dim = grad.shape[axis + axis_offset];
            if (dim != grad_dim and dim != 1) return error.IncompatibleShapes;
        }

        var out = try Tensor.zeroes(allocator, shape);
        errdefer out.deinit(allocator);

        if (shape.len == 0) {
            for (0..grad.numel()) |i| {
                out.storage.data[0] += try grad.getFlat(i);
            }
            return out;
        }

        const grad_strides = try logicalStrides(allocator, grad.shape);
        defer allocator.free(grad_strides);
        const out_strides = try logicalStrides(allocator, shape);
        defer allocator.free(out_strides);

        for (0..grad.numel()) |linear| {
            var remaining = linear;
            var out_linear: usize = 0;

            for (0..grad.shape.len) |axis| {
                const coord = remaining / grad_strides[axis];
                remaining %= grad_strides[axis];

                if (axis < axis_offset) continue;

                const out_axis = axis - axis_offset;
                const out_coord = if (shape[out_axis] == 1) 0 else coord;
                out_linear += out_coord * out_strides[out_axis];
            }

            out.storage.data[out_linear] += try grad.getFlat(linear);
        }

        return out;
    }

    fn accumulateGrad(allocator: std.mem.Allocator, target: *Val, contribution: *const Tensor) !void {
        if (target.grad == null) {
            target.grad = try Tensor.zeroes(allocator, target.value.shape);
        }
        if (!Tensor.shapeMatch(&target.grad.?, contribution)) return error.IncompatibleShapes;

        for (0..contribution.numel()) |i| {
            target.grad.?.storage.data[i] += try contribution.getFlat(i);
        }
    }

    fn accumulateReducedGrad(allocator: std.mem.Allocator, target: *Val, contribution: *const Tensor) !void {
        var reduced = try reduceToShape(allocator, contribution, target.value.shape);
        defer reduced.deinit(allocator);
        try accumulateGrad(allocator, target, &reduced);
    }

    fn backwardOp(allocator: std.mem.Allocator, node: *Val) !void {
        const node_grad = if (node.grad) |*grad| grad else return;

        switch (node.op) {
            .leaf => {},
            .add => {
                try accumulateReducedGrad(allocator, node.parents[0], node_grad);
                try accumulateReducedGrad(allocator, node.parents[1], node_grad);
            },
            .contiguous => {
                try accumulateGrad(allocator, node.parents[0], node_grad);
            },
            .mul => {
                const a = node.parents[0];
                const b = node.parents[1];

                var grad_a_raw = try Tensor.mul(allocator, node_grad, &b.value);
                defer grad_a_raw.deinit(allocator);
                try accumulateReducedGrad(allocator, a, &grad_a_raw);

                var grad_b_raw = try Tensor.mul(allocator, node_grad, &a.value);
                defer grad_b_raw.deinit(allocator);
                try accumulateReducedGrad(allocator, b, &grad_b_raw);
            },
            .rope => |tables| {
                const x = node.parents[0];
                if (x.grad == null) {
                    x.grad = try Tensor.zeroes(allocator, x.value.shape);
                }
                const n_heads = x.value.shape[0];
                const seq = x.value.shape[1];
                const d_head = x.value.shape[2];
                for (0..n_heads) |h| {
                    for (0..seq) |pos| {
                        for (0..d_head / 2) |i| {
                            const base = h * seq * d_head + pos * d_head + 2 * i;
                            const g0 = node_grad.storage.data[base];
                            const g1 = node_grad.storage.data[base + 1];
                            const c = tables.cos.storage.data[pos * d_head + 2 * i];
                            const s = tables.sin.storage.data[pos * d_head + 2 * i];
                            x.grad.?.storage.data[base] += g0 * c + g1 * s;
                            x.grad.?.storage.data[base + 1] += g1 * c - g0 * s;
                        }
                    }
                }
            },
            .silu => {
                const x = node.parents[0];
                if (x.grad == null) {
                    x.grad = try Tensor.zeroes(allocator, x.value.shape);
                }
                for (0..x.value.numel()) |i| {
                    const xv = x.value.storage.data[i];
                    const sig = 1.0 / (1.0 + @exp(-xv));
                    const deriv = sig * (1.0 + xv * (1.0 - sig));
                    x.grad.?.storage.data[i] += node_grad.storage.data[i] * deriv;
                }
            },
            .reshape => {
                const x = node.parents[0];
                var grad_back = try Tensor.reshape(allocator, node_grad, x.value.shape);
                defer grad_back.deinit(allocator);
                try accumulateGrad(allocator, x, &grad_back);
            },

            .matmul_t => |flags| {
                const a = node.parents[0];
                const b = node.parents[1];
                const dY = node_grad;

                var grad_a = blk: {
                    if (!flags.trans_l and !flags.trans_r) break :blk try Tensor.matmul(allocator, dY, &b.value, false, true);
                    if (!flags.trans_l and flags.trans_r) break :blk try Tensor.matmul(allocator, dY, &b.value, false, false);
                    if (flags.trans_l and !flags.trans_r) break :blk try Tensor.matmul(allocator, &b.value, dY, false, true);
                    break :blk try Tensor.matmul(allocator, &b.value, dY, true, true);
                };
                defer grad_a.deinit(allocator);
                try accumulateGrad(allocator, a, &grad_a);

                var grad_b = blk: {
                    if (!flags.trans_l and !flags.trans_r) break :blk try Tensor.matmul(allocator, &a.value, dY, true, false);
                    if (!flags.trans_l and flags.trans_r) break :blk try Tensor.matmul(allocator, dY, &a.value, true, false);
                    if (flags.trans_l and !flags.trans_r) break :blk try Tensor.matmul(allocator, &a.value, dY, false, false);
                    break :blk try Tensor.matmul(allocator, dY, &a.value, true, true);
                };
                defer grad_b.deinit(allocator);
                try accumulateGrad(allocator, b, &grad_b);
            },
            .embedding => |ids| {
                const W = node.parents[0];
                if (W.grad == null) {
                    W.grad = try Tensor.zeroes(allocator, W.value.shape);
                }
                const d_model = W.value.shape[1];
                for (ids, 0..) |id, s| {
                    const dst_off = @as(usize, id) * d_model;
                    const src_off = s * d_model;
                    const dst = W.grad.?.storage.data[dst_off .. dst_off + d_model];
                    const src = node_grad.storage.data[src_off .. src_off + d_model];
                    for (dst, src) |*g, c| {
                        g.* += c;
                    }
                }
            },
            .rsqrt => {
                const x = node.parents[0];
                if (x.grad == null) {
                    x.grad = try Tensor.zeroes(allocator, x.value.shape);
                }
                for (0..x.value.numel()) |i| {
                    const y = node.value.storage.data[i];
                    x.grad.?.storage.data[i] += node_grad.storage.data[i] * (-0.5) * y * y * y;
                }
            },
            .sum_keepdim => |axis| {
                const x = node.parents[0];
                if (x.grad == null) {
                    x.grad = try Tensor.zeroes(allocator, x.value.shape);
                }

                const in_strides = try logicalStrides(allocator, x.value.shape);
                defer allocator.free(in_strides);
                const out_strides = try logicalStrides(allocator, node.value.shape);
                defer allocator.free(out_strides);

                for (0..x.value.numel()) |linear_in| {
                    var remaining = linear_in;
                    var linear_out: usize = 0;
                    for (0..x.value.shape.len) |i| {
                        const coord = remaining / in_strides[i];
                        remaining %= in_strides[i];
                        const out_coord = if (i == axis) 0 else coord;
                        linear_out += out_coord * out_strides[i];
                    }
                    x.grad.?.storage.data[linear_in] += node_grad.storage.data[linear_out];
                }
            },
            .add_scalar => {
                try accumulateGrad(allocator, node.parents[0], node_grad);
            },
            .mul_scalar => |scalar| {
                var scaled_grad = try Tensor.scale(allocator, node_grad, scalar);
                defer scaled_grad.deinit(allocator);
                try accumulateGrad(allocator, node.parents[0], &scaled_grad);
            },
            .softmax => {
                const x = node.parents[0];
                if (x.grad == null) {
                    x.grad = try Tensor.zeroes(allocator, x.value.shape);
                }

                const last_dim = node.value.shape[node.value.shape.len - 1];
                const rows = node.value.numel() / last_dim;

                for (0..rows) |r| {
                    const start = r * last_dim;
                    const end = start + last_dim;
                    const out_row = node.value.storage.data[start..end];
                    const grad_row = node_grad.storage.data[start..end];
                    const x_grad_row = x.grad.?.storage.data[start..end];

                    var sum_grad_dot_out: f32 = 0;
                    for (0..last_dim) |i| {
                        sum_grad_dot_out += grad_row[i] * out_row[i];
                    }

                    for (0..last_dim) |i| {
                        x_grad_row[i] += out_row[i] * (grad_row[i] - sum_grad_dot_out);
                    }
                }
            },

            .log_softmax => {
                const x = node.parents[0];
                if (x.grad == null) {
                    x.grad = try Tensor.zeroes(allocator, x.value.shape);
                }

                const last_dim = node.value.shape[node.value.shape.len - 1];
                const rows = node.value.numel() / last_dim;

                for (0..rows) |r| {
                    const start = r * last_dim;
                    const end = start + last_dim;
                    const out_row = node.value.storage.data[start..end];
                    const grad_row = node_grad.storage.data[start..end];
                    const x_grad_row = x.grad.?.storage.data[start..end];

                    var sum_g: f32 = 0;
                    for (grad_row) |g| sum_g += g;

                    for (0..last_dim) |i| {
                        x_grad_row[i] += grad_row[i] - @exp(out_row[i]) * sum_g;
                    }
                }
            },

            .cross_entropy => |ids| {
                const log_probs = node.parents[0];
                if (log_probs.grad == null) {
                    log_probs.grad = try Tensor.zeroes(allocator, log_probs.value.shape);
                }

                const seq = log_probs.value.shape[0];
                const vocab_size = log_probs.value.shape[1];
                const scale: f32 = -1.0 / @as(f32, @floatFromInt(seq));

                for (0..seq) |s| {
                    const correct = ids[s];
                    log_probs.grad.?.storage.data[s * vocab_size + correct] += scale * node_grad.storage.data[0];
                }
            },
            .transpose => |perm| {
                const x = node.parents[0];
                const inv_perm = try allocator.alloc(usize, perm.len);
                defer allocator.free(inv_perm);
                for (perm, 0..) |dst, src| {
                    inv_perm[dst] = src;
                }
                var grad_back = try Tensor.transpose(allocator, node_grad, inv_perm);
                defer grad_back.deinit(allocator);
                var grad_contiguous = try grad_back.contiguous(allocator);
                defer grad_contiguous.deinit(allocator);
                try accumulateGrad(allocator, x, &grad_contiguous);
            },
        }
    }

    pub fn backward(allocator: std.mem.Allocator, root: *Val) !void {
        if (root.grad == null) {
            const ones = try Tensor.init(allocator, root.value.shape);
            @memset(ones.storage.data, 1.0);
            root.grad = ones;
        }

        var ordered = try topoSort(allocator, root);
        defer ordered.deinit(allocator);

        var i = ordered.items.len;
        while (i > 0) {
            i -= 1;
            try backwardOp(allocator, ordered.items[i]);
        }
    }

    pub fn accumulateToParam(self: *Val, target_grad: *?Tensor, allocator: std.mem.Allocator) !void {
        const source = if (self.grad) |*grad| grad else return;

        if (target_grad.* == null) {
            target_grad.* = try source.clone(allocator);
            return;
        }

        if (!Tensor.shapeMatch(&target_grad.*.?, source)) return error.IncompatibleShapes;
        for (0..source.numel()) |i| {
            const existing = try target_grad.*.?.getFlat(i);
            try target_grad.*.?.setFlat(i, existing + try source.getFlat(i));
        }
    }

    pub fn embedding(allocator: std.mem.Allocator, W: *Val, ids: []const u16) !*Val {
        if (W.value.shape.len != 2) return error.UnsupportedRank;
        const d_model = W.value.shape[1];

        var value = try Tensor.init(allocator, &.{ ids.len, d_model });
        errdefer value.deinit(allocator);

        for (ids, 0..) |id, s| {
            const src_off = @as(usize, id) * d_model;
            const dst_off = s * d_model;
            @memcpy(
                value.storage.data[dst_off .. dst_off + d_model],
                W.value.storage.data[src_off .. src_off + d_model],
            );
        }

        const ids_copy = try allocator.dupe(u16, ids);
        errdefer allocator.free(ids_copy);

        const parents = try allocator.alloc(*Val, 1);
        errdefer allocator.free(parents);
        parents[0] = W;

        const self = try allocator.create(Val);
        self.* = .{
            .value = value,
            .grad = null,
            .op = .{ .embedding = ids_copy },
            .parents = parents,
        };
        return self;
    }

    pub fn sumKeepdim(allocator: std.mem.Allocator, x: *Val, axis: usize) !*Val {
        if (axis >= x.value.shape.len) return error.InvalidAxis;

        const out_shape = try allocator.alloc(usize, x.value.shape.len);
        defer allocator.free(out_shape);
        @memcpy(out_shape, x.value.shape);
        out_shape[axis] = 1;

        var value = try Tensor.zeroes(allocator, out_shape);
        errdefer value.deinit(allocator);

        const in_strides = try logicalStrides(allocator, x.value.shape);
        defer allocator.free(in_strides);
        const out_strides = try logicalStrides(allocator, out_shape);
        defer allocator.free(out_strides);

        for (0..x.value.numel()) |linear_in| {
            var remaining = linear_in;
            var linear_out: usize = 0;
            for (0..x.value.shape.len) |i| {
                const coord = remaining / in_strides[i];
                remaining %= in_strides[i];
                const out_coord = if (i == axis) 0 else coord;
                linear_out += out_coord * out_strides[i];
            }
            value.storage.data[linear_out] += x.value.storage.data[linear_in];
        }

        const parents = try allocator.alloc(*Val, 1);
        errdefer allocator.free(parents);
        parents[0] = x;

        const self = try allocator.create(Val);
        self.* = .{
            .value = value,
            .grad = null,
            .op = .{ .sum_keepdim = axis },
            .parents = parents,
        };
        return self;
    }

    pub fn reshape(allocator: std.mem.Allocator, x: *Val, new_shape: []const usize) !*Val {
        var tensor = try Tensor.reshape(allocator, &x.value, new_shape);
        errdefer tensor.deinit(allocator);

        const parents = try allocator.alloc(*Val, 1);
        errdefer allocator.free(parents);
        parents[0] = x;

        const val = try allocator.create(Val);
        val.* = .{
            .value = tensor,
            .grad = null,
            .op = .reshape,
            .parents = parents,
        };
        return val;
    }
};

pub const AdamW = struct {
    params: []*Tensor,
    grads: []*?Tensor,
    ms: []Tensor,
    vs: []Tensor,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    t: usize,

    pub fn init(
        allocator: std.mem.Allocator,
        params: []const *Tensor,
        grads: []const *?Tensor,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
    ) !AdamW {
        if (params.len != grads.len) return error.SizeMisMatch;

        const owned_params = try allocator.alloc(*Tensor, params.len);
        errdefer allocator.free(owned_params);
        @memcpy(owned_params, params);

        const owned_grads = try allocator.alloc(*?Tensor, grads.len);
        errdefer allocator.free(owned_grads);
        @memcpy(owned_grads, grads);

        const ms = try allocator.alloc(Tensor, params.len);
        errdefer allocator.free(ms);
        const vs = try allocator.alloc(Tensor, params.len);
        errdefer allocator.free(vs);

        var initialized_ms: usize = 0;
        errdefer for (0..initialized_ms) |i| ms[i].deinit(allocator);
        var initialized_vs: usize = 0;
        errdefer for (0..initialized_vs) |i| vs[i].deinit(allocator);

        for (params, 0..) |param, i| {
            ms[i] = try Tensor.zeroes(allocator, param.shape);
            initialized_ms += 1;
            vs[i] = try Tensor.zeroes(allocator, param.shape);
            initialized_vs += 1;
        }

        return .{
            .params = owned_params,
            .grads = owned_grads,
            .ms = ms,
            .vs = vs,
            .lr = lr,
            .beta1 = beta1,
            .beta2 = beta2,
            .eps = eps,
            .weight_decay = weight_decay,
            .t = 0,
        };
    }

    pub fn deinit(self: *AdamW, allocator: std.mem.Allocator) void {
        for (self.ms) |*m| m.deinit(allocator);
        for (self.vs) |*v| v.deinit(allocator);
        allocator.free(self.ms);
        allocator.free(self.vs);
        allocator.free(self.params);
        allocator.free(self.grads);
    }

    pub fn zeroGrads(self: *AdamW, allocator: std.mem.Allocator) void {
        for (self.grads) |grad_slot| {
            if (grad_slot.*) |*grad| {
                grad.deinit(allocator);
                grad_slot.* = null;
            }
        }
    }

    pub fn step(self: *AdamW) !void {
        self.t += 1;

        const one_minus_beta1_t = 1.0 - std.math.pow(f32, self.beta1, @floatFromInt(self.t));
        const one_minus_beta2_t = 1.0 - std.math.pow(f32, self.beta2, @floatFromInt(self.t));

        for (self.params, self.grads, self.ms, self.vs) |param, grad_slot, *m, *v| {
            const grad = if (grad_slot.*) |*g| g else continue;
            if (!Tensor.shapeMatch(param, grad) or !Tensor.shapeMatch(param, m) or !Tensor.shapeMatch(param, v)) {
                return error.IncompatibleShapes;
            }

            for (0..param.numel()) |i| {
                const g = try grad.getFlat(i);
                const p = param.storage.data[i];

                m.storage.data[i] = self.beta1 * m.storage.data[i] + (1.0 - self.beta1) * g;
                v.storage.data[i] = self.beta2 * v.storage.data[i] + (1.0 - self.beta2) * g * g;

                const m_hat = m.storage.data[i] / one_minus_beta1_t;
                const v_hat = v.storage.data[i] / one_minus_beta2_t;
                const update = (m_hat / (@sqrt(v_hat) + self.eps)) + self.weight_decay * p;

                param.storage.data[i] = p - self.lr * update;
            }
        }
    }
};

pub fn ropeTable(allocator: std.mem.Allocator, seq_len: usize, d_head: usize) !struct { cos: Tensor, sin: Tensor } {
    var cos_t = try Tensor.zeroes(allocator, &.{ seq_len, d_head });
    errdefer cos_t.deinit(allocator);
    var sin_t = try Tensor.zeroes(allocator, &.{ seq_len, d_head });
    errdefer sin_t.deinit(allocator);

    const dhead_f: f32 = @floatFromInt(d_head);

    for (0..seq_len) |pos| {
        for (0..d_head / 2) |i| {
            const i_f: f32 = @floatFromInt(i);
            const pos_f: f32 = @floatFromInt(pos);
            const theta: f32 = 1.0 / std.math.pow(f32, 10000.0, (2.0 * i_f) / dhead_f);
            const angle: f32 = pos_f * theta;
            cos_t.storage.data[pos * d_head + 2 * i] = @cos(angle);
            cos_t.storage.data[pos * d_head + 2 * i + 1] = @cos(angle);
            sin_t.storage.data[pos * d_head + 2 * i] = @sin(angle);
            sin_t.storage.data[pos * d_head + 2 * i + 1] = @sin(angle);
        }
    }

    return .{ .cos = cos_t, .sin = sin_t };
}

pub fn saveWeights(model: *const Model, io: std.Io, path: []const u8) !void {
    var buf: [1024 * 16]u8 = undefined;
    const file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var writer = file.writer(io, &buf);

    try writer.interface.writeAll(std.mem.sliceAsBytes(model.embedding.weight.storage.data));
    for (model.blocks) |block| {
        try writer.interface.writeAll(std.mem.sliceAsBytes(block.norm1.gamma.storage.data));
        try writer.interface.writeAll(std.mem.sliceAsBytes(block.norm2.gamma.storage.data));
        try writer.interface.writeAll(std.mem.sliceAsBytes(block.attn.w_q.weight.storage.data));
        try writer.interface.writeAll(std.mem.sliceAsBytes(block.attn.w_k.weight.storage.data));
        try writer.interface.writeAll(std.mem.sliceAsBytes(block.attn.w_v.weight.storage.data));
        try writer.interface.writeAll(std.mem.sliceAsBytes(block.attn.w_o.weight.storage.data));
        try writer.interface.writeAll(std.mem.sliceAsBytes(block.ffn.w_gate.weight.storage.data));
        try writer.interface.writeAll(std.mem.sliceAsBytes(block.ffn.w_up.weight.storage.data));
        try writer.interface.writeAll(std.mem.sliceAsBytes(block.ffn.w_down.weight.storage.data));
    }
    try writer.interface.writeAll(std.mem.sliceAsBytes(model.norm.gamma.storage.data));
    try writer.interface.writeAll(std.mem.sliceAsBytes(model.lm_head.weight.storage.data));

    try writer.interface.flush();
}

pub fn loadWeights(model: *Model, io: std.Io, path: []const u8) !void {
    var buf: [1024 * 16]u8 = undefined;
    const file = try std.Io.Dir.cwd().openFile(io, path, .{ .mode = .read_only });
    defer file.close(io);
    var reader = file.reader(io, &buf);

    try reader.interface.readSliceAll(std.mem.sliceAsBytes(model.embedding.weight.storage.data));
    for (model.blocks) |*block| {
        try reader.interface.readSliceAll(std.mem.sliceAsBytes(block.norm1.gamma.storage.data));
        try reader.interface.readSliceAll(std.mem.sliceAsBytes(block.norm2.gamma.storage.data));
        try reader.interface.readSliceAll(std.mem.sliceAsBytes(block.attn.w_q.weight.storage.data));
        try reader.interface.readSliceAll(std.mem.sliceAsBytes(block.attn.w_k.weight.storage.data));
        try reader.interface.readSliceAll(std.mem.sliceAsBytes(block.attn.w_v.weight.storage.data));
        try reader.interface.readSliceAll(std.mem.sliceAsBytes(block.attn.w_o.weight.storage.data));
        try reader.interface.readSliceAll(std.mem.sliceAsBytes(block.ffn.w_gate.weight.storage.data));
        try reader.interface.readSliceAll(std.mem.sliceAsBytes(block.ffn.w_up.weight.storage.data));
        try reader.interface.readSliceAll(std.mem.sliceAsBytes(block.ffn.w_down.weight.storage.data));
    }
    try reader.interface.readSliceAll(std.mem.sliceAsBytes(model.norm.gamma.storage.data));
    try reader.interface.readSliceAll(std.mem.sliceAsBytes(model.lm_head.weight.storage.data));
}

//TESTING STUFF STARTS HERE

fn expectTensorData(tensor: *const Tensor, expected: []const f32) !void {
    const allocator = std.testing.allocator;
    const actual = try tensor.toOwnedSlice(allocator);
    defer allocator.free(actual);
    try std.testing.expectEqualSlices(f32, expected, actual);
}

fn expectTensorApprox(tensor: *const Tensor, expected: []const f32, tolerance: f32) !void {
    const allocator = std.testing.allocator;
    const actual = try tensor.toOwnedSlice(allocator);
    defer allocator.free(actual);
    try std.testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |e, a| {
        try std.testing.expectApproxEqAbs(e, a, tolerance);
    }
}

fn expectTensorFinite(tensor: *const Tensor) !void {
    for (tensor.storage.data) |v| {
        try std.testing.expect(std.math.isFinite(v));
    }
}

const TransformerLeafGradCounts = struct {
    norm_gamma: usize = 0,
    attention_weight: usize = 0,
    ffn_gate_or_up_weight: usize = 0,
    ffn_down_weight: usize = 0,
};

fn transformerLeafGradCounts(allocator: std.mem.Allocator, root: *Val, d_model: usize, d_ffn: usize) !TransformerLeafGradCounts {
    var visited = std.AutoHashMap(*Val, void).init(allocator);
    defer visited.deinit();

    var stack: std.ArrayList(*Val) = .empty;
    defer stack.deinit(allocator);
    try stack.append(allocator, root);

    var counts = TransformerLeafGradCounts{};
    while (stack.pop()) |node| {
        if (visited.contains(node)) continue;
        try visited.put(node, {});

        for (node.parents) |parent| {
            try stack.append(allocator, parent);
        }

        if (node.op != .leaf or node.grad == null) continue;

        if (std.mem.eql(usize, node.value.shape, &.{d_model})) {
            counts.norm_gamma += 1;
        } else if (std.mem.eql(usize, node.value.shape, &.{ d_model, d_model })) {
            counts.attention_weight += 1;
        } else if (std.mem.eql(usize, node.value.shape, &.{ d_model, d_ffn })) {
            counts.ffn_gate_or_up_weight += 1;
        } else if (std.mem.eql(usize, node.value.shape, &.{ d_ffn, d_model })) {
            counts.ffn_down_weight += 1;
        }

        try expectTensorFinite(&node.grad.?);
    }

    return counts;
}

test "Val accumulateToParam copies and accumulates" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const ga = arena.allocator();

    var value = try Tensor.fromSlice(allocator, &.{2}, &.{ 0.0, 0.0 });
    defer value.deinit(allocator);

    const val = try Val.makeLeaf(ga, &value);
    val.grad = try Tensor.fromSlice(allocator, &.{2}, &.{ 1.0, 2.0 });
    defer val.grad.?.deinit(allocator);

    var persistent_grad: ?Tensor = null;
    defer if (persistent_grad) |*grad| grad.deinit(allocator);

    try val.accumulateToParam(&persistent_grad, allocator);
    try expectTensorData(&persistent_grad.?, &.{ 1.0, 2.0 });

    try val.accumulateToParam(&persistent_grad, allocator);
    try expectTensorData(&persistent_grad.?, &.{ 2.0, 4.0 });
}

test "AdamW step and zeroGrads" {
    const allocator = std.testing.allocator;

    var param = try Tensor.fromSlice(allocator, &.{2}, &.{ 1.0, 2.0 });
    defer param.deinit(allocator);

    var grad_slot: ?Tensor = try Tensor.fromSlice(allocator, &.{2}, &.{ 0.1, -0.2 });

    var params = [_]*Tensor{&param};
    var grads = [_]*?Tensor{&grad_slot};
    var optimizer = try AdamW.init(allocator, params[0..], grads[0..], 0.01, 0.9, 0.999, 0.0, 0.01);
    defer optimizer.deinit(allocator);

    try optimizer.step();
    try expectTensorApprox(&param, &.{ 0.9899, 2.0098 }, 1e-6);

    optimizer.zeroGrads(allocator);
    try std.testing.expect(grad_slot == null);
}

test "Model collectParams includes every persistent grad slot" {
    const allocator = std.testing.allocator;

    const vocab_size: usize = 11;
    const d_model: usize = 4;
    const n_heads: usize = 2;
    const d_ffn: usize = 8;
    const n_layers: usize = 2;
    const seq_len: usize = 3;

    var model = try Model.init(allocator, vocab_size, d_model, n_heads, d_ffn, n_layers, seq_len, TESTSEED);
    defer model.deinit(allocator);

    var params: std.ArrayList(*Tensor) = .empty;
    defer params.deinit(allocator);
    var grads: std.ArrayList(*?Tensor) = .empty;
    defer grads.deinit(allocator);

    try model.collectParams(allocator, &params, &grads);

    const expected = 1 + n_layers * 9 + 2;
    try std.testing.expectEqual(@as(usize, expected), params.items.len);
    try std.testing.expectEqual(params.items.len, grads.items.len);

    for (grads.items) |grad_slot| {
        try std.testing.expect(grad_slot.* == null);
    }
}

test "Model accumulateGrads populates and zeroGrads clears persistent grad slots" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const ga = arena.allocator();

    const vocab_size: usize = 11;
    const d_model: usize = 4;
    const n_heads: usize = 2;
    const d_ffn: usize = 8;
    const n_layers: usize = 2;
    const seq_len: usize = 3;

    var model = try Model.init(allocator, vocab_size, d_model, n_heads, d_ffn, n_layers, seq_len, TESTSEED);
    defer model.deinit(allocator);

    const ids = [_]u16{ 1, 5, 2 };

    var mask_tensor = try causalMask(allocator, seq_len);
    defer mask_tensor.deinit(allocator);
    const mask = try Val.makeLeaf(ga, &mask_tensor);

    const result = try model.forward(ga, &ids, mask);
    try Val.backward(ga, result.out);

    try model.accumulateGrads(allocator, &result);

    var params: std.ArrayList(*Tensor) = .empty;
    defer params.deinit(allocator);
    var grads: std.ArrayList(*?Tensor) = .empty;
    defer grads.deinit(allocator);
    try model.collectParams(allocator, &params, &grads);

    for (grads.items) |grad_slot| {
        try std.testing.expect(grad_slot.* != null);
        try expectTensorFinite(&grad_slot.*.?);
    }

    model.zeroGrads(allocator);

    for (grads.items) |grad_slot| {
        try std.testing.expect(grad_slot.* == null);
    }
}

test "Autograd Add Backward" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const graph_allocator = arena.allocator();

    var x_tensor = try Tensor.fromSlice(allocator, &.{ 2, 2 }, &.{ 1.0, 2.0, 3.0, 4.0 });
    defer x_tensor.deinit(allocator);
    var y_tensor = try Tensor.fromSlice(allocator, &.{ 2, 2 }, &.{ 5.0, 6.0, 7.0, 8.0 });
    defer y_tensor.deinit(allocator);

    const x = try Val.makeLeaf(graph_allocator, &x_tensor);
    const y = try Val.makeLeaf(graph_allocator, &y_tensor);
    const z = try Val.add(graph_allocator, x, y);

    try Val.backward(graph_allocator, z);

    try expectTensorData(&x.grad.?, &.{ 1.0, 1.0, 1.0, 1.0 });
    try expectTensorData(&y.grad.?, &.{ 1.0, 1.0, 1.0, 1.0 });
}

test "Autograd Mul Backward With Broadcast" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const graph_allocator = arena.allocator();

    var x_tensor = try Tensor.fromSlice(allocator, &.{ 2, 3 }, &.{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });
    defer x_tensor.deinit(allocator);
    var y_tensor = try Tensor.fromSlice(allocator, &.{3}, &.{ 10.0, 20.0, 30.0 });
    defer y_tensor.deinit(allocator);

    const x = try Val.makeLeaf(graph_allocator, &x_tensor);
    const y = try Val.makeLeaf(graph_allocator, &y_tensor);
    const z = try Val.mul(graph_allocator, x, y);

    try Val.backward(graph_allocator, z);

    try expectTensorData(&x.grad.?, &.{ 10.0, 20.0, 30.0, 10.0, 20.0, 30.0 });
    try expectTensorData(&y.grad.?, &.{ 5.0, 7.0, 9.0 });
}

test "Autograd Matmul Backward 2D" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const graph_allocator = arena.allocator();

    var a_tensor = try Tensor.fromSlice(allocator, &.{ 2, 3 }, &.{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });
    defer a_tensor.deinit(allocator);
    var b_tensor = try Tensor.fromSlice(allocator, &.{ 3, 2 }, &.{ 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 });
    defer b_tensor.deinit(allocator);

    const a = try Val.makeLeaf(graph_allocator, &a_tensor);
    const b = try Val.makeLeaf(graph_allocator, &b_tensor);
    const out = try Val.matmul(graph_allocator, a, b);

    try Val.backward(graph_allocator, out);

    try expectTensorData(&a.grad.?, &.{ 15.0, 19.0, 23.0, 15.0, 19.0, 23.0 });
    try expectTensorData(&b.grad.?, &.{ 5.0, 5.0, 7.0, 7.0, 9.0, 9.0 });
}

test "Autograd Embedding Backward" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const ga = arena.allocator();

    var w_tensor = try Tensor.fromSlice(allocator, &.{ 4, 3 }, &.{
        0.0, 1.0,  2.0,
        3.0, 4.0,  5.0,
        6.0, 7.0,  8.0,
        9.0, 10.0, 11.0,
    });
    defer w_tensor.deinit(allocator);

    const W = try Val.makeLeaf(ga, &w_tensor);
    const ids = [_]u16{ 2, 0, 2 };
    const out = try Val.embedding(ga, W, &ids);

    try expectTensorData(&out.value, &.{
        6.0, 7.0, 8.0,
        0.0, 1.0, 2.0,
        6.0, 7.0, 8.0,
    });

    var ones = try Tensor.init(allocator, out.value.shape);
    defer ones.deinit(allocator);
    @memset(ones.storage.data, 1.0);
    out.grad = try ones.clone(ga);

    try Val.backward(ga, out);

    try expectTensorData(&W.grad.?, &.{
        1.0, 1.0, 1.0,
        0.0, 0.0, 0.0,
        2.0, 2.0, 2.0,
        0.0, 0.0, 0.0,
    });
}

test "Linear forward shape" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const ga = arena.allocator();

    var linear = try Linear.init(allocator, 3, 4, TESTSEED);
    defer linear.deinit(allocator);

    var x_tensor = try Tensor.fromSlice(allocator, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x_tensor.deinit(allocator);

    const x = try Val.makeLeaf(ga, &x_tensor);
    const W = try Val.makeLeaf(ga, &linear.weight);
    const out = try Val.matmul(ga, x, W);

    try std.testing.expectEqualSlices(usize, &.{ 2, 4 }, out.value.shape);
}

test "Sum keepdim forward axis 1" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const ga = arena.allocator();

    var x_tensor = try Tensor.fromSlice(allocator, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x_tensor.deinit(allocator);

    const x = try Val.makeLeaf(ga, &x_tensor);
    const out = try Val.sumKeepdim(ga, x, 1);

    try std.testing.expectEqualSlices(usize, &.{ 2, 1 }, out.value.shape);
    try expectTensorData(&out.value, &.{ 6, 15 });
}

test "Sum keepdim backward" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const ga = arena.allocator();

    var x_tensor = try Tensor.fromSlice(allocator, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x_tensor.deinit(allocator);

    const x = try Val.makeLeaf(ga, &x_tensor);
    const out = try Val.sumKeepdim(ga, x, 1);

    try Val.backward(ga, out);

    try expectTensorData(&x.grad.?, &.{ 1, 1, 1, 1, 1, 1 });
}

test "Sum keepdim axis 0" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const ga = arena.allocator();

    var x_tensor = try Tensor.fromSlice(allocator, &.{ 2, 3 }, &.{ 1, 2, 3, 4, 5, 6 });
    defer x_tensor.deinit(allocator);

    const x = try Val.makeLeaf(ga, &x_tensor);
    const out = try Val.sumKeepdim(ga, x, 0);

    try std.testing.expectEqualSlices(usize, &.{ 1, 3 }, out.value.shape);
    try expectTensorData(&out.value, &.{ 5, 7, 9 });
}

test "Rsqrt forward" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const ga = arena.allocator();

    var x_tensor = try Tensor.fromSlice(allocator, &.{4}, &.{ 1.0, 4.0, 16.0, 0.25 });
    defer x_tensor.deinit(allocator);

    const x = try Val.makeLeaf(ga, &x_tensor);
    const out = try Val.rsqrt(ga, x);

    try expectTensorData(&out.value, &.{ 1.0, 0.5, 0.25, 2.0 });
}

test "Rsqrt backward" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const ga = arena.allocator();

    var x_tensor = try Tensor.fromSlice(allocator, &.{4}, &.{ 1.0, 4.0, 16.0, 0.25 });
    defer x_tensor.deinit(allocator);

    const x = try Val.makeLeaf(ga, &x_tensor);
    const out = try Val.rsqrt(ga, x);

    try Val.backward(ga, out);

    try expectTensorData(&x.grad.?, &.{ -0.5, -0.0625, -0.0078125, -4.0 });
}

test "Add scalar backward" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const ga = arena.allocator();

    var x_tensor = try Tensor.fromSlice(allocator, &.{3}, &.{ 1, 2, 3 });
    defer x_tensor.deinit(allocator);

    const x = try Val.makeLeaf(ga, &x_tensor);
    const out = try Val.addScalar(ga, x, 10.0);

    try expectTensorData(&out.value, &.{ 11, 12, 13 });
    try Val.backward(ga, out);
    try expectTensorData(&x.grad.?, &.{ 1, 1, 1 });
}

test "Mul scalar backward" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const ga = arena.allocator();

    var x_tensor = try Tensor.fromSlice(allocator, &.{3}, &.{ 1, 2, 3 });
    defer x_tensor.deinit(allocator);

    const x = try Val.makeLeaf(ga, &x_tensor);
    const out = try Val.mulScalar(ga, x, 3.0);

    try expectTensorData(&out.value, &.{ 3, 6, 9 });
    try Val.backward(ga, out);
    try expectTensorData(&x.grad.?, &.{ 3, 3, 3 });
}

test "RMSNorm forward" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const ga = arena.allocator();

    var rms = try RMSNorm.init(allocator, 4, 0.0);
    defer rms.deinit(allocator);

    var x_tensor = try Tensor.fromSlice(allocator, &.{ 2, 4 }, &.{
        1.0, 0.0, 0.0, 0.0,
        0.0, 2.0, 0.0, 0.0,
    });
    defer x_tensor.deinit(allocator);

    const x = try Val.makeLeaf(ga, &x_tensor);

    const out_fwd = try rms.forward(ga, x);
    const out = out_fwd.out;

    try expectTensorData(&out.value, &.{
        2.0, 0.0, 0.0, 0.0,
        0.0, 2.0, 0.0, 0.0,
    });
}

test "RMSNorm backward shapes" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const ga = arena.allocator();

    var rms = try RMSNorm.init(allocator, 4, 1e-5);
    defer rms.deinit(allocator);

    var x_tensor = try Tensor.fromSlice(allocator, &.{ 2, 4 }, &.{
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
    });
    defer x_tensor.deinit(allocator);

    const x = try Val.makeLeaf(ga, &x_tensor);

    const out_fwd = try rms.forward(ga, x);
    const out = out_fwd.out;
    try Val.backward(ga, out);

    try std.testing.expectEqualSlices(usize, x.value.shape, x.grad.?.shape);
}

test "SiLU forward" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const ga = arena.allocator();

    var x_tensor = try Tensor.fromSlice(allocator, &.{4}, &.{ 0.0, 1.0, -1.0, 2.0 });
    defer x_tensor.deinit(allocator);

    const x = try Val.makeLeaf(ga, &x_tensor);
    const out = try Val.silu(ga, x);

    try expectTensorApprox(&out.value, &.{ 0.0, 0.73105858, -0.26894142, 1.76159416 }, 1e-5);
}

test "SiLU backward" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const ga = arena.allocator();

    var x_tensor = try Tensor.fromSlice(allocator, &.{4}, &.{ 0.0, 1.0, -1.0, 2.0 });
    defer x_tensor.deinit(allocator);

    const x = try Val.makeLeaf(ga, &x_tensor);
    const out = try Val.silu(ga, x);

    try Val.backward(ga, out);

    try expectTensorApprox(&x.grad.?, &.{ 0.5, 0.92767054, 0.07232946, 1.09078812 }, 1e-5);
}

test "SwiGLU forward shape" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const ga = arena.allocator();

    const d_model = 4;
    const d_ffn = 16;
    const seq = 3;

    var mlp = try SwiGLU.init(allocator, d_model, d_ffn, 0);
    defer mlp.deinit(allocator);

    var x_tensor = try Tensor.init(allocator, &.{ seq, d_model });
    defer x_tensor.deinit(allocator);
    @memset(x_tensor.storage.data, 0.5);

    const x = try Val.makeLeaf(ga, &x_tensor);
    const w_gate = try Val.makeLeaf(ga, &mlp.w_gate.weight);
    const w_up = try Val.makeLeaf(ga, &mlp.w_up.weight);
    const w_down = try Val.makeLeaf(ga, &mlp.w_down.weight);

    const out = try swiglu(ga, x, w_gate, w_up, w_down);

    try std.testing.expectEqualSlices(usize, &.{ seq, d_model }, out.value.shape);
}

test "SwiGLU backward populates all parameter grads" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const ga = arena.allocator();

    const d_model = 4;
    const d_ffn = 16;
    const seq = 3;

    var mlp = try SwiGLU.init(allocator, d_model, d_ffn, 0);
    defer mlp.deinit(allocator);

    var x_tensor = try Tensor.init(allocator, &.{ seq, d_model });
    defer x_tensor.deinit(allocator);
    @memset(x_tensor.storage.data, 0.5);

    const x = try Val.makeLeaf(ga, &x_tensor);
    const w_gate = try Val.makeLeaf(ga, &mlp.w_gate.weight);
    const w_up = try Val.makeLeaf(ga, &mlp.w_up.weight);
    const w_down = try Val.makeLeaf(ga, &mlp.w_down.weight);

    const out = try swiglu(ga, x, w_gate, w_up, w_down);
    try Val.backward(ga, out);

    try std.testing.expect(x.grad != null);
    try std.testing.expect(w_gate.grad != null);
    try std.testing.expect(w_up.grad != null);
    try std.testing.expect(w_down.grad != null);

    try std.testing.expectEqualSlices(usize, w_gate.value.shape, w_gate.grad.?.shape);
    try std.testing.expectEqualSlices(usize, w_up.value.shape, w_up.grad.?.shape);
    try std.testing.expectEqualSlices(usize, w_down.value.shape, w_down.grad.?.shape);
}

test "Softmax forward" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const ga = arena.allocator();

    var x_tensor = try Tensor.fromSlice(allocator, &.{ 2, 3 }, &.{ 0.0, 0.0, 0.0, 1.0, 2.0, 3.0 });
    defer x_tensor.deinit(allocator);

    const x = try Val.makeLeaf(ga, &x_tensor);
    const out = try Val.softmax(ga, x);

    const val1 = 1.0 / 3.0;
    const e1 = @exp(@as(f32, 1.0 - 3.0));
    const e2 = @exp(@as(f32, 2.0 - 3.0));
    const e3 = @exp(@as(f32, 3.0 - 3.0));
    const sum = e1 + e2 + e3;

    try expectTensorApprox(&out.value, &.{
        val1,     val1,     val1,
        e1 / sum, e2 / sum, e3 / sum,
    }, 1e-6);
}

test "Softmax backward" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const ga = arena.allocator();

    // Using a simple case: 1D softmax
    var x_tensor = try Tensor.fromSlice(allocator, &.{3}, &.{ 1.0, 2.0, 3.0 });
    defer x_tensor.deinit(allocator);

    const x = try Val.makeLeaf(ga, &x_tensor);
    const out = try Val.softmax(ga, x);

    // Set grad of out to [1, 0, 0]
    var out_grad = try Tensor.zeroes(allocator, &.{3});
    defer out_grad.deinit(allocator);
    out_grad.storage.data[0] = 1.0;
    out.grad = try out_grad.clone(ga);

    try Val.backward(ga, out);

    // If grad_out = [1, 0, 0], then x_grad_j = S_j * (grad_out_j - sum(grad_out_i * S_i))
    // S = [s1, s2, s3]
    // sum(grad_out_i * S_i) = 1*s1 + 0*s2 + 0*s3 = s1
    // x_grad_0 = s1 * (1 - s1)
    // x_grad_1 = s2 * (0 - s1) = -s2*s1
    // x_grad_2 = s3 * (0 - s1) = -s3*s1

    const e1 = @exp(@as(f32, 1.0 - 3.0));
    const e2 = @exp(@as(f32, 2.0 - 3.0));
    const e3 = @exp(@as(f32, 3.0 - 3.0));
    const sum = e1 + e2 + e3;
    const s1 = e1 / sum;
    const s2 = e2 / sum;
    const s3 = e3 / sum;

    try expectTensorApprox(&x.grad.?, &.{
        s1 * (1.0 - s1),
        -s2 * s1,
        -s3 * s1,
    }, 1e-6);
}

test "LogSoftmax forward/backward" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const ga = arena.allocator();

    var x_tensor = try Tensor.fromSlice(allocator, &.{ 2, 3 }, &.{
        0.0, 0.0, 0.0,
        1.0, 2.0, 3.0,
    });
    defer x_tensor.deinit(allocator);

    const x = try Val.makeLeaf(ga, &x_tensor);
    const out = try Val.logSoftmax(ga, x);

    const log_three = @log(@as(f32, 3.0));
    const e1 = @exp(@as(f32, 1.0 - 3.0));
    const e2 = @exp(@as(f32, 2.0 - 3.0));
    const e3 = @exp(@as(f32, 3.0 - 3.0));
    const sum = e1 + e2 + e3;
    const log_sum = @log(sum);

    try expectTensorApprox(&out.value, &.{
        -log_three,     -log_three,     -log_three,
        -2.0 - log_sum, -1.0 - log_sum, -log_sum,
    }, 1e-6);

    var out_grad = try Tensor.fromSlice(allocator, &.{ 2, 3 }, &.{
        1.0, 0.0,  0.0,
        2.0, -1.0, 3.0,
    });
    defer out_grad.deinit(allocator);
    out.grad = try out_grad.clone(ga);

    try Val.backward(ga, out);

    const s1 = e1 / sum;
    const s2 = e2 / sum;
    const s3 = e3 / sum;
    const second_row_grad_sum = 2.0 - 1.0 + 3.0;

    try expectTensorApprox(&x.grad.?, &.{
        2.0 / 3.0,                      -1.0 / 3.0,                      -1.0 / 3.0,
        2.0 - s1 * second_row_grad_sum, -1.0 - s2 * second_row_grad_sum, 3.0 - s3 * second_row_grad_sum,
    }, 1e-6);
}

test "CrossEntropy forward/backward" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const ga = arena.allocator();

    var logits_tensor = try Tensor.fromSlice(allocator, &.{ 2, 3 }, &.{
        1.0, 2.0,  3.0,
        0.0, -1.0, 1.0,
    });
    defer logits_tensor.deinit(allocator);

    const logits = try Val.makeLeaf(ga, &logits_tensor);
    const ids = [_]u16{ 2, 0 };
    const loss = try Val.crossEntropy(ga, logits, &ids);

    const e1 = @exp(@as(f32, 1.0 - 3.0));
    const e2 = @exp(@as(f32, 2.0 - 3.0));
    const e3 = @exp(@as(f32, 3.0 - 3.0));
    const row0_sum = e1 + e2 + e3;
    const row0_log_sum = @log(row0_sum);
    const row0_target_log_prob = -row0_log_sum;

    const f1 = @exp(@as(f32, 0.0 - 1.0));
    const f2 = @exp(@as(f32, -1.0 - 1.0));
    const f3 = @exp(@as(f32, 1.0 - 1.0));
    const row1_sum = f1 + f2 + f3;
    const row1_log_sum = @log(row1_sum);
    const row1_target_log_prob = -1.0 - row1_log_sum;

    const expected_loss = -(row0_target_log_prob + row1_target_log_prob) / 2.0;

    try std.testing.expectEqualSlices(usize, &.{}, loss.value.shape);
    try expectTensorApprox(&loss.value, &.{expected_loss}, 1e-6);

    try Val.backward(ga, loss);

    const row0_s1 = e1 / row0_sum;
    const row0_s2 = e2 / row0_sum;
    const row0_s3 = e3 / row0_sum;
    const row1_s1 = f1 / row1_sum;
    const row1_s2 = f2 / row1_sum;
    const row1_s3 = f3 / row1_sum;

    try expectTensorApprox(&logits.grad.?, &.{
        row0_s1 / 2.0,         row0_s2 / 2.0, (row0_s3 - 1.0) / 2.0,
        (row1_s1 - 1.0) / 2.0, row1_s2 / 2.0, row1_s3 / 2.0,
    }, 1e-6);
}

test "Softmax 3D" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const ga = arena.allocator();

    // Shape [2, 1, 2] -> 4 total elements, 2 rows of size 2
    var x_tensor = try Tensor.fromSlice(allocator, &.{ 2, 1, 2 }, &.{
        1.0, 2.0, // Row 1
        10.0, 20.0, // Row 2
    });
    defer x_tensor.deinit(allocator);

    const x = try Val.makeLeaf(ga, &x_tensor);
    const out = try Val.softmax(ga, x);

    const s1_row1 = @exp(@as(f32, -1.0)) / (1.0 + @exp(@as(f32, -1.0)));
    const s2_row1 = 1.0 / (1.0 + @exp(@as(f32, -1.0)));

    const s1_row2 = @exp(@as(f32, -10.0)) / (1.0 + @exp(@as(f32, -10.0)));
    const s2_row2 = 1.0 / (1.0 + @exp(@as(f32, -10.0)));

    try expectTensorApprox(&out.value, &.{
        s1_row1, s2_row1,
        s1_row2, s2_row2,
    }, 1e-6);
}

test "Softmax Numerical Stability" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const ga = arena.allocator();

    // Very large values that would overflow exp(x)
    var x_tensor = try Tensor.fromSlice(allocator, &.{2}, &.{ 1000.0, 1001.0 });
    defer x_tensor.deinit(allocator);

    const x = try Val.makeLeaf(ga, &x_tensor);
    const out = try Val.softmax(ga, x);

    const s1 = @exp(@as(f32, -1.0)) / (1.0 + @exp(@as(f32, -1.0)));
    const s2 = 1.0 / (1.0 + @exp(@as(f32, -1.0)));

    try expectTensorApprox(&out.value, &.{ s1, s2 }, 1e-6);

    // Very small negative values
    var x_neg = try Tensor.fromSlice(allocator, &.{2}, &.{ -1000.0, -1001.0 });
    defer x_neg.deinit(allocator);
    const x_n = try Val.makeLeaf(ga, &x_neg);
    const out_n = try Val.softmax(ga, x_n);

    // Should still work because we subtract the max (-1000)
    // -1000 - (-1000) = 0
    // -1001 - (-1000) = -1
    try expectTensorApprox(&out_n.value, &.{ s2, s1 }, 1e-6);
}

test "Softmax Gradient Complex" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const ga = arena.allocator();

    var x_tensor = try Tensor.fromSlice(allocator, &.{2}, &.{ 0.5, 1.5 });
    defer x_tensor.deinit(allocator);

    const x = try Val.makeLeaf(ga, &x_tensor);
    const out = try Val.softmax(ga, x);

    // Set a complex gradient
    var out_grad = try Tensor.fromSlice(allocator, &.{2}, &.{ 2.0, 3.0 });
    defer out_grad.deinit(allocator);
    out.grad = try out_grad.clone(ga);

    try Val.backward(ga, out);

    const s1 = @exp(@as(f32, -1.0)) / (1.0 + @exp(@as(f32, -1.0)));
    const s2 = 1.0 / (1.0 + @exp(@as(f32, -1.0)));
    const dot = 2.0 * s1 + 3.0 * s2;

    try expectTensorApprox(&x.grad.?, &.{
        s1 * (2.0 - dot),
        s2 * (3.0 - dot),
    }, 1e-6);
}

test "Causal Mask" {
    const allocator = std.testing.allocator;
    const seq_len = 3;
    var mask = try causalMask(allocator, seq_len);
    defer mask.deinit(allocator);

    try std.testing.expectEqualSlices(usize, &.{ 3, 3 }, mask.shape);

    const expected = [_]f32{
        0.0, -1e9, -1e9,
        0.0, 0.0,  -1e9,
        0.0, 0.0,  0.0,
    };

    try expectTensorData(&mask, &expected);
}

test "Attention forward/backward" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const ga = arena.allocator();

    const d_model: usize = 8;
    const seq_len: usize = 4;

    var attn_layer = try Attention.init(allocator, d_model, 2, seq_len, TESTSEED);
    defer attn_layer.deinit(allocator);

    var x_tensor = try Tensor.init(allocator, &.{ seq_len, d_model });
    defer x_tensor.deinit(allocator);
    @memset(x_tensor.storage.data, 0.1);

    const x = try Val.makeLeaf(ga, &x_tensor);

    var mask_tensor = try causalMask(allocator, seq_len);
    defer mask_tensor.deinit(allocator);
    const mask = try Val.makeLeaf(ga, &mask_tensor);

    const out_fwd = try attn_layer.forward(ga, x, mask);
    const out = out_fwd.out;

    try std.testing.expectEqualSlices(usize, &.{ seq_len, d_model }, out.value.shape);

    try Val.backward(ga, out);

    // Verify gradients are populated for all components
    // Note: in attn_layer.forward, we create new leaf nodes for weights
    // To check those gradients, we'd need to access the parents of the internal nodes,
    // but we can at least verify x.grad.
    try std.testing.expect(x.grad != null);
    try std.testing.expectEqualSlices(usize, x.value.shape, x.grad.?.shape);
}

test "Reshape" {
    const ga = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(ga);
    const allocator = arena.allocator();
    defer arena.deinit();
    const expected = [_]f32{
        1, 2,
        3, 4,
    };
    var tensor = try Tensor.fromSlice(allocator, &.{ 2, 2 }, &expected);
    defer tensor.deinit(allocator);
    var og = Val{ .value = tensor, .parents = &.{}, .grad = null, .op = .reshape };
    var val = try Val.reshape(allocator, &og, &.{ 2, 2 });
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, val.value.shape);
    try expectTensorData(&val.value, &expected);
}

test "RoPE forward/backward" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const ga = arena.allocator();

    const n_heads = 1;
    const seq = 2;
    const d_head = 4;

    // x shape [1, 2, 4]
    var x_tensor = try Tensor.fromSlice(allocator, &.{ n_heads, seq, d_head }, &.{
        1.0, 2.0, 3.0, 4.0, // pos 0
        5.0, 6.0, 7.0, 8.0, // pos 1
    });
    defer x_tensor.deinit(allocator);
    const x = try Val.makeLeaf(ga, &x_tensor);

    // cos/sin tables [2, 4]
    var cos = try Tensor.zeroes(allocator, &.{ seq, d_head });
    defer cos.deinit(allocator);
    var sin = try Tensor.zeroes(allocator, &.{ seq, d_head });
    defer sin.deinit(allocator);

    // pos 0: theta=0 -> cos=1, sin=0
    cos.storage.data[0] = 1.0;
    cos.storage.data[1] = 1.0;
    cos.storage.data[2] = 1.0;
    cos.storage.data[3] = 1.0;
    // pos 1: use some known angle, e.g., pi/2 -> cos=0, sin=1
    const half_pi = std.math.pi / 2.0;
    cos.storage.data[4] = @cos(half_pi);
    cos.storage.data[5] = @cos(half_pi);
    cos.storage.data[6] = @cos(half_pi);
    cos.storage.data[7] = @cos(half_pi);
    sin.storage.data[4] = @sin(half_pi);
    sin.storage.data[5] = @sin(half_pi);
    sin.storage.data[6] = @sin(half_pi);
    sin.storage.data[7] = @sin(half_pi);

    const out = try Val.rope(ga, x, cos, sin);

    // Expected forward:
    // pos 0: [1, 2, 3, 4] rotated by 0 -> [1, 2, 3, 4]
    // pos 1: [5, 6, 7, 8] rotated by pi/2:
    // pair 0: [5*0 - 6*1, 6*0 + 5*1] = [-6, 5]
    // pair 1: [7*0 - 8*1, 8*0 + 7*1] = [-8, 7]
    try expectTensorApprox(&out.value, &.{
        1.0,  2.0, 3.0,  4.0,
        -6.0, 5.0, -8.0, 7.0,
    }, 1e-6);

    try Val.backward(ga, out);

    // Expected backward:
    // dL/dout = [1, 1, 1, 1, 1, 1, 1, 1]
    // pos 0: rotated by 0 -> [1, 1, 1, 1]
    // pos 1: dy = [1, 1, 1, 1] rotated by -pi/2
    // pair 0: [1*0 + 1*1, 1*0 - 1*1] = [1, -1]
    // pair 1: [1*0 + 1*1, 1*0 - 1*1] = [1, -1]
    try expectTensorApprox(&x.grad.?, &.{
        1.0, 1.0,  1.0, 1.0,
        1.0, -1.0, 1.0, -1.0,
    }, 1e-6);
}

test "Attention with RoPE comprehensive" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const ga = arena.allocator();

    const d_model = 4;
    const n_heads = 2;
    const seq_len = 2;

    var attn = try Attention.init(allocator, d_model, n_heads, seq_len, 42);
    defer attn.deinit(allocator);

    var x_tensor = try Tensor.init(allocator, &.{ seq_len, d_model });
    defer x_tensor.deinit(allocator);
    @memset(x_tensor.storage.data, 1.0);
    const x = try Val.makeLeaf(ga, &x_tensor);

    var mask_tensor = try causalMask(allocator, seq_len);
    defer mask_tensor.deinit(allocator);
    const mask = try Val.makeLeaf(ga, &mask_tensor);

    const out_fwd = try attn.forward(ga, x, mask);
    const out = out_fwd.out;
    try Val.backward(ga, out);

    try std.testing.expect(x.grad != null);
}

test "TransformerBlock forward/backward" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const ga = arena.allocator();

    const d_model: usize = 4;
    const n_heads: usize = 2;
    const d_ffn: usize = 8;
    const seq_len: usize = 3;

    var block = try TransformerBlock.init(allocator, d_model, n_heads, d_ffn, seq_len, TESTSEED);
    defer block.deinit(allocator);

    var x_tensor = try Tensor.fromSlice(allocator, &.{ seq_len, d_model }, &.{
        0.10, -0.20, 0.30, -0.40,
        0.50, -0.60, 0.70, -0.80,
        0.90, -1.00, 1.10, -1.20,
    });
    defer x_tensor.deinit(allocator);
    const x = try Val.makeLeaf(ga, &x_tensor);

    var mask_tensor = try causalMask(allocator, seq_len);
    defer mask_tensor.deinit(allocator);
    const mask = try Val.makeLeaf(ga, &mask_tensor);

    const out_fwd = try block.forward(ga, x, mask);
    const out = out_fwd.out;

    try std.testing.expectEqualSlices(usize, &.{ seq_len, d_model }, out.value.shape);
    try expectTensorFinite(&out.value);

    try Val.backward(ga, out);

    try std.testing.expect(x.grad != null);
    try std.testing.expectEqualSlices(usize, x.value.shape, x.grad.?.shape);
    try expectTensorFinite(&x.grad.?);

    const leaf_counts = try transformerLeafGradCounts(allocator, out, d_model, d_ffn);
    try std.testing.expectEqual(@as(usize, 2), leaf_counts.norm_gamma);
    try std.testing.expectEqual(@as(usize, 4), leaf_counts.attention_weight);
    try std.testing.expectEqual(@as(usize, 2), leaf_counts.ffn_gate_or_up_weight);
    try std.testing.expectEqual(@as(usize, 1), leaf_counts.ffn_down_weight);
}

test "Model forward shape" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const ga = arena.allocator();

    const vocab_size: usize = 11;
    const d_model: usize = 4;
    const n_heads: usize = 2;
    const d_ffn: usize = 8;
    const n_layers: usize = 2;
    const seq_len: usize = 3;

    var model = try Model.init(allocator, vocab_size, d_model, n_heads, d_ffn, n_layers, seq_len, TESTSEED);
    defer model.deinit(allocator);

    const ids = [_]u16{ 1, 5, 2 };

    var mask_tensor = try causalMask(allocator, seq_len);
    defer mask_tensor.deinit(allocator);
    const mask = try Val.makeLeaf(ga, &mask_tensor);

    const logits_fwd = try model.forward(ga, &ids, mask);
    const logits = logits_fwd.out;

    try std.testing.expectEqualSlices(usize, &.{ seq_len, vocab_size }, logits.value.shape);
    try expectTensorFinite(&logits.value);
}

test "Model weight save/load round-trip" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    const vocab_size: usize = 11;
    const d_model: usize = 4;
    const n_heads: usize = 2;
    const d_ffn: usize = 8;
    const n_layers: usize = 2;
    const seq_len: usize = 3;

    var model1 = try Model.init(allocator, vocab_size, d_model, n_heads, d_ffn, n_layers, seq_len, 42);
    defer model1.deinit(allocator);

    const tmp_weights = "Data/test_weights.bin";
    try saveWeights(&model1, io, tmp_weights);

    var model2 = try Model.init(allocator, vocab_size, d_model, n_heads, d_ffn, n_layers, seq_len, 43); // Different seed
    defer model2.deinit(allocator);

    try loadWeights(&model2, io, tmp_weights);

    // Verify all tensors match
    try std.testing.expectEqualSlices(f32, model1.embedding.weight.storage.data, model2.embedding.weight.storage.data);
    for (model1.blocks, model2.blocks) |b1, b2| {
        try std.testing.expectEqualSlices(f32, b1.norm1.gamma.storage.data, b2.norm1.gamma.storage.data);
        try std.testing.expectEqualSlices(f32, b1.norm2.gamma.storage.data, b2.norm2.gamma.storage.data);
        try std.testing.expectEqualSlices(f32, b1.attn.w_q.weight.storage.data, b2.attn.w_q.weight.storage.data);
        try std.testing.expectEqualSlices(f32, b1.attn.w_k.weight.storage.data, b2.attn.w_k.weight.storage.data);
        try std.testing.expectEqualSlices(f32, b1.attn.w_v.weight.storage.data, b2.attn.w_v.weight.storage.data);
        try std.testing.expectEqualSlices(f32, b1.attn.w_o.weight.storage.data, b2.attn.w_o.weight.storage.data);
        try std.testing.expectEqualSlices(f32, b1.ffn.w_gate.weight.storage.data, b2.ffn.w_gate.weight.storage.data);
        try std.testing.expectEqualSlices(f32, b1.ffn.w_up.weight.storage.data, b2.ffn.w_up.weight.storage.data);
        try std.testing.expectEqualSlices(f32, b1.ffn.w_down.weight.storage.data, b2.ffn.w_down.weight.storage.data);
    }
    try std.testing.expectEqualSlices(f32, model1.norm.gamma.storage.data, model2.norm.gamma.storage.data);
    try std.testing.expectEqualSlices(f32, model1.lm_head.weight.storage.data, model2.lm_head.weight.storage.data);
}
