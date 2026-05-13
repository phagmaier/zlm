const std = @import("std");
const builtin = @import("builtin");
const Tokenizer = @import("tokenizer.zig");
const Tensor = @import("tensor.zig").Tensor;
const ag = @import("autograd.zig");
const Val = ag.Val;
const Model = ag.Model;
const AdamW = ag.AdamW;

const INPUT_FILE = "Data/input.txt";
const SAVED_BIN = "Data/saved.bin";
const ENCODED_BIN = "Data/encoded.bin";
const WEIGHTS_PATH = "Data/weights.bin";
const CHECKPOINT_OPT = "Data/checkpoint_opt.bin";
const VOCAB_SIZE: usize = 4096;

const GRAD_CLIP: f32 = 1.0;
const D_MODEL: usize = 256;
const N_HEADS: usize = 4;
const D_FFN: usize = 1024;
const N_LAYERS: usize = 6;
const SEQ_LEN: usize = 512;
const BATCH_SIZE: usize = 1;
const LR: f32 = 3e-4;
const BETA1: f32 = 0.9;
const BETA2: f32 = 0.999;
const EPS: f32 = 1e-8;
const WEIGHT_DECAY: f32 = 0.01;
const N_STEPS: usize = 300000;
const LOG_EVERY: usize = 100;
const WARMUP_STEPS: usize = 1000;
const CHECKPOINT_EVERY: usize = 1000;
const MIN_LR: f32 = LR * 0.1;

pub fn main(init: std.process.Init) !void {
    const args = init.minimal.args.vector;
    const io = init.io;
    var da = std.heap.DebugAllocator(.{}){};
    const allocator = if (builtin.mode == .Debug) da.allocator() else std.heap.smp_allocator;
    defer _ = da.deinit();

    if (args.len < 2) {
        usage();
        return;
    }

    const subcommand = std.mem.span(args[1]);
    if (std.mem.eql(u8, subcommand, "tokenize")) {
        try runTokenize(allocator, io);
    } else if (std.mem.eql(u8, subcommand, "train")) {
        try runTrain(allocator, io);
    } else if (std.mem.eql(u8, subcommand, "save")) {
        try runSave(allocator, io);
    } else if (std.mem.eql(u8, subcommand, "load")) {
        try runLoad(allocator, io);
    } else if (std.mem.eql(u8, subcommand, "sample")) {
        const prompt = if (args.len > 2) std.mem.span(args[2]) else null;
        try runSample(allocator, io, prompt);
    } else {
        usage();
        std.process.exit(1);
    }
}

fn usage() void {
    std.debug.print("Usage: Ai <subcommand>\n", .{});
    std.debug.print("Subcommands:\n", .{});
    std.debug.print("  tokenize: Train BPE and encode corpus\n", .{});
    std.debug.print("  train:    Run the training loop\n", .{});
    std.debug.print("  save:     Initialize model and save weights\n", .{});
    std.debug.print("  load:     Initialize model and load weights\n", .{});
    std.debug.print("  sample:   [prompt] Generate text from model\n", .{});
}

fn runTokenize(allocator: std.mem.Allocator, io: std.Io) !void {
    const training_data = "Data/training_data.txt";
    const max_train_bytes = 50 * 1024 * 1024;

    std.debug.print("Training tokenizer on first 50MB of {s}...\n", .{training_data});
    var w, var m = try Tokenizer.tokenize(allocator, io, training_data, VOCAB_SIZE, max_train_bytes);
    defer w.deinit();
    defer m.deinit(allocator);

    try Tokenizer.saveMerges(io, SAVED_BIN, m.items);
    std.debug.print("Saved merges to {s}\n", .{SAVED_BIN});

    var encoder = try Tokenizer.Encoder.init(allocator, m.items);
    defer encoder.deinit();

    std.debug.print("Encoding {s} to {s} (streaming)...\n", .{ training_data, ENCODED_BIN });
    try Tokenizer.encodeFile(allocator, io, &encoder, training_data, ENCODED_BIN);
    std.debug.print("Encoded to {s}\n", .{ENCODED_BIN});
}

fn saveCheckpoint(model: *Model, optimizer: *AdamW, io: std.Io, step: usize) !void {
    try ag.saveWeights(model, io, WEIGHTS_PATH);

    var buf: [1024 * 16]u8 = undefined;
    const file = try std.Io.Dir.cwd().createFile(io, CHECKPOINT_OPT, .{});
    defer file.close(io);
    var writer = file.writer(io, &buf);

    var tmp: [8]u8 = undefined;
    std.mem.writeInt(u64, &tmp, @as(u64, step), .little);
    try writer.interface.writeAll(&tmp);
    std.mem.writeInt(u64, &tmp, @as(u64, optimizer.t), .little);
    try writer.interface.writeAll(&tmp);

    for (optimizer.ms) |m| {
        try writer.interface.writeAll(std.mem.sliceAsBytes(m.storage.data));
    }
    for (optimizer.vs) |v| {
        try writer.interface.writeAll(std.mem.sliceAsBytes(v.storage.data));
    }

    try writer.interface.flush();
}

fn loadCheckpoint(model: *Model, optimizer: *AdamW, io: std.Io) !usize {
    try ag.loadWeights(model, io, WEIGHTS_PATH);

    var buf: [1024 * 16]u8 = undefined;
    const file = try std.Io.Dir.cwd().openFile(io, CHECKPOINT_OPT, .{ .mode = .read_only });
    defer file.close(io);
    var reader = file.reader(io, &buf);

    var tmp: [8]u8 = undefined;
    try reader.interface.readSliceAll(&tmp);
    const step = std.mem.readInt(u64, &tmp, .little);
    try reader.interface.readSliceAll(&tmp);
    optimizer.t = @intCast(std.mem.readInt(u64, &tmp, .little));

    for (optimizer.ms) |*m| {
        try reader.interface.readSliceAll(std.mem.sliceAsBytes(m.storage.data));
    }
    for (optimizer.vs) |*v| {
        try reader.interface.readSliceAll(std.mem.sliceAsBytes(v.storage.data));
    }

    return @intCast(step);
}

fn runTrain(allocator: std.mem.Allocator, io: std.Io) !void {
    const corpus_file = try std.Io.Dir.cwd().openFile(io, ENCODED_BIN, .{ .mode = .read_only });
    defer corpus_file.close(io);
    const file_size = (try corpus_file.stat(io)).size;
    const mapped = try std.posix.mmap(null, @intCast(file_size), .{ .READ = true }, .{ .TYPE = .PRIVATE }, corpus_file.handle, 0);
    defer std.posix.munmap(mapped);
    const corpus = std.mem.bytesAsSlice(u16, mapped[8..]);

    var model = try Model.init(allocator, VOCAB_SIZE, D_MODEL, N_HEADS, D_FFN, N_LAYERS, SEQ_LEN, 42);
    defer model.deinit(allocator);

    var param_list: std.ArrayList(*Tensor) = .empty;
    defer param_list.deinit(allocator);
    var grad_list: std.ArrayList(*?Tensor) = .empty;
    defer grad_list.deinit(allocator);
    try model.collectParams(allocator, &param_list, &grad_list);

    var optimizer = try AdamW.init(allocator, param_list.items, grad_list.items, LR, BETA1, BETA2, EPS, WEIGHT_DECAY);
    defer optimizer.deinit(allocator);

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    var step: usize = loadCheckpoint(&model, &optimizer, io) catch 0;
    const start_step = step;
    const start_time = std.Io.Timestamp.now(io, .awake);

    while (step < N_STEPS) : (step += 1) {
        model.zeroGrads(allocator);
        var batch_loss: f32 = 0;

        for (0..BATCH_SIZE) |_| {
            var arena = std.heap.ArenaAllocator.init(allocator);
            defer arena.deinit();
            const ga = arena.allocator();

            const max_start = corpus.len - SEQ_LEN - 1;
            const start = rng.uintLessThan(usize, max_start);
            const input_ids = corpus[start .. start + SEQ_LEN];
            const target_ids = corpus[start + 1 .. start + SEQ_LEN + 1];

            const mask_tensor = try ag.causalMask(ga, SEQ_LEN);
            const mask = try Val.makeLeaf(ga, &mask_tensor);

            const result = try model.forward(ga, input_ids, mask);
            const loss = try Val.crossEntropy(ga, result.out, target_ids);
            try Val.backward(ga, loss);
            try model.accumulateGrads(allocator, &result);

            batch_loss += loss.value.storage.data[0];
        }

        for (grad_list.items) |grad_slot| {
            if (grad_slot.*) |*grad| {
                for (grad.storage.data) |*v| {
                    v.* /= @as(f32, @floatFromInt(BATCH_SIZE));
                }
            }
        }

        clipGrads(grad_list.items, GRAD_CLIP);

        optimizer.lr = getLR(step, WARMUP_STEPS, N_STEPS, LR, MIN_LR);
        try optimizer.step();

        if ((step + 1) % CHECKPOINT_EVERY == 0) try saveCheckpoint(&model, &optimizer, io, step + 1);

        if (step % LOG_EVERY == 0) {
            const now_ts = std.Io.Timestamp.now(io, .awake);
            const elapsed_ns = start_time.durationTo(now_ts).nanoseconds;
            const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / 1e9;
            const completed = step - start_step + 1;
            const sps = @as(f64, @floatFromInt(completed)) / @max(elapsed_s, 1e-9);
            const remaining = N_STEPS - step - 1;
            const eta_s = if (sps > 0) @as(f64, @floatFromInt(remaining)) / sps else 0;
            const eta_h: u64 = @intFromFloat(eta_s / 3600.0);
            const eta_m: u64 = @intFromFloat(@mod(eta_s, 3600.0) / 60.0);
            const eta_sec: u64 = @intFromFloat(@mod(eta_s, 60.0));
            std.debug.print("step {d}/{d}: loss {d:.4} | lr {e:.2} | {d:.2} steps/s | ETA {d:0>2}h{d:0>2}m{d:0>2}s\n", .{
                step,
                N_STEPS,
                batch_loss / @as(f32, @floatFromInt(BATCH_SIZE)),
                optimizer.lr,
                sps,
                eta_h,
                eta_m,
                eta_sec,
            });
        }
    }

    std.debug.print("Training complete. Saving weights to {s}...\n", .{WEIGHTS_PATH});
    try saveCheckpoint(&model, &optimizer, io, step);
    try ag.saveWeights(&model, io, WEIGHTS_PATH);
}

fn runSave(allocator: std.mem.Allocator, io: std.Io) !void {
    var model = try Model.init(allocator, VOCAB_SIZE, D_MODEL, N_HEADS, D_FFN, N_LAYERS, SEQ_LEN, 42);
    defer model.deinit(allocator);
    try ag.saveWeights(&model, io, WEIGHTS_PATH);
    std.debug.print("Model weights saved to {s}\n", .{WEIGHTS_PATH});
}

fn runLoad(allocator: std.mem.Allocator, io: std.Io) !void {
    var model = try Model.init(allocator, VOCAB_SIZE, D_MODEL, N_HEADS, D_FFN, N_LAYERS, SEQ_LEN, 42);
    defer model.deinit(allocator);
    try ag.loadWeights(&model, io, WEIGHTS_PATH);
    std.debug.print("Model weights loaded from {s}\n", .{WEIGHTS_PATH});
}

fn runSample(allocator: std.mem.Allocator, io: std.Io, opt_prompt: ?[]const u8) !void {
    const merges = try Tokenizer.loadMerges(allocator, io, SAVED_BIN);
    defer allocator.free(merges);
    var words = try Tokenizer.Words.fromMerges(allocator, merges);
    defer words.deinit();
    var encoder = try Tokenizer.Encoder.init(allocator, merges);
    defer encoder.deinit();

    var model = try Model.init(allocator, VOCAB_SIZE, D_MODEL, N_HEADS, D_FFN, N_LAYERS, SEQ_LEN, 42);
    defer model.deinit(allocator);
    try ag.loadWeights(&model, io, WEIGHTS_PATH);

    const prompt = opt_prompt orelse "<|user|>";
    const initial_tokens = try encoder.encode(allocator, prompt);
    defer allocator.free(initial_tokens);

    var tokens: std.ArrayList(u16) = .empty;
    defer tokens.deinit(allocator);
    try tokens.appendSlice(allocator, initial_tokens);

    std.debug.print("{s}", .{prompt});

    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    var i: usize = 0;
    while (i < 200) : (i += 1) {
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        const ga = arena.allocator();

        const cur_len = tokens.items.len;
        const start = if (cur_len > SEQ_LEN) cur_len - SEQ_LEN else 0;
        const context = tokens.items[start..];

        const mask_tensor = try ag.causalMask(ga, context.len);
        const mask = try Val.makeLeaf(ga, &mask_tensor);

        const result = try model.forward(ga, context, mask);
        const logits = result.out;

        // Take last row of logits
        const last_logits = try ga.alloc(f32, VOCAB_SIZE);
        const vocab_start = (context.len - 1) * VOCAB_SIZE;
        @memcpy(last_logits, logits.value.storage.data[vocab_start .. vocab_start + VOCAB_SIZE]);

        // Temperature 0.8
        for (last_logits) |*l| l.* /= 0.8;

        // Top-K = 40
        try topK(ga, last_logits, 40);

        // Softmax
        var max: f32 = last_logits[0];
        for (last_logits) |l| if (l > max) {
            max = l;
        };
        var sum: f32 = 0;
        for (last_logits) |*l| {
            l.* = @exp(l.* - max);
            sum += l.*;
        }
        for (last_logits) |*l| l.* /= sum;

        // Sample
        const r = rng.float(f32);
        var acc: f32 = 0;
        var sampled_token: u16 = 0;
        for (last_logits, 0..) |p, idx| {
            acc += p;
            if (r < acc) {
                sampled_token = @intCast(idx);
                break;
            }
        } else {
            sampled_token = @intCast(last_logits.len - 1);
        }

        try tokens.append(allocator, sampled_token);
        std.debug.print("{s}", .{words.data[sampled_token]});
    }
    std.debug.print("\n", .{});
}

fn topK(allocator: std.mem.Allocator, logits: []f32, k: usize) !void {
    if (k >= logits.len) return;
    const indices = try allocator.alloc(usize, logits.len);
    for (0..logits.len) |idx| indices[idx] = idx;

    const Context = struct {
        l: []f32,
        pub fn lessThan(ctx: @This(), a_idx: usize, b_idx: usize) bool {
            return ctx.l[a_idx] > ctx.l[b_idx];
        }
    };
    std.mem.sort(usize, indices, Context{ .l = logits }, Context.lessThan);

    const threshold = logits[indices[k - 1]];
    for (logits) |*l| {
        if (l.* < threshold) {
            l.* = -std.math.inf(f32);
        }
    }
}

fn clipGrads(grads: []const *?Tensor, max_norm: f32) void {
    var sum_sq: f32 = 0;
    for (grads) |grad_slot| {
        if (grad_slot.*) |*grad| {
            for (grad.storage.data) |v| {
                sum_sq += v * v;
            }
        }
    }

    const global_norm = @sqrt(sum_sq);
    if (global_norm > max_norm) {
        const scale = max_norm / global_norm;
        for (grads) |grad_slot| {
            if (grad_slot.*) |*grad| {
                for (grad.storage.data) |*v| {
                    v.* *= scale;
                }
            }
        }
    }
}

fn getLR(step: usize, warmup_steps: usize, total_steps: usize, max_lr: f32, min_lr: f32) f32 {
    const f_step: f32 = @floatFromInt(step);
    const f_warmup: f32 = @floatFromInt(warmup_steps);
    const f_total: f32 = @floatFromInt(total_steps);

    if (step < warmup_steps) {
        return max_lr * (f_step / f_warmup);
    } else {
        const progress = (f_step - f_warmup) / (f_total - f_warmup);
        return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + @cos(std.math.pi * progress));
    }
}
