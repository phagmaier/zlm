const std = @import("std");
pub const PairKey = struct { l: u16, r: u16 };
const PairCount = std.AutoHashMap(PairKey, usize);
const PairPositions = std.AutoHashMap(PairKey, std.ArrayList(usize));
const Mergedic = std.AutoHashMap(PairKey, u16);
const NUMBYTES: u16 = 256;
const SPECIAL_START: u16 = 0;
const BYTE_START: u16 = 4;
const MERGE_START: u16 = BYTE_START + NUMBYTES;

pub const SPECIAL_TOKENS = [_][]const u8{
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|endoftext|>",
};

const NIL: usize = std.math.maxInt(usize);
const FILENAME = "/home/phagmaier/Desktop/Code/Ai/Data/input.txt";
const BINNAME = "/home/phagmaier/Desktop/Code/Ai/Data/saved.bin";
const MAGIC: [4]u8 = .{ 'B', 'P', 'E', 0 };
const VERSION: u32 = 1;

pub const Words = struct {
    data: [][]const u8,
    cap: usize,
    index: usize,
    arena: std.heap.ArenaAllocator,

    pub fn init(allocator: std.mem.Allocator, size: usize) !Words {
        if (size < MERGE_START) return error.VocabTooSmall;
        var self = Words{
            .data = undefined,
            .cap = size,
            .index = 0,
            .arena = std.heap.ArenaAllocator.init(allocator),
        };
        errdefer self.arena.deinit();
        const a = self.arena.allocator();
        self.data = try a.alloc([]const u8, size);

        // Special tokens 0-3
        for (SPECIAL_TOKENS, 0..) |str, i| {
            self.data[i] = try a.dupe(u8, str);
        }

        // Bytes 4-259
        for (0..NUMBYTES) |i| {
            const str = try a.alloc(u8, 1);
            str[0] = @as(u8, @intCast(i));
            self.data[BYTE_START + i] = str;
        }
        self.index = MERGE_START;
        return self;
    }

    pub fn fromMerges(allocator: std.mem.Allocator, merges: []const PairKey) !Words {
        var self = try Words.init(allocator, MERGE_START + merges.len);
        errdefer self.deinit();
        for (merges) |pair| {
            try self.appendPairs(pair);
        }
        return self;
    }

    pub fn deinit(self: *Words) void {
        self.arena.deinit();
    }

    pub fn appendPairs(self: *Words, pair: PairKey) !void {
        if (self.index >= self.cap) return error.ExceededCap;
        const a = self.arena.allocator();
        const left = self.data[pair.l];
        const right = self.data[pair.r];
        const size = left.len + right.len;
        const str = try a.alloc(u8, size);
        @memcpy(str[0..left.len], left);
        @memcpy(str[left.len..], right);
        self.data[self.index] = str;
        self.index += 1;
    }

    pub fn print(self: *const Words) void {
        for (self.data[0..self.index]) |str| {
            std.debug.print("{s}\n", .{str});
        }
    }
};

const TokenList = struct {
    data: []u16,
    prev: []usize,
    next: []usize,
    dead: []bool,
    head: usize,

    fn initFromBytes(allocator: std.mem.Allocator, src: []const u8) !TokenList {
        const n = src.len;
        var self = TokenList{
            .data = try allocator.alloc(u16, n),
            .prev = try allocator.alloc(usize, n),
            .next = try allocator.alloc(usize, n),
            .dead = try allocator.alloc(bool, n),
            .head = if (n > 0) 0 else NIL,
        };
        for (src, 0..) |c, i| {
            self.data[i] = @as(u16, c) + BYTE_START;
            self.prev[i] = if (i == 0) NIL else i - 1;
            self.next[i] = if (i + 1 == n) NIL else i + 1;
            self.dead[i] = false;
        }
        return self;
    }

    fn deinit(self: *TokenList, allocator: std.mem.Allocator) void {
        allocator.free(self.data);
        allocator.free(self.prev);
        allocator.free(self.next);
        allocator.free(self.dead);
    }

    fn collect(self: *const TokenList, allocator: std.mem.Allocator) ![]u16 {
        var count: usize = 0;
        var cur = self.head;
        while (cur != NIL) : (cur = self.next[cur]) count += 1;

        const out = try allocator.alloc(u16, count);
        var i: usize = 0;
        cur = self.head;
        while (cur != NIL) : (cur = self.next[cur]) {
            out[i] = self.data[cur];
            i += 1;
        }
        return out;
    }
};

pub fn readFile(allocator: std.mem.Allocator, io: std.Io, fileName: []const u8) ![]const u8 {
    var buf: [4096]u8 = undefined;
    const file = try std.Io.Dir.cwd().openFile(io, fileName, .{ .mode = .read_only });
    defer file.close(io);
    const size = (try file.stat(io)).size;
    var reader = file.reader(io, &buf);
    return reader.interface.readAlloc(allocator, size);
}

fn deinitPositions(allocator: std.mem.Allocator, positions: *PairPositions) void {
    var it = positions.iterator();
    while (it.next()) |entry| {
        entry.value_ptr.deinit(allocator);
    }
    positions.deinit();
}

fn addPosition(allocator: std.mem.Allocator, positions: *PairPositions, key: PairKey, pos: usize) !void {
    const r = try positions.getOrPut(key);
    if (!r.found_existing) {
        r.value_ptr.* = try std.ArrayList(usize).initCapacity(allocator, 0);
    }
    try r.value_ptr.*.append(allocator, pos);
}

fn incCount(counts: *PairCount, key: PairKey) !void {
    const r = try counts.getOrPut(key);
    if (r.found_existing) {
        r.value_ptr.* += 1;
    } else {
        r.value_ptr.* = 1;
    }
}

fn decCount(counts: *PairCount, key: PairKey) void {
    if (counts.getPtr(key)) |c| {
        if (c.* > 0) c.* -= 1;
    }
}

fn findMaxPair(counts: *const PairCount) ?PairKey {
    var best: usize = 0;
    var pair: PairKey = .{ .l = 0, .r = 0 };
    var found = false;
    var it = counts.iterator();
    while (it.next()) |entry| {
        if (entry.value_ptr.* > best) {
            best = entry.value_ptr.*;
            pair = entry.key_ptr.*;
            found = true;
        }
    }
    return if (found) pair else null;
}

pub const Encoder = struct {
    map: Mergedic,

    const HeapItem = struct {
        new_id: u16,
        pos: usize,
        l: u16,
        r: u16,
    };

    fn heapOrder(_: void, a: HeapItem, b: HeapItem) std.math.Order {
        return std.math.order(a.new_id, b.new_id);
    }

    const Heap = std.PriorityQueue(HeapItem, void, heapOrder);

    pub fn init(allocator: std.mem.Allocator, merges: []const PairKey) !Encoder {
        var map = Mergedic.init(allocator);
        errdefer map.deinit();
        for (merges, 0..) |pair, i| {
            try map.put(pair, @as(u16, @intCast(MERGE_START + i)));
        }
        return .{ .map = map };
    }

    pub fn deinit(self: *Encoder) void {
        self.map.deinit();
    }

    pub fn encode(self: *const Encoder, allocator: std.mem.Allocator, text: []const u8) ![]u16 {
        if (text.len == 0) return allocator.alloc(u16, 0);

        var tokens: std.ArrayList(u16) = .empty;
        defer tokens.deinit(allocator);

        var last_idx: usize = 0;
        var i: usize = 0;
        while (i < text.len) {
            var found_special: ?usize = null;
            for (SPECIAL_TOKENS, 0..) |special, id| {
                if (std.mem.startsWith(u8, text[i..], special)) {
                    found_special = id;
                    break;
                }
            }

            if (found_special) |id| {
                if (i > last_idx) {
                    const sub_tokens = try self.encodeNormal(allocator, text[last_idx..i]);
                    defer allocator.free(sub_tokens);
                    try tokens.appendSlice(allocator, sub_tokens);
                }
                try tokens.append(allocator, @as(u16, @intCast(id)));
                i += SPECIAL_TOKENS[id].len;
                last_idx = i;
            } else {
                i += 1;
            }
        }

        if (last_idx < text.len) {
            const sub_tokens = try self.encodeNormal(allocator, text[last_idx..]);
            defer allocator.free(sub_tokens);
            try tokens.appendSlice(allocator, sub_tokens);
        }

        return tokens.toOwnedSlice(allocator);
    }

    fn encodeNormal(self: *const Encoder, allocator: std.mem.Allocator, text: []const u8) ![]u16 {
        if (text.len == 0) return allocator.alloc(u16, 0);

        var ll = try TokenList.initFromBytes(allocator, text);
        defer ll.deinit(allocator);

        var heap = Heap.initContext({});
        defer heap.deinit(allocator);

        if (text.len >= 2) {
            var cur = ll.head;
            while (cur != NIL) {
                const next = ll.next[cur];
                if (next == NIL) break;

                const key = PairKey{ .l = ll.data[cur], .r = ll.data[next] };
                if (self.map.get(key)) |id| {
                    try heap.push(allocator, .{ .new_id = id, .pos = cur, .l = key.l, .r = key.r });
                }
                cur = next;
            }
        }

        while (heap.pop()) |item| {
            if (ll.dead[item.pos]) continue;
            const r = ll.next[item.pos];
            if (r == NIL or ll.dead[r]) continue;
            if (ll.data[item.pos] != item.l or ll.data[r] != item.r) continue;

            const left_pos = ll.prev[item.pos];
            const far_pos = ll.next[r];

            ll.data[item.pos] = item.new_id;
            ll.dead[r] = true;
            ll.next[item.pos] = far_pos;
            if (far_pos != NIL) ll.prev[far_pos] = item.pos;

            if (left_pos != NIL) {
                const key = PairKey{ .l = ll.data[left_pos], .r = item.new_id };
                if (self.map.get(key)) |id| {
                    try heap.push(allocator, .{ .new_id = id, .pos = left_pos, .l = key.l, .r = key.r });
                }
            }
            if (far_pos != NIL) {
                const key = PairKey{ .l = item.new_id, .r = ll.data[far_pos] };
                if (self.map.get(key)) |id| {
                    try heap.push(allocator, .{ .new_id = id, .pos = item.pos, .l = key.l, .r = key.r });
                }
            }
        }

        return ll.collect(allocator);
    }
};

pub fn tokenize(allocator: std.mem.Allocator, io: std.Io, fileName: []const u8, vocabSize: usize, max_bytes: ?usize) !struct { Words, std.ArrayList(PairKey) } {
    const file = try std.Io.Dir.cwd().openFile(io, fileName, .{ .mode = .read_only });
    defer file.close(io);
    const total_size = (try file.stat(io)).size;
    const read_size = if (max_bytes) |mb| @min(total_size, mb) else total_size;

    var buf: [4096]u8 = undefined;
    var reader = file.reader(io, &buf);
    const fileData = try reader.interface.readAlloc(allocator, read_size);
    defer allocator.free(fileData);

    var words = try Words.init(allocator, vocabSize);
    errdefer words.deinit();

    var merges = try std.ArrayList(PairKey).initCapacity(allocator, vocabSize - MERGE_START);
    errdefer merges.deinit(allocator);

    if (fileData.len < 2 or vocabSize <= MERGE_START) {
        return .{ words, merges };
    }

    var ll = try TokenList.initFromBytes(allocator, fileData);
    defer ll.deinit(allocator);

    var counts = PairCount.init(allocator);
    defer counts.deinit();

    var positions = PairPositions.init(allocator);
    defer deinitPositions(allocator, &positions);

    var i: usize = 0;
    while (i + 1 < fileData.len) : (i += 1) {
        const key = PairKey{ .l = ll.data[i], .r = ll.data[i + 1] };
        try incCount(&counts, key);
        try addPosition(allocator, &positions, key, i);
    }

    while (words.index < words.cap) {
        const best_pair = findMaxPair(&counts) orelse break;
        const new_id = @as(u16, @intCast(words.index));
        try words.appendPairs(best_pair);
        try merges.append(allocator, best_pair);

        const entry = positions.fetchRemove(best_pair) orelse {
            _ = counts.remove(best_pair);
            continue;
        };
        var pos_arr = entry.value;
        defer pos_arr.deinit(allocator);

        for (pos_arr.items) |pos| {
            if (ll.dead[pos]) continue;
            const right_idx = ll.next[pos];
            if (right_idx == NIL or ll.dead[right_idx]) continue;
            if (ll.data[pos] != best_pair.l or ll.data[right_idx] != best_pair.r) continue;

            const left_pos = ll.prev[pos];
            const far_pos = ll.next[right_idx];

            decCount(&counts, best_pair);
            if (left_pos != NIL) {
                const old_left = PairKey{ .l = ll.data[left_pos], .r = best_pair.l };
                decCount(&counts, old_left);
            }
            if (far_pos != NIL) {
                const old_right = PairKey{ .l = best_pair.r, .r = ll.data[far_pos] };
                decCount(&counts, old_right);
            }

            ll.data[pos] = new_id;
            ll.dead[right_idx] = true;
            ll.next[pos] = far_pos;
            if (far_pos != NIL) ll.prev[far_pos] = pos;

            if (left_pos != NIL) {
                const new_left = PairKey{ .l = ll.data[left_pos], .r = new_id };
                try incCount(&counts, new_left);
                try addPosition(allocator, &positions, new_left, left_pos);
            }
            if (far_pos != NIL) {
                const new_right = PairKey{ .l = new_id, .r = ll.data[far_pos] };
                try incCount(&counts, new_right);
                try addPosition(allocator, &positions, new_right, pos);
            }
        }

        _ = counts.remove(best_pair);
    }

    return .{ words, merges };
}

// Layout:
//   [0..4)   magic   "BPE\0"
//   [4..8)   version u32 little-endian
//   [8..12)  count   u32 little-endian (number of merges)
//   then `count` records of: l u16 LE, r u16 LE
// Vocab is reconstructed from merges via Words.fromMerges.
pub fn encodeFile(allocator: std.mem.Allocator, io: std.Io, encoder: *const Encoder, input_path: []const u8, output_path: []const u8) !void {
    const input_file = try std.Io.Dir.cwd().openFile(io, input_path, .{ .mode = .read_only });
    defer input_file.close(io);

    const output_file = try std.Io.Dir.cwd().createFile(io, output_path, .{});
    defer output_file.close(io);

    var out_buf: [1024 * 64]u8 = undefined;
    var writer = output_file.writer(io, &out_buf);

    // Placeholder for length
    try writer.interface.writeInt(u64, 0, .little);

    const chunk_size = 64 * 1024;
    const overlap_size = 64;
    var read_buf = try allocator.alloc(u8, chunk_size + overlap_size);
    defer allocator.free(read_buf);

    var r_buf: [4096]u8 = undefined;
    var reader = input_file.reader(io, &r_buf);

    var total_tokens: u64 = 0;
    var leftover_len: usize = 0;
    var offset: u64 = 0;
    const file_size = (try input_file.stat(io)).size;

    while (true) {
        const remaining = file_size - offset;
        if (remaining == 0 and leftover_len == 0) break;

        const to_read = @min(remaining, chunk_size);
        var n: usize = 0;
        if (to_read > 0) {
            const data = try reader.interface.readAlloc(allocator, to_read);
            defer allocator.free(data);
            @memcpy(read_buf[leftover_len .. leftover_len + data.len], data);
            n = data.len;
            offset += n;
        }

        const current_total = leftover_len + n;
        if (current_total == 0) break;

        var safe_len = current_total;
        if (n > 0) {
            // Find a safe boundary: look back for '<' which might start a special token
            var i: usize = 0;
            while (i < overlap_size and i < current_total) : (i += 1) {
                const pos = current_total - 1 - i;
                if (read_buf[pos] == '<') {
                    safe_len = pos;
                    break;
                }
            }
        }

        const text_to_encode = read_buf[0..safe_len];
        if (text_to_encode.len > 0) {
            const tokens = try encoder.encode(allocator, text_to_encode);
            defer allocator.free(tokens);

            try writer.interface.writeAll(std.mem.sliceAsBytes(tokens));
            total_tokens += tokens.len;
        }

        const new_leftover_len = current_total - safe_len;
        std.mem.copyForwards(u8, read_buf[0..new_leftover_len], read_buf[safe_len..current_total]);
        leftover_len = new_leftover_len;

        if (n == 0) break;
    }

    try writer.interface.flush();

    // Overwrite length header at offset 0
    var count_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &count_buf, total_tokens, .little);
    try output_file.writePositionalAll(io, &count_buf, 0);
}

pub fn saveMerges(io: std.Io, fileName: []const u8, merges: []const PairKey) !void {
    var buf: [4096]u8 = undefined;
    const file = try std.Io.Dir.cwd().createFile(io, fileName, .{});
    defer file.close(io);
    var _writer = file.writer(io, &buf);
    const writer = &_writer.interface;

    try writer.writeAll(&MAGIC);
    try writer.writeInt(u32, VERSION, .little);
    try writer.writeInt(u32, @intCast(merges.len), .little);
    for (merges) |m| {
        try writer.writeInt(u16, m.l, .little);
        try writer.writeInt(u16, m.r, .little);
    }
    try writer.flush();
}

pub fn loadMerges(allocator: std.mem.Allocator, io: std.Io, fileName: []const u8) ![]PairKey {
    const data = try readFile(allocator, io, fileName);
    defer allocator.free(data);

    if (data.len < 12) return error.FileTooSmall;
    if (!std.mem.eql(u8, data[0..4], &MAGIC)) return error.BadMagic;
    const version = std.mem.readInt(u32, data[4..][0..4], .little);
    if (version != VERSION) return error.UnsupportedVersion;
    const count = std.mem.readInt(u32, data[8..][0..4], .little);

    const expected = 12 + @as(usize, count) * 4;
    if (data.len < expected) return error.TruncatedFile;

    const merges = try allocator.alloc(PairKey, count);
    errdefer allocator.free(merges);
    var i: usize = 0;
    while (i < count) : (i += 1) {
        const off = 12 + i * 4;
        merges[i] = .{
            .l = std.mem.readInt(u16, data[off..][0..2], .little),
            .r = std.mem.readInt(u16, data[off + 2 ..][0..2], .little),
        };
    }
    return merges;
}

pub fn saveEncoding(io: std.Io, fileName: []const u8, data: []const u16) !void {
    var buf: [4096]u8 = undefined;
    const file = try std.Io.Dir.cwd().createFile(io, fileName, .{});
    defer file.close(io);
    var _writer = file.writer(io, &buf);
    const writer = &_writer.interface;
    try writer.writeInt(u64, data.len, .little);
    try writer.writeAll(std.mem.sliceAsBytes(data));
    try writer.flush();
}

pub fn loadEncoding(allocator: std.mem.Allocator, io: std.Io, fileName: []const u8) ![]u16 {
    const data = try readFile(allocator, io, fileName);
    defer allocator.free(data);
    if (data.len < 8) return error.FileTooSmall;
    const len = std.mem.readInt(u64, data[0..8], .little);
    const result = try allocator.alloc(u16, len);
    @memcpy(std.mem.sliceAsBytes(result), data[8..]);
    return result;
}

test "Test Run" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var words, var merges = try tokenize(allocator, io, FILENAME, 300, null);
    defer words.deinit();
    defer merges.deinit(allocator);
}

test "encode roundtrip" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var words, var merges = try tokenize(allocator, io, FILENAME, 512, null);
    defer words.deinit();
    defer merges.deinit(allocator);

    var encoder = try Encoder.init(allocator, merges.items);
    defer encoder.deinit();

    const sample = "Hello, world! This is a tokenizer test.";
    const tokens = try encoder.encode(allocator, sample);
    defer allocator.free(tokens);

    var total: usize = 0;
    for (tokens) |t| total += words.data[t].len;

    const decoded = try allocator.alloc(u8, total);
    defer allocator.free(decoded);
    var off: usize = 0;
    for (tokens) |t| {
        const w = words.data[t];
        @memcpy(decoded[off..][0..w.len], w);
        off += w.len;
    }

    try std.testing.expectEqualStrings(sample, decoded);
}

test "SAVE FILE" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var words, var merges = try tokenize(allocator, io, FILENAME, 512, null);
    defer words.deinit();
    defer merges.deinit(allocator);
    try saveMerges(io, BINNAME, merges.items);
}

test "save/load encoding round trip" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var words, var merges = try tokenize(allocator, io, FILENAME, 512, null);
    defer words.deinit();
    defer merges.deinit(allocator);

    var encoder = try Encoder.init(allocator, merges.items);
    defer encoder.deinit();

    const sample = "Hello, world! This is a tokenizer test.";
    const tokens = try encoder.encode(allocator, sample);
    defer allocator.free(tokens);

    const tmp_path = "/tmp/bpe_encoding_roundtrip.bin";
    try saveEncoding(io, tmp_path, tokens);

    const loaded = try loadEncoding(allocator, io, tmp_path);
    defer allocator.free(loaded);

    try std.testing.expectEqual(tokens.len, loaded.len);
    for (tokens, loaded) |a, b| try std.testing.expectEqual(a, b);
}

test "Read Save" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var words, var merges = try tokenize(allocator, io, FILENAME, 512, null);
    defer words.deinit();
    defer merges.deinit(allocator);
    try saveMerges(io, BINNAME, merges.items);

    const loaded = try loadMerges(allocator, io, BINNAME);
    defer allocator.free(loaded);

    try std.testing.expectEqual(merges.items.len, loaded.len);
    for (merges.items, loaded) |a, b| {
        try std.testing.expectEqual(a.l, b.l);
        try std.testing.expectEqual(a.r, b.r);
    }

    var rebuilt = try Words.fromMerges(allocator, loaded);
    defer rebuilt.deinit();
    try std.testing.expectEqual(words.index, rebuilt.index);
    for (0..words.index) |idx| {
        try std.testing.expectEqualStrings(words.data[idx], rebuilt.data[idx]);
    }

    var encoder = try Encoder.init(allocator, loaded);
    defer encoder.deinit();
    const sample = "Hello, world! This is a tokenizer test.";
    const tokens = try encoder.encode(allocator, sample);
    defer allocator.free(tokens);

    var total: usize = 0;
    for (tokens) |t| total += rebuilt.data[t].len;
    const decoded = try allocator.alloc(u8, total);
    defer allocator.free(decoded);
    var off: usize = 0;
    for (tokens) |t| {
        const w = rebuilt.data[t];
        @memcpy(decoded[off..][0..w.len], w);
        off += w.len;
    }
    try std.testing.expectEqualStrings(sample, decoded);
}

test "special tokens" {
    const allocator = std.testing.allocator;
    const merges = [_]PairKey{};
    var encoder = try Encoder.init(allocator, &merges);
    defer encoder.deinit();

    const sample = "<|system|>Hello<|user|>Hi<|assistant|>...<|endoftext|>";
    const tokens = try encoder.encode(allocator, sample);
    defer allocator.free(tokens);

    // <|system|> (0), H(72+4=76), e(101+4=105), l(108+4=112), l(112), o(111+4=115), <|user|> (1), H(76), i(105+4=109), <|assistant|> (2), .(46+4=50), .(50), .(50), <|endoftext|> (3)
    try std.testing.expectEqual(@as(usize, 14), tokens.len);
    try std.testing.expectEqual(@as(u16, 0), tokens[0]);
    try std.testing.expectEqual(@as(u16, 76), tokens[1]);
    try std.testing.expectEqual(@as(u16, 1), tokens[6]);
    try std.testing.expectEqual(@as(u16, 2), tokens[9]);
    try std.testing.expectEqual(@as(u16, 3), tokens[13]);
}
