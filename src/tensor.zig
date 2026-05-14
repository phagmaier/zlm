const std = @import("std");
const cblas = @import("cblas");

const TensorErrs = error{
    ZeroShape,
    AccessDenied,
    SizeMisMatch,
    IncompatibleShapes,
    InvalidPermutation,
    InvalidSlice,
    UnsupportedRank,
    DimensionOverflow,
    NotContiguous,
} || error{OutOfMemory};

pub const Tensor = struct {
    storage: *Storage,
    offset: isize,
    shape: []usize,
    strides: []isize,

    const BinaryOp = enum {
        add,
        mul,
    };

    pub const Slice = struct {
        start: ?isize = null,
        end: ?isize = null,
        step: isize = 1,
    };

    const Storage = struct {
        data: []f32,
        allocator: std.mem.Allocator,
        ref_count: usize,

        fn create(allocator: std.mem.Allocator, size: usize) !*Storage {
            const storage = try allocator.create(Storage);
            errdefer allocator.destroy(storage);

            const data = try allocator.alloc(f32, size);
            errdefer allocator.free(data);

            storage.* = .{
                .data = data,
                .allocator = allocator,
                .ref_count = 1,
            };
            return storage;
        }

        fn retain(self: *Storage) void {
            self.ref_count += 1;
        }

        fn release(self: *Storage) void {
            std.debug.assert(self.ref_count > 0);
            self.ref_count -= 1;
            if (self.ref_count == 0) {
                self.allocator.free(self.data);
                self.allocator.destroy(self);
            }
        }
    };

    fn numelFromShape(shape: []const usize) TensorErrs!usize {
        if (shape.len == 0) return 1;

        var size: usize = 1;
        for (shape) |dim| {
            if (dim == 0) return TensorErrs.ZeroShape;
            size *= dim;
        }
        return size;
    }

    fn fillContiguousStrides(shape: []const usize, strides: []isize) !void {
        std.debug.assert(shape.len == strides.len);
        var stride: usize = 1;
        var i = shape.len;
        while (i > 0) {
            i -= 1;
            strides[i] = try checkedCastToIsize(stride);
            stride *= shape[i];
        }
    }

    fn broadcastShape(
        allocator: std.mem.Allocator,
        left_shape: []const usize,
        right_shape: []const usize,
    ) ![]usize {
        const out_rank = @max(left_shape.len, right_shape.len);
        const out_shape = try allocator.alloc(usize, out_rank);
        errdefer allocator.free(out_shape);

        for (0..out_rank) |i| {
            const left_has_dim = i < left_shape.len;
            const right_has_dim = i < right_shape.len;
            const left_dim = if (left_has_dim) left_shape[left_shape.len - 1 - i] else 1;
            const right_dim = if (right_has_dim) right_shape[right_shape.len - 1 - i] else 1;

            if (left_dim != right_dim and left_dim != 1 and right_dim != 1) {
                return TensorErrs.IncompatibleShapes;
            }

            out_shape[out_rank - 1 - i] = @max(left_dim, right_dim);
        }

        return out_shape;
    }

    fn broadcastOffset(
        out_index: usize,
        out_shape: []const usize,
        out_strides: []const usize,
        in_shape: []const usize,
        in_strides: []const isize,
    ) !isize {
        const axis_offset = out_shape.len - in_shape.len;
        var remaining = out_index;
        var input_offset: isize = 0;

        for (0..out_shape.len) |axis| {
            const coord = remaining / out_strides[axis];
            remaining %= out_strides[axis];

            if (axis < axis_offset) continue;

            const in_axis = axis - axis_offset;
            const source_coord = if (in_shape[in_axis] == 1) 0 else coord;
            input_offset += try checkedMulCoord(source_coord, in_strides[in_axis]);
        }

        return input_offset;
    }

    fn validatePermutation(rank: usize, permutation: []const usize) !void {
        if (permutation.len != rank) return TensorErrs.InvalidPermutation;
        for (permutation, 0..) |axis, i| {
            if (axis >= rank) return TensorErrs.InvalidPermutation;
            for (permutation[(i + 1)..]) |other_axis| {
                if (axis == other_axis) return TensorErrs.InvalidPermutation;
            }
        }
    }

    fn checkedCastToCInt(value: usize) TensorErrs!c_int {
        return std.math.cast(c_int, value) orelse TensorErrs.DimensionOverflow;
    }

    fn checkedCastToIsize(value: usize) TensorErrs!isize {
        return std.math.cast(isize, value) orelse TensorErrs.DimensionOverflow;
    }

    fn checkedMulCoord(coord: usize, stride: isize) TensorErrs!isize {
        const signed_coord = try checkedCastToIsize(coord);
        return std.math.mul(isize, signed_coord, stride) catch TensorErrs.DimensionOverflow;
    }

    fn checkedOffsetToIndex(offset: isize) TensorErrs!usize {
        if (offset < 0) return TensorErrs.AccessDenied;
        return std.math.cast(usize, offset) orelse TensorErrs.DimensionOverflow;
    }

    fn checkedMulIsize(left: isize, right: isize) TensorErrs!isize {
        return std.math.mul(isize, left, right) catch TensorErrs.DimensionOverflow;
    }

    fn normalizeAxisIndex(dim: usize, index: isize) TensorErrs!usize {
        const signed_dim = try checkedCastToIsize(dim);
        var normalized = index;
        if (normalized < 0) normalized += signed_dim;
        if (normalized < 0 or normalized >= signed_dim) return TensorErrs.AccessDenied;
        return checkedOffsetToIndex(normalized);
    }

    fn sliceLength(start: isize, end: isize, step: isize) usize {
        if (step > 0) {
            if (start >= end) return 0;
            return @intCast(@divTrunc((end - start) + (step - 1), step));
        }

        const abs_step = -step;
        if (start <= end) return 0;
        return @intCast(@divTrunc((start - end) + (abs_step - 1), abs_step));
    }

    fn normalizeSliceForDim(dim: usize, spec: Slice) TensorErrs!struct {
        start: isize,
        len: usize,
        step: isize,
    } {
        if (spec.step == 0) return TensorErrs.InvalidSlice;

        const signed_dim = try checkedCastToIsize(dim);
        var start = spec.start orelse if (spec.step > 0) 0 else signed_dim - 1;
        var end = spec.end orelse if (spec.step > 0) signed_dim else -1;

        if (spec.step > 0) {
            if (start < 0) start += signed_dim;
            if (end < 0) end += signed_dim;

            if (start < 0) start = 0;
            if (start > signed_dim) start = signed_dim;
            if (end < 0) end = 0;
            if (end > signed_dim) end = signed_dim;
        } else {
            if (spec.start != null and start < 0) start += signed_dim;
            if (spec.end != null and end < 0) end += signed_dim;

            if (start < -1) start = -1;
            if (start >= signed_dim) start = signed_dim - 1;
            if (end < -1) end = -1;
            if (end >= signed_dim) end = signed_dim - 1;
        }

        const len = sliceLength(start, end, spec.step);
        if (len == 0) return TensorErrs.InvalidSlice;

        return .{
            .start = start,
            .len = len,
            .step = spec.step,
        };
    }

    fn allocMetadata(
        allocator: std.mem.Allocator,
        shape: []const usize,
        strides: []const isize,
    ) !struct { shape: []usize, strides: []isize } {
        std.debug.assert(shape.len == strides.len);

        const owned_shape = try allocator.alloc(usize, shape.len);
        errdefer allocator.free(owned_shape);

        const owned_strides = try allocator.alloc(isize, strides.len);
        errdefer allocator.free(owned_strides);

        @memcpy(owned_shape, shape);
        @memcpy(owned_strides, strides);

        return .{ .shape = owned_shape, .strides = owned_strides };
    }

    fn initView(
        allocator: std.mem.Allocator,
        storage: *Storage,
        offset: isize,
        shape: []const usize,
        strides: []const isize,
    ) !Tensor {
        const meta = try allocMetadata(allocator, shape, strides);
        errdefer {
            allocator.free(meta.shape);
            allocator.free(meta.strides);
        }

        storage.retain();
        return .{
            .storage = storage,
            .offset = offset,
            .shape = meta.shape,
            .strides = meta.strides,
        };
    }

    fn storageIndexFromLinear(self: *const Tensor, linear_index: usize) !usize {
        if (linear_index >= self.numel()) return TensorErrs.AccessDenied;
        if (self.shape.len == 0) return checkedOffsetToIndex(self.offset);

        var remaining = linear_index;
        var logical_stride = self.numel();
        var offset = self.offset;

        for (0..self.shape.len) |axis| {
            logical_stride /= self.shape[axis];
            const coord = remaining / logical_stride;
            remaining %= logical_stride;
            offset += try checkedMulCoord(coord, self.strides[axis]);
        }

        const storage_index = try checkedOffsetToIndex(offset);
        if (storage_index >= self.storage.data.len) return TensorErrs.AccessDenied;
        return storage_index;
    }

    fn storageIndexFromIndices(self: *const Tensor, indices: []const usize) !usize {
        if (indices.len != self.shape.len) return TensorErrs.AccessDenied;

        var offset = self.offset;
        for (indices, 0..) |coord, axis| {
            if (coord >= self.shape[axis]) return TensorErrs.AccessDenied;
            offset += try checkedMulCoord(coord, self.strides[axis]);
        }

        const storage_index = try checkedOffsetToIndex(offset);
        if (storage_index >= self.storage.data.len) return TensorErrs.AccessDenied;
        return storage_index;
    }

    fn batchShapeForMatmul(shape: []const usize, promoted: bool) []const usize {
        return if (promoted) &.{} else shape[0 .. shape.len - 2];
    }

    fn isValidStorageRange(self: *const Tensor) bool {
        const total = self.numel();
        if (total == 0) return false;
        if (total == 1) {
            const idx = checkedOffsetToIndex(self.offset) catch return false;
            return idx < self.storage.data.len;
        }

        _ = self.storageIndexFromLinear(0) catch return false;
        _ = self.storageIndexFromLinear(total - 1) catch return false;
        return true;
    }

    pub fn init(allocator: std.mem.Allocator, shape: []const usize) !Tensor {
        const size = try numelFromShape(shape);
        const storage = try Storage.create(allocator, size);
        errdefer storage.release();

        const owned_shape = try allocator.alloc(usize, shape.len);
        errdefer allocator.free(owned_shape);

        const owned_strides = try allocator.alloc(isize, shape.len);
        errdefer allocator.free(owned_strides);

        @memcpy(owned_shape, shape);
        try fillContiguousStrides(owned_shape, owned_strides);

        return .{
            .storage = storage,
            .offset = 0,
            .shape = owned_shape,
            .strides = owned_strides,
        };
    }

    pub fn deinit(self: *Tensor, allocator: std.mem.Allocator) void {
        allocator.free(self.shape);
        allocator.free(self.strides);
        self.storage.release();
    }

    pub fn numel(self: *const Tensor) usize {
        return numelFromShape(self.shape) catch unreachable;
    }

    pub fn isContiguous(self: *const Tensor) bool {
        var expected: isize = 1;
        var i = self.shape.len;
        while (i > 0) {
            i -= 1;
            if (self.shape[i] != 1 and self.strides[i] != expected) return false;
            expected *= @as(isize, @intCast(self.shape[i]));
        }
        return self.isValidStorageRange();
    }

    pub fn getFlat(self: *const Tensor, linear_index: usize) !f32 {
        const storage_index = try self.storageIndexFromLinear(linear_index);
        return self.storage.data[storage_index];
    }

    /// If the tensor is contiguous in memory, return the backing slice for its
    /// logical elements (length == numel). Otherwise return null. Callers should
    /// use this to bypass strided indexing in hot loops.
    pub fn contiguousSlice(self: *const Tensor) ?[]const f32 {
        if (!self.isContiguous()) return null;
        const base = std.math.cast(usize, self.offset) orelse return null;
        const n = self.numel();
        if (base + n > self.storage.data.len) return null;
        return self.storage.data[base .. base + n];
    }

    pub fn contiguousSliceMut(self: *Tensor) ?[]f32 {
        if (!self.isContiguous()) return null;
        const base = std.math.cast(usize, self.offset) orelse return null;
        const n = self.numel();
        if (base + n > self.storage.data.len) return null;
        return self.storage.data[base .. base + n];
    }

    pub fn get(self: *const Tensor, indices: []const usize) !f32 {
        const storage_index = try self.storageIndexFromIndices(indices);
        return self.storage.data[storage_index];
    }

    pub fn setFlat(self: *Tensor, linear_index: usize, value: f32) !void {
        const storage_index = try self.storageIndexFromLinear(linear_index);
        self.storage.data[storage_index] = value;
    }

    pub fn set(self: *Tensor, indices: []const usize, value: f32) !void {
        const storage_index = try self.storageIndexFromIndices(indices);
        self.storage.data[storage_index] = value;
    }

    pub fn toOwnedSlice(self: *const Tensor, allocator: std.mem.Allocator) ![]f32 {
        const n = self.numel();
        const out = try allocator.alloc(f32, n);
        errdefer allocator.free(out);

        if (self.contiguousSlice()) |src| {
            @memcpy(out, src);
            return out;
        }

        for (0..n) |i| {
            out[i] = try self.getFlat(i);
        }

        return out;
    }

    pub fn clone(self: *const Tensor, allocator: std.mem.Allocator) !Tensor {
        const flat = try self.toOwnedSlice(allocator);
        defer allocator.free(flat);
        return Tensor.fromSlice(allocator, self.shape, flat);
    }

    pub fn contiguous(self: *const Tensor, allocator: std.mem.Allocator) !Tensor {
        return self.clone(allocator);
    }

    pub fn zeroes(allocator: std.mem.Allocator, shape: []const usize) !Tensor {
        const self = try init(allocator, shape);
        @memset(self.storage.data, 0.0);
        return self;
    }

    pub fn fromSlice(allocator: std.mem.Allocator, shape: []const usize, vals: []const f32) !Tensor {
        var self = try init(allocator, shape);
        errdefer self.deinit(allocator);

        if (self.numel() != vals.len) return TensorErrs.SizeMisMatch;
        @memcpy(self.storage.data, vals);
        return self;
    }

    pub inline fn shapeMatch(l: *const Tensor, r: *const Tensor) bool {
        return std.mem.eql(usize, l.shape, r.shape);
    }

    fn broadcastBinaryOp(
        allocator: std.mem.Allocator,
        l: *const Tensor,
        r: *const Tensor,
        comptime op: BinaryOp,
    ) !Tensor {
        const out_shape = if (shapeMatch(l, r))
            try allocator.dupe(usize, l.shape)
        else
            try broadcastShape(allocator, l.shape, r.shape);
        defer allocator.free(out_shape);

        var new = try Tensor.init(allocator, out_shape);
        errdefer new.deinit(allocator);

        // Fast path: Identical shapes and both contiguous
        if (shapeMatch(l, r) and l.isContiguous() and r.isContiguous()) {
            const left_base = try checkedOffsetToIndex(l.offset);
            const right_base = try checkedOffsetToIndex(r.offset);
            const left_data = l.storage.data[left_base..];
            const right_data = r.storage.data[right_base..];

            for (0..new.numel()) |i| {
                new.storage.data[i] = switch (op) {
                    .add => left_data[i] + right_data[i],
                    .mul => left_data[i] * right_data[i],
                };
            }
            return new;
        }

        // General broadcasting with incremental offset updates
        const rank = out_shape.len;
        const coords = try allocator.alloc(usize, rank);
        defer allocator.free(coords);
        @memset(coords, 0);

        var left_offset = l.offset;
        var right_offset = r.offset;

        const l_axis_off = rank - l.shape.len;
        const r_axis_off = rank - r.shape.len;

        for (0..new.numel()) |i| {
            const left_index = try checkedOffsetToIndex(left_offset);
            const right_index = try checkedOffsetToIndex(right_offset);

            new.storage.data[i] = switch (op) {
                .add => l.storage.data[left_index] + r.storage.data[right_index],
                .mul => l.storage.data[left_index] * r.storage.data[right_index],
            };

            // Increment coordinates and update offsets
            var j = rank;
            while (j > 0) {
                j -= 1;
                coords[j] += 1;
                const l_stride = if (j >= l_axis_off and l.shape[j - l_axis_off] > 1) l.strides[j - l_axis_off] else 0;
                const r_stride = if (j >= r_axis_off and r.shape[j - r_axis_off] > 1) r.strides[j - r_axis_off] else 0;

                if (coords[j] < out_shape[j]) {
                    left_offset += l_stride;
                    right_offset += r_stride;
                    break;
                } else {
                    left_offset -= l_stride * @as(isize, @intCast(out_shape[j] - 1));
                    right_offset -= r_stride * @as(isize, @intCast(out_shape[j] - 1));
                    coords[j] = 0;
                }
            }
        }

        return new;
    }

    pub fn add(allocator: std.mem.Allocator, l: *const Tensor, r: *const Tensor) !Tensor {
        return broadcastBinaryOp(allocator, l, r, .add);
    }

    pub fn mul(allocator: std.mem.Allocator, l: *const Tensor, r: *const Tensor) !Tensor {
        return broadcastBinaryOp(allocator, l, r, .mul);
    }

    pub fn reshape(allocator: std.mem.Allocator, tensor: *const Tensor, shape: []const usize) !Tensor {
        if (tensor.numel() != try numelFromShape(shape)) return TensorErrs.IncompatibleShapes;
        if (!tensor.isContiguous()) return TensorErrs.NotContiguous;

        const new_strides = try allocator.alloc(isize, shape.len);
        defer allocator.free(new_strides);
        try fillContiguousStrides(shape, new_strides);

        return initView(allocator, tensor.storage, tensor.offset, shape, new_strides);
    }

    pub fn unsqueeze(allocator: std.mem.Allocator, tensor: *const Tensor, axis: usize) !Tensor {
        if (axis > tensor.shape.len) return TensorErrs.AccessDenied;

        const out_shape = try allocator.alloc(usize, tensor.shape.len + 1);
        defer allocator.free(out_shape);
        const out_strides = try allocator.alloc(isize, tensor.strides.len + 1);
        defer allocator.free(out_strides);

        var old_axis: usize = 0;
        for (0..out_shape.len) |new_axis| {
            if (new_axis == axis) {
                out_shape[new_axis] = 1;
                out_strides[new_axis] = if (new_axis == out_shape.len - 1)
                    1
                else if (old_axis < tensor.strides.len)
                    tensor.strides[old_axis]
                else if (tensor.strides.len == 0)
                    1
                else
                    tensor.strides[tensor.strides.len - 1];
                continue;
            }

            out_shape[new_axis] = tensor.shape[old_axis];
            out_strides[new_axis] = tensor.strides[old_axis];
            old_axis += 1;
        }

        return initView(allocator, tensor.storage, tensor.offset, out_shape, out_strides);
    }

    pub fn squeeze(allocator: std.mem.Allocator, tensor: *const Tensor, axis: ?usize) !Tensor {
        if (axis) |selected_axis| {
            if (selected_axis >= tensor.shape.len) return TensorErrs.AccessDenied;
            if (tensor.shape[selected_axis] != 1) return TensorErrs.IncompatibleShapes;

            const out_shape = try allocator.alloc(usize, tensor.shape.len - 1);
            defer allocator.free(out_shape);
            const out_strides = try allocator.alloc(isize, tensor.strides.len - 1);
            defer allocator.free(out_strides);

            var out_axis: usize = 0;
            for (0..tensor.shape.len) |in_axis| {
                if (in_axis == selected_axis) continue;
                out_shape[out_axis] = tensor.shape[in_axis];
                out_strides[out_axis] = tensor.strides[in_axis];
                out_axis += 1;
            }

            return initView(allocator, tensor.storage, tensor.offset, out_shape, out_strides);
        }

        var kept_dims: usize = 0;
        for (tensor.shape) |dim| {
            if (dim != 1) kept_dims += 1;
        }

        const out_shape = try allocator.alloc(usize, kept_dims);
        defer allocator.free(out_shape);
        const out_strides = try allocator.alloc(isize, kept_dims);
        defer allocator.free(out_strides);

        var out_axis: usize = 0;
        for (0..tensor.shape.len) |in_axis| {
            if (tensor.shape[in_axis] == 1) continue;
            out_shape[out_axis] = tensor.shape[in_axis];
            out_strides[out_axis] = tensor.strides[in_axis];
            out_axis += 1;
        }

        return initView(allocator, tensor.storage, tensor.offset, out_shape, out_strides);
    }

    pub fn select(allocator: std.mem.Allocator, tensor: *const Tensor, axis: usize, index: isize) !Tensor {
        if (axis >= tensor.shape.len) return TensorErrs.AccessDenied;
        const coord = try normalizeAxisIndex(tensor.shape[axis], index);

        const out_shape = try allocator.alloc(usize, tensor.shape.len - 1);
        defer allocator.free(out_shape);
        const out_strides = try allocator.alloc(isize, tensor.strides.len - 1);
        defer allocator.free(out_strides);

        var out_axis: usize = 0;
        for (0..tensor.shape.len) |in_axis| {
            if (in_axis == axis) continue;
            out_shape[out_axis] = tensor.shape[in_axis];
            out_strides[out_axis] = tensor.strides[in_axis];
            out_axis += 1;
        }

        const offset = tensor.offset + try checkedMulCoord(coord, tensor.strides[axis]);
        return initView(allocator, tensor.storage, offset, out_shape, out_strides);
    }

    pub fn slice(allocator: std.mem.Allocator, tensor: *const Tensor, specs: []const Slice) !Tensor {
        if (specs.len != tensor.shape.len) return TensorErrs.InvalidSlice;

        const out_shape = try allocator.alloc(usize, tensor.shape.len);
        defer allocator.free(out_shape);
        const out_strides = try allocator.alloc(isize, tensor.strides.len);
        defer allocator.free(out_strides);

        var offset = tensor.offset;
        for (specs, 0..) |spec, axis| {
            const normalized = try normalizeSliceForDim(tensor.shape[axis], spec);
            offset += try checkedMulIsize(normalized.start, tensor.strides[axis]);
            out_shape[axis] = normalized.len;
            out_strides[axis] = try checkedMulIsize(tensor.strides[axis], normalized.step);
        }

        return initView(allocator, tensor.storage, offset, out_shape, out_strides);
    }

    pub fn narrow(
        allocator: std.mem.Allocator,
        tensor: *const Tensor,
        axis: usize,
        start: isize,
        length: usize,
    ) !Tensor {
        if (axis >= tensor.shape.len) return TensorErrs.AccessDenied;
        if (length == 0) return TensorErrs.InvalidSlice;

        const specs = try allocator.alloc(Slice, tensor.shape.len);
        defer allocator.free(specs);

        for (0..tensor.shape.len) |i| {
            specs[i] = .{};
        }
        const end = std.math.add(isize, start, try checkedCastToIsize(length)) catch return TensorErrs.DimensionOverflow;
        specs[axis] = .{
            .start = start,
            .end = end,
            .step = 1,
        };

        return slice(allocator, tensor, specs);
    }

    pub fn flip(allocator: std.mem.Allocator, tensor: *const Tensor, axis: usize) !Tensor {
        if (axis >= tensor.shape.len) return TensorErrs.AccessDenied;

        const specs = try allocator.alloc(Slice, tensor.shape.len);
        defer allocator.free(specs);

        for (0..tensor.shape.len) |i| {
            specs[i] = .{};
        }
        specs[axis] = .{ .step = -1 };

        return slice(allocator, tensor, specs);
    }

    pub fn transpose(
        allocator: std.mem.Allocator,
        tensor: *const Tensor,
        permutation: ?[]const usize,
    ) !Tensor {
        const rank = tensor.shape.len;
        const owned_axes = if (permutation == null) try allocator.alloc(usize, rank) else null;
        defer if (owned_axes) |axes| allocator.free(axes);

        const axes = if (permutation) |perm| blk: {
            try validatePermutation(rank, perm);
            break :blk perm;
        } else blk: {
            for (0..rank) |i| {
                owned_axes.?[i] = rank - 1 - i;
            }
            break :blk owned_axes.?;
        };

        const out_shape = try allocator.alloc(usize, rank);
        defer allocator.free(out_shape);
        const out_strides = try allocator.alloc(isize, rank);
        defer allocator.free(out_strides);

        for (axes, 0..) |axis, i| {
            out_shape[i] = tensor.shape[axis];
            out_strides[i] = tensor.strides[axis];
        }

        return initView(allocator, tensor.storage, tensor.offset, out_shape, out_strides);
    }

    pub fn addScalar(allocator: std.mem.Allocator, tensor: *const Tensor, scalar: f32) !Tensor {
        var new = try Tensor.init(allocator, tensor.shape);
        errdefer new.deinit(allocator);

        const n = tensor.numel();
        if (tensor.contiguousSlice()) |src| {
            for (0..n) |i| new.storage.data[i] = src[i] + scalar;
            return new;
        }

        for (0..n) |i| {
            new.storage.data[i] = try tensor.getFlat(i) + scalar;
        }
        return new;
    }

    pub fn scale(allocator: std.mem.Allocator, tensor: *const Tensor, scalar: f32) !Tensor {
        var new = try Tensor.init(allocator, tensor.shape);
        errdefer new.deinit(allocator);

        const n = tensor.numel();
        if (tensor.contiguousSlice()) |src| {
            for (0..n) |i| new.storage.data[i] = src[i] * scalar;
            return new;
        }

        for (0..n) |i| {
            new.storage.data[i] = try tensor.getFlat(i) * scalar;
        }
        return new;
    }

    pub fn sumAll(allocator: std.mem.Allocator, tensor: *const Tensor) !Tensor {
        var out = try Tensor.zeroes(allocator, &.{});
        errdefer out.deinit(allocator);

        var total: f32 = 0.0;
        const n = tensor.numel();
        if (tensor.contiguousSlice()) |src| {
            for (0..n) |i| total += src[i];
        } else {
            for (0..n) |i| {
                total += try tensor.getFlat(i);
            }
        }
        out.storage.data[0] = total;
        return out;
    }

    pub fn sum(allocator: std.mem.Allocator, tensor: *const Tensor, axis: usize) !Tensor {
        if (axis >= tensor.shape.len) return TensorErrs.AccessDenied;

        const out_shape = try allocator.alloc(usize, tensor.shape.len - 1);
        defer allocator.free(out_shape);
        var out_axis: usize = 0;
        for (0..tensor.shape.len) |in_axis| {
            if (in_axis == axis) continue;
            out_shape[out_axis] = tensor.shape[in_axis];
            out_axis += 1;
        }

        var out = try Tensor.zeroes(allocator, out_shape);
        errdefer out.deinit(allocator);

        if (out.shape.len == 0) {
            const full_indices = try allocator.alloc(usize, tensor.shape.len);
            defer allocator.free(full_indices);
            @memset(full_indices, 0);

            for (0..tensor.shape[axis]) |reduce_idx| {
                full_indices[axis] = reduce_idx;
                out.storage.data[0] += try tensor.get(full_indices);
            }
            return out;
        }

        const reduce_dim = tensor.shape[axis];
        const out_indices = try allocator.alloc(usize, out.shape.len);
        defer allocator.free(out_indices);
        const full_indices = try allocator.alloc(usize, tensor.shape.len);
        defer allocator.free(full_indices);

        for (0..out.numel()) |out_linear| {
            var remaining = out_linear;
            var logical_stride = out.numel();

            for (0..out.shape.len) |idx_axis| {
                logical_stride /= out.shape[idx_axis];
                out_indices[idx_axis] = remaining / logical_stride;
                remaining %= logical_stride;
            }

            for (0..reduce_dim) |reduce_idx| {
                var source_axis: usize = 0;
                for (0..tensor.shape.len) |tensor_axis| {
                    if (tensor_axis == axis) {
                        full_indices[tensor_axis] = reduce_idx;
                    } else {
                        full_indices[tensor_axis] = out_indices[source_axis];
                        source_axis += 1;
                    }
                }
                out.storage.data[out_linear] += try tensor.get(full_indices);
            }
        }

        return out;
    }

    pub fn matmulTransposed(
        allocator: std.mem.Allocator,
        l: *const Tensor,
        r: *const Tensor,
        transpose_left: bool,
        transpose_right: bool,
    ) !Tensor {
        return matmul(allocator, l, r, transpose_left, transpose_right);
    }

    pub fn matmul(allocator: std.mem.Allocator, l: *const Tensor, r: *const Tensor, transpose_left: bool, transpose_right: bool) !Tensor {
        if (l.shape.len == 0 or r.shape.len == 0) return TensorErrs.UnsupportedRank;
        if (transpose_left and l.shape.len < 2) return TensorErrs.UnsupportedRank;
        if (transpose_right and r.shape.len < 2) return TensorErrs.UnsupportedRank;

        // Transposes are handled by BLAS flags (CblasTrans), not by materializing
        // a permuted copy. We only materialize when an input isn't contiguous in
        // its own logical row-major layout.
        var left_tmp: ?Tensor = null;
        defer if (left_tmp) |*t| t.deinit(allocator);
        const left = blk: {
            if (l.isContiguous()) break :blk l;
            left_tmp = try l.contiguous(allocator);
            break :blk &left_tmp.?;
        };

        var right_tmp: ?Tensor = null;
        defer if (right_tmp) |*t| t.deinit(allocator);
        const right = blk: {
            if (r.isContiguous()) break :blk r;
            right_tmp = try r.contiguous(allocator);
            break :blk &right_tmp.?;
        };

        const left_promoted = left.shape.len == 1;
        const right_promoted = right.shape.len == 1;

        // Physical (stored) row/col counts for the matrix part of each input.
        const l_phys_rows: usize = if (left_promoted) 1 else left.shape[left.shape.len - 2];
        const l_phys_cols: usize = left.shape[left.shape.len - 1];
        const r_phys_rows: usize = if (right_promoted) 1 else right.shape[right.shape.len - 2];
        const r_phys_cols: usize = right.shape[right.shape.len - 1];

        // Logical (post-transpose) dims. Promoted (1D) operands cannot be transposed.
        const apply_trans_left = transpose_left and !left_promoted;
        const apply_trans_right = transpose_right and !right_promoted;

        const m = if (left_promoted) 1 else if (apply_trans_left) l_phys_cols else l_phys_rows;
        const k_left = if (left_promoted) l_phys_cols else if (apply_trans_left) l_phys_rows else l_phys_cols;
        const k_right = if (right_promoted) r_phys_cols else if (apply_trans_right) r_phys_cols else r_phys_rows;
        const n = if (right_promoted) 1 else if (apply_trans_right) r_phys_rows else r_phys_cols;
        const k = k_left;

        if (k_left != k_right) return TensorErrs.IncompatibleShapes;

        const left_batch_shape = batchShapeForMatmul(left.shape, left_promoted);
        const right_batch_shape = batchShapeForMatmul(right.shape, right_promoted);
        const out_batch_shape = try broadcastShape(allocator, left_batch_shape, right_batch_shape);
        defer allocator.free(out_batch_shape);

        const squeeze_left: usize = if (left_promoted) 1 else 0;
        const squeeze_right: usize = if (right_promoted) 1 else 0;
        const out_rank = out_batch_shape.len + 2 - squeeze_left - squeeze_right;
        const out_shape = try allocator.alloc(usize, out_rank);
        defer allocator.free(out_shape);

        var out_axis: usize = 0;
        @memcpy(out_shape[0..out_batch_shape.len], out_batch_shape);
        out_axis += out_batch_shape.len;
        if (!left_promoted) {
            out_shape[out_axis] = m;
            out_axis += 1;
        }
        if (!right_promoted) {
            out_shape[out_axis] = n;
        }

        var out = try Tensor.init(allocator, out_shape);
        errdefer out.deinit(allocator);

        const batch_count = try numelFromShape(out_batch_shape);
        const matrix_size = m * n;

        var out_batch_strides = try allocator.alloc(usize, out_batch_shape.len);
        defer allocator.free(out_batch_strides);
        if (out_batch_shape.len > 0) {
            var stride: usize = 1;
            var i = out_batch_shape.len;
            while (i > 0) {
                i -= 1;
                out_batch_strides[i] = stride;
                stride *= out_batch_shape[i];
            }
        }

        const cm = try checkedCastToCInt(m);
        const ck = try checkedCastToCInt(k);
        const cn = try checkedCastToCInt(n);
        const left_base = try checkedOffsetToIndex(left.offset);
        const right_base = try checkedOffsetToIndex(right.offset);

        // Leading dims are the row stride of each operand as BLAS sees it.
        // For a 1D right operand interpreted as a (k x 1) column, the stride is 1.
        const lda = try checkedCastToCInt(l_phys_cols);
        const ldb = try checkedCastToCInt(if (right_promoted) @as(usize, 1) else r_phys_cols);

        const trans_a: cblas.CBLAS_TRANSPOSE = if (apply_trans_left)
            @intCast(cblas.CblasTrans)
        else
            @intCast(cblas.CblasNoTrans);
        const trans_b: cblas.CBLAS_TRANSPOSE = if (apply_trans_right)
            @intCast(cblas.CblasTrans)
        else
            @intCast(cblas.CblasNoTrans);

        for (0..batch_count) |batch_idx| {
            const left_offset = if (out_batch_shape.len == 0)
                0
            else
                try broadcastOffset(batch_idx, out_batch_shape, out_batch_strides, left_batch_shape, left.strides[0..left_batch_shape.len]);
            const right_offset = if (out_batch_shape.len == 0)
                0
            else
                try broadcastOffset(batch_idx, out_batch_shape, out_batch_strides, right_batch_shape, right.strides[0..right_batch_shape.len]);

            const left_index = left_base + try checkedOffsetToIndex(left_offset);
            const right_index = right_base + try checkedOffsetToIndex(right_offset);
            const out_index = batch_idx * matrix_size;

            cblas.cblas_sgemm(
                @as(cblas.CBLAS_ORDER, @intCast(cblas.CblasRowMajor)),
                trans_a,
                trans_b,
                cm,
                cn,
                ck,
                1.0,
                left.storage.data.ptr + left_index,
                lda,
                right.storage.data.ptr + right_index,
                ldb,
                0.0,
                out.storage.data.ptr + out_index,
                cn,
            );
        }

        return out;
    }
};

fn expectTensorData(tensor: *const Tensor, expected: []const f32) !void {
    const allocator = std.testing.allocator;
    const actual = try tensor.toOwnedSlice(allocator);
    defer allocator.free(actual);
    try std.testing.expectEqualSlices(f32, expected, actual);
}

test "Init" {
    const allocator = std.testing.allocator;
    var shape = [_]usize{ 2, 3 };
    var expected_zeroes = std.mem.zeroes([6]f32);
    var expected_strides = [_]isize{ 3, 1 };
    var tensor = try Tensor.zeroes(allocator, &shape);
    defer tensor.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 6), tensor.numel());
    try expectTensorData(&tensor, &expected_zeroes);
    try std.testing.expectEqualSlices(isize, &expected_strides, tensor.strides);
    try std.testing.expect(tensor.isContiguous());
}

test "Scalar Support" {
    const allocator = std.testing.allocator;
    var vals = [_]f32{42.0};
    var tensor = try Tensor.fromSlice(allocator, &.{}, &vals);
    defer tensor.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 1), tensor.numel());
    try std.testing.expectEqual(@as(f32, 42.0), try tensor.getFlat(0));
    try std.testing.expectEqual(@as(f32, 42.0), try tensor.get(&.{}));
    try std.testing.expect(tensor.isContiguous());
}

test "From Slice" {
    const allocator = std.testing.allocator;
    var shape = [_]usize{ 2, 2 };
    var vals = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var tensor = try Tensor.fromSlice(allocator, &shape, &vals);
    defer tensor.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 4), tensor.numel());
    try expectTensorData(&tensor, &vals);
}

test "Invalid Shape" {
    const allocator = std.testing.allocator;
    var shape = [_]usize{ 2, 3 };
    var vals = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    try std.testing.expectError(TensorErrs.SizeMisMatch, Tensor.fromSlice(allocator, &shape, &vals));
}

test "Scaling" {
    const allocator = std.testing.allocator;
    var shape = [_]usize{ 2, 2 };
    var vals = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var tensor = try Tensor.fromSlice(allocator, &shape, &vals);
    defer tensor.deinit(allocator);
    var new = try Tensor.scale(allocator, &tensor, 2);
    defer new.deinit(allocator);

    var expecting = [_]f32{ 2.0, 4.0, 6.0, 8.0 };
    try expectTensorData(&new, &expecting);
}

test "Addition" {
    const allocator = std.testing.allocator;
    var shape = [_]usize{ 2, 2 };
    var vals = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var tensor = try Tensor.fromSlice(allocator, &shape, &vals);
    defer tensor.deinit(allocator);
    var new = try Tensor.add(allocator, &tensor, &tensor);
    defer new.deinit(allocator);

    var expecting = [_]f32{ 2.0, 4.0, 6.0, 8.0 };
    try expectTensorData(&new, &expecting);
}

test "Mul" {
    const allocator = std.testing.allocator;
    var shape = [_]usize{ 2, 2 };
    var vals = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var tensor = try Tensor.fromSlice(allocator, &shape, &vals);
    defer tensor.deinit(allocator);
    var new = try Tensor.mul(allocator, &tensor, &tensor);
    defer new.deinit(allocator);

    var expecting = [_]f32{ 1.0, 4.0, 9.0, 16.0 };
    try expectTensorData(&new, &expecting);
}

test "Invalid Ops" {
    const allocator = std.testing.allocator;
    var shape = [_]usize{ 2, 2 };
    var vals = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var tensor = try Tensor.fromSlice(allocator, &shape, &vals);
    defer tensor.deinit(allocator);
    var shape2 = [_]usize{4};
    var tensor2 = try Tensor.fromSlice(allocator, &shape2, &vals);
    defer tensor2.deinit(allocator);

    try std.testing.expectError(TensorErrs.IncompatibleShapes, Tensor.add(allocator, &tensor, &tensor2));
    try std.testing.expectError(TensorErrs.IncompatibleShapes, Tensor.mul(allocator, &tensor, &tensor2));
}

test "Broadcast Addition" {
    const allocator = std.testing.allocator;
    var left_shape = [_]usize{ 2, 3 };
    var left_vals = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    var right_shape = [_]usize{3};
    var right_vals = [_]f32{ 10.0, 20.0, 30.0 };

    var left = try Tensor.fromSlice(allocator, &left_shape, &left_vals);
    defer left.deinit(allocator);
    var right = try Tensor.fromSlice(allocator, &right_shape, &right_vals);
    defer right.deinit(allocator);

    var new = try Tensor.add(allocator, &left, &right);
    defer new.deinit(allocator);

    var expected_shape = [_]usize{ 2, 3 };
    var expected_vals = [_]f32{ 11.0, 22.0, 33.0, 14.0, 25.0, 36.0 };
    try std.testing.expectEqualSlices(usize, &expected_shape, new.shape);
    try expectTensorData(&new, &expected_vals);
}

test "Broadcast Multiplication" {
    const allocator = std.testing.allocator;
    var left_shape = [_]usize{ 2, 1, 3 };
    var left_vals = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    var right_shape = [_]usize{ 1, 4, 1 };
    var right_vals = [_]f32{ 10.0, 20.0, 30.0, 40.0 };

    var left = try Tensor.fromSlice(allocator, &left_shape, &left_vals);
    defer left.deinit(allocator);
    var right = try Tensor.fromSlice(allocator, &right_shape, &right_vals);
    defer right.deinit(allocator);

    var new = try Tensor.mul(allocator, &left, &right);
    defer new.deinit(allocator);

    var expected_shape = [_]usize{ 2, 4, 3 };
    var expected_vals = [_]f32{
        10.0,  20.0,  30.0,
        20.0,  40.0,  60.0,
        30.0,  60.0,  90.0,
        40.0,  80.0,  120.0,
        40.0,  50.0,  60.0,
        80.0,  100.0, 120.0,
        120.0, 150.0, 180.0,
        160.0, 200.0, 240.0,
    };
    try std.testing.expectEqualSlices(usize, &expected_shape, new.shape);
    try expectTensorData(&new, &expected_vals);
}

test "Transpose 2D View" {
    const allocator = std.testing.allocator;
    var shape = [_]usize{ 2, 3 };
    var vals = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    var tensor = try Tensor.fromSlice(allocator, &shape, &vals);
    defer tensor.deinit(allocator);

    var new = try Tensor.transpose(allocator, &tensor, null);
    defer new.deinit(allocator);

    var expected_shape = [_]usize{ 3, 2 };
    var expected_vals = [_]f32{ 1.0, 4.0, 2.0, 5.0, 3.0, 6.0 };
    try std.testing.expectEqualSlices(usize, &expected_shape, new.shape);
    try expectTensorData(&new, &expected_vals);
    try std.testing.expect(!new.isContiguous());
}

test "Transpose 3D Permutation View" {
    const allocator = std.testing.allocator;
    var shape = [_]usize{ 2, 3, 4 };
    var vals = [_]f32{
        0.0,  1.0,  2.0,  3.0,
        4.0,  5.0,  6.0,  7.0,
        8.0,  9.0,  10.0, 11.0,
        12.0, 13.0, 14.0, 15.0,
        16.0, 17.0, 18.0, 19.0,
        20.0, 21.0, 22.0, 23.0,
    };
    var permutation = [_]usize{ 1, 0, 2 };
    var tensor = try Tensor.fromSlice(allocator, &shape, &vals);
    defer tensor.deinit(allocator);

    var new = try Tensor.transpose(allocator, &tensor, &permutation);
    defer new.deinit(allocator);

    var expected_shape = [_]usize{ 3, 2, 4 };
    var expected_vals = [_]f32{
        0.0,  1.0,  2.0,  3.0,
        12.0, 13.0, 14.0, 15.0,
        4.0,  5.0,  6.0,  7.0,
        16.0, 17.0, 18.0, 19.0,
        8.0,  9.0,  10.0, 11.0,
        20.0, 21.0, 22.0, 23.0,
    };
    try std.testing.expectEqualSlices(usize, &expected_shape, new.shape);
    try expectTensorData(&new, &expected_vals);
}

test "Invalid Transpose Permutation" {
    const allocator = std.testing.allocator;
    var shape = [_]usize{ 2, 2 };
    var vals = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var bad_permutation = [_]usize{ 0, 0 };
    var tensor = try Tensor.fromSlice(allocator, &shape, &vals);
    defer tensor.deinit(allocator);

    try std.testing.expectError(TensorErrs.InvalidPermutation, Tensor.transpose(allocator, &tensor, &bad_permutation));
}

test "Reshape View" {
    const allocator = std.testing.allocator;
    var shape = [_]usize{ 2, 3 };
    var vals = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    var tensor = try Tensor.fromSlice(allocator, &shape, &vals);
    defer tensor.deinit(allocator);

    var reshaped = try Tensor.reshape(allocator, &tensor, &.{ 3, 2 });
    defer reshaped.deinit(allocator);

    var expected = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    try std.testing.expectEqualSlices(usize, &.{ 3, 2 }, reshaped.shape);
    try expectTensorData(&reshaped, &expected);

    try reshaped.setFlat(1, 20.0);
    try std.testing.expectEqual(@as(f32, 20.0), try tensor.getFlat(1));
}

test "Multidimensional Get Set" {
    const allocator = std.testing.allocator;
    var tensor = try Tensor.fromSlice(allocator, &.{ 2, 3 }, &.{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });
    defer tensor.deinit(allocator);

    try std.testing.expectEqual(@as(f32, 6.0), try tensor.get(&.{ 1, 2 }));
    try tensor.set(&.{ 0, 1 }, 20.0);
    try std.testing.expectEqual(@as(f32, 20.0), try tensor.get(&.{ 0, 1 }));
    try expectTensorData(&tensor, &.{ 1.0, 20.0, 3.0, 4.0, 5.0, 6.0 });
}

test "Select View" {
    const allocator = std.testing.allocator;
    var tensor = try Tensor.fromSlice(allocator, &.{ 2, 3 }, &.{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });
    defer tensor.deinit(allocator);

    var row = try Tensor.select(allocator, &tensor, 0, 1);
    defer row.deinit(allocator);

    try std.testing.expectEqualSlices(usize, &.{3}, row.shape);
    try expectTensorData(&row, &.{ 4.0, 5.0, 6.0 });
    try row.set(&.{1}, 50.0);
    try std.testing.expectEqual(@as(f32, 50.0), try tensor.get(&.{ 1, 1 }));
}

test "Slice View Positive Step" {
    const allocator = std.testing.allocator;
    var tensor = try Tensor.fromSlice(
        allocator,
        &.{ 3, 4 },
        &.{
            0.0, 1.0, 2.0,  3.0,
            4.0, 5.0, 6.0,  7.0,
            8.0, 9.0, 10.0, 11.0,
        },
    );
    defer tensor.deinit(allocator);

    var view = try Tensor.slice(allocator, &tensor, &.{
        .{ .start = 1, .end = 3, .step = 1 },
        .{ .start = 0, .end = null, .step = 2 },
    });
    defer view.deinit(allocator);

    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, view.shape);
    try expectTensorData(&view, &.{ 4.0, 6.0, 8.0, 10.0 });
    try view.set(&.{ 1, 0 }, 80.0);
    try std.testing.expectEqual(@as(f32, 80.0), try tensor.get(&.{ 2, 0 }));
}

test "Slice View Negative Step" {
    const allocator = std.testing.allocator;
    var tensor = try Tensor.fromSlice(allocator, &.{5}, &.{ 1.0, 2.0, 3.0, 4.0, 5.0 });
    defer tensor.deinit(allocator);

    var reversed = try Tensor.slice(allocator, &tensor, &.{
        .{ .start = null, .end = null, .step = -1 },
    });
    defer reversed.deinit(allocator);

    try std.testing.expectEqualSlices(usize, &.{5}, reversed.shape);
    try expectTensorData(&reversed, &.{ 5.0, 4.0, 3.0, 2.0, 1.0 });
    try reversed.set(&.{0}, 50.0);
    try std.testing.expectEqual(@as(f32, 50.0), try tensor.get(&.{4}));
}

test "Slice Invalid Step" {
    const allocator = std.testing.allocator;
    var tensor = try Tensor.fromSlice(allocator, &.{3}, &.{ 1.0, 2.0, 3.0 });
    defer tensor.deinit(allocator);

    try std.testing.expectError(TensorErrs.InvalidSlice, Tensor.slice(allocator, &tensor, &.{
        .{ .step = 0 },
    }));
}

test "Slice Rejects Empty View" {
    const allocator = std.testing.allocator;
    var tensor = try Tensor.fromSlice(allocator, &.{5}, &.{ 1.0, 2.0, 3.0, 4.0, 5.0 });
    defer tensor.deinit(allocator);

    try std.testing.expectError(TensorErrs.InvalidSlice, Tensor.slice(allocator, &tensor, &.{
        .{ .start = 2, .end = 2, .step = 1 },
    }));
}

test "Unsqueeze View" {
    const allocator = std.testing.allocator;
    var tensor = try Tensor.fromSlice(allocator, &.{ 2, 3 }, &.{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });
    defer tensor.deinit(allocator);

    var expanded = try Tensor.unsqueeze(allocator, &tensor, 1);
    defer expanded.deinit(allocator);

    try std.testing.expectEqualSlices(usize, &.{ 2, 1, 3 }, expanded.shape);
    try expectTensorData(&expanded, &.{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });
}

test "Squeeze View" {
    const allocator = std.testing.allocator;
    var tensor = try Tensor.fromSlice(allocator, &.{ 2, 1, 3, 1 }, &.{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });
    defer tensor.deinit(allocator);

    var squeezed_all = try Tensor.squeeze(allocator, &tensor, null);
    defer squeezed_all.deinit(allocator);
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, squeezed_all.shape);
    try expectTensorData(&squeezed_all, &.{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });

    var squeezed_axis = try Tensor.squeeze(allocator, &tensor, 1);
    defer squeezed_axis.deinit(allocator);
    try std.testing.expectEqualSlices(usize, &.{ 2, 3, 1 }, squeezed_axis.shape);
}

test "Narrow View" {
    const allocator = std.testing.allocator;
    var tensor = try Tensor.fromSlice(allocator, &.{ 3, 4 }, &.{
        0.0, 1.0, 2.0,  3.0,
        4.0, 5.0, 6.0,  7.0,
        8.0, 9.0, 10.0, 11.0,
    });
    defer tensor.deinit(allocator);

    var narrowed = try Tensor.narrow(allocator, &tensor, 1, 1, 2);
    defer narrowed.deinit(allocator);

    try std.testing.expectEqualSlices(usize, &.{ 3, 2 }, narrowed.shape);
    try expectTensorData(&narrowed, &.{ 1.0, 2.0, 5.0, 6.0, 9.0, 10.0 });
}

test "Narrow Rejects Empty View" {
    const allocator = std.testing.allocator;
    var tensor = try Tensor.fromSlice(allocator, &.{4}, &.{ 1.0, 2.0, 3.0, 4.0 });
    defer tensor.deinit(allocator);

    try std.testing.expectError(TensorErrs.InvalidSlice, Tensor.narrow(allocator, &tensor, 0, 1, 0));
}

test "Flip View" {
    const allocator = std.testing.allocator;
    var tensor = try Tensor.fromSlice(allocator, &.{ 2, 3 }, &.{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });
    defer tensor.deinit(allocator);

    var flipped = try Tensor.flip(allocator, &tensor, 1);
    defer flipped.deinit(allocator);

    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, flipped.shape);
    try expectTensorData(&flipped, &.{ 3.0, 2.0, 1.0, 6.0, 5.0, 4.0 });
}

test "Sum All Returns Scalar" {
    const allocator = std.testing.allocator;
    var tensor = try Tensor.fromSlice(allocator, &.{ 2, 3 }, &.{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });
    defer tensor.deinit(allocator);

    var out = try Tensor.sumAll(allocator, &tensor);
    defer out.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 0), out.shape.len);
    try std.testing.expectEqual(@as(f32, 21.0), try out.getFlat(0));
}

test "Sum Along Axis" {
    const allocator = std.testing.allocator;
    var tensor = try Tensor.fromSlice(allocator, &.{ 2, 3 }, &.{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });
    defer tensor.deinit(allocator);

    var sum0 = try Tensor.sum(allocator, &tensor, 0);
    defer sum0.deinit(allocator);
    try std.testing.expectEqualSlices(usize, &.{3}, sum0.shape);
    try expectTensorData(&sum0, &.{ 5.0, 7.0, 9.0 });

    var sum1 = try Tensor.sum(allocator, &tensor, 1);
    defer sum1.deinit(allocator);
    try std.testing.expectEqualSlices(usize, &.{2}, sum1.shape);
    try expectTensorData(&sum1, &.{ 6.0, 15.0 });
}

test "Reshape Requires Contiguous" {
    const allocator = std.testing.allocator;
    var shape = [_]usize{ 2, 3 };
    var vals = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    var tensor = try Tensor.fromSlice(allocator, &shape, &vals);
    defer tensor.deinit(allocator);
    var transposed = try Tensor.transpose(allocator, &tensor, null);
    defer transposed.deinit(allocator);

    try std.testing.expectError(TensorErrs.NotContiguous, Tensor.reshape(allocator, &transposed, &.{6}));
}

test "Contiguous Materializes View" {
    const allocator = std.testing.allocator;
    var shape = [_]usize{ 2, 3 };
    var vals = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    var tensor = try Tensor.fromSlice(allocator, &shape, &vals);
    defer tensor.deinit(allocator);
    var transposed = try Tensor.transpose(allocator, &tensor, null);
    defer transposed.deinit(allocator);
    var materialized = try transposed.contiguous(allocator);
    defer materialized.deinit(allocator);

    var expected_vals = [_]f32{ 1.0, 4.0, 2.0, 5.0, 3.0, 6.0 };
    try std.testing.expect(materialized.isContiguous());
    try expectTensorData(&materialized, &expected_vals);
}

test "Matmul 2D" {
    const allocator = std.testing.allocator;
    var left_shape = [_]usize{ 2, 3 };
    var left_vals = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    var right_shape = [_]usize{ 3, 2 };
    var right_vals = [_]f32{ 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 };

    var left = try Tensor.fromSlice(allocator, &left_shape, &left_vals);
    defer left.deinit(allocator);
    var right = try Tensor.fromSlice(allocator, &right_shape, &right_vals);
    defer right.deinit(allocator);

    var new = try Tensor.matmul(allocator, &left, &right, false, false);
    defer new.deinit(allocator);

    var expected_shape = [_]usize{ 2, 2 };
    var expected_vals = [_]f32{ 58.0, 64.0, 139.0, 154.0 };
    try std.testing.expectEqualSlices(usize, &expected_shape, new.shape);
    try expectTensorData(&new, &expected_vals);
}

test "Matmul Transposed 2D" {
    const allocator = std.testing.allocator;
    var canonical_right = try Tensor.fromSlice(allocator, &.{ 3, 2 }, &.{ 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 });
    defer canonical_right.deinit(allocator);
    var canonical_left = try Tensor.fromSlice(allocator, &.{ 2, 3 }, &.{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });
    defer canonical_left.deinit(allocator);

    var left_physical_t = try Tensor.fromSlice(allocator, &.{ 3, 2 }, &.{ 1.0, 4.0, 2.0, 5.0, 3.0, 6.0 });
    defer left_physical_t.deinit(allocator);
    var right_physical_t = try Tensor.fromSlice(allocator, &.{ 2, 3 }, &.{ 7.0, 9.0, 11.0, 8.0, 10.0, 12.0 });
    defer right_physical_t.deinit(allocator);

    const expected = &.{ 58.0, 64.0, 139.0, 154.0 };

    var right_trans = try Tensor.matmulTransposed(allocator, &canonical_left, &right_physical_t, false, true);
    defer right_trans.deinit(allocator);
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, right_trans.shape);
    try expectTensorData(&right_trans, expected);

    var no_trans = try Tensor.matmulTransposed(allocator, &canonical_left, &canonical_right, false, false);
    defer no_trans.deinit(allocator);
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, no_trans.shape);
    try expectTensorData(&no_trans, expected);

    var left_trans = try Tensor.matmulTransposed(allocator, &left_physical_t, &canonical_right, true, false);
    defer left_trans.deinit(allocator);
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, left_trans.shape);
    try expectTensorData(&left_trans, expected);

    var both_trans = try Tensor.matmulTransposed(allocator, &left_physical_t, &right_physical_t, true, true);
    defer both_trans.deinit(allocator);
    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, both_trans.shape);
    try expectTensorData(&both_trans, expected);
}

test "Matmul Vector Dot Returns Scalar" {
    const allocator = std.testing.allocator;
    var left = try Tensor.fromSlice(allocator, &.{3}, &.{ 1.0, 2.0, 3.0 });
    defer left.deinit(allocator);
    var right = try Tensor.fromSlice(allocator, &.{3}, &.{ 4.0, 5.0, 6.0 });
    defer right.deinit(allocator);

    var out = try Tensor.matmul(allocator, &left, &right, false, false);
    defer out.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 0), out.shape.len);
    try std.testing.expectEqual(@as(f32, 32.0), try out.getFlat(0));
}

test "Matmul Matrix Vector" {
    const allocator = std.testing.allocator;
    var left = try Tensor.fromSlice(allocator, &.{ 2, 3 }, &.{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });
    defer left.deinit(allocator);
    var right = try Tensor.fromSlice(allocator, &.{3}, &.{ 7.0, 8.0, 9.0 });
    defer right.deinit(allocator);

    var out = try Tensor.matmul(allocator, &left, &right, false, false);
    defer out.deinit(allocator);

    try std.testing.expectEqualSlices(usize, &.{2}, out.shape);
    try expectTensorData(&out, &.{ 50.0, 122.0 });
}

test "Matmul Vector Matrix" {
    const allocator = std.testing.allocator;
    var left = try Tensor.fromSlice(allocator, &.{3}, &.{ 1.0, 2.0, 3.0 });
    defer left.deinit(allocator);
    var right = try Tensor.fromSlice(allocator, &.{ 3, 2 }, &.{ 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 });
    defer right.deinit(allocator);

    var out = try Tensor.matmul(allocator, &left, &right, false, false);
    defer out.deinit(allocator);

    try std.testing.expectEqualSlices(usize, &.{2}, out.shape);
    try expectTensorData(&out, &.{ 40.0, 46.0 });
}

test "Matmul Batched" {
    const allocator = std.testing.allocator;
    var left_shape = [_]usize{ 2, 2, 3 };
    var left_vals = [_]f32{
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        1.0, 0.0, 1.0,
        0.0, 1.0, 0.0,
    };
    var right_shape = [_]usize{ 2, 3, 2 };
    var right_vals = [_]f32{
        7.0,  8.0,
        9.0,  10.0,
        11.0, 12.0,
        1.0,  2.0,
        3.0,  4.0,
        5.0,  6.0,
    };

    var left = try Tensor.fromSlice(allocator, &left_shape, &left_vals);
    defer left.deinit(allocator);
    var right = try Tensor.fromSlice(allocator, &right_shape, &right_vals);
    defer right.deinit(allocator);

    var new = try Tensor.matmul(allocator, &left, &right, false, false);
    defer new.deinit(allocator);

    var expected_shape = [_]usize{ 2, 2, 2 };
    var expected_vals = [_]f32{
        58.0,  64.0,
        139.0, 154.0,
        6.0,   8.0,
        3.0,   4.0,
    };
    try std.testing.expectEqualSlices(usize, &expected_shape, new.shape);
    try expectTensorData(&new, &expected_vals);
}

test "Matmul Broadcast Batch Dimensions" {
    const allocator = std.testing.allocator;
    var left_shape = [_]usize{ 2, 1, 2, 2 };
    var left_vals = [_]f32{
        1.0, 2.0,
        3.0, 4.0,
        2.0, 0.0,
        1.0, 2.0,
    };
    var right_shape = [_]usize{ 1, 3, 2, 2 };
    var right_vals = [_]f32{
        1.0, 0.0,
        0.0, 1.0,
        0.0, 1.0,
        1.0, 0.0,
        2.0, 0.0,
        0.0, 2.0,
    };

    var left = try Tensor.fromSlice(allocator, &left_shape, &left_vals);
    defer left.deinit(allocator);
    var right = try Tensor.fromSlice(allocator, &right_shape, &right_vals);
    defer right.deinit(allocator);

    var new = try Tensor.matmul(allocator, &left, &right, false, false);
    defer new.deinit(allocator);

    var expected_shape = [_]usize{ 2, 3, 2, 2 };
    var expected_vals = [_]f32{
        1.0, 2.0,
        3.0, 4.0,
        2.0, 1.0,
        4.0, 3.0,
        2.0, 4.0,
        6.0, 8.0,
        2.0, 0.0,
        1.0, 2.0,
        0.0, 2.0,
        2.0, 1.0,
        4.0, 0.0,
        2.0, 4.0,
    };
    try std.testing.expectEqualSlices(usize, &expected_shape, new.shape);
    try expectTensorData(&new, &expected_vals);
}

test "Matmul Works With Non Contiguous Inputs" {
    const allocator = std.testing.allocator;
    var left = try Tensor.fromSlice(allocator, &.{ 3, 2 }, &.{ 1.0, 4.0, 2.0, 5.0, 3.0, 6.0 });
    defer left.deinit(allocator);
    var right = try Tensor.fromSlice(allocator, &.{ 3, 2 }, &.{ 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 });
    defer right.deinit(allocator);
    var left_t = try Tensor.transpose(allocator, &left, null);
    defer left_t.deinit(allocator);

    var out = try Tensor.matmul(allocator, &left_t, &right, false, false);
    defer out.deinit(allocator);

    try std.testing.expectEqualSlices(usize, &.{ 2, 2 }, out.shape);
    try expectTensorData(&out, &.{ 58.0, 64.0, 139.0, 154.0 });
}

test "Matmul Invalid Shapes" {
    const allocator = std.testing.allocator;
    var left_shape = [_]usize{ 2, 3 };
    var left_vals = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    var right_shape = [_]usize{ 4, 2 };
    var right_vals = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };

    var left = try Tensor.fromSlice(allocator, &left_shape, &left_vals);
    defer left.deinit(allocator);
    var right = try Tensor.fromSlice(allocator, &right_shape, &right_vals);
    defer right.deinit(allocator);

    try std.testing.expectError(TensorErrs.IncompatibleShapes, Tensor.matmul(allocator, &left, &right, false, false));
}

test "Matmul Unsupported Rank" {
    const allocator = std.testing.allocator;
    var scalar = try Tensor.fromSlice(allocator, &.{}, &.{2.0});
    defer scalar.deinit(allocator);
    var matrix = try Tensor.fromSlice(allocator, &.{ 1, 1 }, &.{3.0});
    defer matrix.deinit(allocator);

    try std.testing.expectError(TensorErrs.UnsupportedRank, Tensor.matmul(allocator, &scalar, &matrix, false, false));
}
