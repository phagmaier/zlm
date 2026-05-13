pub const tensor = @import("tensor.zig");
pub const Tensor = tensor.Tensor;
pub const autograd = @import("autograd.zig");
pub const Val = autograd.Val;

test {
    _ = @import("tensor.zig");
    _ = @import("autograd.zig");
}
