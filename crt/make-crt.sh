#!/usr/bin/env bash
# Regenerate the stripped crt object files and libc.txt that build.zig uses.
# See the big comment in ../build.zig for why this exists.
#
# Run this whenever the system glibc is upgraded (or when first cloning on a
# new machine). Requires `objcopy` (from binutils) and `zig` on PATH.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sys_crt="$(dirname "$(cc -print-file-name=crt1.o)")"
echo "system crt dir: $sys_crt"

for f in crt1.o Scrt1.o crti.o crtn.o; do
    src="$sys_crt/$f"
    if [ ! -e "$src" ]; then
        echo "warning: $src missing, skipping" >&2
        continue
    fi
    objcopy --remove-section=.sframe --remove-section=.rela.sframe "$src" "$here/$f"
    echo "  stripped $f"
done

# Symlink the libraries Zig wants to find next to the crt objects.
for f in libc libdl librt libutil libpthread libm; do
    for ext in .so .so.6 .a; do
        src="/usr/lib/${f}${ext}"
        if [ -e "$src" ]; then
            ln -sfn "$src" "$here/${f}${ext}"
        fi
    done
done

# Write a libc.txt seeded from `zig libc` defaults, with crt_dir pointing at us.
zig libc | sed "s|^crt_dir=.*|crt_dir=$here|" > "$here/libc.txt"
echo "wrote $here/libc.txt"
