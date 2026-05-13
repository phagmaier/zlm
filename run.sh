#!/usr/bin/env bash
set -euo pipefail

case "$1" in
    test)
        zig build test --summary all
        ;;
    release)
        zig build run -Doptimize=ReleaseFast
        ;;
    *)
        echo "Usage: $0 {test|release}"
        exit 1
        ;;
esac
