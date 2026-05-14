#!/usr/bin/env bash
set -euo pipefail

case "${1:-}" in
    test)
        zig build test --summary all
        ;;
    build)
        zig build -Doptimize=ReleaseFast -Dcpu=native
        ;;
    release)
        shift
        zig build run -Doptimize=ReleaseFast -Dcpu=native -- "$@"
        ;;
    *)
        echo "Usage: $0 {test|build|release [args...]}"
        echo "  test                  - run unit tests"
        echo "  build                 - build optimized binary to zig-out/bin/Ai"
        echo "  release <subcommand>  - build and run (e.g. release train, release bench, release sample 'hi')"
        exit 1
        ;;
esac
