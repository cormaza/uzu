#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

DEFAULT_MODEL="liquidai:lfm2.5:230m:mirai:mirai-m:4"

SERVE_MODE=0
HAS_MODEL=0
SERVER_ARGS=()

ARGS=("$@")
i=0
while [[ $i -lt ${#ARGS[@]} ]]; do
    arg="${ARGS[$i]}"
    case "$arg" in
        --serve|serve|server)
            SERVE_MODE=1
            ;;
        --model|-m)
            HAS_MODEL=1
            if [[ $((i + 1)) -lt ${#ARGS[@]} ]]; then
                SERVER_ARGS+=("--model" "${ARGS[$((i + 1))]}")
                i=$((i + 1))
            else
                SERVER_ARGS+=("--model")
            fi
            ;;
        --model=*)
            HAS_MODEL=1
            SERVER_ARGS+=("$arg")
            ;;
        *)
            SERVER_ARGS+=("$arg")
            ;;
    esac
    i=$((i + 1))
done

if [[ $SERVE_MODE -eq 1 ]]; then
    if [[ $HAS_MODEL -eq 0 ]]; then
        SERVER_ARGS=("--model" "$DEFAULT_MODEL" "${SERVER_ARGS[@]}")
    fi
    exec cargo run --release -p cli -- server "${SERVER_ARGS[@]}"
else
    exec cargo run --release -p uzu --example mirai_rocm_chat -- "$@"
fi
