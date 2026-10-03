#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_NAME="ai_model_server"

if ! command -v conda >/dev/null 2>&1; then
    echo "Conda is not installed. Please run install/install-conda-linux.sh first." >&2
    exit 1
fi

# Ensure conda shell functions are available
eval "$(conda shell.bash hook)"

MODE="nvidia"
if [[ $# -gt 0 ]]; then
    case "$1" in
        --amd)
            MODE="amd"
            ;;
        --intel)
            MODE="intel"
            ;;
        --help|-h)
            echo "Usage: $0 [--amd|--intel]" >&2
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 [--amd|--intel]" >&2
            exit 1
            ;;
    esac
fi

echo "Preparing Conda environment for mode: $MODE"
conda deactivate >/dev/null 2>&1 || true

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    echo "Removing existing conda environment '$ENV_NAME'..."
    conda remove -n "$ENV_NAME" --all -y
fi

case "$MODE" in
    nvidia)
        ENV_FILE="$SCRIPT_DIR/environment-linux.yml"
        ;;
    amd)
        ENV_FILE="$SCRIPT_DIR/environment-linux-amd.yml"
        ;;
    intel)
        ENV_FILE="$SCRIPT_DIR/environment-linux-intel.yml"
        ;;
esac

echo "Creating conda environment from $ENV_FILE..."
conda env create -f "$ENV_FILE"

echo "Activating environment '$ENV_NAME'..."
conda activate "$ENV_NAME"

if [[ "$MODE" == "amd" ]]; then
    echo "Running AMD post-install steps..."
    source "$SCRIPT_DIR/install-amd-post.sh"
fi

echo "Running database migrations..."
python "$PROJECT_ROOT/migrate.py"

echo "Starting server..."
python "$PROJECT_ROOT/server.py"