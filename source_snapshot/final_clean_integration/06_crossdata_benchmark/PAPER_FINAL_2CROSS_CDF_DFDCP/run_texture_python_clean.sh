#!/usr/bin/env bash
set -euo pipefail

TEXENV="/home/aiotlab/miniconda3/envs/TextureExpert"
TEXPY="/home/aiotlab/miniconda3/envs/TextureExpert/bin/python"
TEX_SITE="/home/aiotlab/miniconda3/envs/TextureExpert/lib/python3.8/site-packages"
TEX_LD="/home/aiotlab/miniconda3/envs/TextureExpert/lib:/home/aiotlab/miniconda3/envs/TextureExpert/lib/python3.8/site-packages/torch/lib"

ENV_ARGS=(
  HOME="${HOME:-/home/aiotlab}"
  USER="${USER:-aiotlab}"
  LANG="${LANG:-C.UTF-8}"
  PATH="$TEXENV/bin:/usr/bin:/bin"
  PYTHONNOUSERSITE=1
  PYTHONPATH="$TEX_SITE"
  LD_LIBRARY_PATH="$TEX_LD"
)

# Preserve GPU selection only if caller intentionally set it.
if [[ -n "${CUDA_VISIBLE_DEVICES+x}" ]]; then
    ENV_ARGS+=(
      CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
    )
fi

exec env -i   "${ENV_ARGS[@]}"   "$TEXPY"   -S   "$@"
