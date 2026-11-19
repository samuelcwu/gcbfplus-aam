#!/bin/bash
# One-time setup of the gcbfplus conda env on MIT ORCD Engaging. Run on a LOGIN node from the repo root:
#     cd ~/gcbfplus-aam && bash cluster/setup_env.sh
# Takes ~10-15 min (mostly pip downloads). Re-running is safe: it skips creating the env if it exists.
# Versions are pinned to the Mac env (jax 0.4.38 etc.) so results match as closely as possible; the CUDA build of jax
# brings its own CUDA libraries as pip wheels, so no `module load cuda` is needed (and loading one can conflict).
set -euo pipefail

ENV_NAME=gcbfplus

module load miniforge
# Engaging's miniforge module puts conda on PATH but we don't run `conda init` (it edits ~/.bashrc);
# sourcing conda.sh gives `conda activate` for this shell only.
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
    conda create -y -n "$ENV_NAME" python=3.11
fi
conda activate "$ENV_NAME"

# ffmpeg: matplotlib's video writer shells out to it
conda install -y -c conda-forge ffmpeg

python -m pip install --upgrade pip
python -m pip install "jax[cuda12]==0.4.38" \
    flax==0.10.4 optax==0.2.5 orbax-checkpoint==0.11.5 chex==0.1.90 equinox==0.13.8 jaxtyping==0.3.9 \
    jraph==0.0.6.dev0 tensorflow-probability==0.25.0 \
    numpy scipy matplotlib==3.10.9 einops pyyaml rich loguru attrs seaborn control colour tqdm \
    opencv-python-headless wandb ipdb
python -m pip install "git+https://github.com/oswinso/jaxproxqp.git"
python -m pip install -e .

echo
echo "Installed. jax sees (login nodes have no GPU, so CPU here is expected):"
python -c "import jax; print(jax.__version__, jax.devices())"
