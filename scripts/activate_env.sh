#!/usr/bin/env bash
# Source from any directory: source /path/to/abc/scripts/activate_env.sh
_abc_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export UV_CACHE_DIR="${_abc_repo_root}/.uv/cache"
export UV_PYTHON_INSTALL_DIR="${_abc_repo_root}/.uv/python"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
if [[ -d "${_abc_repo_root}/.uv/ffmpeg/root" ]]; then
    _abc_ffmpeg_root="${_abc_repo_root}/.uv/ffmpeg/root/usr"
    export PATH="${_abc_ffmpeg_root}/bin:${PATH}"
    export LD_LIBRARY_PATH="${_abc_ffmpeg_root}/lib/x86_64-linux-gnu:${_abc_ffmpeg_root}/lib/x86_64-linux-gnu/blas:${_abc_ffmpeg_root}/lib/x86_64-linux-gnu/lapack${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    unset _abc_ffmpeg_root
fi
source "${_abc_repo_root}/.venv/bin/activate"
unset _abc_repo_root
