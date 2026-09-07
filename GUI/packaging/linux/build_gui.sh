#!/usr/bin/env bash
set -euo pipefail

python_bin="python3"
output_dir=""
icon_path=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      python_bin="$2"
      shift 2
      ;;
    --output-dir)
      output_dir="$2"
      shift 2
      ;;
    --icon)
      icon_path="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
if [[ -z "$output_dir" ]]; then
  output_dir="$repo_root/artifacts/packaging/linux"
fi

mkdir -p "$output_dir"

spec_template="$repo_root/packaging/gui/pysidedeploy.spec.in"
spec_path="$output_dir/pysidedeploy.spec"
input_file="$repo_root/launch_gui.py"
python_exe="$("$python_bin" -c 'import sys; print(sys.executable)')"

render_args=(
  "$repo_root/packaging/render_pyside6_spec.py"
  --template "$spec_template"
  --output "$spec_path"
  --title "Surrogate Model Training Suite"
  --project-dir "$repo_root"
  --input-file "$input_file"
  --exec-directory "$output_dir"
  --python-path "$python_exe"
)

if [[ -n "$icon_path" ]]; then
  render_args+=(--icon "$icon_path")
fi

"$python_bin" "${render_args[@]}"

deploy_args=(-c "$spec_path" --name mlp-training-studio --keep-deployment-files --force --verbose)

# Prefer the console script belonging to the interpreter we were told to build with.
# `python -m PySide6.scripts.deploy` is not a usable entry point: deploy.py does a bare
# `from deploy_lib import ...`, which only resolves because the console script puts its
# own directory on sys.path. Probing the bare PATH is wrong for the same reason -- with
# --python pointing at a venv it finds either nothing or another environment's copy, and
# the old fallback then failed with "No module named 'deploy_lib'".
python_bindir="$(dirname "$python_exe")"
if [[ -x "$python_bindir/pyside6-deploy" ]]; then
  deploy_bin="$python_bindir/pyside6-deploy"
elif command -v pyside6-deploy >/dev/null 2>&1; then
  deploy_bin="$(command -v pyside6-deploy)"
else
  echo "pyside6-deploy was not found next to $python_exe or on PATH." >&2
  echo "Install PySide6 into that interpreter, or point --python at one that has it." >&2
  exit 1
fi

"$deploy_bin" "${deploy_args[@]}"

# pyside6-deploy exits 0 even when Nuitka fails: a missing Python.h aborts the C
# backend with "FATAL: Failed unexpectedly in Scons C backend compilation", deploy
# reports the exception and still returns success, and this script would then
# announce a completed build over an empty folder. Verify the artifact instead of
# trusting the exit code.
dist_dir="$output_dir/mlp-training-studio.dist"
if [[ ! -d "$dist_dir" ]]; then
  echo "Build FAILED: $dist_dir was not created. See the Nuitka output above." >&2
  exit 1
fi

missing=()
for entry in mlp-training-studio xfmr_v2 numpy torch pyqtgraph PySide6 matplotlib onnx onnxruntime onnxscript; do
  compgen -G "$dist_dir/$entry*" >/dev/null || missing+=("$entry")
done
# matplotlib needs its mpl-data tree (matplotlibrc, fonts) on disk at runtime, which
# only --include-package-data copies; without it every run dies when it saves a plot.
[[ -d "$dist_dir/matplotlib/mpl-data" ]] || missing+=("matplotlib/mpl-data")

if (( ${#missing[@]} )); then
  echo "Build FAILED: $dist_dir is missing ${missing[*]}." >&2
  echo "Check the --include-package flags in packaging/gui/pysidedeploy.spec.in." >&2
  exit 1
fi

echo "Standalone Linux build completed: $dist_dir ($(du -sh "$dist_dir" | cut -f1))."
