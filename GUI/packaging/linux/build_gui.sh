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

# pyside6-deploy exits 0 even when Nuitka fails: a missing Python.h, an Anaconda
# python without libpython-static, or a LIBDIR with no libpython3.12.so symlink each
# abort the C backend with "FATAL: Failed unexpectedly in Scons C backend
# compilation", deploy reports the exception and still returns success, and this
# script would then announce a completed build over an empty folder. Verify the
# artifact instead of trusting the exit code.
dist_dir="$output_dir/mlp-training-studio.dist"
if [[ ! -d "$dist_dir" ]]; then
  echo "Build FAILED: $dist_dir was not created. See the Nuitka output above." >&2
  exit 1
fi

# --name sets the .dist FOLDER name, not the executable: Nuitka emits
# <input-stem>.bin, i.e. launch_gui.bin. build_bundle.sh, the launcher it writes and
# the .desktop Exec= line all expect "mlp-training-studio", so normalise it here
# instead of leaving every consumer to guess. Without this, build_bundle.sh refuses
# the dist with "Standalone directory has no 'mlp-training-studio'".
binary_path=""
for candidate in "$dist_dir/mlp-training-studio" "$dist_dir/launch_gui.bin" "$dist_dir/launch_gui"; do
  if [[ -f "$candidate" ]]; then binary_path="$candidate"; break; fi
done
if [[ -z "$binary_path" ]]; then
  echo "Build FAILED: no executable found in $dist_dir. See the Nuitka output above." >&2
  exit 1
fi
if [[ "$binary_path" != "$dist_dir/mlp-training-studio" ]]; then
  mv "$binary_path" "$dist_dir/mlp-training-studio"
  echo "Renamed $(basename "$binary_path") to mlp-training-studio for the bundler and the .desktop entry."
fi
chmod +x "$dist_dir/mlp-training-studio"

# Packages that MUST be on disk: they ship extension modules or data files that
# cannot be compiled into the executable. Pure-Python packages (xfmr_v2, onnxscript)
# are compiled INTO the binary and deliberately absent here -- requiring them as
# directories reported a false failure on a perfectly good build.
missing=()
for entry in numpy torch pyqtgraph PySide6 matplotlib onnx onnxruntime; do
  compgen -G "$dist_dir/$entry*" >/dev/null || missing+=("$entry")
done
# matplotlib needs its mpl-data tree (matplotlibrc, fonts) on disk at runtime, which
# only --include-package-data copies; without it every run dies when it saves a plot.
[[ -d "$dist_dir/matplotlib/mpl-data" ]] || missing+=("matplotlib/mpl-data")

# The compiled-in packages are verified by looking for their module names in the
# binary, which is where Nuitka puts them. The symbol table is dumped once to a
# temp file rather than piped per entry: `strings | grep -q` makes grep exit early,
# strings then dies of SIGPIPE, and `set -o pipefail` turns that into a spurious
# build failure on a perfectly good binary.
symbols_file="$(mktemp)"
trap 'rm -f "$symbols_file"' EXIT
strings -a "$dist_dir/mlp-training-studio" >"$symbols_file" 2>/dev/null || true
for entry in xfmr_v2.gui_window xfmr_v2.runner onnxscript; do
  grep -qF "$entry" "$symbols_file" || missing+=("$entry (compiled-in)")
done

if (( ${#missing[@]} )); then
  echo "Build FAILED: $dist_dir is missing ${missing[*]}." >&2
  echo "Check the --include-package flags in packaging/gui/pysidedeploy.spec.in." >&2
  exit 1
fi

echo "Standalone Linux build completed: $dist_dir ($(du -sh "$dist_dir" | cut -f1))."
echo "Executable: $dist_dir/mlp-training-studio"
