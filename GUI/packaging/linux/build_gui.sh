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
  --title "Surrogate Model Traning Suite"
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
if command -v pyside6-deploy >/dev/null 2>&1; then
  pyside6-deploy "${deploy_args[@]}"
else
  "$python_bin" -m PySide6.scripts.deploy "${deploy_args[@]}"
fi

echo "Standalone Linux build completed. Inspect $output_dir for the deployment folder."
