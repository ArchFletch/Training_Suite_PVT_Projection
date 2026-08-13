#!/usr/bin/env bash
set -euo pipefail

standalone_dir=""
output_dir=""
install_root="/opt/mlp-training-studio"
bundle_name="mlp-training-studio"
binary_name="mlp-training-studio"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --standalone-dir)
      standalone_dir="$2"
      shift 2
      ;;
    --output-dir)
      output_dir="$2"
      shift 2
      ;;
    --install-root)
      install_root="$2"
      shift 2
      ;;
    --bundle-name)
      bundle_name="$2"
      shift 2
      ;;
    --binary-name)
      binary_name="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$standalone_dir" ]]; then
  echo "--standalone-dir is required" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
standalone_dir="$(cd "$standalone_dir" && pwd)"
if [[ -z "$output_dir" ]]; then
  output_dir="$repo_root/artifacts/packaging/linux"
fi

staging_root="$output_dir/$bundle_name"
tarball_path="$output_dir/${bundle_name}-linux.tar.gz"
desktop_template="$repo_root/packaging/linux/mlp-training-studio.desktop.in"
desktop_output="$staging_root/share/applications/mlp-training-studio.desktop"

rm -rf "$staging_root"
mkdir -p "$staging_root/app" "$staging_root/bin" "$staging_root/share/applications"

cp -R "$standalone_dir"/. "$staging_root/app/"

cat >"$staging_root/bin/mlp-training-studio" <<EOF
#!/usr/bin/env bash
set -euo pipefail
script_dir="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
app_dir="\$(cd "\$script_dir/../app" && pwd)"
exec "\$app_dir/$binary_name" "\$@"
EOF
chmod +x "$staging_root/bin/mlp-training-studio"

sed "s|__INSTALL_ROOT__|$install_root|g" "$desktop_template" >"$desktop_output"

cat >"$staging_root/INSTALL.txt" <<EOF
Extract this bundle to the final install root, for example:

  $install_root

Then launch:

  $install_root/bin/mlp-training-studio

If you want a desktop launcher, copy:

  $install_root/share/applications/mlp-training-studio.desktop

into either:

  /usr/share/applications/
  ~/.local/share/applications/
EOF

mkdir -p "$output_dir"
tar -C "$output_dir" -czf "$tarball_path" "$bundle_name"

echo "Linux bundle created at $tarball_path"
