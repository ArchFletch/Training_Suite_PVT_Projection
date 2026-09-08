#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  build_bundle.sh [options]

Options:
  --output-dir PATH     Directory receiving the staged bundle and the tarball.
                        Default: <repo>/artifacts/packaging/license_server/linux
  --bundle-name NAME    Staged directory and archive base name.
                        Default: mlp-license-server
  --skip-archive        Stage the bundle directory without writing the tarball
  --help                Show this help text
EOF
}

output_dir=""
bundle_name="mlp-license-server"
skip_archive=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      output_dir="$2"
      shift 2
      ;;
    --bundle-name)
      bundle_name="$2"
      shift 2
      ;;
    --skip-archive)
      skip_archive=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

# An empty bundle name would turn the staging cleanup below into "rm -rf <output_dir>/".
# The rest of the character restriction keeps the name safe in the three places it is
# interpolated: the sed replacement that renders INSTALL.md, the tar member name, and
# the extraction commands printed in INSTALL.md itself. A name with spaces renders
# "tar -xzf MLP License Server-linux.tar.gz", which the customer cannot run.
if [[ -z "$bundle_name" ]]; then
  echo "--bundle-name must not be empty" >&2
  exit 2
fi
if [[ ! "$bundle_name" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "--bundle-name may only contain letters, digits, dot, underscore and hyphen: $bundle_name" >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir"/../../.. && pwd)"
if [[ -z "$output_dir" ]]; then
  output_dir="$repo_root/artifacts/packaging/license_server/linux"
fi

mkdir -p "$output_dir"
output_dir="$(cd "$output_dir" && pwd)"

package_source="$repo_root/license_server"
install_script_source="$script_dir/install_linux_service.sh"
env_sample_source="$script_dir/mlp-license-server.env.sample"
unit_template_source="$script_dir/mlp-license-server.service.template"
config_sample_source="$repo_root/packaging/license_server/shared/config.linux.toml.sample"
install_doc_source="$script_dir/INSTALL.md.template"

for required in \
  "$install_script_source" \
  "$env_sample_source" \
  "$unit_template_source" \
  "$config_sample_source" \
  "$install_doc_source" \
  "$package_source/requirements.txt"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing bundle input: $required" >&2
    exit 1
  fi
done

staging_root="$output_dir/$bundle_name"
package_root="$staging_root/license_server"
tarball_path="$output_dir/${bundle_name}-linux.tar.gz"

rm -rf "$staging_root"
if [[ $skip_archive -eq 0 ]]; then
  rm -f "$tarball_path"
fi

# The staged tree is the whole customer delivery, so only the files copied below
# ship. license_vendor/ (the vendor signing tooling), tests and internal reports
# are never staged; keeping them off customer hosts is the point of this script.
mkdir -p "$staging_root/linux" "$staging_root/shared"
cp -R "$package_source" "$staging_root/"
find "$package_root" -type d -name '__pycache__' -prune -exec rm -rf {} +
find "$package_root" -type f -name '*.pyc' -delete

# install_linux_service.sh resolves its templates from its own directory and
# from ../shared, so those two directories stay side by side here and the
# customer can run the script straight out of the extracted tarball.
install -m 0755 "$install_script_source" "$staging_root/linux/install_linux_service.sh"
install -m 0644 "$env_sample_source" "$staging_root/linux/mlp-license-server.env.sample"
install -m 0644 "$unit_template_source" "$staging_root/linux/mlp-license-server.service.template"
install -m 0644 "$config_sample_source" "$staging_root/shared/config.linux.toml.sample"

sed \
  -e "s|__BUNDLE_NAME__|${bundle_name}|g" \
  -e "s|__TARBALL_NAME__|$(basename "$tarball_path")|g" \
  "$install_doc_source" > "$staging_root/INSTALL.md"
chmod 0644 "$staging_root/INSTALL.md"

# license_server/ is copied wholesale, so this guard has to cover more than the
# directories that should never be staged: a working tree can hold untracked or
# git-ignored files, and the vendor commands in license_vendor/README.md write
# their key material with fixed names into whatever directory they are run from.
# These are the same names the repo .gitignore refuses to track.
leaked="$(find "$staging_root" \( \
  -name 'license_vendor' -o -name 'tests' -o -name '__pycache__' -o -name '*.pyc' \
  -o -name '*signing_key*' -o -name '*.pem' -o -name '*.jsonl' -o -name 'license.json' \
  \) -print)"
if [[ -n "$leaked" ]]; then
  echo "Refusing to package, excluded paths present under $staging_root:" >&2
  echo "$leaked" >&2
  exit 1
fi

if [[ $skip_archive -eq 0 ]]; then
  # --mode='go-w' strips group and world write from every recorded member. It is a
  # no-op for a build on a native filesystem, and it is what makes a build run from
  # a Windows mount safe to ship: drvfs reports every file 0777 and silently ignores
  # chmod, so the staged tree cannot be corrected in place. Without this the customer
  # extracts a license_server/ whose .py files any local user can rewrite.
  # --owner/--group/--numeric-owner: without them tar records the build host's
  # numeric uid/gid, and `sudo tar -xzf ... -C /opt` reproduces them on the
  # customer's machine -- every file showed as UNKNOWN:users (81256:100).
  # It still runs, because the modes are 0755, but it reads as a broken install
  # and trips file-integrity tooling. root:root is what a system package ships.
  tar -C "$output_dir" --mode='go-w' --owner=0 --group=0 --numeric-owner \
    -czf "$tarball_path" "$bundle_name"
fi

archive_summary="$tarball_path"
if [[ $skip_archive -eq 1 ]]; then
  archive_summary="skipped (--skip-archive)"
fi

cat <<EOF
Staged Linux license server bundle:
  bundle root:    $staging_root
  bundle archive: $archive_summary
  package root:   $package_root
  install script: $staging_root/linux/install_linux_service.sh
  install notes:  $staging_root/INSTALL.md

The bundle carries no vendor signing tooling: license_vendor/ is never staged.
EOF
