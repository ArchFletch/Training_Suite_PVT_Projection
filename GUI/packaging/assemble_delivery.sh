#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  assemble_delivery.sh [options]

Collects the customer-facing artifacts already built under artifacts/packaging
into one delivery tree grouped by recipient, and reports the artifacts this
machine has not built. A missing artifact is not an error.

No single host builds all four artifacts, so one --output-dir is normally filled
in by several runs on several machines. An artifact that is already in the tree
but cannot be rebuilt here is kept, checksummed and reported as CARRIED OVER.
This script never deletes a destination it is not about to replace, so running
it on a machine with a partial toolchain cannot destroy another machine's
contribution. --clean does discard the whole tree, on purpose.

Options:
  --output-dir PATH     Directory receiving the delivery tree.
                        Default: <repo>/artifacts/delivery
  --clean               Remove a previous delivery tree before assembling.
                        Discards artifacts other machines contributed.
  --help                Show this help text

Exit status:
  0  tree assembled; every file in it is accounted for
  1  hard failure (vendor key material found, or a copy/checksum failed)
  2  usage error
  3  tree assembled, but the recipient directories hold files this run did not
     place. They are listed in the console output, in checksums.txt and in the
     generated README.md. Nothing was deleted; the tree must not ship until
     they are explained, removed, or the tree is rebuilt with --clean.
EOF
}

# A flag whose "value" is the next flag silently ate that flag in an earlier
# version and then died at mkdir with exit 1. Anything starting with a dash is
# rejected as a value: it is far likelier to be a mistyped command line than a
# path, and a path that really does start with a dash can be written ./-name.
require_value() {
  local flag="$1"
  local remaining="$2"
  local value="${3-}"

  if [[ $remaining -lt 2 ]]; then
    echo "$flag requires a value" >&2
    usage >&2
    exit 2
  fi
  if [[ -z "$value" ]]; then
    echo "$flag requires a non-empty value" >&2
    usage >&2
    exit 2
  fi
  case "$value" in
    -*)
      echo "$flag requires a value, but the next argument is another option: $value" >&2
      echo "For a path that really begins with a dash, write it as ./$value" >&2
      usage >&2
      exit 2
      ;;
  esac
}

output_dir=""
clean=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      require_value "$1" "$#" "${2-}"
      output_dir="$2"
      shift 2
      ;;
    --clean)
      clean=1
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

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir"/.. && pwd)"
if [[ -z "$output_dir" ]]; then
  output_dir="$repo_root/artifacts/delivery"
fi

mkdir -p "$output_dir"
output_dir="$(cd "$output_dir" && pwd)"

engineers_dir="$output_dir/engineers-gui"
server_dir="$output_dir/server-admin"
readme_path="$output_dir/README.md"
checksums_path="$output_dir/checksums.txt"

if [[ $clean -eq 1 ]]; then
  # Only the fixed paths this script writes are removed. "rm -rf $output_dir" would
  # take out anything else a release manager parked in a --output-dir of their own.
  rm -rf "$engineers_dir" "$server_dir"
  rm -f "$readme_path" "$checksums_path"
fi

# The recipient directories are created on demand, when something is actually put
# in them. An empty engineers-gui/ next to an empty checksums.txt reads like a
# delivery with nothing wrong in it.

placed_count=0
carried_count=0
missing_count=0
missing_labels=""
carried_labels=""
checksum_lines=""
readme_body=""
# Newline-delimited list of output-dir-relative paths this run vouches for.
accounted_rels=""

readme_line() {
  readme_body+="$1"$'\n'
}

print_indented() {
  local prefix="$1"
  local line
  while IFS= read -r line; do
    printf '%s%s\n' "$prefix" "$line"
  done <<<"$2"
}

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum -- "$1" | cut -d' ' -f1
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 -- "$1" | cut -d' ' -f1
  else
    echo "Neither sha256sum nor shasum is available, cannot checksum $1" >&2
    return 1
  fi
}

size_of() {
  wc -c < "$1" | tr -d '[:space:]'
}

# These are the names license_vendor/README.md writes: vendor_signing_key.json,
# issuance_log.jsonl, vendor_public_key.pem. They are the same names the repo refuses
# to track and the same set build_bundle.sh refuses to stage. This script is the one
# place that assembles a complete customer shipment, so the guard belongs here too.
reject_key_material() {
  case "$1" in
    *signing_key*|*.pem|*.jsonl|license.json)
      echo "Refusing to copy vendor key material into the delivery tree: $1" >&2
      exit 1
      ;;
  esac
}

# Documentation the recipient needs in their hands. The README used to point
# engineers at `doc/customer/gui_install_run.md`, which is in the repo and not in
# the delivery -- so the one instruction they were given named a file they did not
# have. Docs are checksummed alongside the archives so the tree stays fully
# accounted for.
collect_doc() {
  local recipient="$1"
  local source_rel="$2"

  local source_path="$repo_root/$source_rel"
  local base; base="$(basename "$source_rel")"
  local dest="$output_dir/$recipient/$base"
  local rel="$recipient/$base"

  reject_key_material "$base"

  if [[ ! -f "$source_path" ]]; then
    echo "Missing delivery document: $source_rel" >&2
    exit 1
  fi
  mkdir -p "$output_dir/$recipient"
  if ! cp "$source_path" "$dest"; then
    echo "Failed to copy $source_rel into the delivery tree" >&2
    exit 1
  fi
  local digest; digest="$(sha256_of "$dest")"
  checksum_lines+="$digest  $rel"$'\n'
  accounted_rels+="$rel"$'\n'
  printf '  INCLUDED %s\n' "$rel"
}

collect() {
  local recipient="$1"
  local label="$2"
  local source_rel="$3"
  local build_command="$4"
  local toolchain="$5"

  local source_path="$repo_root/$source_rel"
  local base
  base="$(basename "$source_rel")"
  local recipient_dir="$output_dir/$recipient"
  local dest="$recipient_dir/$base"
  local rel="$recipient/$base"

  reject_key_material "$base"

  readme_line "### $label"
  readme_line ""

  local digest size

  if [[ ! -f "$source_path" ]]; then
    if [[ -f "$dest" ]]; then
      # Built elsewhere and already contributed to this shared tree. Deleting it
      # here, on a machine that cannot rebuild it, would make assembling the tree
      # from several hosts impossible. Keep it, checksum what is actually on disk,
      # and say plainly that this run did not produce it.
      digest="$(sha256_of "$dest")"
      size="$(size_of "$dest")"
      carried_count=$((carried_count + 1))
      carried_labels+="    $label ($rel)"$'\n'
      checksum_lines+="$digest  $rel"$'\n'
      accounted_rels+="$rel"$'\n'

      printf '  CARRIED  %s\n' "$label"
      printf '           %s\n' "$rel"
      printf '           not built on this machine, kept from an earlier run\n'
      printf '           sha256 %s\n' "$digest"
      printf '           %s bytes\n' "$size"

      readme_line "Status: PRESENT, carried over."
      readme_line ""
      readme_line "The machine that assembled this tree did not build this artifact. The copy"
      readme_line "below was already in the tree, contributed by another build host, and was"
      readme_line "kept and re-checksummed as it sits on disk. Confirm the SHA256 against the"
      readme_line "machine that built it before shipping."
      readme_line ""
      readme_line "File: \`$rel\`"
      readme_line ""
      readme_line "SHA256: \`$digest\`"
      readme_line ""
      return 0
    fi

    missing_count=$((missing_count + 1))
    missing_labels+="    $label"$'\n'

    printf '  MISSING  %s\n' "$label"
    printf '           expected at %s\n' "$source_rel"
    printf '           nothing at %s either\n' "$rel"
    printf '           build with:\n'
    print_indented '             ' "$build_command"
    printf '           needs: %s\n' "$toolchain"

    readme_line "Status: MISSING. Not built on the machine that assembled this tree, and no"
    readme_line "copy from another machine was present in the tree either."
    readme_line ""
    readme_line "Build:"
    readme_line ""
    readme_line '```'
    readme_line "$build_command"
    readme_line '```'
    readme_line ""
    readme_line "Needs: $toolchain"
    readme_line ""
    return 0
  fi

  mkdir -p "$recipient_dir"

  # Stage beside the destination, then rename over it. The destination is removed
  # only at the moment a complete replacement exists, so a failed copy cannot leave
  # the tree short of an artifact this machine can no longer supply.
  local staged="$dest.incoming"
  rm -f "$staged"
  if ! install -m 0644 "$source_path" "$staged"; then
    rm -f "$staged"
    echo "Failed to copy $source_rel into the delivery tree" >&2
    exit 1
  fi
  mv -f "$staged" "$dest"

  digest="$(sha256_of "$dest")"

  placed_count=$((placed_count + 1))
  checksum_lines+="$digest  $rel"$'\n'
  accounted_rels+="$rel"$'\n'

  printf '  PRESENT  %s\n' "$label"
  printf '           %s\n' "$rel"
  printf '           sha256 %s\n' "$digest"

  readme_line "Status: PRESENT, built on the machine that assembled this tree."
  readme_line ""
  readme_line "File: \`$rel\`"
  readme_line ""
  readme_line "SHA256: \`$digest\`"
  readme_line ""
}

windows_server_zip_rel="artifacts/packaging/license_server/windows/MLP License Server.zip"
windows_server_zip_dest="$server_dir/$(basename "$windows_server_zip_rel")"

printf 'Repo root:     %s\n' "$repo_root"
printf 'Delivery tree: %s\n' "$output_dir"
printf '\n'

readme_line "## engineers-gui/"
readme_line ""
readme_line "Goes to the design engineers who run the desktop app, one file per platform."
readme_line "Install and first-run notes for them are in \`engineers-gui/gui_install_run.md\`,"
readme_line "included in this delivery. On Linux, read its \"Linux System Prerequisites\" section"
readme_line "before the first launch: an X11 desktop needs system libraries a stock install may"
readme_line "not have, and Qt's own error names the wrong one. The same list is in the bundle's"
readme_line "\`INSTALL.txt\`, and the launcher checks it and names what is actually missing."
readme_line "The app needs the license server below to be running before it can take a seat."
readme_line ""

printf 'engineers-gui/  (design engineers running the desktop app)\n'

collect "engineers-gui" "Windows desktop installer" \
  "artifacts/packaging/windows/installer/SurrogateModelTrainingSuite-Windows.exe" \
  'packaging\windows\build_gui.ps1
packaging\windows\build_installer.ps1 -StandaloneDir artifacts\packaging\windows\SurrogateModelTrainingSuite.dist' \
  'Windows host, PowerShell, a Python with the GUI dependencies, PySide6, Nuitka, Inno Setup 6'

readme_line "Two Linux bundles are provided. They are the same application and extract to the"
readme_line "same directory; they differ only in the bundled PyTorch build, which decides which"
readme_line "GPUs can be used. No single PyTorch wheel covers both ends of the range, so pick by"
readme_line "the oldest GPU generation in the fleet:"
readme_line ""
readme_line "| Bundle | GPU architectures | Cards |"
readme_line "| --- | --- | --- |"
readme_line "| \`...-cu128.tar.gz\` | sm_75 - sm_120 | Turing, Ampere, Ada, Hopper, Blackwell (RTX 50-series) |"
readme_line "| \`...-cu121.tar.gz\` | sm_50 - sm_90 | Maxwell, Pascal, Volta, Turing, Ampere, Ada, Hopper |"
readme_line ""
readme_line "A card outside the bundled range still runs the app; training falls back to the CPU"
readme_line "and the app prints a \"CUDA capability sm_NNN is not compatible\" warning at startup."
readme_line "Install only one: both extract to the same path and would overwrite each other."
readme_line ""

collect "engineers-gui" "Linux desktop bundle (cu128 - Turing through Blackwell)" \
  "artifacts/packaging/linux-cu128/mlp-training-studio-linux-cu128.tar.gz" \
  'pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128   # pinned: 2.11.0 crashes Nuitka in torch/_dynamo/pgo.py
bash packaging/linux/build_gui.sh --python <build venv python> --output-dir artifacts/packaging/linux-cu128
bash packaging/linux/build_bundle.sh --standalone-dir <deployment directory reported by build_gui.sh> \
  --output-dir artifacts/packaging/linux-cu128 --tarball-name mlp-training-studio-linux-cu128.tar.gz' \
  'Linux host, a Python with the GUI dependencies (numpy 2.3.4 -- see the PEP 695 guard in build_gui.sh), PySide6 (pyside6-deploy), Nuitka'

collect "engineers-gui" "Linux desktop bundle (cu121 - Maxwell through Hopper)" \
  "artifacts/packaging/linux-cu121/mlp-training-studio-linux-cu121.tar.gz" \
  'pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
bash packaging/linux/build_gui.sh --python <build venv python> --output-dir artifacts/packaging/linux-cu121
bash packaging/linux/build_bundle.sh --standalone-dir <deployment directory reported by build_gui.sh> \
  --output-dir artifacts/packaging/linux-cu121 --tarball-name mlp-training-studio-linux-cu121.tar.gz' \
  'Linux host, a Python with the GUI dependencies (numpy 2.3.4 -- see the PEP 695 guard in build_gui.sh), PySide6 (pyside6-deploy), Nuitka'

collect_doc "engineers-gui" "doc/customer/gui_install_run.md"

readme_line "## server-admin/"
readme_line ""
readme_line "Goes to customer IT, who installs the on-prem floating license server on one LAN"
readme_line "host. Send the file matching the server OS. The Linux bundle carries its own"
readme_line "\`INSTALL.md\`; the Windows bundle carries \`install-service.ps1\`. This half of the"
readme_line "delivery goes out first: no engineer gets a seat until the server is licensed."
readme_line ""

printf 'server-admin/   (customer IT installing the floating license server)\n'

collect "server-admin" "Windows license server bundle" \
  "$windows_server_zip_rel" \
  'packaging\license_server\windows\build_bundle.ps1 -WinSWExePath <path to WinSW-x64.exe>' \
  'Windows host, PowerShell, Python 3.12 (its base prefix is copied into the bundle), a downloaded WinSW x64 executable, and network access to PyPI (the build runs ensurepip and pip install -r runtime_requirements.txt into the bundled runtime)'

collect "server-admin" "Linux license server bundle" \
  "artifacts/packaging/license_server/linux/mlp-license-server-linux.tar.gz" \
  'bash packaging/license_server/linux/build_bundle.sh' \
  'any host with bash and tar, no Python or Qt toolchain, no network access'

present_count=$((placed_count + carried_count))
total_count=$((present_count + missing_count))

# The copies above are name-checked one by one, but the tree can also hold files from a
# previous run or dropped in by hand, and a delivery tree is exactly where someone would
# stage a vendor response. Re-check everything that is about to be handed over.
leaked="$(find "$output_dir" \( \
  -name '*signing_key*' -o -name '*.pem' -o -name '*.jsonl' -o -name 'license.json' \
  -o -name 'license_vendor' \
  \) -print)"
if [[ -n "$leaked" ]]; then
  echo "Refusing to publish the delivery tree, vendor key material present under $output_dir:" >&2
  echo "$leaked" >&2
  exit 1
fi

# Everything above is matched by destination name. Anything else in the recipient
# directories -- last release's installer, a renamed hand-built copy, a leftover
# .incoming from a failed run -- is invisible to that matching, stays out of
# checksums.txt and out of the README, and ships anyway.
#
# It is not removed. Deleting files this script did not write is the same mistake
# as deleting a destination it cannot rebuild, and a release manager may have
# parked something there deliberately. It is not merely warned about either: a
# console warning scrolls past and CI ignores it. So the tree is assembled in
# full, the strays are recorded in the two places anyone verifying this tree
# reads (checksums.txt and the delivery README), and the run exits 3 so nothing
# automated can treat the tree as shippable.
scan_dirs=()
if [[ -d "$engineers_dir" ]]; then
  scan_dirs+=("$engineers_dir")
fi
if [[ -d "$server_dir" ]]; then
  scan_dirs+=("$server_dir")
fi

unaccounted_rels=""
unaccounted_count=0
if [[ ${#scan_dirs[@]} -gt 0 ]]; then
  while IFS= read -r -d '' found_path; do
    found_rel="${found_path#"$output_dir/"}"
    if ! printf '%s' "$accounted_rels" | grep -Fxq -- "$found_rel"; then
      unaccounted_count=$((unaccounted_count + 1))
      unaccounted_rels+="$found_rel"$'\n'
    fi
  done < <(find "${scan_dirs[@]}" \( -type f -o -type l \) -print0)
fi

if [[ $present_count -eq 0 ]]; then
  # Nothing was collected. A zero-byte checksums.txt is not an empty manifest, it
  # is a file that makes "sha256sum -c checksums.txt" fail with "no properly
  # formatted SHA256 checksum lines found". Do not write one, and drop a stale one
  # from an earlier run, since it can only describe files that are no longer
  # vouched for.
  rm -f "$checksums_path"
else
  {
    if [[ $unaccounted_count -gt 0 ]]; then
      # sha256sum -c and shasum -c skip lines beginning with '#', so the warning
      # travels with the manifest without breaking verification.
      printf '# WARNING: this delivery tree is not ready to ship.\n'
      printf '# %d file(s) in the recipient directories were not placed by\n' "$unaccounted_count"
      printf '# packaging/assemble_delivery.sh and are NOT listed below:\n'
      printf '%s' "$unaccounted_rels" | while IFS= read -r stray; do
        printf '#   %s\n' "$stray"
      done
      printf '# Verifying this file therefore does not cover the whole tree.\n'
      printf '# Remove them, or re-assemble with --clean, then re-run the script.\n'
    fi
    printf '%s' "$checksum_lines"
  } > "$checksums_path"
  chmod 0644 "$checksums_path"
fi

if [[ $missing_count -eq 0 ]]; then
  status_line="All $total_count expected artifacts are present in this tree."
elif [[ $missing_count -eq 1 ]]; then
  status_line="$present_count of $total_count expected artifacts are present. The remaining one is marked MISSING below and still has to be built."
else
  status_line="$present_count of $total_count expected artifacts are present. The remaining $missing_count are marked MISSING below and still have to be built."
fi
if [[ $carried_count -gt 0 ]]; then
  status_line="$status_line $carried_count of the present artifacts were carried over from an earlier run on another machine rather than built for this one."
fi

{
  cat <<'EOF'
# Customer Delivery

Everything under this directory goes to a customer. It is assembled by
`packaging/assemble_delivery.sh` from whatever has already been built under
`artifacts/packaging/`, plus whatever earlier runs on other machines already
contributed here. This file and `checksums.txt` are rewritten on every run, so
nothing here should be edited by hand.

The two folders go to two different people at the customer site, usually at
different times. Build commands quoted below run from the repository root.

EOF
  printf '%s\n\n' "$status_line"
  printf '%s' "$readme_body"

  if [[ $unaccounted_count -gt 0 ]]; then
    cat <<'EOF'
## Files this tree does not account for

DO NOT SHIP until these are explained. The files below are in the recipient
directories but were not placed by the assembly script, so they are absent from
`checksums.txt` and from the artifact list above, and no one has said what they
are. A stale installer from a previous release looks exactly like this.

EOF
    printf '%s' "$unaccounted_rels" | while IFS= read -r stray; do
      printf -- '- `%s`\n' "$stray"
    done
    cat <<'EOF'

Delete them, or re-assemble the tree with `--clean` on a machine that can
rebuild every artifact, then re-run `packaging/assemble_delivery.sh`.

EOF
  fi

  cat <<'EOF'
## License handshake

`license.json` and `vendor_public_key.pem` are not in this tree and cannot be.
They are issued per customer against one specific server, which has to exist
first:

1. The server admin installs the `server-admin/` bundle, runs `init`, then
   `export-request`, which writes `license_request.json`. That file carries a
   server ID, a host fingerprint, the hostname and the OS family, nothing else.
2. The admin sends `license_request.json` back to the vendor.
3. The vendor signs a license for that request and returns `license.json` plus
   `vendor_public_key.pem` out of band. The public key is delivered once per
   customer; renewals reuse it.
4. The admin imports both on the server, then enables and starts the service.

Step 3 cannot be done in advance, so every delivery is two shipments: this tree
first, the license files after the request comes back. The signing key never
leaves the vendor.

## Checksums

EOF

  if [[ $present_count -eq 0 ]]; then
    cat <<'EOF'
No artifact was collected into this tree, so there is no `checksums.txt`. One is
written on the first run that finds something to collect.
EOF
  else
    cat <<'EOF'
`checksums.txt` has one line per file collected here. Verify from this
directory:

```bash
sha256sum -c checksums.txt
```
EOF
    if [[ $unaccounted_count -gt 0 ]]; then
      cat <<'EOF'

That check covers only the files listed in `checksums.txt`. It says nothing
about the unaccounted files above, which are also in this tree.
EOF
    fi
  fi

  cat <<'EOF'

## Never shipped

The vendor signing tooling (`license_vendor/`), `vendor_signing_key.json` and
the issuance log are not part of any delivery. The assembly script refuses to
copy `*signing_key*`, `*.pem`, `*.jsonl` and `license.json` into this tree and
fails if it finds a file with one of those names here.

That guard matches file names in this tree. It does not look inside the
installer, the zip or the tarballs: what goes into each archive is the job of
the build script that stages it, and the Windows server zip, whose build has no
such guard, is reviewed by hand before a release goes out.
EOF
} > "$readme_path"
chmod 0644 "$readme_path"

cat <<EOF

Assembled $present_count of $total_count customer artifacts.
  delivery tree:  $output_dir
  delivery notes: $readme_path
EOF
if [[ $present_count -eq 0 ]]; then
  printf '  checksums:      not written, nothing was collected\n'
else
  printf '  checksums:      %s\n' "$checksums_path"
fi
cat <<EOF
  built here:     $placed_count
  carried over:   $carried_count
  missing:        $missing_count
EOF

if [[ $carried_count -gt 0 ]]; then
  printf '\nCarried over from an earlier run:\n'
  printf '%s' "$carried_labels"
  cat <<'EOF'

Carried-over artifacts were already in the delivery tree and could not be built
on this machine, so they were kept and checksummed as found. Nothing is deleted
just because this host lacks the toolchain for it. Check their SHA256 against
the machine that built them.
EOF
fi

if [[ $missing_count -gt 0 ]]; then
  printf '\nNot built here and not in the tree:\n'
  printf '%s' "$missing_labels"
  cat <<'EOF'

Missing artifacts are not an error: each machine can only build the ones its
toolchain supports. Build the entries above on a host that has them and re-run
this script against the same --output-dir, which collects them alongside what is
already here.
EOF
fi

cat <<'EOF'

Vendor key material: file names were checked, archive contents were not.
  Checked:     every path under the delivery tree against *signing_key*, *.pem,
               *.jsonl, license.json and license_vendor. Nothing matched.
  Not checked: what is inside the .exe, .zip and .tar.gz files. This script does
               not open them, so it cannot vouch for their contents.
EOF

if [[ -f "$windows_server_zip_dest" ]]; then
  cat <<'EOF'
  Open by hand before shipping: server-admin/"MLP License Server.zip".
               packaging/license_server/windows/build_bundle.ps1 copies the build
               machine's whole license_server/ directory into that zip and has no
               leak guard of its own, so a vendor_public_key.pem, license.json or
               issuance_log.jsonl left in that directory during development
               would be inside the archive. The Linux server bundle needs no
               such check:
               packaging/license_server/linux/build_bundle.sh applies the same
               name guard to its staging before it writes the tarball.
EOF
fi

cat <<'EOF'

license.json and vendor_public_key.pem reach the customer out of band, after the
handshake described in the delivery README.
EOF

if [[ $unaccounted_count -gt 0 ]]; then
  cat <<EOF

NOT SHIPPABLE: $unaccounted_count file(s) in the recipient directories were not placed by
this run. They are not in checksums.txt and not described in the delivery
README's artifact list, but they are in the tree and would go out with it:
EOF
  printf '%s' "$unaccounted_rels" | while IFS= read -r stray; do
    printf '    %s\n' "$stray"
  done
  cat <<'EOF'

Nothing was deleted. Identify each one, then either remove it or re-assemble
with --clean, and re-run this script.
EOF
  exit 3
fi
