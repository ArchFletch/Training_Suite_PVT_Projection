#!/usr/bin/env bash
set -euo pipefail

standalone_dir=""
output_dir=""
install_root="/opt/mlp-training-studio"
bundle_name="mlp-training-studio"
binary_name="mlp-training-studio"
# Archive filename, separate from bundle_name on purpose. Two bundles built from
# different torch wheels must extract to the SAME directory, carry the same
# launcher and the same desktop entry -- the install docs and StartupWMClass name
# that path -- and differ only in the file the customer downloads.
tarball_name=""

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
    --tarball-name)
      tarball_name="$2"
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

# The launcher written below execs "$binary_name" out of app/, so a standalone
# directory without it yields a bundle that cannot start. That is not theoretical:
# when the Nuitka step fails, pyside6-deploy still leaves an empty .dist behind and
# exits, and this script happily packaged it into a 690-byte tarball and reported
# success -- a dead bundle that looks like a delivery.
if [[ ! -f "$standalone_dir/$binary_name" ]]; then
  echo "Standalone directory has no '$binary_name': $standalone_dir" >&2
  echo "The GUI build did not complete. Refusing to package an unusable bundle." >&2
  exit 1
fi

staging_root="$output_dir/$bundle_name"
tarball_path="$output_dir/${tarball_name:-${bundle_name}-linux.tar.gz}"
desktop_template="$repo_root/packaging/linux/mlp-training-studio.desktop.in"
desktop_output="$staging_root/share/applications/mlp-training-studio.desktop"

rm -rf "$staging_root"
mkdir -p "$staging_root/app" "$staging_root/bin" "$staging_root/share/applications"

cp -R "$standalone_dir"/. "$staging_root/app/"

# The launcher checks the host's X11 libraries before handing over to Qt. Qt's own
# diagnostic misnames the culprit -- a missing libxcb-icccm or libxcb-keysyms makes
# it print "From 6.5.0, xcb-cursor0 or libxcb-cursor0 is needed" -- and only
# QT_DEBUG_PLUGINS=1 reveals the real one. Checking here names the right package.
requirements_file="$repo_root/packaging/linux/x11_runtime_requirements.txt"
if [[ ! -f "$requirements_file" ]]; then
  echo "Missing bundle input: $requirements_file" >&2
  exit 1
fi
required_sonames="$(grep -vE '^\s*#|^\s*$' "$requirements_file" | awk '$4 == "required" {print $1}' | tr '\n' ' ')"
deb_packages="$(grep -vE '^\s*#|^\s*$' "$requirements_file" | awk '$4 == "required" {print $2}' | sort -u | tr '\n' ' ')"
rpm_packages="$(grep -vE '^\s*#|^\s*$' "$requirements_file" | awk '$4 == "required" {print $3}' | sort -u | tr '\n' ' ')"

cat >"$staging_root/bin/mlp-training-studio" <<EOF
#!/usr/bin/env bash
set -euo pipefail
script_dir="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
app_dir="\$(cd "\$script_dir/../app" && pwd)"

# Preflight: an X11 session needs these from the host. Skipped under Wayland, where
# Qt uses its own plugin and none of them are involved -- which is also why this
# class of failure is invisible on a Wayland desktop and only bites in X11.
#
# Wayland is detected by WAYLAND_DISPLAY as well as XDG_SESSION_TYPE. Testing this
# on XDG_SESSION_TYPE alone made the check block a perfectly runnable app: WSLg,
# bare compositors, \`su\` into another user and systemd-user launches all leave
# XDG_SESSION_TYPE unset while still exporting WAYLAND_DISPLAY, so the check ran,
# found the X11 helpers absent, and exited 1 on a session that never needed them.
if [[ "\${MLP_SKIP_LIBRARY_CHECK:-0}" != "1" \\
   && -z "\${WAYLAND_DISPLAY:-}" \\
   && "\${XDG_SESSION_TYPE:-}" != "wayland" \\
   && -n "\${DISPLAY:-}" ]]; then
  # The cache is read ONCE into a variable. Piping it per soname into \`grep -q\`
  # makes grep exit on the first match, ldconfig then dies of SIGPIPE, and the
  # script's own \`set -o pipefail\` turns that into "missing" for every library --
  # which reported 22 missing files on a host that only lacked one.
  library_cache="\$( { /sbin/ldconfig -p 2>/dev/null || ldconfig -p 2>/dev/null || true; } )"
  missing=()
  for soname in $required_sonames; do
    case "\$library_cache" in
      *"\$soname"*) ;;
      *) missing+=("\$soname") ;;
    esac
  done
  if (( \${#missing[@]} )); then
    echo "Surrogate Model Training Suite cannot start: your system is missing \${#missing[@]} library file(s)" >&2
    echo "the Qt X11 plugin needs:" >&2
    printf '  %s\n' "\${missing[@]}" >&2
    echo "" >&2
    echo "Install them with one of:" >&2
    echo "  Debian/Ubuntu:  sudo apt install $deb_packages" >&2
    echo "  RHEL/Rocky 8:   sudo dnf install $rpm_packages" >&2
    echo "                  (xcb-util-cursor is in EPEL on RHEL 8: sudo dnf install epel-release)" >&2
    echo "" >&2
    echo "Qt's own error message names xcb-cursor whatever is actually missing, so trust" >&2
    echo "the list above. To bypass this check: MLP_SKIP_LIBRARY_CHECK=1" >&2
    exit 1
  fi
fi

exec "\$app_dir/$binary_name" "\$@"
EOF
chmod +x "$staging_root/bin/mlp-training-studio"

sed "s|__INSTALL_ROOT__|$install_root|g" "$desktop_template" >"$desktop_output"

cat >"$staging_root/INSTALL.txt" <<EOF
Extract this bundle to the final install root, for example:

  $install_root

Then launch:

  $install_root/bin/mlp-training-studio

SYSTEM REQUIREMENTS (X11 sessions)

The bundle carries Python, Qt, PyTorch and everything else it needs, but the Qt
X11 plugin links against your distribution's own X libraries. A minimal or stock
desktop install is often missing a few of them, and the application will not open
a window until they are present.

  Debian/Ubuntu:  sudo apt install $deb_packages
  RHEL/Rocky 8:   sudo dnf install $rpm_packages
                  (xcb-util-cursor lives in EPEL on RHEL 8:
                   sudo dnf install epel-release)

The launcher checks for these before starting and names the exact files if any
are missing, so you do not have to guess. Qt's own message is misleading here: it
reports "xcb-cursor0 is needed" no matter which of these is actually absent.

A Wayland session does not use any of them -- Qt falls back to its Wayland
plugin -- so the application can work on a Wayland desktop and fail on an X11
one on the same machine.

If you want a desktop launcher, copy:

  $install_root/share/applications/mlp-training-studio.desktop

into either:

  /usr/share/applications/
  ~/.local/share/applications/
EOF

mkdir -p "$output_dir"
# --mode='go-w' strips group and world write from every recorded member. It is a no-op
# for a build on a native filesystem, and it is what makes a build run from a Windows
# mount safe to ship: drvfs reports every file 0777 and silently ignores chmod, so the
# staged tree cannot be corrected in place. u+x is preserved, which the launcher and the
# Nuitka binary both need.
# --owner/--group/--numeric-owner: without them tar records the build host's
# numeric uid/gid, and `sudo tar -xzf ... -C /opt` reproduces them on the
# customer's machine -- every file showed as UNKNOWN:users (81256:100).
# It still runs, because the modes are 0755, but it reads as a broken install
# and trips file-integrity tooling. root:root is what a system package ships.
tar -C "$output_dir" --mode='go-w' --owner=0 --group=0 --numeric-owner \
  -czf "$tarball_path" "$bundle_name"

echo "Linux bundle created at $tarball_path"
