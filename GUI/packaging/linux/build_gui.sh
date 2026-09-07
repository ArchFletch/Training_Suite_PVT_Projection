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

# The bundled torch decides which GPUs work, and a wheel that is too old fails at
# RUN time -- every launch on the customer's card prints "CUDA capability sm_NNN
# is not compatible with the current PyTorch installation" and the app then trains
# on the CPU. A cu121 wheel shipped that way once. Report the arch list here, in
# the build log, where there is still time to change the wheel.
#
# Advisory by default: a CPU-only bundle is a legitimate build. Set
# MLP_REQUIRE_GPU_ARCH to the architecture a shippable bundle must cover
# (MLP_REQUIRE_GPU_ARCH=sm_120 for RTX 50-series) to make it a hard gate.
gpu_arch_list="$("$python_bin" -c 'import torch; print(" ".join(torch.cuda.get_arch_list()))' 2>/dev/null || true)"
required_gpu_arch="${MLP_REQUIRE_GPU_ARCH:-}"
if [[ -z "$gpu_arch_list" ]]; then
  echo "NOTE: bundled torch reports no CUDA architectures -- this will be a CPU-only bundle." >&2
else
  echo "Bundled torch CUDA architectures: $gpu_arch_list" >&2
fi
if [[ -n "$required_gpu_arch" ]]; then
  if [[ " $gpu_arch_list " != *" $required_gpu_arch "* ]]; then
    echo "Refusing to build: MLP_REQUIRE_GPU_ARCH=$required_gpu_arch is not in the bundled torch." >&2
    echo "  bundled: ${gpu_arch_list:-<none>}" >&2
    echo "  install a matching wheel first, e.g. for sm_120 (Blackwell):" >&2
    echo "    $python_bin -m pip install torch --index-url https://download.pytorch.org/whl/cu128" >&2
    exit 1
  fi
  echo "Required GPU architecture $required_gpu_arch is present." >&2
fi

# Nuitka 2.7.11 cannot parse PEP 695 generic type aliases (`type Name[T] = ...`).
# It does not report the file: it aborts with `AssertionError: [<ast.TypeVar
# object ...>]` from buildTypeAliasNode, twenty frames deep in its own importer,
# and pyside6-deploy then re-raises that as a CalledProcessError. numpy 2.5.3
# introduced them in numpy/_typing/, which is enough to kill the whole build.
# Find them here and name the file instead.
pep695_report="$("$python_bin" - <<'PYEOF' || true
import re, sys, importlib.util

# The packages Nuitka is told to compile whole; those are the ones it parses.
PACKAGES = ("numpy", "pyqtgraph", "torch", "matplotlib", "onnx", "onnxruntime", "onnxscript", "xfmr_v2")
ALIAS = re.compile(r"^[ \t]*type[ \t]+[A-Za-z_][A-Za-z0-9_]*[ \t]*\[", re.M)

for name in PACKAGES:
    try:
        spec = importlib.util.find_spec(name)
    except Exception:
        continue
    if spec is None or not spec.submodule_search_locations:
        continue
    for root in spec.submodule_search_locations:
        import pathlib
        for path in pathlib.Path(root).rglob("*.py"):
            if "/tests/" in str(path) or "/test/" in str(path):
                continue  # Nuitka does not follow test trees into the bundle
            try:
                if ALIAS.search(path.read_text(encoding="utf-8", errors="ignore")):
                    version = getattr(__import__(name), "__version__", "?")
                    print(f"{name} {version}: {path}")
                    break
            except OSError:
                continue
        else:
            continue
        break
PYEOF
)"
if [[ -n "$pep695_report" ]]; then
  echo "Refusing to build: a package to be compiled uses PEP 695 type aliases, which the" >&2
  echo "pinned Nuitka cannot parse. It would abort with a bare AssertionError naming no file." >&2
  printf '  %s\n' "$pep695_report" >&2
  echo "Pin that package lower in the build environment (numpy 2.3.4 is known good), or" >&2
  echo "raise the Nuitka pin in packaging/gui/pysidedeploy.spec.in and re-rehearse the build." >&2
  exit 1
fi

# Nuitka links the final binary with -lpython3.<minor> but passes no matching -L,
# so gcc falls back to its default search path. That works for a system Python and
# fails for a relocated one -- and it fails at the LINK step, after the entire
# compile, with nothing but "/usr/bin/ld: cannot find -lpython3.12". Forty minutes
# of compilation are thrown away to learn one path is missing, and because the
# variables below are ambient, a build that worked in one shell fails in the next
# with no change to the tree.
#
# Take the directory from the build interpreter itself, and refuse in two seconds
# if the library genuinely is not there.
python_libdir="$("$python_bin" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")')"
python_ldlibrary="$("$python_bin" -c 'import sysconfig; print(sysconfig.get_config_var("LDLIBRARY") or "")')"
if [[ -n "$python_libdir" && -n "$python_ldlibrary" ]]; then
  if [[ ! -e "$python_libdir/$python_ldlibrary" ]]; then
    echo "Refusing to build: the build interpreter reports its shared library at" >&2
    echo "  $python_libdir/$python_ldlibrary" >&2
    echo "but no such file exists. Nuitka would compile everything and then fail at" >&2
    echo "the link step with 'cannot find -l${python_ldlibrary#lib}'." >&2
    echo "Install the matching python3-devel, or fix LIBDIR in the interpreter's" >&2
    echo "_sysconfigdata module if this is a relocated CPython." >&2
    exit 1
  fi
  # gcc resolves -l against LIBRARY_PATH; the loader uses LD_LIBRARY_PATH.
  export LIBRARY_PATH="${python_libdir}${LIBRARY_PATH:+:$LIBRARY_PATH}"
  export LD_LIBRARY_PATH="${python_libdir}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  echo "Linking against the build interpreter's libdir: $python_libdir" >&2
fi

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
strings -a "$dist_dir/mlp-training-studio" >"$symbols_file" 2>/dev/null || true
for entry in xfmr_v2.gui_window xfmr_v2.runner onnxscript; do
  grep -qF "$entry" "$symbols_file" || missing+=("$entry (compiled-in)")
done

if (( ${#missing[@]} )); then
  echo "Build FAILED: $dist_dir is missing ${missing[*]}." >&2
  echo "Check the --include-package flags in packaging/gui/pysidedeploy.spec.in." >&2
  exit 1
fi

# Every external library the Qt xcb (X11) plugin chain pulls in must be declared in
# x11_runtime_requirements.txt, which INSTALL.txt and the launcher's preflight are
# both rendered from. An `ldd` check on the build host cannot catch this: 26 of the
# 27 happen to be installed here, so they resolve and the dist looks self-contained
# while a stock Ubuntu X11 session fails to start. Comparing against the declared
# list instead of the build host's /usr/lib64 is what makes the check meaningful.
requirements_file="$repo_root/packaging/linux/x11_runtime_requirements.txt"
if [[ ! -f "$requirements_file" ]]; then
  echo "Build FAILED: $requirements_file is missing; it is the source of truth for the" >&2
  echo "system libraries INSTALL.txt and the launcher preflight tell the customer to install." >&2
  exit 1
fi

declared="$(mktemp)"; shipped="$(mktemp)"; external="$(mktemp)"
trap 'rm -f "$symbols_file" "$declared" "$shipped" "$external"' EXIT
grep -vE '^\s*#|^\s*$' "$requirements_file" | awk '{print $1}' | sort -u >"$declared"
find "$dist_dir" -name '*.so*' -type f -printf '%f\n' | sort -u >"$shipped"

# Transitively walk the xcb plugin chain and collect what it needs from outside.
: >"$external"
pending="$(find "$dist_dir" \( -name 'libqxcb.so' -o -name 'libQt6XcbQpa.so.6' \) -type f)"
visited=""
while [[ -n "$pending" ]]; do
  current="$(echo "$pending" | head -1)"; pending="$(echo "$pending" | tail -n +2)"
  case " $visited " in *" $current "*) continue;; esac
  visited="$visited $current"
  while read -r soname; do
    [[ -z "$soname" ]] && continue
    if grep -qxF "$soname" "$shipped"; then
      next="$(find "$dist_dir" -name "$soname" -type f | head -1)"
      [[ -n "$next" ]] && pending="$(printf '%s\n%s' "$pending" "$next")"
    else
      echo "$soname" >>"$external"
    fi
  done < <(objdump -p "$current" 2>/dev/null | awk '/NEEDED/{print $2}')
done

# glibc and the C++ runtime are on every Linux host worth shipping to.
undeclared="$(sort -u "$external" \
  | grep -vE '^(libc|libm|libdl|librt|libpthread|libutil|libresolv|libnsl|libcrypt|libgcc_s|libstdc\+\+|ld-linux-x86-64)\.so' \
  | grep -vxFf "$declared" || true)"
if [[ -n "$undeclared" ]]; then
  echo "Build FAILED: the Qt xcb plugin needs system libraries that are not declared in" >&2
  echo "packaging/linux/x11_runtime_requirements.txt:" >&2
  echo "$undeclared" | sed 's/^/  /' >&2
  echo "Add them there (with their Debian and RHEL package names) so INSTALL.txt and the" >&2
  echo "launcher preflight tell the customer to install them." >&2
  exit 1
fi

echo "Standalone Linux build completed: $dist_dir ($(du -sh "$dist_dir" | cut -f1))."
echo "Executable: $dist_dir/mlp-training-studio"
echo "X11 system libraries required at runtime: $(wc -l <"$declared") declared, all accounted for."
