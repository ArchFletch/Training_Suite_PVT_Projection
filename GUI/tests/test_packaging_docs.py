"""Guards on the documents a customer actually follows.

Both files here were found wrong by a real install: the license-server INSTALL.md
had a step that failed three times and then reported success, and the forwarded
GUI install notes listed none of the X11 packages the app needs. Neither is
generated, so the only thing that keeps them honest is a test.
"""

from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_INSTALL = REPO_ROOT / "packaging" / "license_server" / "linux" / "INSTALL.md.template"
GUI_INSTALL = REPO_ROOT / "doc" / "customer" / "gui_install_run.md"
X11_REQUIREMENTS = REPO_ROOT / "packaging" / "linux" / "x11_runtime_requirements.txt"


def _bash_blocks(document: Path) -> list[str]:
    return re.findall(r"```bash\n(.*?)```", document.read_text(encoding="utf-8"), re.S)


def _required_packages() -> tuple[list[str], list[str]]:
    """Debian and RHEL package names for the mandatory sonames."""

    rows = [
        line.split()
        for line in X11_REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    required = [row for row in rows if row[3] == "required"]
    return sorted({row[1] for row in required}), sorted({row[2] for row in required})


# --------------------------------------------------------------------------- #
# license server INSTALL.md
# --------------------------------------------------------------------------- #


def test_the_venv_step_updates_the_package_lists_first() -> None:
    """On a freshly provisioned host the lists are empty, so `apt install
    python3-venv` returns "no installation candidate", `venv` then fails with
    "ensurepip is not available", and the `pip` the next line calls does not
    exist. All three steps failed on a clean Ubuntu 24.04 install."""

    staging = next(block for block in _bash_blocks(SERVER_INSTALL) if "python3-venv" in block)
    lines = [line.strip() for line in staging.splitlines() if line.strip()]
    update_at = next(i for i, line in enumerate(lines) if line.startswith("sudo apt update"))
    install_at = next(i for i, line in enumerate(lines) if "apt install" in line)
    assert update_at < install_at, "apt update must come before apt install"


def test_the_import_gate_loads_the_module_the_service_loads() -> None:
    """`import license_server` passes against an empty venv: `env -C` puts the
    staged package on the import path, so it resolves to the install root and
    never touches site-packages. It reported success on a venv with none of the
    five dependencies installed, directly after three visible failures."""

    text = SERVER_INSTALL.read_text(encoding="utf-8")
    assert "import license_server.main" in text, (
        "the verification step must import the module uvicorn loads, so a missing "
        "dependency fails the check instead of passing it"
    )


def test_hostname_stability_is_stated_before_the_identity_is_created() -> None:
    """The fingerprint is recomputed live from the hostname, so a rename after
    `init` invalidates a working license."""

    text = SERVER_INSTALL.read_text(encoding="utf-8")
    warning_at = text.index("Settle the hostname")
    init_at = text.index("svc init")
    assert warning_at < init_at, "the hostname warning must precede `svc init`"


def test_the_runtime_directory_is_called_out_for_backup() -> None:
    """Losing /var/lib/mlp-license-server loses server_id, which only a re-issued
    license can restore."""

    text = SERVER_INSTALL.read_text(encoding="utf-8")
    assert "Back up `/var/lib/mlp-license-server/`" in text
    assert "server_id" in text


def test_the_lan_url_step_does_not_invite_a_rename() -> None:
    """Step 7 used to offer `http://mlp-license-01:27850` as the URL to hand
    out, three steps after `init` locked the hostname into the binding."""

    text = SERVER_INSTALL.read_text(encoding="utf-8")
    # The document is hard-wrapped, so match on collapsed whitespace.
    tail = " ".join(text[text.index("## 7."):].split())
    assert "mlp-license-01" in tail, "the example is fine; the missing caveat was not"
    assert "Do not rename the host to match an example" in tail


def test_the_documented_python_floor_matches_what_the_code_needs() -> None:
    """`license_server.service.config` imports `tomllib` at module scope with no
    `tomli` fallback, so 3.11 is a hard floor. The document used to name Ubuntu
    22.04 as supported and Python 3.12 as the requirement; 22.04's default
    `python3` is 3.10, which builds a venv that installs every dependency and
    then fails at service start."""

    config_source = (REPO_ROOT / "license_server" / "service" / "config.py").read_text(encoding="utf-8")
    assert "\nimport tomllib" in config_source, (
        "if the tomllib dependency is gone, the documented Python floor can be lowered"
    )

    text = SERVER_INSTALL.read_text(encoding="utf-8")
    assert "Python 3.11 or newer" in text
    assert "tomllib" in text, "the floor needs its reason, or someone will lower it again"
    # 22.04 may be named, but only with the interpreter caveat attached.
    if "22.04" in text:
        assert "python3.12-venv" in text, "naming 22.04 without telling the admin to install 3.12+"


def test_the_venv_step_checks_the_interpreter_version() -> None:
    staging = next(block for block in _bash_blocks(SERVER_INSTALL) if "python3 -m venv" in block)
    assert "python3 --version" in staging


# --------------------------------------------------------------------------- #
# forwarded GUI install notes
# --------------------------------------------------------------------------- #


def test_the_forwarded_gui_notes_list_every_required_x11_package() -> None:
    """This document is the one that gets forwarded to an engineer. The package
    list existed only in the bundle's INSTALL.txt, which is read after
    extracting -- and after the app has already failed to start."""

    deb, rpm = _required_packages()
    blocks = _bash_blocks(GUI_INSTALL)
    apt = next((b for b in blocks if "apt install" in b), "")
    dnf = next((b for b in blocks if "dnf install" in b and "epel" not in b.split("\n")[-2]), "")
    dnf_all = " ".join(b for b in blocks if "dnf install" in b)

    missing_deb = [name for name in deb if not re.search(rf"(?<![\w.-]){re.escape(name)}(?![\w.-])", apt)]
    missing_rpm = [name for name in rpm if not re.search(rf"(?<![\w.-]){re.escape(name)}(?![\w.-])", dnf_all)]
    assert not missing_deb, f"absent from the Debian list: {missing_deb}"
    assert not missing_rpm, f"absent from the RHEL list: {missing_rpm}"
    assert dnf


def test_the_forwarded_gui_notes_explain_the_two_traps() -> None:
    """Both were established in testing and both cost time on their own: Qt
    misnames the missing library, and a Wayland session needs none of them, so
    the same bundle works for one engineer and fails for another."""

    text = GUI_INSTALL.read_text(encoding="utf-8")
    assert "xcb-cursor0 or libxcb-cursor0 is needed" in text
    assert "Wayland session uses none of these" in text
    assert "MLP_SKIP_LIBRARY_CHECK" in text
    assert "epel-release" in text


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_every_documented_shell_block_parses() -> None:
    """A wrapped `apt install` list is easy to get wrong by one backslash, and a
    broken continuation silently installs a prefix of the list."""

    for document in (SERVER_INSTALL, GUI_INSTALL):
        for index, block in enumerate(_bash_blocks(document)):
            # The server template still carries its __PLACEHOLDER__ tokens; they
            # are ordinary words to bash, so the syntax check is unaffected.
            result = subprocess.run(["bash", "-n"], input=block, capture_output=True, text=True)
            assert result.returncode == 0, f"{document.name} block {index}: {result.stderr}"


def test_the_delivery_readme_names_a_section_that_exists() -> None:
    """The round-1 finding was a README pointing at a document that was not in
    the delivery. Pointing at a *section* that does not exist is the same defect
    one level down, and the README is generated, so the reference and the
    heading live in different files."""

    assembler = (REPO_ROOT / "packaging" / "assemble_delivery.sh").read_text(encoding="utf-8")
    referenced = "Linux System Prerequisites"
    assert referenced in assembler, "the README no longer names the prerequisites section"
    headings = [
        line.strip("# ").strip()
        for line in GUI_INSTALL.read_text(encoding="utf-8").splitlines()
        if line.startswith("#")
    ]
    assert any(heading.startswith(referenced) for heading in headings), (
        f"no heading starting with {referenced!r} in {GUI_INSTALL.name}: {headings}"
    )


def test_every_shipment_directory_gets_its_own_checksum_manifest() -> None:
    """A recipient gets one folder, not the tree, so the manifest they run has to
    live in that folder and name only its files.

    The tree-level checksums.txt lists all three shipment directories. Customer
    IT, sent gui-cu128/ and server/, ran it and got

        gui-cu121/mlp-training-studio-linux-cu121.tar.gz: FAILED open or read
        sha256sum: WARNING: 2 listed files could not be read

    and exit 1, on the first command the instructions give them, for a tree in
    which every file present verified fine.
    """

    assembler = (REPO_ROOT / "packaging" / "assemble_delivery.sh").read_text(encoding="utf-8")

    # The per-recipient manifest is written, and by bare name rather than a path
    # relative to the tree -- a tree-relative path fails the same way.
    assert 'recipient_manifest="$output_dir/$recipient/checksums.txt"' in assembler
    assert 'recipient_checksum_lines["$recipient"]+="$digest  $base"' in assembler

    # Every artifact goes through the one recorder, so none can miss the manifest.
    assert "record_checksum " in assembler
    appends = {
        line.strip()
        for line in assembler.splitlines()
        if line.strip().startswith("checksum_lines+=")
    }
    expected = {
        # Inside record_checksum, which is the only route an artifact may take.
        """checksum_lines+="$digest  $rel"$'\\n'""",
        # The manifest files listing themselves, written after the artifacts.
        """checksum_lines+="$(sha256_of "$recipient_manifest")  $recipient/checksums.txt"$'\\n'""",
    }
    assert appends == expected, (
        "checksum bookkeeping changed; an artifact appended to the tree-level "
        f"manifest without going through record_checksum would miss its shipment "
        f"folder's checksums.txt. Unexpected: {sorted(appends - expected)}"
    )

    # And the README has to send the recipient into their own folder.
    assert "cd gui-cu128 && sha256sum -c checksums.txt" in assembler, (
        "the delivery README still tells the recipient to verify from the tree root"
    )
