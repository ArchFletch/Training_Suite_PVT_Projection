# Customer Delivery

The release manager's reference for a customer shipment: what is sent, to whom,
which command produced it, and in what order.

A shipment is never a single package sent once. It goes to two different people,
and the license file cannot be produced until one of them has already installed
and run their half.

## Recipients

| Recipient | Receives | Installs on | Follows |
| --- | --- | --- | --- |
| design engineers | Windows GUI installer or Linux GUI bundle, plus the LAN URL of the license server | each engineer's own workstation | `doc/customer/gui_install_run.md` |
| customer IT / server admin | license server bundle for their server OS, then `license.json` and `vendor_public_key.pem` | one LAN host, typically a server | Linux: `INSTALL.md` inside the bundle. Windows: `install-service.ps1` inside the bundle, no install document. Both: `doc/operations/license_server_customer_install.md` |

These are different people and normally different machines. Engineers never
receive the license server bundle, the license file, or the public key; the
server admin does not need the GUI installed to complete their half. The only
thing that crosses between them is the LAN URL, for example
`http://mlp-license-01:27850`, which the admin publishes after the service is
running. Customer IT may pre-seed that URL into each workstation's
`license_client.json` rather than have engineers type it.

Only the Linux bundle carries install instructions.
`packaging/license_server/linux/build_bundle.sh` renders `INSTALL.md` into the
staged tree from its `INSTALL.md.template`, so the Linux admin can work entirely
from the extracted tarball. The Windows bundle stages
`install-service.ps1`, `WinSW-x64.exe`, `mlp-license-server-service.xml.template`,
`config.windows.toml.sample` and the `.cmd` wrappers, and nothing that reads as
an install document.

That is a gap, not a choice: a Windows shipment currently depends on the admin
being sent `doc/operations/license_server_customer_install.md` separately,
because there is nothing in the zip to open. Send it explicitly with a Windows
bundle until `build_bundle.ps1` stages an install document of its own.

## Delivery Sequence

The license handshake is two-phase. `license.json` is bound to a `server_id` and
a host fingerprint that do not exist until the admin runs `init` on their own
machine, so there is nothing to sign before step 3.

| # | Direction | Item | Produced by |
| --- | --- | --- | --- |
| 1 | vendor to admin | license server bundle | `build_bundle.sh` / `build_bundle.ps1` (see build matrix) |
| 2 | admin, locally | install service, `init`, `export-request` | bundle install script, then the platform's admin entry point (below) |
| 3 | admin to vendor | `license_request.json` | `export-request` on the customer host |
| 4 | vendor to admin | `license.json` and `vendor_public_key.pem` | `python -m license_vendor issue-eval` / `issue-paid`, `export-public-key` |
| 5 | admin, locally | `import-license`, start service, open the port, publish the LAN URL | the same platform admin entry point |
| 6 | vendor to engineers | GUI installer or GUI bundle | `build_gui` plus `build_installer` / `build_bundle` |
| 7 | engineers, locally | install the GUI, point it at the LAN URL | `doc/customer/gui_install_run.md` |

Steps 6 and 7 do not depend on the handshake and can ship in parallel with step
1. Engineers cannot check out a seat until step 5 completes, so shipping the GUI
early only means the app is installed and idle, not that anything is broken.

What goes out in steps 1 and 6 is taken from `artifacts/delivery/`, produced by
`packaging/assemble_delivery.sh`, not copied out of `artifacts/packaging/` by
hand. See Assembling The Shipment below.

### The Admin Entry Point Differs By Platform

Only Windows ships wrappers. `build_bundle.ps1` writes
`mlp-license-server-admin.cmd` into the bundle, plus `init-server.cmd`,
`export-license-request.cmd`, `import-license.cmd` and `show-status.cmd`, each
of which calls the first one. They run `license_server.cli` through the bundled
`python\python.exe` and default `MLP_LICENSE_SERVER_RUNTIME_DIR` to
`%PROGRAMDATA%\MLP License Server`, so the Windows admin types
`init-server.cmd` and `export-license-request.cmd` for step 2 and
`import-license.cmd` for step 5.

The Linux bundle ships no wrapper at all. `INSTALL.md` has the admin define a
shell function by hand and use it for every admin command:

```bash
svc() { sudo -u mlp-license-server env -C /opt/mlp-license-server \
        /opt/mlp-license-server/.venv/bin/python -m license_server.cli "$@"; }
```

Step 2 is then `svc init` and `svc export-request`, step 5 is
`svc import-license`, and `svc show-status` is the admin view. Dropping to the
service account is the point of that function: `INSTALL.md` warns that a
root-created `license.db` leaves the service unable to write it, which presents
as a successful activation followed by failures on every seat request.

`license_request.json` carries `schema_version`, `product`, `server_id`,
`host_fingerprint`, `hostname`, `os_family`, `generated_at`, and an optional
`requested_by`. That last field is free text the admin supplies with
`export-request --requested-by`, and the CLI describes it as an operator name,
so a request is not automatically free of personal or internal detail. Treat it
as low-sensitivity rather than as not sensitive: it carries no key material and
no entitlement, but read `requested_by` before the file travels over anything
other than the channel already agreed with the customer. It is also per-server:
a request from one host cannot be used to license another.

## Why The License Cannot Ship Up Front

`init` mints a fresh random `server_id` and derives the host fingerprint from
hostname, OS family, and a machine token. `import-license` rejects any license
whose `product`, `server_id`, or `host_fingerprint` does not match the local
identity, so a pre-issued license imports nowhere.

### Re-Issue Triggers

| Event | What changes | Action |
| --- | --- | --- |
| server rebuilt, runtime root wiped, or `license.db` recreated | new `server_id` from `init` | full handshake again: new `export-request`, new `license.json` |
| server moved to another host, or hostname changed | new host fingerprint | full handshake again |
| seat count or term change on the same server | nothing | issue a replacement against the existing request and import it in place |

The vendor public key is delivered once per customer. Renewals and seat changes
reuse it; only `license.json` is reissued.

## Vendor Issuance

Run from the `GUI/` directory on the vendor machine, where `license_server` and
`license_vendor` are siblings. The signing key is created once with `init-key`
and reused for every customer.

```powershell
python -m license_vendor export-public-key --signing-key-file vendor_signing_key.json --output vendor_public_key.pem

python -m license_vendor issue-eval `
  --request-file license_request.json `
  --signing-key-file vendor_signing_key.json `
  --company-name "Acme Design House" `
  --seat-count 2 `
  --term-days 14 `
  --output license.json `
  --issuance-log license_vendor_issuance_log.jsonl
```

`issue-paid` takes the same arguments and defaults to a 365-day term. Send the
output file to the customer named `license.json`; the server's import command
does not care about the file name, but every install doc uses that one.

## Build Matrix

All commands run from the `GUI/` directory. Everything lands under
`artifacts/packaging/`, which is git-ignored: artifacts are rebuilt, never
committed.

| Artifact | Command | Output |
| --- | --- | --- |
| Linux license server bundle | `packaging/license_server/linux/build_bundle.sh` | `artifacts/packaging/license_server/linux/mlp-license-server-linux.tar.gz` |
| Windows license server bundle | `packaging/license_server/windows/build_bundle.ps1 -WinSWExePath <winsw.exe>` | `artifacts/packaging/license_server/windows/MLP License Server.zip` |
| Linux GUI bundle | `packaging/linux/build_gui.sh --output-dir <build dir>`, then `packaging/linux/build_bundle.sh --standalone-dir <standalone folder inside that build dir>` | `artifacts/packaging/linux/mlp-training-studio-linux.tar.gz` |
| Windows GUI installer | `packaging/windows/build_gui.ps1`, then `packaging/windows/build_installer.ps1 -StandaloneDir artifacts\packaging\windows\SurrogateModelTrainingSuite.dist` | `artifacts/packaging/windows/installer/SurrogateModelTrainingSuite-Windows.exe` |

| Artifact | Build host | Toolchain |
| --- | --- | --- |
| Linux license server bundle | any host with bash | bash, tar, and the usual POSIX file tools; no Python and no Qt. The bundle ships application code and the customer builds the virtual environment on their host |
| Windows license server bundle | Windows | PowerShell, a Windows Python whose base installation is a copyable tree, network access to PyPI, and a WinSW x64 executable supplied by path |
| Linux GUI bundle | Linux | Python with the GUI dependencies, PySide6, Nuitka via `pyside6-deploy` |
| Windows GUI installer | Windows | Python with the GUI dependencies, PySide6, Nuitka, and Inno Setup 6 for `ISCC.exe` |

### Which Python The Windows Server Bundle Needs

A virtual environment is fine. `build_bundle.ps1` inspects the interpreter named
by `-Python` and reads `sys.base_prefix`, which resolves to the base
installation a venv was created from, never to the venv itself, and it is that
base installation tree it copies into the bundle. The requirement is only that
the interpreter's base installation be a normal copyable Python directory.

No version is enforced anywhere in the script, and `runtime_requirements.txt`
pins no Python version either, so whatever `-Python` resolves to becomes the
customer's runtime by default rather than by decision. Choose it deliberately:
the Linux `INSTALL.md` asks the customer for Python 3.12, and matching that
keeps both platforms on one runtime. After copying, the script runs `ensurepip`,
upgrades `pip`, and installs `runtime_requirements.txt` into the bundled copy,
which is what the PyPI access is for.

### Finding The Linux GUI Standalone Folder

`build_gui.sh` does not print the deployment folder. Its final line is
`Inspect <output-dir> for the deployment folder`, which names only the
containing directory, never the folder itself, so the path `--standalone-dir`
wants has to be found by listing that directory once the Nuitka build finishes.

Give the two scripts separate directories. Both default `--output-dir` to
`artifacts/packaging/linux`, and `build_bundle.sh` starts with
`rm -rf <output-dir>/<bundle-name>`, where the default bundle name is
`mlp-training-studio`, before it copies anything. A `--standalone-dir` pointing
inside the bundler's own output directory at a folder of that name is therefore
deleted before it is read, destroying the build that was about to be packaged.
That collision is the likely one rather than a contrived one, because
`build_gui.sh` invokes the deploy step with `--name mlp-training-studio`.
Building the GUI into its own directory and passing an explicit path out of it
avoids the whole class.

### Cross-Build Limits

Windows artifacts cannot be produced from a Linux host. Both Windows commands
are PowerShell and both consume Windows-only inputs: the license server bundle
copies the build host's own Python runtime tree and a WinSW executable into the
bundle, and the GUI build produces a native Windows executable through Nuitka
before Inno Setup wraps it. A Windows build machine is a hard prerequisite for a
Windows shipment, and the GUI installer additionally needs Inno Setup 6
installed on it; `build_installer.ps1` fails outright if it cannot find
`ISCC.exe`.

The reverse holds for the Linux GUI bundle, which must be built on Linux: it is
a native Linux executable produced by Nuitka.

The Linux license server bundle is the exception in both directions. It stages
files and calls `tar`; there is no compiler, no Python and no Linux-only tool in
the path, and the script is written to survive a build from a Windows mount. It
passes `--mode='go-w'` to `tar` precisely because drvfs reports every file as
`0777` and ignores `chmod`, so the staged tree cannot be corrected in place and
the permissions have to be fixed as the archive is written. A Windows
workstation with Git Bash or WSL can therefore build it.

## Assembling The Shipment

Nothing is copied out of `artifacts/packaging/` by hand.
`packaging/assemble_delivery.sh` collects whatever has been built into one tree
grouped by recipient:

```bash
bash packaging/assemble_delivery.sh --clean
```

It writes `artifacts/delivery/engineers-gui/` (the Windows installer, the Linux
GUI bundle), `artifacts/delivery/server-admin/` (the two license server
bundles), a generated `README.md` and `checksums.txt`. `artifacts/delivery/` is
git-ignored the same way `artifacts/packaging/` is, and the whole tree is
rewritten on every run, so nothing in it should be edited by hand. `--clean`
removes only the paths the script itself writes, so a `--output-dir` holding
anything else survives.

A missing artifact is not an error. Each build host can only produce what its
toolchain supports, so the script prints `MISSING` for the rest, records the
same status in the generated `README.md` along with that artifact's build
command and required toolchain, and still exits 0. Run it on each build host and
re-run it after the last build: every run recollects the whole tree.

This is also where the key-material check stops depending on anyone remembering
to look. The script refuses any file whose name matches `*signing_key*`, `*.pem`,
`*.jsonl` or `license.json` on the way in, and then re-scans the finished tree
for those names plus `license_vendor` and fails the assembly outright if one is
present. The second sweep is the one that earns its place: it catches a file
dropped into the delivery tree by hand or left behind by an earlier run, which
is how a signing key would actually escape. It does not replace inspecting the
Windows staging tree described under Never Send, which is a different directory
that the assembly step never reads.

`checksums.txt` is the customer-verifiable half: one `sha256` line per collected
file, verified from the delivery directory with

```bash
sha256sum -c checksums.txt
```

Send it with the shipment. It is the only mechanical check that what the
customer received is what was assembled.

## Never Send

| Item | Why |
| --- | --- |
| `license_vendor/` | the vendor signing tooling; a customer host has no reason to hold license-minting code |
| `vendor_signing_key.json` | holds the Ed25519 private seed that signs every license |
| `license_vendor_issuance_log.jsonl` | the vendor-side record of which customer was issued what |
| repo-root internal reports (`BUGFIX_REPORT.md`, `LINUX_SERVER_INSTALL_TEST_REPORT.md`) | internal defect and rehearsal findings, including known gaps |
| the source checkout, and `tests/` | customer runtime behavior must not depend on a checkout |

The signing key is the one item with no recovery path. It mints a valid license
for any `server_id`, any seat count, and any term, and the server verifies that
signature offline against `vendor_public_key.pem` with no revocation list and no
callback to the vendor. A leaked key therefore cannot be revoked. Recovering
from a leak means generating a new key, distributing a new public key to every
fielded server, and reissuing every live license.

The Linux bundle script enforces this: it refuses to package if a
`license_vendor` directory, a `*signing_key*` file, a `.pem`, a `.jsonl`, or a
`license.json` appears anywhere in the staged tree. The Windows bundle script has
no equivalent guard; it copies `license_server/` wholesale, excluding only
`__pycache__` and `.pyc`. Inspect the Windows staging tree by hand before
shipping it.

## Release Status

| Artifact | Built | Install rehearsed from the packaged artifact |
| --- | --- | --- |
| Linux license server bundle | yes | yes, on a wiped rehearsal host, twice |
| Linux GUI bundle | no | no |
| Windows license server bundle | no | no |
| Windows GUI installer | no | no |

The Linux license server bundle is the only artifact that exists.
`packaging/license_server/linux/build_bundle.sh` produced
`artifacts/packaging/license_server/linux/mlp-license-server-linux.tar.gz`, 50
entries, and that tarball is what was installed.

The rehearsal host was WSL2 Ubuntu 26.04. It is a rehearsal host and not a
production one: no customer server and no standalone Linux server install has
run this bundle. The host was wiped before each run — service account, `/opt`,
`/etc` and `/var/lib` state removed — and the bundle was then extracted into a
fresh directory and installed from that directory alone, with no source checkout
present anywhere on the host. That was done twice: once scripted, once by hand
following the bundle's own `INSTALL.md`. A license was issued with
`license_vendor`, including its `export-public-key` command, imported on the
server, and the service was enabled and started; `GET /api/v1/status` returned
`ok: true`. Seat checkout, heartbeat, over-capacity denial and release were each
exercised over HTTP.

The client half of that path is rehearsed too, from Windows. A real PySide6
desktop GUI running on Windows connected to that server, took a seat
(`lease_52914779a941`, `machine_id gui_8f3e3627e3b5`, platform `windows`) and
released it on a clean exit: `POST /api/v1/release` appears in the journal 98
seconds before the lease TTL would have expired, so the seat came back because
the client returned it, not because it timed out. What that does not cover is
the packaged GUI. No GUI artifact has been built on either platform, so the
client in that rehearsal cannot have been an installed one, and the
installer-to-server path remains unrehearsed on both platforms.

The Linux GUI bundle does not exist. A build was attempted this session and
`artifacts/packaging/linux/` was never created. The first attempt failed on a
real bug in `packaging/linux/build_gui.sh`: it probed the bare `PATH` for
`pyside6-deploy` and fell back to `python -m PySide6.scripts.deploy`, which is
not a usable entry point, because `deploy.py` does a bare
`from deploy_lib import ...` that only resolves when the console script puts its
own directory on `sys.path`. The fallback died with `No module named
'deploy_lib'`. The script now prefers the console script sitting next to the
interpreter given by `--python` and fails with an explicit message if it finds
none. Treat this bundle as not built until a tarball is on disk.

The Windows GUI installer and the Windows license server bundle have never been
built and never been rehearsed. Neither has been produced even once, so nothing
about the Windows install path has been observed from packaged artifacts, only
from source. The machine used this session has no toolchain for either: no
standalone Python installation, no PySide6, no Nuitka, no Inno Setup 6, and the
Windows server bundle additionally needs a WinSW x64 executable supplied from
outside the repo. `doc/integration/windows_packaged_install_rehearsal.md`
defines the rehearsal that must pass before a first Windows shipment.

Two things follow for a shipment assembled today. `artifacts/delivery/` will
hold exactly one file, the Linux server bundle, with the other three reported
`MISSING`. And no engineer-side artifact exists on any platform, so the GUI half
of every shipment is unbuilt, the Linux one included.

None of the above rests on the repo-root `LINUX_SERVER_INSTALL_TEST_REPORT.md`.
That report predates this work and describes an earlier state; do not cite it as
evidence for the current one, and do not ship it either — see Never Send.

## Related Documents

- `doc/operations/license_server_customer_install.md`: the admin-side install path in full
- `doc/customer/gui_install_run.md`: the engineer-side install path and per-user file locations
- `doc/integration/windows_packaged_install_rehearsal.md`: the Windows rehearsal that gates a Windows shipment
- `packaging/README.md`: build script inventory and prerequisites
- `license_vendor/README.md`: the vendor CLI, including key creation
