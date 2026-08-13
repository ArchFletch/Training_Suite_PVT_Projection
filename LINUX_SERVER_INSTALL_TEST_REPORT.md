# Linux License Server — Install, License Check, and Renewal Test Report

**Date:** 2026-08-12
**Target:** WSL2 Ubuntu 24.04 (`Zenbook`), Python 3.12.3
**Source:** `ArchFletch/Training_Suite_PVT_Projection` @ `7d4bb94`
**Scope:** Linux server install, license checking (seat enforcement), license renewal. Windows not tested.

---

## Verdict

The license server **works correctly once installed** — seat enforcement, heartbeat, release, feature gating, signature verification, in-place renewal, and downgrade eviction all behave exactly as the PRD specifies.

The **packaging does not work as shipped**. A customer following the documented Linux procedure gets a service that reports success and does nothing. Three defects block a clean install; two more make a broken state look healthy. All five are small fixes.

| # | Severity | Defect | Status |
|---|---|---|---|
| F1 | **Blocker** | systemd unit never actually starts the server — exits 0, reports success, binds nothing | **Fixed** |
| F2 | **Blocker** | CRLF line endings break the installer script (no `.gitattributes`) | **Fixed** |
| F3 | Major | Installer script is not executable in git — `./install_linux_service.sh` fails | **Fixed** |
| F4 | Major | Unit sets an environment variable nothing reads, omits the one that matters | **Fixed** |
| F5 | Major | An expired license imports successfully and `/status` still reports `"ok": true` | **Fixed** |
| F6 | Minor | `config.toml` is never parsed — port and lease settings in it are inert | **Fixed** |
| F7 | Minor | `/var/log/mlp-license-server/` is created but never written to | **Fixed** |
| F8 | Minor | The documented install command cannot work as written | **Fixed** |
| F9 | Major | `config.toml` is written `0640 root:root` — the service account cannot read its own config | **Fixed** |

F9 was found only while verifying the F6 fix: the config file parsed correctly in tests but had no
effect on the running service, because the service user got `Permission denied` opening it and the
loader fell back to defaults silently. See "Fixes applied" below.

---

## Environment note

The sudo password supplied was never used — **WSL grants root to the owning Windows user with no authentication**:

```bash
wsl.exe -d Ubuntu-24.04 -u root -- id
# uid=0(root) gid=0(root) groups=0(root)
```

Every step below ran as root through that path. Since the password was pasted into a chat transcript, rotate it (`passwd` inside WSL) regardless.

This also matters for the product: a WSL-hosted license server offers no protection from the workstation's own user. WSL2 is fine for rehearsing an install, but it is not a viable host for a real floating-seat deployment — it also sits behind a NAT'd adapter, so no other machine on the LAN can reach port 27850 without a `netsh portproxy` rule that breaks on every restart.

---

## Procedure

### 1. Teardown — start from nothing

**Why:** the box carried a partial install from an earlier attempt. A defect found on a dirty box proves nothing, so everything was removed including the service account, forcing the installer to exercise its full path.

```bash
systemctl disable --now mlp-license-server
rm -rf /etc/systemd/system/mlp-license-server.service \
       /etc/systemd/system/mlp-license-server.service.d \
       /etc/mlp-license-server /var/lib/mlp-license-server \
       /var/log/mlp-license-server /opt/mlp-license-server
systemctl daemon-reload
userdel mlp-license-server; groupdel mlp-license-server
```

```
removed: /etc/systemd/system/mlp-license-server.service
removed: service account
Unit mlp-license-server.service could not be found.
```

### 2. Obtain a clean source checkout

**Why:** the previous failure was suspected to be CRLF contamination from the Windows working tree. Cloning natively on Linux isolates that variable — `core.autocrlf` is a Windows-side setting, so a Linux clone must produce LF.

```bash
apt-get install -y git python3-venv curl iproute2
git clone https://github.com/ArchFletch/Training_Suite_PVT_Projection.git /home/lei/mlp-repo
```

```
HEAD: 7d4bb94 PVT corner projection: SpectraHydraProj, design-level splits, prebuilt-npz cache
LF    packaging/license_server/linux/install_linux_service.sh
LF    packaging/license_server/linux/mlp-license-server.service.template
LF    packaging/license_server/linux/mlp-license-server.env.sample
LF    packaging/license_server/shared/config.linux.toml.sample
```

**Confirms F2.** All four files are LF from a Linux clone. The same four files in the Windows working tree are CRLF, which is what produced the earlier failure:

```
: invalid option name.sh: line 2: set: pipefail
```

That is `set -euo pipefail\r` — bash reads `pipefail\r` as an invalid option, and the stray CR is what mangles the error message itself. This is not cosmetic: the script `sed`s three CRLF templates into `/etc/systemd/system/mlp-license-server.service`, `config.toml`, and `service.env`. A systemd unit with CRLF gets a trailing `\r` on every value — `User=mlp-license-server\r` is an invalid user — so the unit fails to load even after the script runs clean.

### 3. Run the packaged installer

**Why:** this is the one command the customer documentation gives for Linux, so it is the thing under test.

```bash
bash /home/lei/mlp-repo/GUI/packaging/license_server/linux/install_linux_service.sh \
  --install-root /opt/mlp-license-server \
  --service-command "/opt/mlp-license-server/.venv/bin/python -m uvicorn license_server.main:app --host 0.0.0.0 --port 27850"
```

```
Prepared Linux service assets:
  systemd unit: /etc/systemd/system/mlp-license-server.service
  app config:   /etc/mlp-license-server/config.toml
  env file:     /etc/mlp-license-server/service.env
installer exit: 0
```

Invoked as `bash <script>`, **not** `./script`. **F3:** the file is committed mode `100644`, so it has no executable bit on any fresh clone, and `sudo ./install_linux_service.sh` fails with `command not found` — sudo's message when a target is not executable, which reads misleadingly like a missing file.

What it produced:

```
-rw-r--r-- 1 root root 667 /etc/systemd/system/mlp-license-server.service
-rw-r----- 1 root root 617 /etc/mlp-license-server/config.toml
-rw-r----- 1 root root 221 /etc/mlp-license-server/service.env
drwxr-xr-x 2 root               root               /opt/mlp-license-server
drwxr-x--- 2 mlp-license-server mlp-license-server /var/lib/mlp-license-server
drwxr-x--- 2 mlp-license-server mlp-license-server /var/log/mlp-license-server
mlp-license-server:x:999:989::/var/lib/mlp-license-server:/usr/sbin/nologin
```

Note what it did **not** do: no application code, no virtualenv, no dependencies. `/opt/mlp-license-server` is empty. The `--service-command` string is never validated — it is only quoted into `service.env`.

### 4. Stage the application

**Why:** the installer leaves the install root empty, so the service has nothing to run.

```bash
cp -rT /home/lei/mlp-repo/GUI/license_server /opt/mlp-license-server/license_server
python3 -m venv /opt/mlp-license-server/.venv
/opt/mlp-license-server/.venv/bin/pip install -r /opt/mlp-license-server/license_server/requirements.txt
```

```
cryptography 50.0.0   fastapi 0.141.1   pydantic 2.13.4   typer 0.27.1   uvicorn 0.52.1
```

**Why this specific check next:** it reproduces exactly what systemd does — same account, same working directory — so an import or permission problem surfaces here rather than as an opaque unit failure.

```bash
runuser -u mlp-license-server -- env -C /opt/mlp-license-server \
  /opt/mlp-license-server/.venv/bin/python -c "import license_server, fastapi, uvicorn; print('ok')"
```

```
package importable as service user
```

### 5. Control experiment — start with the shipped unit

**Why:** to establish whether the unit works unmodified, before applying any fix. Without this the fix would be an unproven assertion.

```bash
systemctl start mlp-license-server
systemctl is-active mlp-license-server
systemctl show -p Result,ExecMainCode,ExecMainStatus --value mlp-license-server
ss -ltn | grep -c ':27850 '
```

```
is-active : inactive
Result    : success
ExecCode  : 0
ExecStatus: 0
listening on 27850: 0

Aug 12 16:33:49 Zenbook systemd[1]: Started mlp-license-server.service - MLP License Server.
Aug 12 16:33:49 Zenbook systemd[1]: mlp-license-server.service: Deactivated successfully.
```

**F1 confirmed empirically.** A perfectly correct `--service-command` produced a service that started, exited 0, and bound nothing.

Cause — the rendered unit contains:

```ini
ExecStart=/bin/sh -lc "$MLP_LICENSE_SERVER_COMMAND"
```

systemd strips the quotes at parse time, so the expander sees the bare word `$MLP_LICENSE_SERVER_COMMAND`. A bare `$VAR` is substituted **split at whitespace into multiple arguments**; only `${VAR}` yields a single argument. The effective argv becomes:

```
/bin/sh -lc /opt/.../python -m uvicorn license_server.main:app --host 0.0.0.0 --port 27850
```

`sh -c` takes only its first operand as the command string — the rest become `$0`, `$1`, … So Python is launched with no arguments, reads its program from stdin (`/dev/null` under systemd), gets EOF, and exits 0. `Type=simple` plus exit 0 means the unit goes `inactive (dead)` with `status=0/SUCCESS`, and `Restart=on-failure` never fires.

This is the worst possible failure mode: `systemctl status` shows no error, the journal says "Started", and the exit code is success.

### 6. Repair the unit

**Why:** fixes F1 and F4 in one file, without editing the shipped template.

```bash
mkdir -p /etc/systemd/system/mlp-license-server.service.d
cat > /etc/systemd/system/mlp-license-server.service.d/override.conf <<'EOF'
[Service]
Environment=MLP_LICENSE_SERVER_RUNTIME_DIR=/var/lib/mlp-license-server
ExecStart=
ExecStart=/opt/mlp-license-server/.venv/bin/python -m uvicorn license_server.main:app --host 0.0.0.0 --port 27850
EOF
systemctl daemon-reload
```

The bare `ExecStart=` is required — it is a list setting and must be cleared before replacement.

**F4:** the shipped unit sets `Environment=MLP_LICENSE_SERVER_CONFIG=…`, which **no code in the repository reads**. The variable that actually controls where the server looks for its database and license is `MLP_LICENSE_SERVER_RUNTIME_DIR` (`license_server/service/config.py:92`). On a clean box the fallback happens to land on `/var/lib/mlp-license-server`, so it works by luck. But `resolve_repo_root` walks up from the working directory looking for `.git` — so if anyone stages the app by cloning into `/opt/mlp-license-server`, the service silently switches to `/opt/mlp-license-server/runtime/license_server` while the admin CLI stays on `/var/lib`. Two databases, no error, and the server reports "no license" forever after a successful import.

### 7. Server identity and license request

**Why:** the server generates its own identity on first use; the license is bound to it. Run as the service account so SQLite is not left root-owned — a root-owned `license.db` produces a working-looking activation followed by "readonly database" on every seat request.

```bash
runuser -u mlp-license-server -- env -C /opt/mlp-license-server \
  /opt/mlp-license-server/.venv/bin/python -m license_server.cli init \
  --runtime-root /var/lib/mlp-license-server
```

```json
{"ok": true, "identity": {"server_id": "srv_131dad2517a2",
 "host_fingerprint": "host_784d85ae6198", "hostname": "Zenbook", "os_family": "linux"}}
```

```bash
… license_server.cli export-request /var/lib/mlp-license-server/license_request.json \
  --runtime-root /var/lib/mlp-license-server
```

The request contains only `server_id`, `host_fingerprint`, `hostname`, `os_family`, and a timestamp — no usage or personal data.

### 8. Vendor side — sign a license

**Why:** a fresh server is unlicensed and denies every checkout. In production this half runs on the vendor's machine and `license_vendor/` is never shipped to a customer; it was run here only to produce a test license.

```bash
python3 -m license_vendor init-key --key-id 2026-08 --output ~/mlp-vendor/vendor_signing_key.json
```

```json
{"status": "ok", "algorithm": "Ed25519", "key_id": "2026-08",
 "public_key": "ZCo4usNKa0FtsPpa/NsHiHpYY9AaRIEkqGsnoa7aPEM="}
```

**Gap worth noting:** `init-key` emits the public key as raw base64 inside a JSON file, but the server verifies with `load_pem_public_key` and needs PEM. No CLI command produces the PEM — the conversion exists only inside a Windows PowerShell helper. On Linux it must be done by hand:

```python
pub = ed25519.Ed25519PublicKey.from_public_bytes(base64.b64decode(data["public_key"]))
out.write_bytes(pub.public_bytes(serialization.Encoding.PEM,
                                 serialization.PublicFormat.SubjectPublicKeyInfo))
```

```bash
python3 -m license_vendor issue-eval --request-file … --signing-key-file … \
  --company-name "Internal Test" --seat-count 2 --term-days 14 --output license_eval.json
```

```json
{"license_type": "evaluation", "seat_count": 2, "features": ["baseline", "transfer"],
 "starts_at": "2026-08-12T21:34:54Z", "ends_at": "2026-08-26T21:34:53Z"}
```

### 9. Import and start

```bash
install -m 0644 vendor_public_key.pem /etc/mlp-license-server/vendor_public_key.pem
… license_server.cli import-license … --vendor-public-key /etc/mlp-license-server/vendor_public_key.pem
systemctl enable --now mlp-license-server
```

```
is-active: active
LISTEN 0 2048 0.0.0.0:27850 0.0.0.0:* users:(("python",pid=530,fd=13))
```

```json
{"ok": true, "product": "Surrogate Model Training Suite", "company_name": "Internal Test",
 "license_type": "evaluation", "seat_count": 2, "seats_in_use": 0}
```

**Verified with `ss`, not `systemctl is-active`** — F1 showed the unit can report success while nothing listens, so the bound socket is the only trustworthy signal.

---

## License checking

**Why these tests:** the entire commercial model rests on the server enforcing a concurrency limit and reclaiming seats. Each behaviour was exercised against the live HTTP API.

### Seat limit

```
checkout alpha   -> granted=True  lease=lease_ef4e4a6bfe8d  seats=1/2
checkout bravo   -> granted=True  lease=lease_c3df0867d232  seats=2/2
checkout charlie -> granted=False reason='all_seats_in_use'
                    msg='All floating seats are currently in use.'
```

Exact enforcement at the seat count, with the documented reason code.

### Heartbeat

```
heartbeat alpha -> ok=True  expires_at=2026-08-12T21:38:05Z
heartbeat bogus -> ok=False reason='invalid_lease'
```

Valid lease extends; unknown lease is rejected authoritatively rather than silently tolerated.

### Release

```
release alpha    -> ok=True
status           -> seats=1/2
checkout charlie -> granted=True lease=lease_6604d6e00449
```

Clean release frees the seat immediately; the previously-denied client succeeds.

### Feature gating

```
feature 'baseline'    -> granted=False reason='all_seats_in_use'
feature 'quicksearch' -> granted=False reason='feature_not_enabled'
```

Feature validity is checked **before** seat availability — `baseline` passed the feature check and then hit the seat limit, while `quicksearch` failed on the feature. Both correct.

Worth knowing: the desktop client never sends `requested_feature` (`xfmr_v2/licensing/models.py:206`), so the `features` list in every license is currently declared but unenforced. The server side works; the client side does not use it.

### Tamper rejection

```
seat_count 5 -> 999, re-signed envelope untouched
import -> {"ok": false, "error": "License signature verification failed."}  exit 1
```

Ed25519 verification does its job.

---

## License renewal

### Evaluation → paid, 2 seats → 5, in place

**Why:** the PRD promises eval-to-paid conversion on the same server without changing client settings. The test checks whether that costs a restart or drops live sessions.

```
service MainPID before import: 214
issue-paid --seat-count 5 --term-days 365
import: ok=True license_id=lic_paid_20260812T213605Z_feabdfd3 evicted=[]
service MainPID after import : 214   (restart required: False)
after renewal: type=paid seats=2/5 ends=2027-08-12T21:36:04Z
```

**Same PID, license swapped, live leases untouched.** Renewal is genuinely non-disruptive — engineers keep working through it. This is the strongest part of the implementation.

### Downgrade eviction, 5 seats → 1 with 4 in use

```
before downgrade: type=paid seats=4/5
import 1-seat: ok=True
  evicted_lease_ids=['lease_4abd16fa5e26','lease_4105ef1b5b94','lease_c3df0867d232']
after downgrade: seats=1/1
```

Excess leases evicted immediately on import, exactly as specified. Eviction is newest-first, so the longest-running session survives — a sensible choice that is not documented anywhere.

### Expired license — **F5**

```
issue-eval --starts-at 2026-01-01 --ends-at 2026-01-15   (already in the past)
import expired: ok=True          <-- import does NOT check the term
status now: ok=true  ends_at=2026-01-15T23:59:59Z
checkout golf -> granted=False reason='license_expired'
```

Import validates signature, product, `server_id`, and `host_fingerprint` — but **not the license term**. So an admin importing a stale or wrong-dated file gets `"ok": true` and a success message, `/api/v1/status` continues to report `"ok": true`, and the first hint of trouble is every engineer being blocked at launch.

`get_status()` never compares `ends_at` to the clock; it reports `ok: true` whenever any license row exists. The GUI's "Test Connection" would therefore show green against an expired license.

### Audit trail

```
license_imported / lease_evicted ×3 / checkout_granted / checkout_denied / lease_released
```

Every state change is recorded with a timestamp. Support-grade and complete.

---

## Final state

```
enabled for boot : enabled
active           : active
listening        : 0.0.0.0:27850  (pid 206)
license          : paid, "Internal Test", 5 seats, 0 in use, ends 2027-08-12
data dir         : all files owned by mlp-license-server
```

`/var/log/mlp-license-server/` is empty and always will be — **F7**. The installer creates it and the unit grants it write access, but the code resolves its log directory to `<runtime_root>/logs` and application output goes to journald. Use `journalctl -u mlp-license-server`.

**F6:** `/etc/mlp-license-server/config.toml` is never parsed by any code. Its `[server] host`/`port` and the whole `[leases]` block are inert — the real port comes only from the uvicorn argument, and lease timings are hardcoded to 30 s / 120 s / 300 s in `service/config.py:13-15`. The sample also says `port = 8090` while every document says `27850`; the installer's built-in default service command uses 8090 too, so omitting `--service-command` silently produces a server on the wrong port.

**F8:** the documented command in `doc/operations/license_server_customer_install.md:83` cannot work as written — it uses a relative `./packaging/...` path that presupposes a working directory inside the source checkout (contradicting the same file's "rooted outside the repo"), invokes a non-executable file via `./`, and passes `--enable --start` before any application code exists.

---

## Recommended fixes

Ranked by cost-to-benefit. The first three are one-liners that convert a failed install into a working one.

1. **`mlp-license-server.service.template:14`** — change `"$MLP_LICENSE_SERVER_COMMAND"` to `"${MLP_LICENSE_SERVER_COMMAND}"`, or drop the shell wrapper entirely. *(F1)*
2. **Add `.gitattributes`** at the repo root *(F2)*:
   ```gitattributes
   * text=auto eol=lf
   *.bat text eol=crlf
   *.cmd text eol=crlf
   ```
   Then `git add --renormalize .`. This also protects `packaging/linux/build_bundle.sh` and `build_gui.sh`, which have the same problem.
3. **`git update-index --chmod=+x`** on `install_linux_service.sh`, `build_bundle.sh`, `build_gui.sh`. *(F3)*
4. **Add `Environment=MLP_LICENSE_SERVER_RUNTIME_DIR=__DATA_DIR__`** to the unit template and remove the unread `MLP_LICENSE_SERVER_CONFIG`. *(F4)*
5. **Refuse or loudly warn on importing an expired license**, and make `get_status()` return `ok: false` outside the term. *(F5)*
6. Either parse `config.toml` or delete it; today it is a support trap. Align the 8090/27850 defaults. *(F6)*
7. Add a `export-public-key` command to `license_vendor` so the PEM does not require a hand-written snippet on Linux.
8. Rewrite the Linux section of the customer install doc around the real sequence: install script → stage app → venv → drop-in → init → license → start, verified with `ss`. *(F8)*

---

## Fixes applied

All nine defects were fixed in the source tree and re-validated by tearing the machine down
completely and installing again from the corrected sources. Changes are **staged but not
committed** — 12 files, +340 / −85.

| File | Change |
|---|---|
| `.gitattributes` *(new)* | `* text=auto eol=lf`, CRLF kept only for `.bat`/`.cmd`, binaries marked. Fixes F2 for every Linux-destined file including `build_bundle.sh` and `build_gui.sh`. |
| `mlp-license-server.service.template` | `"$MLP_…"` → `"exec ${MLP_…}"`, plus `Environment=MLP_LICENSE_SERVER_RUNTIME_DIR=__DATA_DIR__`. Fixes F1 and F4. |
| `install_linux_service.sh` | Default port 8090 → 27850; `config.toml` now `root:<service_group>` so the service can read it. Fixes F6's real cause (F9). |
| `config.linux.toml.sample`, `config.windows.toml.sample` | Reduced to the `[leases]` block that is actually read, with the port and path ownership documented instead of implied. Fixes F6. |
| `service/config.py` | Reads `[leases]` from the config file via `tomllib` (explicit argument > file > default); Linux system log root resolves to `/var/log/mlp-license-server`. Fixes F6 and F7. |
| `services/license_service.py` | New `resolve_license_state()`; `get_status()` reports `ok: false` outside the term; import returns `license_state` and a `warning`. Fixes F5. |
| `core_models.py` | `LicenseImportResult` gains `license_state` and `warning`. |
| `cli/main.py` | Admin status reads the term from the payload directly and adds `license_state` / `can_grant_seats`; import prints the warning to stderr. Keeps admin visibility once the client API goes quiet. |
| `doc/operations/license_server_customer_install.md` | Linux section rewritten as the sequence that actually works. Fixes F8. |
| three `*.sh` | `git update-index --chmod=+x` → mode `100755`. Fixes F3. |

Design note on F5: `StatusResponse` already required that license fields be omitted when `ok`
is false, so "expired" is expressed as *not active* rather than by adding a wire field. `ok` now
means "this server can grant a seat right now". The detail an admin needs did not disappear — it
moved to `show-status`, which reports `license_state`, `can_grant_seats`, and the full term.

### Regression suite

```
42 passed, 1 skipped
tests/license_server  tests/license_vendor  tests/gui_license  tests/test_license_client_config.py
```

One failure surfaced during the work and was fixed: reading `starts_at`/`ends_at` straight off the
payload yields `datetime` objects, which the CLI's `json.dumps` rejected — the previous code path
had been serializing them via `to_dict()`.

### Re-validation from scratch

Full teardown (unit, drop-in, `/etc`, `/var/lib`, `/var/log`, `/opt`, service account), then a
clean install from the fixed sources **with no manual drop-in of any kind**:

```
--- rendered unit, the two lines that were broken ---
13:Environment=MLP_LICENSE_SERVER_RUNTIME_DIR=/var/lib/mlp-license-server
20:ExecStart=/bin/sh -lc "exec ${MLP_LICENSE_SERVER_COMMAND}"

--- start with the shipped unit only ---
ls: cannot access '/etc/systemd/system/mlp-license-server.service.d': No such file or directory
is-active : active
MainPID   : 732
LISTEN 0 2048 0.0.0.0:27850 0.0.0.0:* users:(("python",pid=732,fd=13))
```

The unit now starts the server unaided — the single most important result in this report.

`config.toml` edited from 120/30/300 to 90/15/200 and the service restarted:

```
checkout reports ttl=90 heartbeat=15 grace=200
```

Expired license imported:

```
state   : expired
warning : This license ended at 2026-01-15 23:59:59+00:00 and is already expired. It is now the
          active license, but every seat request will be denied until a license covering the
          current date is imported.
stderr  : WARNING: <same text>

client /status : {"ok": false, "seat_count": 0, "seats_in_use": 0}
checkout       : granted=False reason=license_expired
admin          : license_state=expired can_grant_seats=False ends_at=2026-01-15T23:59:59Z
```

Every source file under `packaging/` and `license_server/` is LF; the only remaining files
containing CR bytes are `__pycache__/*.pyc`, which are compiled bytecode.

### Note on git configuration

`core.autocrlf=true` was set in the system-wide gitconfig and **overrode** the new `eol=lf`
attribute on checkout, so the working tree kept coming back CRLF. It is now `false` for this
repository only (`git config --local core.autocrlf false`); `.gitattributes` is the single source
of truth. No global git setting was changed.

---

## Not tested

- Windows server install (out of scope by request)
- Real multi-machine concurrency — WSL2's NAT prevents LAN access
- Desktop GUI client against this server
- Lease expiry by timeout (120 s TTL) and the client's 300 s grace window
- Reboot persistence (unit is `enabled`, but WSL was not restarted)
- `server_binding_mismatch` — would require moving the data directory to another host
