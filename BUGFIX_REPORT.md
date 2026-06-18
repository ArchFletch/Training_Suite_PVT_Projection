# Bug-Fix Report — Code Audit of June 2026

**Branch:** `training-pipeline-improvements` (on top of `11ac21b`)
**Scope:** full audit of the codebase (~16k lines): ML training pipeline (`xfmr_v2`), GUI layer, CLI entry points, and the licensing subsystem (client, server, vendor tooling).
**Verification:** full test suite passes — 86 tests, 0 failures (previously 78 passing, 6 pre-existing failures, 2 uncollectable modules).
**Change size:** 59 files, +410/−186 lines (uncommitted at time of writing).

---

## 1. Data pipeline (`xfmr_v2/data.py`)

### 1.1 Tiny datasets silently trained to garbage (empty validation split)
- **Where:** `split_indices`
- **Symptom:** for datasets of ≤ 5 samples at default fractions (0.8/0.1), the validation split came out **empty**. `_eval_loss` on an empty loader returns `0.0`, so epoch 1 always "improved" and no later epoch could beat `0.0 < 0.0` — the run trained its full budget but silently saved the **epoch-1 weights**. The same empty split crashed quick search (`torch.cat` on an empty list).
- **Fix:** the split now guarantees ≥ 1 training sample always, ≥ 1 validation sample for n ≥ 2, and ≥ 1 test sample for n ≥ 3. Behavior for normal dataset sizes is unchanged. `_eval_metrics` additionally raises a clear "Evaluation split is empty" error instead of an opaque tensor crash.

### 1.2 Cadence multi-channel alignment silently substituted the wrong curve
- **Where:** `_build_cadence_auto`
- **Symptom:** when a channel CSV's parameter tuple did not match channel 0's, the fallback paired sample *i* with the positionally-*i*-th curve — and when the channel had fewer rows, with **curve 0** — duplicating one sample's data across all unmatched rows with no warning. Silently corrupted training targets.
- **Fix:** row-order pairing is still used as the best-available guess for genuinely unmatched keys, but it now emits a per-channel warning with the unmatched count, and a channel with fewer samples than channel 0 raises a clear `ValueError` instead of fabricating data.

### 1.3 Ground-truth file matching by substring bound the wrong file
- **Where:** `_find_per_sample_file`
- **Symptom:** after the exact `{id}{ext}` check, the fallback used `sample_id in stem` — so sample `"1"` could match `sample_10.s4p` or `12.s4p`, silently pairing the wrong S-parameters with a feature row. Results also depended on filesystem enumeration order.
- **Fix:** the fallback now requires a whole-token match (the ID bounded by non-alphanumerics), and directory iteration is sorted for determinism.

### 1.4 One malformed sample crashed the whole cache build
- **Where:** `_build_per_sample_arrays`
- **Symptom:** the frequency-grid comparison used `np.allclose` on potentially different-length arrays, which **raises** instead of returning `False` — outside the try/except that exists precisely to skip bad samples. A single file with a different grid length (or channel count) killed the build with an opaque broadcast error.
- **Fix:** shapes are compared before values; frequency or target-shape mismatches now skip the sample with a logged reason, as intended.

### 1.5 Stop requests were never honored during cache builds
- **Where:** `build_cache_from_dataset` → `_build_auto` → `_build_touchstone_auto` / `_build_cadence_auto` / `_build_per_sample_arrays`
- **Symptom:** `should_stop` was accepted but never checked anywhere in the build path.
- **Fix:** `should_stop` is plumbed through the entire chain and polled in the per-sample and per-CSV loops (raising `RunCancelled`, consistent with the runner).

### 1.6 `ensure_cache` could no longer build caches (dead CLI pipeline)
- **Where:** `ensure_cache`
- **Symptom:** after the loader refactor it only raised `FileNotFoundError` when the cache was missing, while `train_baseline.py` / `quick_search.py` / `suggest_initial_settings.py` still advertised `--data-root` / `--input-feature-path` / `--ground-truth-data-dir` flags that fed into it. A fresh checkout had **no working CLI path to a trained model**.
- **Fix:** when the cache is missing, `ensure_cache` now builds it from the dataset folder (derived from `data_root`, the ground-truth dir, or the input file's parent) via format auto-detection, with a clear error if no usable folder exists. It also forwards `should_stop`.

### 1.7 Pointing at the SPData folder itself failed auto-detection
- **Where:** `_build_auto`
- **Symptom:** selecting the `.sNp` folder directly (e.g. via the legacy ground-truth field) failed with `FileNotFoundError` because `log.txt` conventionally lives one level up.
- **Fix:** when the chosen folder contains Touchstone files but no `log.txt`, the parent directory is checked for `log.txt`.

## 2. CLI entry points

### 2.1 `prepare_cache.py` crashed at import
- **Symptom:** `from xfmr_v2.data import ... build_cache` — a function deleted in the loader refactor. The script was dead on arrival (`ImportError`).
- **Fix:** rewritten against the current API: takes a dataset folder positionally, auto-detects the format via `build_cache_from_dataset`, reuses an existing cache unless `--overwrite`, prints progress and a JSON summary. `GUI/README.md` updated to the new usage.

## 3. Training runner (`xfmr_v2/runner.py`)

### 3.1 A NaN validation loss at epoch 1 pinned the checkpoint forever
- **Symptom:** `improved = best_state is None or val_loss < best_val` — once `best_val` was NaN (e.g. an AMP fp16 spike), every later `x < NaN` comparison was `False`; the run kept the epoch-1 weights and reported `best_val_loss: NaN`.
- **Fix:** a NaN `best_val` is now always replaceable (`math.isnan(best_val)` counts as improvement).

### 3.2 Reported evaluation loss ignored the configured loss function
- **Symptom:** `_eval_metrics` always evaluated with the default `frequency_rmse`, while training history and `best_val_loss` used `config.loss_function` (MSE in the default SpectraHydra recipe) — `summary.json` and search trial records mixed units.
- **Fix:** `_eval_metrics` accepts and receives the configured `loss_fn`.

### 3.3 Stale GPU selection crashed runs
- **Where:** `resolve_device`
- **Symptom:** a persisted `"cuda:1"` on a one-GPU machine reached `torch.cuda.set_device` → `RuntimeError: invalid device ordinal`, despite the function's documented "stale selection can never crash" guarantee.
- **Fix:** an out-of-range CUDA index falls back to the default CUDA device.

### 3.4 Same-second runs overwrote each other's artifacts
- **Where:** `make_run_dir`
- **Symptom:** second-resolution timestamps with `exist_ok=True` meant two runs started within the same second shared one folder, clobbering `summary.json` and checkpoints.
- **Fix:** `exist_ok=False` with a `-2`, `-3`, … suffix on collision.

### 3.5 Self-transfer mislabeled non-frequency sweep axes by 9 orders of magnitude
- **Symptom:** the transfer path unconditionally divided the sweep axis by 1e9, while the baseline path checks `sweep_label` first. For non-frequency sweeps (e.g. a CTLE VDIFF axis) progress payloads, band labels, and plot axes were off by 1e9.
- **Fix:** the transfer path reads `sweep_label` from the cache and only converts genuine frequency axes, matching `load_split_bundle`.

## 4. GUI layer

### 4.1 The Stop button was a no-op for dataset scans — and the GUI claimed success
- **Where:** `gui_backend.build_and_scan_dataset` (+ §1.5)
- **Symptom:** the GUI enabled Stop for scans and set the badge to "Stopped", but `should_stop` was never forwarded to the cache builder; the build ran to completion and the badge then flipped to "Completed".
- **Fix:** `should_stop` is forwarded end-to-end; combined with §1.5 the scan now actually stops.

### 4.2 User-initiated stops were presented as errors
- **Where:** `gui_workers._TaskRunner` / `ImmediateTaskExecutor`, `gui_window` result handlers
- **Symptom:** tasks without their own `RunCancelled` handling (scan, suggest) surfaced a stop as an **Error dialog with a full traceback**, and the suggest confidence badge froze at "Checking".
- **Fix:** the workers catch `RunCancelled` and report `{"status": "stopped"}` — the same contract the runner uses. Scan/suggest/search result handlers recognize it and show "Stopped" with sensible badge states. The error handler also resets the suggest badge to "Error" instead of leaving it frozen.

### 4.3 Closing the window mid-task could abort the whole process
- **Where:** `gui_window.closeEvent`
- **Symptom:** `thread.quit()` does not interrupt a running worker; after the 3 s `wait()` timed out, Qt destroyed a still-running `QThread` → fatal "QThread: Destroyed while thread is still running".
- **Fix:** stop is requested (now actually honored, §4.1), the wait extended to 10 s, with `terminate()` as a last-resort fallback before teardown.

### 4.4 Search progress throttle filtered nothing
- **Where:** `gui_backend._should_forward_search_progress`
- **Symptom:** the filter read `trial_index` / `epoch` at the top level of the payload, but `emit_progress` nests them under `"data"` — every per-epoch trial event flooded through and the `checkpoint_updated` suppression was unreachable.
- **Fix:** the filter reads both shapes (runner events are nested; scan events are flat). The regression test now covers the nested shape too.

### 4.5 Legacy path fields / Suggest-before-scan dead ends
- **Symptom:** the Advanced input-feature/ground-truth fields were accepted as a data source but ignored by the builder (only used to guess a root), and clicking "Suggest Initial Settings" before scanning raised a raw `FileNotFoundError` dialog.
- **Fix:** with §1.6 and §1.7, both paths now genuinely work — suggest builds the cache on demand, and pointing at the SPData folder resolves correctly. The misleading "README schema" error message was also corrected.

## 5. Licensing — client (`xfmr_v2/licensing/`)

### 5.1 Server-rejected heartbeats were treated as successful renewals (fail-open) — **license bypass**
- **Where:** `controller.send_heartbeat_once`
- **Symptom:** the server returns lease denials as HTTP 200 with `ok: false`; the controller only treated *exceptions* as failures, so a denial was recorded as a healthy heartbeat (`heartbeat_failures=0`, phase `checked_out`). A laptop sleeping past the 120 s lease TTL kept running "licensed" on a seat the server had already given away — N seats could run on N+1 machines.
- **Fix:** `ok: false` is now handled as an authoritative loss of the lease: immediate transition to `license_required` (no grace window — the seat is gone), lease cleared, reason surfaced. `LicenseHeartbeatResult` gained `reason_code`/`message` parsing.

### 5.2 In-flight heartbeats could resurrect a released lease
- **Where:** `controller` (snapshot → network call → unconditional write-back)
- **Symptom:** a `release()` / `clear_configuration()` / reconnect racing a 5 s heartbeat HTTP call was overwritten by the stale pre-release snapshot when the heartbeat completed.
- **Fix:** heartbeat results are applied via `_set_state_for_lease`, which discards the write-back if the current lease/server no longer match the snapshot the heartbeat was sent for.

### 5.3 Replaced heartbeat threads could run forever alongside the new one
- **Symptom:** the loop read `self._heartbeat_stop` per iteration; after a reconnect replaced the event (old thread still in its HTTP call past the 1 s join), the old thread latched onto the new event and never stopped — two concurrent heartbeat loops.
- **Fix:** each loop thread owns its stop event (passed as an argument), so a replaced thread always sees its own, already-set event.

## 6. Licensing — server (`license_server/`)

### 6.1 Heartbeats extended leases indefinitely past the license term
- **Where:** `lease_service.heartbeat` / `checkout`
- **Symptom:** the license term was enforced only at checkout; a client checked out one second before `ends_at` could renew every 30 s forever.
- **Fix:** heartbeat re-validates the license term (denying with `license_expired` and releasing the lease once the term ends), and both checkout and heartbeat clamp the lease expiry to the license `ends_at`.

### 6.2 macOS clients were rejected with an opaque 422
- **Where:** `schemas/common.ClientPlatform` vs client `current_platform_name()`
- **Symptom:** the client reports `"macos"` on darwin; the server enum only accepted `windows`/`linux` (`extra="forbid"`), making every macOS checkout a guaranteed validation error.
- **Fix:** `MACOS` added to the server enum. (The fixture exercising the invalid-platform path now uses `"solaris"`.)

### 6.3 Windows fingerprinting crashed on non-Windows hosts
- **Where:** `crypto/fingerprint.read_machine_token`
- **Symptom:** `import winreg` was guarded by `except OSError`, but a missing module raises `ImportError` — requesting a Windows fingerprint on Linux crashed instead of falling through to `/etc/machine-id`. (This was also the root cause of four long-failing server tests.)
- **Fix:** the guard catches `(OSError, ImportError)`.

## 7. Suggest & search (`xfmr_v2/suggest.py`, `xfmr_v2/search.py`)

### 7.1 Suggestion pipeline used a different feature set than training
- **Symptom:** suggest dropped features with `std ≤ 1e-8` while training uses `max != min` — exactly the failure mode data.py's own comment warns about (tiny SI-unit values like farads ~1e-13 were dropped). Diagnostics, input dims, and overfit estimates were computed on the wrong feature set; valid datasets could be rejected as "all columns constant".
- **Fix:** suggest now uses the same `max != min` mask and the same `std == 0` normalization clamp as training.

### 7.2 Division by zero in `fastest_acceptable` scoring
- **Symptom:** a best trial with MAE exactly 0 (degenerate constant targets) divided by zero in the penalty term.
- **Fix:** guarded.

## 8. Product-name typo (repo-wide)

`"Surrogate Model Traning Suite"` → `"Surrogate Model Training Suite"` in **all** locations at once, keeping every contract consistent: `PRODUCT_NAME` on client and server, `APP_NAME` (config/lease/session and Documents paths), all signed-license test fixtures, the Inno Setup installer, Windows/Linux packaging scripts, the `.desktop` file, Dockerfile, smoke tests, and all documentation.

> **Deployment note:** any license issued *before* this rename carries the old product string and will fail the product-match check against a renamed server (re-issue required), and existing installs will start with fresh config/session state because the storage folder name changed. Both are non-issues pre-release.

## 9. Test-suite repairs (pre-existing failures)

- `tests/license_server/test_core_services.py` — 4 errors fixed by §6.3.
- `tests/test_gui_window.py` — 2 stale tests updated to the current UI (removed README widget, renamed scan button, added "Test Samples" tab, reintroduced avg-MAE transfer plot, scan-completion log message).
- `tests/license_vendor/test_cli.py` → renamed `test_vendor_cli.py` — a pytest basename collision with `tests/license_server/test_cli.py`, previously masked because the latter failed to import (missing `typer`, now installed in the `_GUI` env along with `fastapi`/`httpx`).
- `tests/test_gui_backend.py` — throttle test extended to cover the real (nested) progress payload shape.

---

## Known issues deliberately left open

- **Hard-coded Windows default `DATA_ROOT`** in `data.py` — harmless (only reached when no path is supplied) but pointless on non-Windows machines.
- **Client `default_machine_id` instability** (`hostname + uuid.getnode()`): on hosts with no readable MAC, restarted clients can transiently hold extra seats until the 120 s TTL. Low impact, no bypass.
- **`load_license_client_config`** has no corruption handling and non-atomic writes — currently latent (the GUI persists the server URL through its main config payload; only tests use this module).
- **Server responses with malformed numeric fields** raise plain `ValueError` instead of a `LicenseClientError` subtype — fail-closed, requires a misbehaving server.
