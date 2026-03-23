# License Server MVP Rehearsal

## Purpose

Preserve the historical source-based rehearsal notes for the licensing MVP.

## Current Status

This document is now historical reference only.

For current release acceptance of the Windows delivery slice, use:

- `windows_packaged_install_rehearsal.md`

## What This Historical Doc Still Covers

- source-based local server bring-up
- source-based vendor issuance
- source-based CLI and API happy-path validation
- the pre-packaging integration context that led to the current packaging slice

## What It Does Not Cover

- packaged Windows GUI install
- packaged Windows server service bundle install
- Program Files and ProgramData validation
- clean-install rehearsal outside the repo
- pristine-VM signoff

## Historical Source-Based Baseline

The source-based baseline is still useful for fast regression checks:

- `tests/license_server/*`
- `tests/license_vendor/*`
- `tests/gui_license/*`
- `scripts/smoke_test_license_server.ps1`

That baseline proves the wiring and behavior are healthy in a development
context, but it is not sufficient for customer-facing release readiness.

## Historical Outcome

The source-based happy path for:

`init -> export-request -> issue license -> import-license -> status -> checkout -> heartbeat -> release`

is the baseline that the packaged-install rehearsal now needs to mirror from
installed artifacts.

## Source References

- `windows_packaged_install_rehearsal.md`
- `../../scripts/smoke_test_license_server.ps1`
