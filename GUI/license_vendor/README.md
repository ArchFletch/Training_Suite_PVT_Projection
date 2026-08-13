# Vendor License Tool Package

This package is reserved for the internal-only tooling that will create and sign evaluation and
paid licenses after your team approves a POC or validates procurement payment.

Keeping it separate from `license_server` prevents vendor signing logic from being mixed into
customer-deployed runtime code.

## Prerequisites

Run every command below from the `GUI/` directory, where `license_server` and `license_vendor` are
sibling packages. The vendor tool imports the shared license schemas from `license_server.schemas`,
so copying `license_vendor/` on its own to a vendor workstation fails at import time. Copy the whole
`GUI/` directory instead, or put it on `PYTHONPATH`.

## Send and Never Send

Send to the customer:

- `license.json`, the signed license file
- `vendor_public_key.pem`, written by `export-public-key`

Never send, and never copy off the vendor machine:

- `vendor_signing_key.json`, which holds the Ed25519 private seed that signs every license
- the issuance log, which is the vendor-side record of who was issued what

## CLI

Create a signing key once:

```powershell
python -m license_vendor init-key --key-id 2026-01 --output vendor_signing_key.json
```

Export the public half of that key for the customer server, which verifies licenses from a
SubjectPublicKeyInfo PEM:

```powershell
python -m license_vendor export-public-key --signing-key-file vendor_signing_key.json --output vendor_public_key.pem
```

Issue an evaluation license from a customer request file:

```powershell
python -m license_vendor issue-eval `
  --request-file license_request.json `
  --signing-key-file vendor_signing_key.json `
  --company-name "Acme Design House" `
  --seat-count 2 `
  --output eval_license.json `
  --issuance-log issuance_log.jsonl
```

Issue a paid replacement for the same server:

```powershell
python -m license_vendor issue-paid `
  --request-file license_request.json `
  --signing-key-file vendor_signing_key.json `
  --company-name "Acme Design House" `
  --seat-count 5 `
  --output paid_license.json `
  --issuance-log issuance_log.jsonl
```

## PowerShell Helper

If you want one editable file that reuses the same signing key, issues an
evaluation `license.json`, and creates `vendor_public_key.pem` when needed, use:

```powershell
powershell -ExecutionPolicy Bypass -File .\license_vendor\issue_eval_license.ps1
```
