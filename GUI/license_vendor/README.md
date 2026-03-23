# Vendor License Tool Package

This package is reserved for the internal-only tooling that will create and sign evaluation and
paid licenses after your team approves a POC or validates procurement payment.

Keeping it separate from `license_server` prevents vendor signing logic from being mixed into
customer-deployed runtime code.

## CLI

Create a signing key once:

```powershell
python -m license_vendor init-key --key-id 2026-01 --output vendor_signing_key.json
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
