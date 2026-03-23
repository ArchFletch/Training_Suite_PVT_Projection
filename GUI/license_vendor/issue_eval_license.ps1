[CmdletBinding()]
param(
    [string]$RequestFile = "C:\Users\tc57\Downloads\license_request.json",
    [string]$CompanyName = "Test EXE Company",
    [int]$SeatCount = 1,
    [string]$OutputDir = "",
    [string]$KeyId = "2026-03",
    [string]$PythonExe = "",
    [string]$LicenseFileName = "license.json",
    [string]$IssuanceLogFileName = "issuance_log.jsonl",
    [string]$StartsAt = "",
    [string]$EndsAt = "",
    [int]$TermDays = 14,
    [switch]$RefreshPublicKeyPem
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $OutputDir) {
    $OutputDir = Join-Path (Split-Path -Parent $repoRoot) "MLP_modeling_v2_runtime\vendor"
}
if (-not $PythonExe) {
    $bundledPython = Join-Path $repoRoot "artifacts\packaging\license_server\windows\MLP License Server\python\python.exe"
    if (Test-Path -Path $bundledPython) {
        $PythonExe = $bundledPython
    } else {
        throw "Bundled vendor Python not found. Pass -PythonExe explicitly."
    }
}

$resolvedRequestFile = (Resolve-Path $RequestFile).Path
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$signingKeyFile = Join-Path $OutputDir "vendor_signing_key.json"
$publicKeyPemFile = Join-Path $OutputDir "vendor_public_key.pem"
$licenseFile = Join-Path $OutputDir $LicenseFileName
$issuanceLogFile = Join-Path $OutputDir $IssuanceLogFileName

function Invoke-RepoPython {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    Push-Location $repoRoot
    try {
        & $PythonExe @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed: $PythonExe $($Arguments -join ' ')"
        }
    } finally {
        Pop-Location
    }
}

$keyCreated = $false
if (-not (Test-Path -Path $signingKeyFile)) {
    Invoke-RepoPython -Arguments @(
        "-m", "license_vendor",
        "init-key",
        "--key-id", $KeyId,
        "--output", $signingKeyFile
    )
    $keyCreated = $true
}

$issueArguments = @(
    "-m", "license_vendor",
    "issue-eval",
    "--request-file", $resolvedRequestFile,
    "--signing-key-file", $signingKeyFile,
    "--company-name", $CompanyName,
    "--seat-count", $SeatCount.ToString(),
    "--term-days", $TermDays.ToString(),
    "--output", $licenseFile,
    "--issuance-log", $issuanceLogFile
)
if ($StartsAt) {
    $issueArguments += @("--starts-at", $StartsAt)
}
if ($EndsAt) {
    $issueArguments += @("--ends-at", $EndsAt)
}

Invoke-RepoPython -Arguments $issueArguments

$publicKeyPemCreated = $false
if ($RefreshPublicKeyPem -or -not (Test-Path -Path $publicKeyPemFile)) {
    $pemScript = @'
import base64
import json
import sys
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

key_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
data = json.loads(key_path.read_text(encoding="utf-8"))
public_key = ed25519.Ed25519PublicKey.from_public_bytes(base64.b64decode(data["public_key"]))
out_path.write_bytes(
    public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
)
print(out_path)
'@

    Push-Location $repoRoot
    try {
        $pemScript | & $PythonExe - $signingKeyFile $publicKeyPemFile
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to export vendor_public_key.pem."
        }
    } finally {
        Pop-Location
    }
    $publicKeyPemCreated = $true
}

$summary = [ordered]@{
    ok = $true
    request_file = $resolvedRequestFile
    company_name = $CompanyName
    seat_count = $SeatCount
    signing_key_file = $signingKeyFile
    vendor_public_key_pem = $publicKeyPemFile
    license_file = $licenseFile
    issuance_log = $issuanceLogFile
    key_created = $keyCreated
    public_key_pem_created = $publicKeyPemCreated
}

$summary | ConvertTo-Json -Depth 3
