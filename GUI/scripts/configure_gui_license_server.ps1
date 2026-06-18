[CmdletBinding()]
param(
    [string]$ServerUrl = "http://mlp-license-01:27850",
    [string]$ConfigPath = "$env:APPDATA\Surrogate Model Training Suite\license_client.json"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$configDir = Split-Path -Parent $ConfigPath
if ($configDir) {
    New-Item -ItemType Directory -Force -Path $configDir | Out-Null
}

$payload = @{
    server_url = $ServerUrl
} | ConvertTo-Json -Depth 3

Set-Content -Path $ConfigPath -Value $payload -Encoding UTF8

Write-Host "Wrote GUI license config to $ConfigPath"
Write-Host $payload
