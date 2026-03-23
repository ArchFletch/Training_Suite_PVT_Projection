[CmdletBinding()]
param(
    [string]$GuiInstallerPath = "",
    [string]$ServerBundleArchivePath = "",
    [string]$OutputPath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $GuiInstallerPath) {
    $GuiInstallerPath = Join-Path $repoRoot "artifacts\packaging\windows\installer\SurrogateModelTrainingSuite-Windows.exe"
}
if (-not $ServerBundleArchivePath) {
    $ServerBundleArchivePath = Join-Path $repoRoot "artifacts\packaging\license_server\windows\MLP License Server.zip"
}
if (-not $OutputPath) {
    $OutputPath = Join-Path $repoRoot "artifacts\packaging\windows\release_manifest.json"
}

function Get-ArtifactMetadata {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $resolvedPath = (Resolve-Path $Path).Path
    $item = Get-Item $resolvedPath
    $hash = Get-FileHash -Algorithm SHA256 -Path $resolvedPath

    return [ordered]@{
        path = $resolvedPath
        size_bytes = $item.Length
        last_write_time = $item.LastWriteTime.ToString("o")
        sha256 = $hash.Hash.ToLowerInvariant()
    }
}

$guiMetadata = Get-ArtifactMetadata -Path $GuiInstallerPath
$serverMetadata = Get-ArtifactMetadata -Path $ServerBundleArchivePath

$manifest = [ordered]@{
    generated_at = (Get-Date).ToUniversalTime().ToString("o")
    gui_installer = $guiMetadata
    server_bundle_archive = $serverMetadata
    default_paths = [ordered]@{
        gui_install_root = "$env:LOCALAPPDATA\Programs\Surrogate Model Traning Suite"
        server_install_root = "$env:ProgramFiles\MLP License Server"
        server_runtime_root = "$env:ProgramData\MLP License Server"
    }
    rehearsal_helpers = [ordered]@{
        installed_smoke_wrapper = (Resolve-Path (Join-Path $PSScriptRoot "run_windows_installed_smoke.ps1")).Path
        gui_config_helper = (Resolve-Path (Join-Path $PSScriptRoot "configure_gui_license_server.ps1")).Path
        rehearsal_doc = (Resolve-Path (Join-Path $PSScriptRoot "..\doc\integration\windows_packaged_install_rehearsal.md")).Path
    }
}

$outputDir = Split-Path -Parent $OutputPath
if ($outputDir) {
    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
}

$manifest | ConvertTo-Json -Depth 6 | Set-Content -Path $OutputPath -Encoding UTF8
Write-Host "Wrote manifest to $OutputPath"
