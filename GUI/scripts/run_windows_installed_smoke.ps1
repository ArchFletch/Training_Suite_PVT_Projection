[CmdletBinding()]
param(
    [string]$ServerBundleRoot = "$env:ProgramFiles\MLP License Server",
    [string]$GuiExecutablePath = "$env:LOCALAPPDATA\Programs\Surrogate Model Traning Suite\SurrogateModelTrainingSuite.exe",
    [string]$SmokeRoot = "",
    [string]$LicensePython = "",
    [string]$ServerHost = "127.0.0.1",
    [int]$ServerPort = 27850,
    [switch]$LaunchGui,
    [switch]$SetupOnly,
    [switch]$StopServer
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$smokeScript = Join-Path $repoRoot "scripts\smoke_test_license_server.ps1"
$serverAdminCommand = Join-Path $ServerBundleRoot "mlp-license-server-admin.cmd"
$serverRunCommand = Join-Path $ServerBundleRoot "run-server.cmd"

if (-not (Test-Path $smokeScript)) {
    throw "Smoke script not found: $smokeScript"
}
if (-not (Test-Path $serverAdminCommand)) {
    throw "Bundled admin wrapper not found: $serverAdminCommand"
}
if (-not (Test-Path $serverRunCommand)) {
    throw "Bundled server runner not found: $serverRunCommand"
}
if ($LaunchGui -and -not (Test-Path $GuiExecutablePath)) {
    throw "Installed GUI executable not found: $GuiExecutablePath"
}

$arguments = @(
    "-ExecutionPolicy", "Bypass",
    "-File", $smokeScript,
    "-Mode", "installed",
    "-ServerAdminCommand", $serverAdminCommand,
    "-ServerRunCommand", $serverRunCommand,
    "-ServerHost", $ServerHost,
    "-ServerPort", $ServerPort
)

if ($SmokeRoot) {
    $arguments += @("-SmokeRoot", $SmokeRoot)
}
if ($LicensePython) {
    $arguments += @("-LicensePython", $LicensePython)
}
if ($LaunchGui) {
    $arguments += @("-LaunchGui", "-GuiCommand", $GuiExecutablePath)
}
if ($SetupOnly) {
    $arguments += "-SetupOnly"
}
if ($StopServer) {
    $arguments += "-StopServer"
}

& powershell.exe @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Installed smoke wrapper failed with exit code $LASTEXITCODE"
}
