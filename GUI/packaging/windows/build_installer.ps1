param(
    [Parameter(Mandatory = $true)]
    [string]$StandaloneDir,
    [string]$OutputDir = "",
    [string]$IsccPath = "",
    [string]$AppExecutableBaseName = "SurrogateModelTrainingSuite"
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\\..")).Path
$standaloneDir = (Resolve-Path $StandaloneDir).Path
if (-not $OutputDir) {
    $OutputDir = Join-Path $repoRoot "artifacts\\packaging\\windows\\installer"
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$appExe = Join-Path $standaloneDir "$AppExecutableBaseName.exe"
if (-not (Test-Path $appExe)) {
    throw "Expected packaged executable at $appExe"
}

if (-not $IsccPath) {
    $discoveredIscc = Get-Command "ISCC.exe" -ErrorAction SilentlyContinue
    if ($discoveredIscc) {
        $IsccPath = $discoveredIscc.Source
    } else {
        $defaultIscc = "C:\\Program Files (x86)\\Inno Setup 6\\ISCC.exe"
        if (Test-Path $defaultIscc) {
            $IsccPath = $defaultIscc
        } else {
            $userInstallIscc = Join-Path $env:LOCALAPPDATA "Programs\\Inno Setup 6\\ISCC.exe"
            if (Test-Path $userInstallIscc) {
                $IsccPath = $userInstallIscc
            }
        }
    }
}

if (-not $IsccPath) {
    throw "Could not find ISCC.exe. Install Inno Setup 6 or pass -IsccPath explicitly."
}

$issPath = Join-Path $repoRoot "packaging\\windows\\SurrogateModelTrainingSuite.iss"
& $IsccPath "/DAppBuildDir=$standaloneDir" "/DOutputDir=$OutputDir" "/DAppExecutableBaseName=$AppExecutableBaseName" $issPath

Write-Host "Windows installer created in $OutputDir"
