param(
    [string]$Python = "python",
    [string]$OutputDir = "",
    [string]$BundleName = "MLP License Server",
    [string]$WinSWExePath = "",
    [switch]$SkipArchive
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-ExternalCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [string]$WorkingDirectory = ""
    )

    if ($WorkingDirectory) {
        Push-Location $WorkingDirectory
    }

    try {
        & $FilePath @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed: $FilePath $($Arguments -join ' ')"
        }
    } finally {
        if ($WorkingDirectory) {
            Pop-Location
        }
    }
}

function Get-PythonRuntimeInfo {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PythonCommand
    )

    $pythonInfoJson = & $PythonCommand -c "import json, sys; print(json.dumps({'executable': sys.executable, 'prefix': sys.prefix, 'base_prefix': sys.base_prefix}))"
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to inspect Python runtime: $PythonCommand"
    }

    return $pythonInfoJson | ConvertFrom-Json
}

function Copy-DirectoryTree {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Source,
        [Parameter(Mandatory = $true)]
        [string]$Destination
    )

    $sourceRoot = (Resolve-Path $Source).Path
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null

    foreach ($item in Get-ChildItem -LiteralPath $sourceRoot -Recurse -Force) {
        $relativePath = $item.FullName.Substring($sourceRoot.Length).TrimStart("\")
        if (-not $relativePath) {
            continue
        }

        if ($item.PSIsContainer) {
            if ($item.Name -eq "__pycache__") {
                continue
            }
            New-Item -ItemType Directory -Force -Path (Join-Path $Destination $relativePath) | Out-Null
            continue
        }

        if ($item.Extension -eq ".pyc" -or $item.FullName -like "*\__pycache__\*") {
            continue
        }

        $destinationPath = Join-Path $Destination $relativePath
        $destinationDir = Split-Path -Parent $destinationPath
        if ($destinationDir) {
            New-Item -ItemType Directory -Force -Path $destinationDir | Out-Null
        }
        Copy-Item -LiteralPath $item.FullName -Destination $destinationPath -Force
    }
}

function Write-CmdScript {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string[]]$Lines
    )

    Set-Content -Path $Path -Value ($Lines -join "`r`n") -Encoding ascii
}

function Remove-PathWithRetry {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PathToRemove,
        [int]$Attempts = 5
    )

    for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
        if (-not (Test-Path -Path $PathToRemove)) {
            return
        }

        try {
            Remove-Item -Path $PathToRemove -Recurse -Force
            return
        } catch {
            if ($attempt -eq $Attempts) {
                throw
            }
            Start-Sleep -Seconds 2
        }
    }
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
if (-not $OutputDir) {
    $OutputDir = Join-Path $repoRoot "artifacts\packaging\license_server\windows"
}

if (-not $WinSWExePath) {
    throw "Provide -WinSWExePath to the WinSW executable that should be bundled."
}

$winswSource = (Resolve-Path $WinSWExePath).Path
$requirementsPath = Join-Path $PSScriptRoot "runtime_requirements.txt"
$bundleRoot = Join-Path $OutputDir $BundleName
$bundleArchiveName = "$BundleName.zip"
$bundleArchivePath = Join-Path $OutputDir $bundleArchiveName
$runtimeInfo = Get-PythonRuntimeInfo -PythonCommand $Python
$pythonRuntimeSource = $runtimeInfo.base_prefix
$pythonRuntimeDir = Join-Path $bundleRoot "python"
$bundlePython = Join-Path $pythonRuntimeDir "python.exe"
$packageRoot = Join-Path $bundleRoot "license_server"
$installScriptSource = Join-Path $PSScriptRoot "install_windows_service.ps1"
$templateSource = Join-Path $PSScriptRoot "mlp-license-server-service.xml.template"
$configTemplateSource = Join-Path $repoRoot "packaging\license_server\shared\config.windows.toml.sample"

if (Test-Path -Path $bundleRoot) {
    Remove-PathWithRetry -PathToRemove $bundleRoot
}
if ((-not $SkipArchive) -and (Test-Path -Path $bundleArchivePath)) {
    Remove-PathWithRetry -PathToRemove $bundleArchivePath
}

New-Item -ItemType Directory -Force -Path $bundleRoot | Out-Null
Copy-DirectoryTree -Source (Join-Path $repoRoot "license_server") -Destination $packageRoot
Copy-DirectoryTree -Source $pythonRuntimeSource -Destination $pythonRuntimeDir

Copy-Item -Path $installScriptSource -Destination (Join-Path $bundleRoot "install-service.ps1") -Force
Copy-Item -Path $templateSource -Destination (Join-Path $bundleRoot "mlp-license-server-service.xml.template") -Force
Copy-Item -Path $configTemplateSource -Destination (Join-Path $bundleRoot "config.windows.toml.sample") -Force
Copy-Item -Path $winswSource -Destination (Join-Path $bundleRoot "WinSW-x64.exe") -Force

Invoke-ExternalCommand -FilePath $bundlePython -Arguments @("-m", "ensurepip", "--upgrade") -WorkingDirectory $repoRoot
Invoke-ExternalCommand -FilePath $bundlePython -Arguments @("-m", "pip", "install", "--upgrade", "pip") -WorkingDirectory $repoRoot
Invoke-ExternalCommand -FilePath $bundlePython -Arguments @("-m", "pip", "install", "-r", $requirementsPath) -WorkingDirectory $repoRoot

Write-CmdScript -Path (Join-Path $bundleRoot "mlp-license-server-admin.cmd") -Lines @(
    "@echo off",
    "setlocal",
    'set "SCRIPT_DIR=%~dp0"',
    'set "PYTHON_EXE=%SCRIPT_DIR%python\python.exe"',
    'if not defined MLP_LICENSE_SERVER_RUNTIME_DIR set "MLP_LICENSE_SERVER_RUNTIME_DIR=%PROGRAMDATA%\MLP License Server"',
    'pushd "%SCRIPT_DIR%" >nul',
    '"%PYTHON_EXE%" -m license_server.cli %*',
    'set "EXIT_CODE=%ERRORLEVEL%"',
    'popd >nul',
    'exit /b %EXIT_CODE%'
)

Write-CmdScript -Path (Join-Path $bundleRoot "run-server.cmd") -Lines @(
    "@echo off",
    "setlocal",
    'set "SCRIPT_DIR=%~dp0"',
    'set "PYTHON_EXE=%SCRIPT_DIR%python\python.exe"',
    'if not defined MLP_LICENSE_SERVER_RUNTIME_DIR set "MLP_LICENSE_SERVER_RUNTIME_DIR=%PROGRAMDATA%\MLP License Server"',
    'pushd "%SCRIPT_DIR%" >nul',
    '"%PYTHON_EXE%" -m uvicorn license_server.main:app --host 0.0.0.0 --port 27850 %*',
    'set "EXIT_CODE=%ERRORLEVEL%"',
    'popd >nul',
    'exit /b %EXIT_CODE%'
)

Write-CmdScript -Path (Join-Path $bundleRoot "init-server.cmd") -Lines @(
    "@echo off",
    'call "%~dp0mlp-license-server-admin.cmd" init %*'
)

Write-CmdScript -Path (Join-Path $bundleRoot "export-license-request.cmd") -Lines @(
    "@echo off",
    'call "%~dp0mlp-license-server-admin.cmd" export-request %*'
)

Write-CmdScript -Path (Join-Path $bundleRoot "import-license.cmd") -Lines @(
    "@echo off",
    'call "%~dp0mlp-license-server-admin.cmd" import-license %*'
)

Write-CmdScript -Path (Join-Path $bundleRoot "show-status.cmd") -Lines @(
    "@echo off",
    'call "%~dp0mlp-license-server-admin.cmd" show-status %*'
)

if (-not $SkipArchive) {
    Push-Location $OutputDir
    try {
        Compress-Archive -Path $BundleName -DestinationPath $bundleArchivePath -CompressionLevel Optimal
    } finally {
        Pop-Location
    }
}

$bundleSummary = [ordered]@{
    bundle_root = $bundleRoot
    bundle_archive = if ($SkipArchive) { "" } else { $bundleArchivePath }
    bundled_python = $bundlePython
    python_runtime_source = $pythonRuntimeSource
    winsw_executable = Join-Path $bundleRoot "WinSW-x64.exe"
    install_script = Join-Path $bundleRoot "install-service.ps1"
    admin_wrapper = Join-Path $bundleRoot "mlp-license-server-admin.cmd"
    service_runner = Join-Path $bundleRoot "run-server.cmd"
    package_root = $packageRoot
}

$bundleSummary | ConvertTo-Json -Depth 3
