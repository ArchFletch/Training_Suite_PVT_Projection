[CmdletBinding()]
param(
    [ValidateSet("source", "installed")]
    [string]$Mode = "source",
    [string]$SmokeRoot,
    [string]$LicensePython,
    [string]$GuiPython,
    [string]$ServerAdminCommand = "",
    [string]$ServerAdminWorkingDir = "",
    [string]$ServerRunCommand = "",
    [string]$ServerRunWorkingDir = "",
    [string[]]$ServerRunArguments = @(),
    [string]$GuiCommand = "",
    [string]$GuiWorkingDir = "",
    [string[]]$GuiLaunchArguments = @(),
    [string]$LicenseClientConfigPath = "",
    [string]$ServerHost = "127.0.0.1",
    [int]$ServerPort = 27850,
    [string]$CompanyName = "Acme Design House",
    [int]$SeatCount = 2,
    [string]$RequestedBy = "local-smoke",
    [string]$LicenseId = "lic_eval_smoke_001",
    [string]$StartDate = "",
    [int]$TermDays = 14,
    [switch]$SetupOnly,
    [switch]$LaunchGui,
    [switch]$KeepExistingFiles,
    [switch]$StopServer
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-DefaultPath {
    param(
        [string]$Value,
        [string]$DefaultValue
    )

    if ([string]::IsNullOrWhiteSpace($Value)) {
        return $DefaultValue
    }
    return $Value
}

function Resolve-CommandPath {
    param([string]$Value)

    if ([string]::IsNullOrWhiteSpace($Value)) {
        return ""
    }
    if (Test-Path $Value) {
        return (Resolve-Path $Value).Path
    }

    $command = Get-Command $Value -ErrorAction SilentlyContinue
    if ($command) {
        if ($command.Source) {
            return $command.Source
        }
        return $command.Name
    }

    return $Value
}

function Test-CommandAvailable {
    param([string]$Value)

    if ([string]::IsNullOrWhiteSpace($Value)) {
        return $false
    }
    if (Test-Path $Value) {
        return $true
    }
    return $null -ne (Get-Command $Value -ErrorAction SilentlyContinue)
}

function Resolve-WorkingDirectory {
    param(
        [string]$Value,
        [string]$CommandPath,
        [string]$DefaultValue
    )

    if (-not [string]::IsNullOrWhiteSpace($Value)) {
        return $Value
    }
    if (-not [string]::IsNullOrWhiteSpace($CommandPath) -and (Test-Path $CommandPath)) {
        $parent = Split-Path -Parent (Resolve-Path $CommandPath).Path
        if ($parent) {
            return $parent
        }
    }
    return $DefaultValue
}

function Invoke-ExternalCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [string]$WorkingDirectory = ""
    )

    if ($WorkingDirectory) {
        Push-Location $WorkingDirectory
    }

    try {
        & $Command @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed: $Command $($Arguments -join ' ')"
        }
    } finally {
        if ($WorkingDirectory) {
            Pop-Location
        }
    }
}

function Invoke-ExternalCommandCapture {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [string]$WorkingDirectory = ""
    )

    if ($WorkingDirectory) {
        Push-Location $WorkingDirectory
    }

    try {
        $output = & $Command @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed: $Command $($Arguments -join ' ')"
        }
        return $output
    } finally {
        if ($WorkingDirectory) {
            Pop-Location
        }
    }
}

function Stop-ManagedServer {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PidPath
    )

    if (-not (Test-Path $PidPath)) {
        return $false
    }

    $pidText = (Get-Content -Path $PidPath -Raw -Encoding UTF8).Trim()
    if ($pidText) {
        try {
            $rootProcessId = [int]$pidText
            $processIdsToStop = [System.Collections.Generic.HashSet[int]]::new()
            $pending = [System.Collections.Generic.Queue[int]]::new()
            $pending.Enqueue($rootProcessId)

            while ($pending.Count -gt 0) {
                $currentProcessId = $pending.Dequeue()
                if (-not $processIdsToStop.Add($currentProcessId)) {
                    continue
                }

                $childProcesses = Get-CimInstance Win32_Process -Filter "ParentProcessId = $currentProcessId" -ErrorAction SilentlyContinue
                foreach ($childProcess in $childProcesses) {
                    $pending.Enqueue([int]$childProcess.ProcessId)
                }
            }

            $processIds = $processIdsToStop.ToArray()
            [Array]::Reverse($processIds)
            foreach ($processId in $processIds) {
                Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
            }
            foreach ($processId in $processIds) {
                Wait-Process -Id $processId -Timeout 5 -ErrorAction SilentlyContinue
            }
        } catch {
            # The process may already be gone, which is fine for cleanup.
        }
    }

    Remove-Item -Path $PidPath -Force -ErrorAction SilentlyContinue
    return $true
}

function Wait-ForHttpStatus {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Url,
        [int]$TimeoutSeconds = 15
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            return Invoke-RestMethod -Uri $Url -Method Get -TimeoutSec 2
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }

    throw "Timed out waiting for $Url"
}

function Stop-SmokeServerOnPort {
    param(
        [int]$Port,
        [string]$SmokeRoot
    )

    try {
        $listeners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop
    } catch {
        return $false
    }

    foreach ($listener in $listeners) {
        $processId = [int]$listener.OwningProcess
        $processInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $processId" -ErrorAction SilentlyContinue
        if ($null -eq $processInfo) {
            continue
        }

        $commandLine = $processInfo.CommandLine
        if (($commandLine -like "*license_server.main:app*") -or ($commandLine -like "*$SmokeRoot*")) {
            try {
                Stop-Process -Id $processId -Force -ErrorAction Stop
                Start-Sleep -Seconds 1
                return $true
            } catch {
                # Fall through to the final failure below.
            }
        }

        throw "Port $Port is already in use by process $processId and it does not look like the managed smoke server."
    }

    return $false
}

function Remove-PathWithRetry {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PathToRemove,
        [int]$Attempts = 5
    )

    for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
        if (-not (Test-Path $PathToRemove)) {
            return
        }
        try {
            Remove-Item -Path $PathToRemove -Recurse -Force
            return
        } catch {
            if ($attempt -eq $Attempts) {
                throw
            }
            Start-Sleep -Milliseconds 750
        }
    }
}

function Save-LicenseClientConfig {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$ServerUrl
    )

    $parentDir = Split-Path -Parent $Path
    if ($parentDir) {
        New-Item -ItemType Directory -Force -Path $parentDir | Out-Null
    }

    $payload = @{ server_url = $ServerUrl } | ConvertTo-Json -Depth 3
    Set-Content -Path $Path -Value $payload -Encoding UTF8
}

function Convert-ToCmdArgument {
    param([string]$Value)

    if ($null -eq $Value -or $Value -eq "") {
        return '""'
    }
    if ($Value -match '[\s"]') {
        return '"' + $Value.Replace('"', '""') + '"'
    }
    return $Value
}

function Join-CmdArguments {
    param([string[]]$Arguments)

    if ($null -eq $Arguments -or $Arguments.Count -eq 0) {
        return ""
    }

    return (($Arguments | ForEach-Object { Convert-ToCmdArgument -Value $_ }) -join " ")
}

function Write-BackgroundCommandScript {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Command,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [Parameter(Mandatory = $true)]
        [string]$WorkingDirectory,
        [Parameter(Mandatory = $true)]
        [string]$RuntimeRoot
    )

    $quotedCommand = Convert-ToCmdArgument -Value $Command
    $quotedWorkingDirectory = Convert-ToCmdArgument -Value $WorkingDirectory
    $joinedArguments = Join-CmdArguments -Arguments $Arguments
    $commandExtension = [System.IO.Path]::GetExtension($Command)
    $commandLine = if ($commandExtension -in @(".cmd", ".bat")) {
        "call $quotedCommand"
    } else {
        $quotedCommand
    }
    if ($joinedArguments) {
        $commandLine = "$commandLine $joinedArguments"
    }

    $scriptLines = @(
        "@echo off",
        "setlocal",
        "set MLP_LICENSE_SERVER_RUNTIME_DIR=$RuntimeRoot",
        "pushd $quotedWorkingDirectory >nul",
        $commandLine,
        'set "EXIT_CODE=%ERRORLEVEL%"',
        "popd >nul",
        "exit /b %EXIT_CODE%"
    )

    Set-Content -Path $Path -Value ($scriptLines -join "`r`n") -Encoding ascii
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$projectsRoot = Split-Path -Parent $repoRoot
$runtimeBase = Join-Path $projectsRoot "MLP_modeling_v2_runtime"
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"

$defaultSmokeLeaf = if ($Mode -eq "installed") {
    Join-Path "install-smoke" $timestamp
} else {
    "local-smoke"
}

$SmokeRoot = Resolve-DefaultPath -Value $SmokeRoot -DefaultValue (Join-Path $runtimeBase $defaultSmokeLeaf)
$defaultLicensePython = Join-Path $runtimeBase "integration-qa\license-test-venv\Scripts\python.exe"
if (-not (Test-Path $defaultLicensePython)) {
    $defaultLicensePython = Join-Path $repoRoot ".venv\Scripts\python.exe"
}
$LicensePython = Resolve-DefaultPath -Value $LicensePython -DefaultValue $defaultLicensePython
$defaultGuiPython = Join-Path $repoRoot ".venv\Scripts\pythonw.exe"
if (-not (Test-Path $defaultGuiPython)) {
    $defaultGuiPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
}
$GuiPython = Resolve-DefaultPath -Value $GuiPython -DefaultValue $defaultGuiPython
$StartDate = Resolve-DefaultPath -Value $StartDate -DefaultValue (Get-Date -Format "yyyy-MM-dd")

$vendorDir = Join-Path $SmokeRoot "vendor"
$serverRoot = Join-Path $SmokeRoot "server"
$requestPath = Join-Path $SmokeRoot "license_request.json"
$licensePath = Join-Path $SmokeRoot "eval_license.json"
$keyJson = Join-Path $vendorDir "vendor_signing_key.json"
$keyPem = Join-Path $vendorDir "vendor_public_key.pem"
$issuanceLog = Join-Path $vendorDir "issuance_log.jsonl"
$pidPath = Join-Path $serverRoot "uvicorn.pid"
$serverOutLog = Join-Path $serverRoot "uvicorn.out.log"
$serverErrLog = Join-Path $serverRoot "uvicorn.err.log"
$statusUrl = "http://$ServerHost`:$ServerPort/api/v1/status"
$checkoutUrl = "http://$ServerHost`:$ServerPort/api/v1/checkout"
$releaseUrl = "http://$ServerHost`:$ServerPort/api/v1/release"
$serverUrl = "http://$ServerHost`:$ServerPort"

if (-not (Test-Path $LicensePython)) {
    throw "Vendor/runtime Python not found: $LicensePython"
}

$serverAdminBaseArgs = @()
$serverRunBaseArgs = @()
$guiLaunchBaseArgs = @()

if ($Mode -eq "source") {
    if (-not (Test-Path $GuiPython)) {
        throw "GUI Python not found: $GuiPython"
    }

    $ServerAdminCommand = Resolve-CommandPath (Resolve-DefaultPath -Value $ServerAdminCommand -DefaultValue $LicensePython)
    $ServerRunCommand = Resolve-CommandPath (Resolve-DefaultPath -Value $ServerRunCommand -DefaultValue $LicensePython)
    $GuiCommand = Resolve-CommandPath (Resolve-DefaultPath -Value $GuiCommand -DefaultValue $GuiPython)

    $ServerAdminWorkingDir = Resolve-WorkingDirectory -Value $ServerAdminWorkingDir -CommandPath $ServerAdminCommand -DefaultValue $repoRoot
    $ServerRunWorkingDir = Resolve-WorkingDirectory -Value $ServerRunWorkingDir -CommandPath $ServerRunCommand -DefaultValue $repoRoot
    $GuiWorkingDir = Resolve-WorkingDirectory -Value $GuiWorkingDir -CommandPath $GuiCommand -DefaultValue $repoRoot

    $serverAdminBaseArgs = @("-m", "license_server.cli")
    $serverRunBaseArgs = @("-m", "uvicorn", "license_server.main:app", "--host", $ServerHost, "--port", $ServerPort)
    $guiLaunchBaseArgs = @("launch_gui.py")

    $LicenseClientConfigPath = Resolve-DefaultPath -Value $LicenseClientConfigPath -DefaultValue (Join-Path $repoRoot "artifacts\gui\license_client.json")
} else {
    $ServerAdminCommand = Resolve-CommandPath $ServerAdminCommand
    $ServerRunCommand = Resolve-CommandPath $ServerRunCommand
    $GuiCommand = Resolve-CommandPath $GuiCommand

    if (-not (Test-CommandAvailable $ServerAdminCommand)) {
        throw "Installed mode requires -ServerAdminCommand to point to the bundled admin wrapper."
    }
    if (-not (Test-CommandAvailable $ServerRunCommand)) {
        throw "Installed mode requires -ServerRunCommand to point to the bundled server runner."
    }
    if ($LaunchGui -and -not (Test-CommandAvailable $GuiCommand)) {
        throw "Installed mode requires -GuiCommand when using -LaunchGui."
    }

    $ServerAdminWorkingDir = Resolve-WorkingDirectory -Value $ServerAdminWorkingDir -CommandPath $ServerAdminCommand -DefaultValue $SmokeRoot
    $ServerRunWorkingDir = Resolve-WorkingDirectory -Value $ServerRunWorkingDir -CommandPath $ServerRunCommand -DefaultValue $SmokeRoot
    $GuiWorkingDir = Resolve-WorkingDirectory -Value $GuiWorkingDir -CommandPath $GuiCommand -DefaultValue $SmokeRoot

    $serverRunCommandName = [System.IO.Path]::GetFileName($ServerRunCommand).ToLowerInvariant()
    if ($ServerRunArguments.Count -gt 0) {
        $serverRunBaseArgs = @()
    } elseif ($serverRunCommandName -in @("python.exe", "pythonw.exe")) {
        $serverRunBaseArgs = @("-m", "uvicorn", "license_server.main:app", "--host", $ServerHost, "--port", $ServerPort)
    } else {
        $serverRunBaseArgs = @("--host", $ServerHost, "--port", $ServerPort)
    }
    $LicenseClientConfigPath = Resolve-DefaultPath -Value $LicenseClientConfigPath -DefaultValue (Join-Path $env:APPDATA "Surrogate Model Traning Suite\license_client.json")
}

if ($StopServer) {
    $stopped = Stop-ManagedServer -PidPath $pidPath
    if ($stopped) {
        Write-Host "Stopped smoke-test license server."
    } else {
        Write-Host "No managed smoke-test license server was running."
    }
    return
}

if (-not $KeepExistingFiles) {
    Stop-ManagedServer -PidPath $pidPath | Out-Null
    Stop-SmokeServerOnPort -Port $ServerPort -SmokeRoot $SmokeRoot | Out-Null
    if (Test-Path $SmokeRoot) {
        Remove-PathWithRetry -PathToRemove $SmokeRoot
    }
}

New-Item -ItemType Directory -Force -Path $vendorDir | Out-Null
New-Item -ItemType Directory -Force -Path $serverRoot | Out-Null
Save-LicenseClientConfig -Path $LicenseClientConfigPath -ServerUrl $serverUrl

Push-Location $repoRoot
try {
    Invoke-ExternalCommand -Command $LicensePython -Arguments @(
        "-m", "license_vendor.cli",
        "init-key",
        "--key-id", "local-smoke",
        "--output", $keyJson
    ) -WorkingDirectory $repoRoot

    $pemScript = @"
import base64
import json
from pathlib import Path
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

key_json = Path(r"$keyJson")
key_pem = Path(r"$keyPem")
data = json.loads(key_json.read_text(encoding="utf-8"))
public_key = base64.b64decode(data["public_key"])
pem = ed25519.Ed25519PublicKey.from_public_bytes(public_key).public_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PublicFormat.SubjectPublicKeyInfo,
)
key_pem.write_bytes(pem)
print(key_pem)
"@
    $pemScript | & $LicensePython -
    if ($LASTEXITCODE -ne 0) {
        throw "Could not convert vendor public key to PEM."
    }

    Invoke-ExternalCommand -Command $ServerAdminCommand -Arguments ($serverAdminBaseArgs + @(
        "init",
        "--runtime-root", $serverRoot
    )) -WorkingDirectory $ServerAdminWorkingDir

    Invoke-ExternalCommand -Command $ServerAdminCommand -Arguments ($serverAdminBaseArgs + @(
        "export-request", $requestPath,
        "--runtime-root", $serverRoot,
        "--requested-by", $RequestedBy
    )) -WorkingDirectory $ServerAdminWorkingDir

    Invoke-ExternalCommand -Command $LicensePython -Arguments @(
        "-m", "license_vendor.cli",
        "issue-eval",
        "--request-file", $requestPath,
        "--signing-key-file", $keyJson,
        "--company-name", $CompanyName,
        "--seat-count", $SeatCount.ToString(),
        "--starts-at", $StartDate,
        "--term-days", $TermDays.ToString(),
        "--license-id", $LicenseId,
        "--output", $licensePath,
        "--issuance-log", $issuanceLog
    ) -WorkingDirectory $repoRoot

    Invoke-ExternalCommand -Command $ServerAdminCommand -Arguments ($serverAdminBaseArgs + @(
        "import-license", $licensePath,
        "--runtime-root", $serverRoot,
        "--vendor-public-key", $keyPem
    )) -WorkingDirectory $ServerAdminWorkingDir

    $statusJson = Invoke-ExternalCommandCapture -Command $ServerAdminCommand -Arguments ($serverAdminBaseArgs + @(
        "show-status",
        "--runtime-root", $serverRoot
    )) -WorkingDirectory $ServerAdminWorkingDir
    Write-Host $statusJson

    if (-not $SetupOnly) {
        Stop-ManagedServer -PidPath $pidPath | Out-Null

        $serverLaunchScript = Join-Path $serverRoot "start-server.cmd"
        Write-BackgroundCommandScript `
            -Path $serverLaunchScript `
            -Command $ServerRunCommand `
            -Arguments ($serverRunBaseArgs + $ServerRunArguments) `
            -WorkingDirectory $ServerRunWorkingDir `
            -RuntimeRoot $serverRoot

        $process = Start-Process `
            -FilePath $env:ComSpec `
            -ArgumentList @("/c", $serverLaunchScript) `
            -WorkingDirectory $serverRoot `
            -RedirectStandardOutput $serverOutLog `
            -RedirectStandardError $serverErrLog `
            -PassThru
        Set-Content -Path $pidPath -Value $process.Id -Encoding UTF8

        try {
            $status = Wait-ForHttpStatus -Url $statusUrl
        } catch {
            Stop-ManagedServer -PidPath $pidPath | Out-Null
            throw
        }

        $checkoutPayload = @{
            product = "Surrogate Model Traning Suite"
            product_version = "0.1.0"
            machine_id = "smoke-client-1"
            hostname = $env:COMPUTERNAME
            username = $env:USERNAME
            platform = "windows"
        } | ConvertTo-Json -Compress
        $checkout = Invoke-RestMethod -Method Post -Uri $checkoutUrl -ContentType "application/json" -Body $checkoutPayload
        if (-not $checkout.granted) {
            $checkoutJson = $checkout | ConvertTo-Json -Depth 6
            Stop-ManagedServer -PidPath $pidPath | Out-Null
            throw "Checkout was not granted: $checkoutJson"
        }
        Start-Sleep -Seconds 2
        $releasePayload = @{ lease_id = $checkout.lease_id } | ConvertTo-Json -Compress
        $release = Invoke-RestMethod -Method Post -Uri $releaseUrl -ContentType "application/json" -Body $releasePayload

        $summary = [ordered]@{
            mode = $Mode
            smoke_root = $SmokeRoot
            server_url = $serverUrl
            runtime_root = $serverRoot
            request_file = $requestPath
            license_file = $licensePath
            vendor_key_json = $keyJson
            vendor_key_pem = $keyPem
            issuance_log = $issuanceLog
            license_client_config = $LicenseClientConfigPath
            server_admin_command = $ServerAdminCommand
            server_run_command = $ServerRunCommand
            server_pid = $process.Id
            status_ok = $status.ok
            checkout_granted = $checkout.granted
            checkout_lease_id = $checkout.lease_id
            release_ok = $release.ok
            gui_command = if ($GuiCommand) { "$GuiCommand $($guiLaunchBaseArgs + $GuiLaunchArguments -join ' ')" } else { "" }
        }
        $summary | ConvertTo-Json -Depth 4

        if ($LaunchGui) {
            Start-Process -FilePath $GuiCommand -ArgumentList ($guiLaunchBaseArgs + $GuiLaunchArguments) -WorkingDirectory $GuiWorkingDir | Out-Null
        }
    }
} finally {
    Pop-Location
}
