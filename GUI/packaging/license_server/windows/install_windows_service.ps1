[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$ServiceCommand = "",

    [string]$ServiceArguments = "",
    [string]$ServiceId = "mlp-license-server",
    [string]$ServiceName = "MLP License Server",
    [string]$ServiceDescription = "On-prem floating license server for Surrogate Model Training Suite",
    [string]$ServiceRoot = "$env:ProgramFiles\MLP License Server",
    [string]$ProgramDataRoot = "$env:ProgramData\MLP License Server",
    [string]$ConfigTemplatePath = "",
    [string]$WinSWExePath = "",
    [string]$ServiceHost = "0.0.0.0",
    [int]$ServicePort = 27850,
    [switch]$InstallService,
    [switch]$StartService
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Test-IsAdministrator {
    $currentIdentity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($currentIdentity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not $ConfigTemplatePath) {
    $bundledTemplatePath = Join-Path $PSScriptRoot "config.windows.toml.sample"
    if (Test-Path -Path $bundledTemplatePath) {
        $ConfigTemplatePath = $bundledTemplatePath
    } else {
        $ConfigTemplatePath = Join-Path $PSScriptRoot "..\shared\config.windows.toml.sample"
    }
}

if (-not $ServiceCommand) {
    $ServiceCommand = Join-Path $ServiceRoot "python\python.exe"
}

if (-not $ServiceArguments) {
    $ServiceArguments = "-m uvicorn license_server.main:app --host $ServiceHost --port $ServicePort"
}

if (-not $WinSWExePath) {
    $bundledWinSWExePath = Join-Path $PSScriptRoot "WinSW-x64.exe"
    if (Test-Path -Path $bundledWinSWExePath) {
        $WinSWExePath = $bundledWinSWExePath
    }
}

function Escape-Xml {
    param([string]$Value)

    return [System.Security.SecurityElement]::Escape($Value)
}

function Render-Config {
    param(
        [string]$TemplatePath,
        [string]$DestinationPath,
        [string]$DataRoot,
        [string]$LogRoot,
        [string]$ListenHost,
        [int]$Port
    )

    $configText = Get-Content -Path $TemplatePath -Raw
    $configText = $configText.Replace("__HOST__", $ListenHost)
    $configText = $configText.Replace("__PORT__", $Port.ToString())
    $configText = $configText.Replace("__DATA_DIR__", $DataRoot)
    $configText = $configText.Replace("__LOG_DIR__", $LogRoot)
    Set-Content -Path $DestinationPath -Value $configText -Encoding ascii
}

$templatePath = Join-Path $PSScriptRoot "mlp-license-server-service.xml.template"
if (-not (Test-Path -Path $templatePath)) {
    throw "Missing service template: $templatePath"
}
if (-not (Test-Path -Path $ConfigTemplatePath)) {
    throw "Missing config template: $ConfigTemplatePath"
}
if (($InstallService -or $StartService) -and -not (Test-Path -Path $ServiceCommand)) {
    throw "Service command not found: $ServiceCommand"
}
if (($InstallService -or $StartService) -and -not $WinSWExePath) {
    throw "Provide -WinSWExePath when using -InstallService or -StartService."
}

$logRoot = Join-Path $ProgramDataRoot "logs"
$configPath = Join-Path $ProgramDataRoot "config.toml"
$serviceXmlPath = Join-Path $ServiceRoot "$ServiceId.xml"
$serviceExePath = Join-Path $ServiceRoot "$ServiceId.exe"
$serviceRootFullPath = [System.IO.Path]::GetFullPath($ServiceRoot)
$programFilesFullPath = [System.IO.Path]::GetFullPath($env:ProgramFiles)

$requiresAdministrator = $InstallService -or $StartService -or $serviceRootFullPath.StartsWith($programFilesFullPath, [System.StringComparison]::OrdinalIgnoreCase)
if ($requiresAdministrator -and -not (Test-IsAdministrator)) {
    throw "Administrator privileges are required to prepare or install service assets under '$ServiceRoot'. Re-run this script from an elevated PowerShell session."
}

foreach ($path in @($ServiceRoot, $ProgramDataRoot, $logRoot)) {
    if ($PSCmdlet.ShouldProcess($path, "Create directory")) {
        New-Item -ItemType Directory -Path $path -Force | Out-Null
    }
}

if (-not (Test-Path -Path $configPath)) {
    if ($PSCmdlet.ShouldProcess($configPath, "Write sample config")) {
        Render-Config -TemplatePath $ConfigTemplatePath `
            -DestinationPath $configPath `
            -DataRoot $ProgramDataRoot `
            -LogRoot $logRoot `
            -ListenHost $ServiceHost `
            -Port $ServicePort
    }
}

$serviceXml = Get-Content -Path $templatePath -Raw
$replacements = @{
    "{{SERVICE_ID}}" = Escape-Xml -Value $ServiceId
    "{{SERVICE_NAME}}" = Escape-Xml -Value $ServiceName
    "{{SERVICE_DESCRIPTION}}" = Escape-Xml -Value $ServiceDescription
    "{{SERVICE_COMMAND}}" = Escape-Xml -Value $ServiceCommand
    "{{SERVICE_ARGUMENTS}}" = Escape-Xml -Value $ServiceArguments
    "{{WORKING_DIRECTORY}}" = Escape-Xml -Value $ServiceRoot
    "{{LOG_DIRECTORY}}" = Escape-Xml -Value $logRoot
    "{{CONFIG_PATH}}" = Escape-Xml -Value $configPath
    "{{RUNTIME_ROOT}}" = Escape-Xml -Value $ProgramDataRoot
}

foreach ($token in $replacements.Keys) {
    $serviceXml = $serviceXml.Replace($token, $replacements[$token])
}

if ($PSCmdlet.ShouldProcess($serviceXmlPath, "Write WinSW service definition")) {
    Set-Content -Path $serviceXmlPath -Value $serviceXml -Encoding ascii
}

if ($WinSWExePath) {
    if (-not (Test-Path -Path $WinSWExePath)) {
        throw "WinSW executable not found: $WinSWExePath"
    }

    if ($PSCmdlet.ShouldProcess($serviceExePath, "Copy WinSW executable")) {
        Copy-Item -Path $WinSWExePath -Destination $serviceExePath -Force
    }
}

if ($InstallService) {
    if ($PSCmdlet.ShouldProcess($ServiceId, "Install Windows service")) {
        & $serviceExePath install
    }
}

if ($StartService) {
    if ($PSCmdlet.ShouldProcess($ServiceId, "Start Windows service")) {
        & $serviceExePath start
    }
}

Write-Host "Prepared Windows service assets:"
if ($WinSWExePath) {
    Write-Host "  Service wrapper: $serviceExePath"
}
else {
    Write-Host "  Service wrapper target: $serviceExePath"
    Write-Host "  Wrapper status:         WinSW executable not copied yet"
}
Write-Host "  Service config:  $serviceXmlPath"
Write-Host "  App config:      $configPath"
Write-Host ""
Write-Host "Current example service command assumption:"
Write-Host "  $ServiceCommand $ServiceArguments"
Write-Host ""
Write-Host "If the runtime entrypoint changes later, rerun this script with the updated command."
