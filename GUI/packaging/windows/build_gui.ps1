param(
    [string]$Python = "python",
    [string]$OutputDir = "",
    [string]$IconPath = "",
    [string]$AppTitle = "Surrogate Model Training Suite",
    [string]$AppExecutableBaseName = "SurrogateModelTrainingSuite"
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\\..")).Path
if (-not $OutputDir) {
    $OutputDir = Join-Path $repoRoot "artifacts\\packaging\\windows"
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$specTemplate = Join-Path $repoRoot "packaging\\gui\\pysidedeploy.spec.in"
$specPath = Join-Path $OutputDir "pysidedeploy.spec"
$inputFile = Join-Path $repoRoot "launch_gui.py"
$pythonExe = (& $Python -c "import sys; print(sys.executable)").Trim()

$renderArgs = @(
    (Join-Path $repoRoot "packaging\\render_pyside6_spec.py"),
    "--template", $specTemplate,
    "--output", $specPath,
    "--title", $AppTitle,
    "--project-dir", $repoRoot,
    "--input-file", $inputFile,
    "--exec-directory", $OutputDir,
    "--python-path", $pythonExe
)
if ($IconPath) {
    $renderArgs += @("--icon", $IconPath)
}

& $Python @renderArgs

$generatedRoot = Join-Path $OutputDir "_nuitka"
$generatedBuildDir = Join-Path $generatedRoot "launch_gui.build"
$generatedDistDir = Join-Path $generatedRoot "launch_gui.dist"
if (Test-Path $generatedDistDir) {
    Remove-Item -Recurse -Force $generatedDistDir
}

# Build directly with Nuitka. This avoids the PySide6 deploy helper's default
# Windows icon post-processing path, which produced a non-launchable EXE here.
$nuitkaArgs = @(
    "-m", "nuitka",
    $inputFile,
    "--follow-imports",
    "--enable-plugin=pyside6",
    "--output-dir=$generatedRoot",
    "--output-filename=$AppExecutableBaseName.exe",
    "--quiet",
    "--assume-yes-for-downloads",
    "--module-parameter=torch-disable-jit=yes",
    "--noinclude-qt-translations",
    "--include-module=PySide6.QtOpenGL",
    "--include-module=PySide6.QtOpenGLWidgets",
    "--include-module=PySide6.QtSvg",
    "--include-module=PySide6.QtTest",
    "--include-package=xfmr_v2",
    "--include-package=numpy",
    "--include-package=pyqtgraph",
    "--include-package=torch",
    "--standalone",
    "--noinclude-dlls=*.cpp.o",
    "--noinclude-dlls=*.qsb",
    "--include-qt-plugins=platforminputcontexts"
)
if ($IconPath) {
    $resolvedIcon = (Resolve-Path $IconPath).Path
    $nuitkaArgs += "--windows-icon-from-ico=$resolvedIcon"
}

& $Python @nuitkaArgs
if ($LASTEXITCODE -ne 0) {
    throw "Nuitka build failed with exit code $LASTEXITCODE"
}

$generatedExe = Join-Path $generatedDistDir "$AppExecutableBaseName.exe"
if (-not (Test-Path $generatedExe)) {
    $legacyExe = Join-Path $generatedDistDir "launch_gui.exe"
    if (Test-Path $legacyExe) {
        Rename-Item -Path $legacyExe -NewName "$AppExecutableBaseName.exe"
    }
}

$standaloneDir = Join-Path $OutputDir "$AppExecutableBaseName.dist"
if (Test-Path $standaloneDir) {
    Remove-Item -Recurse -Force $standaloneDir
}
Copy-Item -Recurse -Force $generatedDistDir $standaloneDir

$standaloneExe = Join-Path $standaloneDir "$AppExecutableBaseName.exe"
if (-not (Test-Path $standaloneExe)) {
    throw "Standalone Windows build did not produce the expected executable at $standaloneExe"
}

Write-Host "Standalone Windows build completed. Standalone folder: $standaloneDir"
