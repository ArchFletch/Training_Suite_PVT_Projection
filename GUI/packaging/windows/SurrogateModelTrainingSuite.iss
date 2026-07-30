#ifndef AppBuildDir
  #error AppBuildDir define is required
#endif

#ifndef OutputDir
  #define OutputDir AddBackslash(SourcePath) + "..\..\artifacts\packaging\windows\installer"
#endif

#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif

#ifndef AppExecutableBaseName
  #define AppExecutableBaseName "SurrogateModelTrainingSuite"
#endif

[Setup]
AppId={{94127F4D-A637-4B80-9307-E463796C92F8}
AppName=Surrogate Model Training Suite
AppVersion={#AppVersion}
DefaultDirName={localappdata}\Programs\Surrogate Model Training Suite
DefaultGroupName=Surrogate Model Training Suite
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutputDir}
OutputBaseFilename=SurrogateModelTrainingSuite-Windows
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\{#AppExecutableBaseName}.exe

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Files]
Source: "{#AppBuildDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\Surrogate Model Training Suite"; Filename: "{app}\{#AppExecutableBaseName}.exe"
Name: "{autodesktop}\Surrogate Model Training Suite"; Filename: "{app}\{#AppExecutableBaseName}.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExecutableBaseName}.exe"; Description: "Launch Surrogate Model Training Suite"; Flags: nowait postinstall skipifsilent
