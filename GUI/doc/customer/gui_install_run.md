# GUI Install And Run Notes

These notes describe how the packaged desktop app for
`Surrogate Model Training Suite` is expected to be installed and where it stores
user-writable state.

## Windows

1. Run the provided Windows installer.
2. Accept the default per-user install location:
   `%LOCALAPPDATA%\Programs\Surrogate Model Training Suite\`
3. Launch `Surrogate Model Training Suite` from the Start menu or desktop shortcut.

The Windows GUI install is intended for a normal design-engineer user account
and should not require administrator rights.

### Windows User Data Locations

- license server settings: `%APPDATA%\Surrogate Model Training Suite\license_client.json`
- last GUI session: `%LOCALAPPDATA%\Surrogate Model Training Suite\last_session.json`
- logs: `%LOCALAPPDATA%\Surrogate Model Training Suite\logs\`
- default outputs and auto-managed cache: `%USERPROFILE%\Documents\Surrogate Model Training Suite\Runs\`

### Silent Install And Uninstall

The Windows installer should support standard Inno Setup silent-operation
switches for enterprise rollout and release validation.

Representative examples:

```powershell
SurrogateModelTrainingSuite-Windows.exe /VERYSILENT /NORESTART
%LOCALAPPDATA%\Programs\Surrogate Model Training Suite\unins000.exe /VERYSILENT /NORESTART
```

## Linux

1. Extract the provided tar.gz bundle to the chosen install root, for example:
   `/opt/mlp-training-studio`
2. Launch the app with:
   `/opt/mlp-training-studio/bin/mlp-training-studio`
3. Optionally copy the rendered desktop entry into:
   `/usr/share/applications/` or `~/.local/share/applications/`

Writable files are not stored in the extracted bundle directory.

### Linux User Data Locations

- license server settings: `~/.config/mlp-training-studio/license_client.json`
- last GUI session: `~/.local/state/mlp-training-studio/last_session.json`
- default outputs and auto-managed cache: `~/Documents/Surrogate Model Training Suite/runs/`

## License Server Settings

The packaged app uses one per-user JSON file named `license_client.json` for
the on-prem server URL.

Customer IT may pre-seed the file with:

```json
{
  "server_url": "http://mlp-license-01:27850"
}
```

The packaged GUI continues to read and write this same file path when the user
updates the server URL inside the app.

## Notes

- Changing the output folder in the GUI moves future outputs and auto-managed caches to that chosen folder.
- The packaged app should never write runtime state into the install directory.
- Uninstalling the app removes binaries and shortcuts, but it should not delete user configs, caches, logs, or run outputs.
- Clean-install validation must confirm that the packaged app runs without relying on the source checkout.
