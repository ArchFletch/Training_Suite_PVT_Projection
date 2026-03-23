# GUI Licensing Package

This package is reserved for desktop-side licensing code inside the existing GUI application.

Planned responsibilities:

- store the configured on-prem server URL
- call `status`, `checkout`, `heartbeat`, and `release`
- expose GUI-friendly status and error models
- manage the background heartbeat loop without mixing licensing code into the training core
- use `xfmr_v2.licensing.client_config` so packaged builds store `license_client.json`
  in the OS-specific per-user config location defined by `xfmr_v2.app_paths`
