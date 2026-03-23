# PRD: On-Prem Floating License Server

## Product

On-prem floating license server for `Surrogate Model Traning Suite`.

## Goal

Support an EDA-style company licensing model with:

- 14-day floating-license POC
- offline procurement and manual payment validation
- customer-hosted on-prem server
- floating concurrent seats for Windows and Linux desktop users
- eval-to-paid conversion on the same server
- a Windows service-bundle install path that customer IT can operate without a source checkout

## Users

- Vendor licensing admin: issues eval and paid licenses
- Customer IT admin: installs the server, exports the request, imports the license, checks status, and operates the service
- Customer engineer: points the app at the server and consumes a seat

## In Scope

- one active license per customer server
- signed evaluation and paid licenses
- floating-seat checkout, heartbeat, and release
- Windows and Linux server support
- Windows and Linux desktop-client support
- CLI-only customer admin workflow for MVP
- immediate eviction when a replacement license lowers seat count below active use
- same feature set for eval and paid licenses in MVP
- Windows service-bundle delivery with bundled runtime and bundled WinSW

## Out Of Scope

- self-serve payment
- named-user licensing
- cloud-hosted runtime licensing
- offline borrowing
- server redundancy or triad failover
- SSO or LDAP
- customer-facing web admin console
- packaging the internal vendor tool for customer delivery

## Primary Workflow

1. Customer IT installs the license server.
2. Server exports a request file.
3. Vendor issues a signed 14-day eval or paid license.
4. Customer IT imports the license through the CLI.
5. Desktop apps check out and heartbeat seats while running.
6. If the customer buys, the vendor issues a replacement paid license for the same server.

## Product Requirements

### Licensing Model

- Licensing is company-based, not user-account-based.
- One app instance consumes one seat in MVP.
- The runtime model for eval and paid licenses must be the same.

### License Files

- The server must generate a persistent server identity and export a request file.
- Vendor-issued licenses must be signed and bound to that server identity.
- License import must verify signature and server binding before activation.

### Seat Control

- Client startup must request a seat.
- Seat grant requires:
  - valid license
  - current date inside the license term
  - active leases below the seat count
- Clients must heartbeat while running.
- Seats must be reclaimed automatically on lease expiry.
- Clean shutdown should release seats immediately.

### Admin And Operations

- Customer admin must be possible from the command line only in MVP.
- The Windows customer build must be a self-contained service bundle.
- The Windows customer build must include a bundled runtime and bundled WinSW wrapper.
- Customer IT must be able to run `init`, `export-request`, `import-license`, `show-status`, and `show-audit` after install.
- Clear install and operating instructions are required.
- Customer IT must be able to inspect:
  - active license summary
  - current seat usage
  - audit events

### Deployment Expectations

- Windows install root: `C:\Program Files\MLP License Server\`
- Windows runtime root: `%PROGRAMDATA%\MLP License Server\`
- default example LAN URL: `http://mlp-license-01:27850`
- packaged runtime behavior must not require a source checkout

### Replacement Behavior

- Replacing a license on the same server must be supported.
- If seat count is reduced, excess active seats must be evicted immediately.

## Non-Functional Requirements

- no vendor internet dependency during normal runtime
- signed-license trust model
- stable Windows and Linux service behavior
- auditable operational logs
- lightweight local persistence suitable for on-prem install
- packaging docs and runtime behavior must agree on paths and defaults

## Success Criteria

- A customer can complete a 14-day floating POC on the same server they would use after purchase.
- A paid license can replace an eval license without changing desktop client settings.
- Seat limits are enforced on both Windows and Linux clients.
- Vendor staff can issue and replace licenses without modifying application code.
- Customer IT can complete a fresh Windows service-bundle install without relying on a separate Python install.

## Resolved MVP Decisions

- Windows and Linux are both required.
- Customer admin is CLI-only for MVP.
- Evaluation licenses use the same feature set as paid licenses.
- License replacement immediately evicts excess seats.
- The default IT-facing example URL uses port `27850`.

## Key Risks

- customer IT setup friction slows evaluations
- OS-specific service packaging increases operational complexity
- weak host binding increases abuse risk; overly strict binding increases support burden
- poorly explained lease-loss behavior frustrates users
- packaging drift between docs and runtime will create avoidable support incidents

## Source References

- `deployment_prd.md`
- `../architecture/deployment_architecture.md`
- `../tech_stack/tech_stack_license_server.md`
