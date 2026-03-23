# Tech Stack: License Server

## Recommended Stack

- Python 3.12
- FastAPI
- Uvicorn
- Pydantic v2
- SQLAlchemy 2.0
- SQLite for MVP
- Typer for CLI admin tooling
- `cryptography` with Ed25519
- Python `logging`
- bundled Python runtime or dedicated bundled virtual environment for Windows delivery
- WinSW for Windows service hosting
- PowerShell for Windows install and service bootstrap

## Why This Fits

- It matches the rest of the codebase and keeps the operational model simple.
- FastAPI + Pydantic is a strong fit for a small JSON API with clear contracts.
- SQLite is enough for a customer-local floating-seat service at MVP scale.
- Typer fits the CLI-only admin decision well.
- Ed25519 signatures are a good fit for signed license files.
- A bundled Windows runtime avoids forcing customer IT to manage a separate host Python install.
- WinSW is a pragmatic Windows service host for the MVP.

## Keep

- one local service per customer environment
- CLI-first admin model
- signed JSON licenses
- lightweight local persistence
- service bundle delivery for Windows operations
- clean-install validation as part of release readiness

## Avoid For MVP

- PostgreSQL as the default requirement
- Redis
- Docker as a hard dependency
- Kubernetes
- browser-heavy admin console
- requiring customer IT to fetch WinSW separately

## Likely Later Additions

- PostgreSQL if audit and reporting needs grow
- localhost web admin on top of the same service
- stronger server identity using a local keypair
- signed Windows service bundle or alternate enterprise installer formats

## Bottom Line

Use **Python + FastAPI + Uvicorn + Pydantic + SQLAlchemy + SQLite + Typer +
`cryptography` + bundled Windows runtime + WinSW + PowerShell**.

## References

- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Uvicorn Docs](https://www.uvicorn.org/)
- [Typer Docs](https://typer.tiangolo.com/)
- [Pydantic Docs](https://docs.pydantic.dev/)
- [SQLAlchemy Docs](https://docs.sqlalchemy.org/20/)
- [SQLite Docs](https://www.sqlite.org/about.html)
- [Cryptography Ed25519 Docs](https://cryptography.io/en/41.0.6/hazmat/primitives/asymmetric/ed25519/)
- `tech_stack_deployment.md`
