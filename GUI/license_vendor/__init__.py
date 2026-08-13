"""Internal vendor tooling for issuing signed evaluation and paid licenses."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

_SUBMODULES = ("cli", "issuance_log", "signer")

# Public names mapped to the submodule that defines them, resolved on first
# access instead of at import time. `python -m license_vendor` executes this
# module before __main__, so eagerly importing .issuance_log here -- which needs
# license_server.schemas -- made a vendor workstation holding only license_vendor/
# die with a bare ModuleNotFoundError raised three levels down, before __main__
# could catch it and say what to do about it.
_EXPORTS = {
    "VendorSigningKey": "signer",
    "append_issuance_record": "issuance_log",
    "build_issuance_record": "issuance_log",
    "derive_public_key": "signer",
    "generate_signing_key": "signer",
    "load_signing_key": "signer",
    "read_issuance_log": "issuance_log",
    "save_signing_key": "signer",
    "sign_license_payload": "signer",
    "sign_message": "signer",
    "verify_license_signature": "signer",
    "verify_message": "signer",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> object:
    if name in _SUBMODULES:
        value: object = import_module(f".{name}", __name__)
    elif name in _EXPORTS:
        value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS, *_SUBMODULES})


if TYPE_CHECKING:  # keep the re-exports visible to type checkers and editors
    from .issuance_log import append_issuance_record, build_issuance_record, read_issuance_log
    from .signer import (
        VendorSigningKey,
        derive_public_key,
        generate_signing_key,
        load_signing_key,
        save_signing_key,
        sign_license_payload,
        sign_message,
        verify_license_signature,
        verify_message,
    )
