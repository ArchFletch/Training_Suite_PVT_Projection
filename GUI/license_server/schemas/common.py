"""Shared primitives for the license-server schema layer."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints

SCHEMA_VERSION = 1
PRODUCT_NAME = "Surrogate Model Traning Suite"
DEFAULT_SCHEMA_VERSION = SCHEMA_VERSION
DEFAULT_PRODUCT = PRODUCT_NAME


def _normalize_utc(value: datetime) -> datetime:
    """Require timezone-aware timestamps and normalize them to UTC."""

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("must include timezone information")
    return value.astimezone(timezone.utc)


def _validate_base64(value: str) -> str:
    """Reject malformed signatures before the crypto layer sees them."""

    try:
        base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise ValueError("must be valid base64") from exc
    return value


UtcDatetime = Annotated[datetime, AfterValidator(_normalize_utc)]
Base64Signature = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
    AfterValidator(_validate_base64),
]
ServerId = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, pattern=r"^srv_[a-z0-9]+$"),
]
HostFingerprint = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, pattern=r"^host_[a-z0-9]+$"),
]
LicenseId = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, pattern=r"^lic_[A-Za-z0-9._-]+$"),
]
LeaseId = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, pattern=r"^lease_[A-Za-z0-9._-]+$"),
]
MachineId = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, pattern=r"^[A-Za-z0-9._:-]+$"),
]
Hostname = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=255, pattern=r"^[A-Za-z0-9._-]+$"),
]
KeyId = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=64, pattern=r"^[A-Za-z0-9._-]+$"),
]
FeatureCode = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=64, pattern=r"^[a-z0-9_][a-z0-9_-]*$"),
]
ProductVersion = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]
NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=255)]
OptionalShortText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=255)]
OptionalLongText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1024)]
PositiveInt = Annotated[int, Field(gt=0)]
PositiveSeatCount = Annotated[int, Field(ge=1)]
NonNegativeInt = Annotated[int, Field(ge=0)]


class OSFamily(str, Enum):
    """Supported operating systems for the on-prem server identity."""

    WINDOWS = "windows"
    LINUX = "linux"


class ClientPlatform(str, Enum):
    """Supported operating systems for desktop clients in the MVP."""

    WINDOWS = "windows"
    LINUX = "linux"


class LicenseType(str, Enum):
    """License kinds supported by the shared license payload."""

    EVALUATION = "evaluation"
    PAID = "paid"


class SigningAlgorithm(str, Enum):
    """Signing algorithms supported by the vendor-issued license envelope."""

    ED25519 = "Ed25519"


class LicenseServerSchemaModel(BaseModel):
    """Common base model for every externally visible license-server contract."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )

    @classmethod
    def from_json(cls, text: str) -> "LicenseServerSchemaModel":
        """Parse one model from a raw JSON string."""

        return cls.model_validate_json(text)

    @classmethod
    def from_path(cls, path: str | Path) -> "LicenseServerSchemaModel":
        """Parse one model from a UTF-8 JSON file on disk."""

        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly dictionary with optional fields removed."""

        return self.model_dump(mode="json", exclude_none=True)

    def to_json(self, *, indent: int = 2) -> str:
        """Return a JSON document string with a trailing newline for file output."""

        return self.model_dump_json(indent=indent, exclude_none=True) + "\n"
