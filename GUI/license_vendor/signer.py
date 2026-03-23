"""Vendor-only Ed25519 signing helpers for license issuance."""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from license_server.schemas import (
    SUPPORTED_ALGORITHM,
    LicensePayload,
    SignedLicenseEnvelope,
    canonical_payload_bytes,
)

_P = 2**255 - 19
_Q = 2**252 + 27742317777372353535851937790883648493
_IDENTITY = (0, 1)


def _sha512(data: bytes) -> bytes:
    return hashlib.sha512(data).digest()


def _modp_inv(value: int) -> int:
    return pow(value, _P - 2, _P)


_D = -121665 * _modp_inv(121666) % _P
_I = pow(2, (_P - 1) // 4, _P)


def _recover_x(y_value: int, sign_bit: int) -> int:
    if y_value >= _P:
        raise ValueError("Compressed point is out of range.")
    x_squared = (y_value * y_value - 1) * _modp_inv(_D * y_value * y_value + 1) % _P
    if x_squared == 0:
        if sign_bit:
            raise ValueError("Compressed point has an invalid sign bit.")
        return 0
    x_value = pow(x_squared, (_P + 3) // 8, _P)
    if (x_value * x_value - x_squared) % _P != 0:
        x_value = x_value * _I % _P
    if (x_value * x_value - x_squared) % _P != 0:
        raise ValueError("Compressed point is not on the curve.")
    if x_value & 1 != sign_bit:
        x_value = _P - x_value
    return x_value


_BASE_Y = 4 * _modp_inv(5) % _P
_BASE_X = _recover_x(_BASE_Y, 0)
_BASE_POINT = (_BASE_X, _BASE_Y)


def _point_add(left_point: tuple[int, int], right_point: tuple[int, int]) -> tuple[int, int]:
    x1, y1 = left_point
    x2, y2 = right_point
    xx = x1 * x2 % _P
    yy = y1 * y2 % _P
    dxxyy = _D * xx * yy % _P
    x3 = (x1 * y2 + x2 * y1) * _modp_inv(1 + dxxyy) % _P
    y3 = (yy + xx) * _modp_inv(1 - dxxyy) % _P
    return (x3, y3)


def _point_mul(scalar: int, point: tuple[int, int]) -> tuple[int, int]:
    result = _IDENTITY
    addend = point
    remaining = scalar
    while remaining > 0:
        if remaining & 1:
            result = _point_add(result, addend)
        addend = _point_add(addend, addend)
        remaining >>= 1
    return result


def _point_compress(point: tuple[int, int]) -> bytes:
    x_value, y_value = point
    return int.to_bytes(y_value | ((x_value & 1) << 255), 32, "little")


def _point_decompress(data: bytes) -> tuple[int, int]:
    if len(data) != 32:
        raise ValueError("Compressed point must be 32 bytes.")
    y_value = int.from_bytes(data, "little") & ((1 << 255) - 1)
    sign_bit = data[31] >> 7
    return (_recover_x(y_value, sign_bit), y_value)


def _expand_secret(seed: bytes) -> tuple[int, bytes]:
    if len(seed) != 32:
        raise ValueError("Ed25519 private key seeds must be 32 bytes.")
    digest = _sha512(seed)
    scalar = int.from_bytes(digest[:32], "little")
    scalar &= (1 << 254) - 8
    scalar |= 1 << 254
    return (scalar, digest[32:])


def derive_public_key(seed: bytes) -> bytes:
    """Derive the Ed25519 public key for a 32-byte private key seed."""

    scalar, _ = _expand_secret(seed)
    return _point_compress(_point_mul(scalar, _BASE_POINT))


def sign_message(seed: bytes, message: bytes) -> bytes:
    """Sign a message with an Ed25519 private key seed."""

    scalar, prefix = _expand_secret(seed)
    public_key = derive_public_key(seed)
    nonce = int.from_bytes(_sha512(prefix + message), "little") % _Q
    encoded_r = _point_compress(_point_mul(nonce, _BASE_POINT))
    challenge = int.from_bytes(_sha512(encoded_r + public_key + message), "little") % _Q
    s_value = (nonce + challenge * scalar) % _Q
    return encoded_r + int.to_bytes(s_value, 32, "little")


def verify_message(public_key: bytes, message: bytes, signature: bytes) -> bool:
    """Verify an Ed25519 message signature."""

    if len(public_key) != 32 or len(signature) != 64:
        return False
    try:
        public_point = _point_decompress(public_key)
        r_point = _point_decompress(signature[:32])
    except ValueError:
        return False
    if _point_mul(_Q, public_point) != _IDENTITY:
        return False
    if _point_mul(_Q, r_point) != _IDENTITY:
        return False
    s_value = int.from_bytes(signature[32:], "little")
    if s_value >= _Q:
        return False
    challenge = int.from_bytes(_sha512(signature[:32] + public_key + message), "little") % _Q
    left_point = _point_mul(8 * s_value, _BASE_POINT)
    right_point = _point_add(_point_mul(8, r_point), _point_mul(8 * challenge, public_point))
    return left_point == right_point


@dataclass(frozen=True, slots=True)
class VendorSigningKey:
    """Vendor-side signing key material stored outside the customer runtime."""

    key_id: str
    private_key_seed: bytes
    public_key: bytes
    created_at: str
    algorithm: str = SUPPORTED_ALGORITHM

    def __post_init__(self) -> None:
        if self.algorithm != SUPPORTED_ALGORITHM:
            raise ValueError(f"algorithm must be {SUPPORTED_ALGORITHM}.")
        key_id = self.key_id.strip()
        if not key_id:
            raise ValueError("key_id must not be empty.")
        object.__setattr__(self, "key_id", key_id)
        if len(self.private_key_seed) != 32:
            raise ValueError("private_key_seed must be 32 bytes.")
        if len(self.public_key) != 32:
            raise ValueError("public_key must be 32 bytes.")
        if derive_public_key(self.private_key_seed) != self.public_key:
            raise ValueError("public_key does not match the private_key_seed.")
        parsed_created_at = datetime.fromisoformat(
            self.created_at.replace("Z", "+00:00")
        ).astimezone(timezone.utc)
        object.__setattr__(
            self,
            "created_at",
            parsed_created_at.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        )

    @property
    def public_key_base64(self) -> str:
        return base64.b64encode(self.public_key).decode("ascii")

    def to_dict(self) -> dict[str, str]:
        return {
            "algorithm": self.algorithm,
            "key_id": self.key_id,
            "created_at": self.created_at,
            "private_key_seed": base64.b64encode(self.private_key_seed).decode("ascii"),
            "public_key": self.public_key_base64,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=True) + "\n"


def generate_signing_key(key_id: str, *, seed: bytes | None = None) -> VendorSigningKey:
    """Create a new vendor signing key with an Ed25519 seed and derived public key."""

    private_seed = seed or secrets.token_bytes(32)
    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )
    return VendorSigningKey(
        key_id=key_id,
        private_key_seed=private_seed,
        public_key=derive_public_key(private_seed),
        created_at=created_at,
    )


def save_signing_key(path: str | Path, signing_key: VendorSigningKey) -> Path:
    """Persist a vendor signing key as JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(signing_key.to_json(), encoding="utf-8")
    return output_path


def load_signing_key(path: str | Path) -> VendorSigningKey:
    """Load a vendor signing key from JSON."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        private_key_seed = base64.b64decode(data["private_key_seed"], validate=True)
        public_key = base64.b64decode(data["public_key"], validate=True)
    except Exception as exc:
        raise ValueError("Signing key file contains invalid base64 key material.") from exc
    return VendorSigningKey(
        algorithm=data.get("algorithm", SUPPORTED_ALGORITHM),
        key_id=data.get("key_id", ""),
        created_at=data.get("created_at", ""),
        private_key_seed=private_key_seed,
        public_key=public_key,
    )


def sign_license_payload(
    payload: LicensePayload, signing_key: VendorSigningKey
) -> SignedLicenseEnvelope:
    """Create a signed license envelope from a shared license payload."""

    payload_bytes = canonical_payload_bytes(payload)
    signature_bytes = sign_message(signing_key.private_key_seed, payload_bytes)
    signature = base64.b64encode(signature_bytes).decode("ascii")
    return SignedLicenseEnvelope(
        algorithm=SUPPORTED_ALGORITHM,
        key_id=signing_key.key_id,
        payload=payload,
        signature=signature,
    )


def verify_license_signature(
    envelope: SignedLicenseEnvelope, public_key: bytes | str
) -> bool:
    """Verify a signed license envelope against an Ed25519 public key."""

    if isinstance(public_key, str):
        public_key_bytes = base64.b64decode(public_key.encode("ascii"), validate=True)
    else:
        public_key_bytes = public_key
    signature_bytes = base64.b64decode(envelope.signature.encode("ascii"), validate=True)
    return verify_message(public_key_bytes, canonical_payload_bytes(envelope.payload), signature_bytes)
