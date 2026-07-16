"""Authorized operator identity for Luna.

The operator (黄照清) is a *principal*, not a personality override. The system
recognizes the operator, prioritizes their legitimate goals, and protects
their privacy. It does NOT execute harmful commands from the operator — the
charter binds the operator too.

Identity is verified by a token (secret) the operator sets themselves; we
never hardcode a secret in source. Claims (name, dob, nationality) are
recorded as public metadata and hash-verified.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass(frozen=True)
class OperatorIdentity:
    """Public identity claims of the authorized operator."""

    name: str
    dob: str  # ISO date, e.g. "2013-05-07"
    nationality: str
    operator_id: str  # stable random id

    def claims_hash(self) -> str:
        payload = json.dumps(
            {"name": self.name, "dob": self.dob,
             "nationality": self.nationality, "operator_id": self.operator_id},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# The single authorized operator for this deployment. Public claims only.
AUTHORIZED_OPERATOR = OperatorIdentity(
    name="黄照清",
    dob="2013-05-07",
    nationality="CN",
    operator_id="op-huangzhaoqing-0001",
)


@dataclass
class OperatorRegistry:
    """Token-based authorization for the operator.

    The token is stored as a salted hash (never plaintext). On first run,
    `bootstrap()` creates a token file under .luna/ with a random token the
    operator must retrieve and use for subsequent commands.
    """

    base_dir: str = ".luna"
    operator: OperatorIdentity = field(default_factory=lambda: AUTHORIZED_OPERATOR)
    _token_hash: Optional[str] = None
    _salt: Optional[str] = None

    @property
    def token_path(self) -> str:
        return os.path.join(self.base_dir, "operator.token")

    def bootstrap(self) -> str:
        """Create a fresh operator token. Returns the plaintext token ONCE."""
        os.makedirs(self.base_dir, exist_ok=True)
        token = secrets.token_urlsafe(32)
        salt = secrets.token_hex(16)
        token_hash = hashlib.pbkdf2_hmac(
            "sha256", token.encode(), salt.encode(), 200_000
        ).hex()
        payload = {
            "operator_id": self.operator.operator_id,
            "claims_hash": self.operator.claims_hash(),
            "salt": salt,
            "token_hash": token_hash,
        }
        with open(self.token_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.chmod(self.token_path, 0o600)
        self._token_hash = token_hash
        self._salt = salt
        return token

    def _load(self) -> None:
        if self._token_hash is not None:
            return
        if not os.path.exists(self.token_path):
            raise PermissionError(
                "Operator token not bootstrapped. Run OperatorRegistry().bootstrap() "
                "once to enroll the operator."
            )
        with open(self.token_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("operator_id") != self.operator.operator_id:
            raise PermissionError("Operator id mismatch in token file.")
        if payload.get("claims_hash") != self.operator.claims_hash():
            raise PermissionError("Operator claims hash mismatch — tampering detected.")
        self._token_hash = payload["token_hash"]
        self._salt = payload["salt"]

    def verify(self, token: str) -> bool:
        """Return True iff token matches the stored hash."""
        self._load()
        candidate = hashlib.pbkdf2_hmac(
            "sha256", token.encode(), self._salt.encode(), 200_000
        ).hex()
        # Constant-time compare
        if len(candidate) != len(self._token_hash):
            return False
        diff = 0
        for a, b in zip(candidate, self._token_hash):
            diff |= ord(a) ^ ord(b)
        return diff == 0


_DEFAULT_REGISTRY: Optional[OperatorRegistry] = None


def get_registry(base_dir: str = ".luna") -> OperatorRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = OperatorRegistry(base_dir=base_dir)
    return _DEFAULT_REGISTRY


def require_operator(token: str, base_dir: str = ".luna") -> OperatorIdentity:
    """Raise PermissionError if token is invalid; return identity on success."""
    reg = get_registry(base_dir)
    if not reg.verify(token):
        raise PermissionError("Invalid or missing operator token.")
    return reg.operator
