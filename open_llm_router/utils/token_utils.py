from __future__ import annotations

import base64
import json
import time
from typing import Any


class TokenMetadataParser:
    CHATGPT_ACCOUNT_CLAIM_PATH = "https://api.openai.com/auth"

    @staticmethod
    def is_token_expiring(expires_at: int | None, skew_seconds: int = 60) -> bool:
        if expires_at is None:
            return False
        return expires_at <= int(time.time()) + skew_seconds

    @staticmethod
    def extract_expires_at(token_response: dict[str, Any]) -> int | None:
        now = int(time.time())

        raw_expires_in = token_response.get("expires_in")
        if raw_expires_in is not None:
            try:
                return now + int(float(raw_expires_in))
            except (TypeError, ValueError):
                pass

        raw_expires_at = token_response.get("expires_at")
        if raw_expires_at is not None:
            try:
                return int(float(raw_expires_at))
            except (TypeError, ValueError):
                pass

        return None

    @classmethod
    def extract_chatgpt_account_id(
        cls,
        token: str | None,
        *,
        claim_path: str | None = None,
    ) -> str | None:
        if not token:
            return None
        parts = token.split(".")
        if len(parts) != 3:
            return None

        payload_b64 = parts[1]
        padding = "=" * ((4 - len(payload_b64) % 4) % 4)

        try:
            payload_raw = base64.urlsafe_b64decode(payload_b64 + padding)
            payload = json.loads(payload_raw.decode("utf-8"))
        except Exception:
            return None

        claim = payload.get(claim_path or cls.CHATGPT_ACCOUNT_CLAIM_PATH)
        if isinstance(claim, dict):
            account_id = claim.get("chatgpt_account_id")
            if isinstance(account_id, str) and account_id.strip():
                return account_id.strip()
        return None
