from __future__ import annotations

from urllib.parse import urlparse

_ALLOWED_HOSTS: frozenset[str] = frozenset({"localhost", "127.0.0.1"})


def validate_url(url: str) -> str:
    """Reject any URL that does not point to localhost.

    Raises ValueError for external URLs so that no accidental calls
    leave the Codespace during development or testing.
    """
    parsed = urlparse(url)
    if parsed.hostname not in _ALLOWED_HOSTS:
        raise ValueError(f"Non-localhost URL rejected by NetworkGuard: {url!r}")
    return url
