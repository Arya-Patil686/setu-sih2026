"""Environment fix-ups applied on import of `setu`.

One concession to reality: the python.org macOS builds ship without a usable CA bundle,
so the first attempt to fetch a model checkpoint fails with an opaque SSL error rather
than anything that names the problem. Pointing the standard variables at `certifi` when
they are unset costs nothing and removes a failure mode that would otherwise surface
during a live demo.
"""

from __future__ import annotations

import os


def ensure_ca_bundle() -> str | None:
    """Point SSL_CERT_FILE and REQUESTS_CA_BUNDLE at certifi if the caller has not."""
    if os.environ.get("SSL_CERT_FILE") and os.environ.get("REQUESTS_CA_BUNDLE"):
        return os.environ["SSL_CERT_FILE"]
    try:
        import certifi
    except Exception:
        return None
    path = certifi.where()
    os.environ.setdefault("SSL_CERT_FILE", path)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", path)
    return path
