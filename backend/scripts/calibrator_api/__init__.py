from __future__ import annotations

import sys
import types
import warnings


def _install_scproxy_fallback() -> None:
    """Install a minimal _scproxy fallback when macOS blocks the extension module."""
    if sys.platform != "darwin":
        return
    if "_scproxy" in sys.modules:
        return

    try:
        import _scproxy  # noqa: F401
        return
    except Exception as exc:  # pragma: no cover - depends on local macOS policy/runtime
        fallback = types.ModuleType("_scproxy")

        def _get_proxy_settings() -> dict:
            return {}

        def _get_proxies() -> dict:
            return {}

        fallback._get_proxy_settings = _get_proxy_settings  # type: ignore[attr-defined]
        fallback._get_proxies = _get_proxies  # type: ignore[attr-defined]
        sys.modules["_scproxy"] = fallback
        warnings.warn(
            f"Using _scproxy fallback because native module failed to load: {exc}. "
            "System proxy auto-detection is disabled for this process.",
            RuntimeWarning,
            stacklevel=2,
        )


_install_scproxy_fallback()
