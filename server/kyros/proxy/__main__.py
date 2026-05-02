"""V14 — kyros.proxy.start() one-liner API.

Provides the simplest possible way to start the memory proxy:

    import kyros
    kyros.proxy.start(api_key="mk_live_...", port=8080)

Or from the command line:

    python -m kyros.proxy --api-key mk_live_... --port 8080
"""

from __future__ import annotations

import argparse
import os

from kyros.proxy.server import start as _start_server


def start(
    api_key: str | None = None,
    port: int = 8080,
    host: str = "0.0.0.0",
    kyros_url: str | None = None,
    injection: bool = True,
    extraction: bool = True,
):
    """Start the Kyros Memory Proxy."""
    resolved_key = api_key or os.environ.get("KYROS_API_KEY", "")
    resolved_url = kyros_url or os.environ.get("KYROS_BASE_URL", "http://localhost:8000")

    if not resolved_key:
        raise ValueError(
            "API key required. Pass api_key= or set KYROS_API_KEY environment variable."
        )
    if not (1 <= port <= 65535):
        raise ValueError(f"port must be between 1 and 65535, got {port}")

    kyros_url_display = resolved_url[:44] if len(resolved_url) > 44 else resolved_url
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  Kyros Zero-Code Memory Proxy                                ║
║  ─────────────────────────────                               ║
║  Proxy:    http://{host}:{port:<5}                           ║
║  Kyros:    {kyros_url_display:<48} ║
║  Inject:   {'✅ ON' if injection else '❌ OFF':<48} ║
║  Extract:  {'✅ ON' if extraction else '❌ OFF':<48} ║
║                                                              ║
║  Usage:                                                      ║
║  client = openai.OpenAI(                                     ║
║      base_url="http://localhost:{port}/v1"                   ║
║  )  # Add header: X-Agent-ID: my-agent                      ║
╚══════════════════════════════════════════════════════════════╝
""")

    _start_server(
        api_key=resolved_key,
        port=port,
        host=host,
        kyros_url=resolved_url,
    )


# ─── CLI Entry Point ──────────────────────────

def main():
    """CLI entry point: python -m kyros.proxy"""
    parser = argparse.ArgumentParser(
        description="Kyros Zero-Code Memory Proxy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m kyros.proxy --api-key mk_live_abc123
  python -m kyros.proxy --port 9090 --kyros-url http://kyros.internal:8000
  KYROS_API_KEY=mk_live_abc123 python -m kyros.proxy
        """,
    )
    parser.add_argument("--api-key", help="Kyros API key (or set KYROS_API_KEY)")
    parser.add_argument("--port", type=int, default=8080, help="Proxy port (default: 8080)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--kyros-url", default=None, help="Kyros API URL")
    parser.add_argument("--no-inject", action="store_true", help="Disable memory injection")
    parser.add_argument("--no-extract", action="store_true", help="Disable memory extraction")

    args = parser.parse_args()

    start(
        api_key=args.api_key,
        port=args.port,
        host=args.host,
        kyros_url=args.kyros_url,
        injection=not args.no_inject,
        extraction=not args.no_extract,
    )


if __name__ == "__main__":
    main()
