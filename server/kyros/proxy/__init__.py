"""Kyros Zero-Code Memory Proxy — package init.

The proxy intercepts LLM API calls, automatically injects relevant
memories into prompts, and extracts new memories from responses.

Quick start:
    # Start the proxy (reads KYROS_API_KEY and KYROS_BASE_URL from env)
    python -m kyros.proxy --port 8080

    # Then point your OpenAI client at the proxy:
    client = openai.OpenAI(base_url="http://<proxy-host>:8080/v1")
    # Add header: X-Agent-ID: <your-agent-id>

Environment variables:
    KYROS_API_KEY       Your Kyros API key (required)
    KYROS_BASE_URL      URL of the Kyros API server (default: http://localhost:8000)
    KYROS_PROXY_PORT    Port for the proxy to listen on (default: 8080)
    OPENAI_API_KEY      OpenAI API key (for forwarding to OpenAI)
    ANTHROPIC_API_KEY   Anthropic API key (for forwarding to Anthropic)
    GEMINI_API_KEY      Google Gemini API key (for forwarding to Gemini)
"""
