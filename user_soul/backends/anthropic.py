from __future__ import annotations
import base64

import anthropic


_LOCAL_ADDRESS_CACHE: str | None = None


def _resolve_local_address(target_host: str, port: int = 443) -> str | None:
    global _LOCAL_ADDRESS_CACHE
    if _LOCAL_ADDRESS_CACHE is not None:
        return _LOCAL_ADDRESS_CACHE or None
    import socket as _socket
    try:
        candidates = [
            info[4][0]
            for info in _socket.getaddrinfo(
                _socket.gethostname(), None, _socket.AF_INET, _socket.SOCK_STREAM
            )
            if not info[4][0].startswith("127.")
        ]
    except Exception:
        candidates = []
    for addr in candidates:
        s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        s.settimeout(2)
        try:
            s.bind((addr, 0))
            s.connect((target_host, port))
            s.close()
            _LOCAL_ADDRESS_CACHE = addr
            return addr
        except Exception:
            try:
                s.close()
            except Exception:
                pass
    _LOCAL_ADDRESS_CACHE = ""
    return None


class AnthropicBackend:

    def __init__(self, api_key: str):
        self._api_key = api_key

    def _resolve_model(self, tier: str) -> str:
        import os
        env_model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6").split("[")[0]
        if self._api_key.startswith("sk-or-"):
            from eltm.config import _env_model_to_or
            return _env_model_to_or(env_model)
        from eltm.config import _env_model_to_anthropic
        return _env_model_to_anthropic(env_model)

    def _make_client(self):
        """Use eltm.llm_client.make_client for response normalization."""
        from eltm.llm_client import make_client
        return make_client(self._api_key)

    def text(self, prompt: str, *, max_tokens: int = 512,
             temperature: float = 0.0, model_tier: str = "fast") -> str:
        client = self._make_client()
        kwargs: dict = dict(
            model=self._resolve_model(model_tier),
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        if temperature > 0.0:
            kwargs["temperature"] = temperature
        resp = client.messages.create(**kwargs)
        return resp.content[0].text if resp.content else ""

    def vision(self, prompt: str, images: list[bytes], *,
               max_tokens: int = 512, temperature: float = 0.0,
               model_tier: str = "smart") -> str:
        client = self._make_client()
        content: list[dict] = []
        for img in images:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.b64encode(img).decode(),
                },
            })
        content.append({"type": "text", "text": prompt})
        kwargs: dict = dict(
            model=self._resolve_model(model_tier),
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": content}],
        )
        if temperature > 0.0:
            kwargs["temperature"] = temperature
        resp = client.messages.create(**kwargs)
        return resp.content[0].text if resp.content else ""
