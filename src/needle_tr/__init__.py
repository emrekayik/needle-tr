"""Needle 3 Türkçe Araç Çağırıcı ve Fine-Tuning Kütüphanesi."""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path
from typing import Any, Callable, List, Optional

from .tools import ACTIVE_TOOLS, calculate, get_weather, send_message, set_alarm


class _TurkishAgentProxy:
    """Adds deterministic handling for common Turkish queries before model inference."""

    def __init__(self, delegate: Any):
        self._delegate = delegate

    def run(self, query: str, *args, **kwargs):
        normalized = query.strip()
        weather_match = re.search(
            r"\b([A-ZÇĞİÖŞÜ][a-zçğıöşü]+)(?:['’](?:da|de|ta|te)|\s+(?:da|de|ta|te))\b",
            normalized,
        )
        if weather_match is None:
            weather_match = re.search(
                r"(?:hava|hava durumu).*?\b([A-ZÇĞİÖŞÜ][a-zçğıöşü]+)\b",
                normalized,
                flags=re.IGNORECASE,
            )
        if weather_match and "hava" in normalized.lower():
            return {"results": [get_weather(weather_match.group(1))]}

        expression = re.sub(
            r"^(?:lütfen\s+)?(.+?)\s*(?:hesapla|kaç eder)\s*[?.!]*$",
            r"\1",
            normalized,
            flags=re.IGNORECASE,
        )
        if expression != normalized and re.fullmatch(r"[0-9\s+\-*/().%]+", expression):
            return {"results": [calculate(expression)]}

        return self._delegate.run(query, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._delegate, name)


def get_weights_path() -> Optional[str]:
    """Needle-TR Türkçe model ağırlıklarını (.cact) arar veya Hugging Face Hub'dan otomatik önbelleğe alır."""
    # 1. Ortam değişkeni ile belirtilmişse
    if "NEEDLE_TR_WEIGHTS" in os.environ and Path(os.environ["NEEDLE_TR_WEIGHTS"]).exists():
        return str(Path(os.environ["NEEDLE_TR_WEIGHTS"]).resolve())

    # 2. Yerel dosya adayları kontrolü
    candidates = [
        Path("models/needle3-tr.cact"),
        Path("needle3-tr.cact"),
        Path(__file__).parent / "needle3-tr.cact",
        Path(__file__).resolve().parent.parent.parent / "models" / "needle3-tr.cact",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())

    # 3. Hugging Face Hub kontrolü (varsa)
    repo_id = os.environ.get("NEEDLE_HF_REPO", "emrekayik/needle-tr")
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(
            repo_id=repo_id,
            filename="needle3-tr.cact",
            repo_type="model",
        )
    except Exception:
        # Hub'da henüz yoksa temel Needle 3 modeli kullanılır
        return None


def create_agent(
    tools: Optional[List[Callable[..., Any]]] = None,
    weights: Optional[str] = None,
    stateless: bool = True,
    suppress_confidence_warning: bool = True,
    **kwargs,
) -> Any:
    """Needle 3 Türkçe ajanını başlatır.

    Args:
        tools: Kullanılacak araç fonksiyonları (varsayılan: ACTIVE_TOOLS)
        weights: .cact ağırlık dosya yolu (None ise otomatik tespit edilir)
        stateless: Her çalıştırmada bağlamı sıfırlama (varsayılan: True)
        suppress_confidence_warning: Yerel fine-tune modellerinde beklenen confidence uyarısını filtreler.

    Kullanım:
        ```python
        from needle_tr import create_agent

        agent = create_agent()
        print(agent.run("Lagos'ta hava nasıl?")["results"])
        ```
    """
    import needle

    selected_tools = tools if tools is not None else ACTIVE_TOOLS
    selected_weights = weights if weights is not None else get_weights_path()

    if suppress_confidence_warning:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*these weights carry no confidence head.*")
            return _TurkishAgentProxy(needle.Needle(
                tools=selected_tools,
                weights=selected_weights,
                stateless=stateless,
                **kwargs,
            ))

    return _TurkishAgentProxy(needle.Needle(
        tools=selected_tools,
        weights=selected_weights,
        stateless=stateless,
        **kwargs,
    ))


class _LazyAgent:
    """Modül seviyesinde lazy agent örneği."""
    def __init__(self):
        self._instance = None

    def _get(self):
        if self._instance is None:
            self._instance = create_agent()
        return self._instance

    def run(self, *args, **kwargs):
        return self._get().run(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._get(), name)


agent = _LazyAgent()


def main():
    print("Hava durumu:", agent.run("Lagos'ta hava nasıl?")["results"])
    print("Mesaj gönderme:", agent.run("Ahmet'e 'Toplantı başladı' mesajı gönder")["results"])
    print("Alarm kurma:", agent.run("Saat 07:30'a alarm kur")["results"])
    print("Hesaplama:", agent.run("25 * 4 hesapla")["results"])


if __name__ == "__main__":
    main()
