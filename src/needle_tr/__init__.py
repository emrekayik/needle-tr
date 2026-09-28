import os
from pathlib import Path
from typing import Optional
import needle

from .tools import ACTIVE_TOOLS, calculate, get_weather, send_message, set_alarm


def get_weights_path() -> Optional[str]:
    """Needle-TR Türkçe model ağırlıklarını arar veya Hugging Face Hub'dan otomatik önbelleğe alır."""
    # 1. Ortam değişkeni ile belirtilmişse
    if "NEEDLE_TR_WEIGHTS" in os.environ and Path(os.environ["NEEDLE_TR_WEIGHTS"]).exists():
        return str(Path(os.environ["NEEDLE_TR_WEIGHTS"]).resolve())

    # 2. Çalışma dizini veya paket dizini kontrolü
    for candidate in [Path("needle3-tr.cact"), Path(__file__).parent / "needle3-tr.cact"]:
        if candidate.exists():
            return str(candidate.resolve())

    # 3. Bulunamazsa Hugging Face Hub'dan otomatik indirip önbellekten kullan
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(
            repo_id="emrekayik/needle-tr",
            filename="needle3-tr.cact",
            repo_type="model",
        )
    except Exception as e:
        print(f"[needle-tr] Uyarı: Türkçe ağırlıklar Hugging Face'den alınamadı ({e}). Temel model kullanılıyor.")
        return None


_weights_path = get_weights_path()

agent = needle.Needle(
    tools=ACTIVE_TOOLS,
    weights=_weights_path,
    stateless=True,
)


def main():
    print("Hava durumu:", agent.run("Lagos'ta hava nasıl?")["results"])
    print("Mesaj gönderme:", agent.run("Ahmet'e 'Toplantı başladı' mesajı gönder")["results"])
    print("Alarm kurma:", agent.run("Saat 07:30'a alarm kur")["results"])
    print("Hesaplama:", agent.run("25 * 4 hesapla")["results"])


if __name__ == "__main__":
    main()

