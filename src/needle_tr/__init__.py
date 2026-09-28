from pathlib import Path
import needle

from .tools import ACTIVE_TOOLS, calculate, get_weather, send_message, set_alarm

_cact_candidates = [
    Path("needle3-tr.cact"),
    Path(__file__).parent / "needle3-tr.cact",
]
_weights_path = None
for _p in _cact_candidates:
    if _p.exists():
        _weights_path = str(_p.resolve())
        break

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

