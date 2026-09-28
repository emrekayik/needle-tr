import needle


from .tools import ACTIVE_TOOLS, calculate, get_weather, send_message, set_alarm

agent = needle.Needle(
    tools=ACTIVE_TOOLS,
    stateless=True,
)


def main():
    print("Hava durumu:", agent.run("Lagos'ta hava nasıl?")["results"])
    print("Mesaj gönderme:", agent.run("Ahmet'e 'Toplantı başladı' mesajı gönder")["results"])
    print("Alarm kurma:", agent.run("Saat 07:30'a alarm kur")["results"])
    print("Hesaplama:", agent.run("25 * 4 hesapla")["results"])


if __name__ == "__main__":
    main()

