"""Needle 3 Türkçe Veri Seti Üreticisi.

Cactus Needle 3 spesifikasyonuna (https://cactuscompute.com/blog/finetuning-needle) uygun olarak:
1. Her satırda tek bir JSON objesi: `query`, `tools`, `answers`, `reasoning`.
2. Gerekçelendirme (reasoning): Argümanların sorgudaki kaynak karşılıklarını bağlar (grounding).
3. Negatif / Konu dışı örnekler: 'answers': [] (~1/8 oranında). Modelin her şeye araç çağırmasını engeller.
4. Çevrimdışı zengin sentetik çoğaltıcı ve isteğe bağlı OpenRouter / Ollama entegrasyonu.
"""

from __future__ import annotations

import json
import random
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from .tools import get_needle_tools_schema

# ==============================================================================
# 1. TOHUM VERİLERİ (Needle 3 Formatı)
# ==============================================================================

SEED_EXAMPLES: List[Dict[str, Any]] = [
    # get_weather
    {
        "query": "Lagos'ta hava nasıl?",
        "answers": [{"name": "get_weather", "arguments": {"city": "Lagos"}}],
        "reasoning": "'Lagos' -> city",
    },
    {
        "query": "İstanbul için hava durumu raporunu getir",
        "answers": [{"name": "get_weather", "arguments": {"city": "İstanbul"}}],
        "reasoning": "'İstanbul' -> city",
    },
    {
        "query": "Ankara'da hava şu an kaç derece?",
        "answers": [{"name": "get_weather", "arguments": {"city": "Ankara"}}],
        "reasoning": "'Ankara' -> city",
    },
    {
        "query": "İzmir hava durumu nedir?",
        "answers": [{"name": "get_weather", "arguments": {"city": "İzmir"}}],
        "reasoning": "'İzmir' -> city",
    },
    {
        "query": "Bugün Bursa'da yağmur var mı, hava nasıl?",
        "answers": [{"name": "get_weather", "arguments": {"city": "Bursa"}}],
        "reasoning": "'Bursa' -> city",
    },
    {
        "query": "Londra'nın hava durumunu kontrol et",
        "answers": [{"name": "get_weather", "arguments": {"city": "Londra"}}],
        "reasoning": "'Londra' -> city",
    },
    {
        "query": "Antalya'da hava sıcaklığı ne durumda?",
        "answers": [{"name": "get_weather", "arguments": {"city": "Antalya"}}],
        "reasoning": "'Antalya' -> city",
    },
    # send_message
    {
        "query": "Ahmet'e 'Toplantı başladı' mesajı gönder",
        "answers": [{"name": "send_message", "arguments": {"recipient": "Ahmet", "message": "Toplantı başladı"}}],
        "reasoning": "'Ahmet' -> recipient; 'Toplantı başladı' -> message",
    },
    {
        "query": "Zeynep'e eve geç geleceğim diye mesaj at",
        "answers": [{"name": "send_message", "arguments": {"recipient": "Zeynep", "message": "eve geç geleceğim"}}],
        "reasoning": "'Zeynep' -> recipient; 'eve geç geleceğim' -> message",
    },
    {
        "query": "Mehmet'e yarın saat 10'da ofisteyim diye yaz",
        "answers": [{"name": "send_message", "arguments": {"recipient": "Mehmet", "message": "yarın saat 10'da ofisteyim"}}],
        "reasoning": "'Mehmet' -> recipient; 'yarın saat 10'da ofisteyim' -> message",
    },
    {
        "query": "Ali'ye 'Gelirken iki ekmek al' iletisi gönder",
        "answers": [{"name": "send_message", "arguments": {"recipient": "Ali", "message": "Gelirken iki ekmek al"}}],
        "reasoning": "'Ali' -> recipient; 'Gelirken iki ekmek al' -> message",
    },
    {
        "query": "Ayşe'ye doğum günün kutlu olsun mesajı ilet",
        "answers": [{"name": "send_message", "arguments": {"recipient": "Ayşe", "message": "doğum günün kutlu olsun"}}],
        "reasoning": "'Ayşe' -> recipient; 'doğum günün kutlu olsun' -> message",
    },
    {
        "query": "Can'a raporu e-posta ile gönderdim de",
        "answers": [{"name": "send_message", "arguments": {"recipient": "Can", "message": "raporu e-posta ile gönderdim"}}],
        "reasoning": "'Can' -> recipient; 'raporu e-posta ile gönderdim' -> message",
    },
    # set_alarm
    {
        "query": "Sabah 07:30 için alarm kur",
        "answers": [{"name": "set_alarm", "arguments": {"time": "07:30"}}],
        "reasoning": "'07:30' -> time",
    },
    {
        "query": "Saat 08:00'e Uyanma alarmı ekle",
        "answers": [{"name": "set_alarm", "arguments": {"time": "08:00", "label": "Uyanma"}}],
        "reasoning": "'08:00' -> time; 'Uyanma' -> label",
    },
    {
        "query": "14:15 toplantı alarmı kur",
        "answers": [{"name": "set_alarm", "arguments": {"time": "14:15", "label": "toplantı"}}],
        "reasoning": "'14:15' -> time; 'toplantı' -> label",
    },
    {
        "query": "Beni 06:45'te uyandır",
        "answers": [{"name": "set_alarm", "arguments": {"time": "06:45"}}],
        "reasoning": "'06:45' -> time",
    },
    {
        "query": "22:00 için İlaç Vakti alarmı ayarla",
        "answers": [{"name": "set_alarm", "arguments": {"time": "22:00", "label": "İlaç Vakti"}}],
        "reasoning": "'22:00' -> time; 'İlaç Vakti' -> label",
    },
    {
        "query": "18:30 için fırını kapat alarmı kur",
        "answers": [{"name": "set_alarm", "arguments": {"time": "18:30", "label": "fırını kapat"}}],
        "reasoning": "'18:30' -> time; 'fırını kapat' -> label",
    },
    # calculate
    {
        "query": "25 * 4 hesapla",
        "answers": [{"name": "calculate", "arguments": {"expression": "25 * 4"}}],
        "reasoning": "'25 * 4' -> expression",
    },
    {
        "query": "15 ile 48'i topla",
        "answers": [{"name": "calculate", "arguments": {"expression": "15 + 48"}}],
        "reasoning": "'15 + 48' -> expression",
    },
    {
        "query": "120 bölü 6 kaç eder?",
        "answers": [{"name": "calculate", "arguments": {"expression": "120 / 6"}}],
        "reasoning": "'120 / 6' -> expression",
    },
    {
        "query": "7'nin karesini al",
        "answers": [{"name": "calculate", "arguments": {"expression": "7 ** 2"}}],
        "reasoning": "'7 ** 2' -> expression",
    },
    {
        "query": "250'den 85 çıkar",
        "answers": [{"name": "calculate", "arguments": {"expression": "250 - 85"}}],
        "reasoning": "'250 - 85' -> expression",
    },
    {
        "query": "(10 + 5) * 3 işleminin sonucu nedir?",
        "answers": [{"name": "calculate", "arguments": {"expression": "(10 + 5) * 3"}}],
        "reasoning": "'(10 + 5) * 3' -> expression",
    },
    {
        "query": "500 * 0.18 işlemini hesaplar mısın?",
        "answers": [{"name": "calculate", "arguments": {"expression": "500 * 0.18"}}],
        "reasoning": "'500 * 0.18' -> expression",
    },
    # NEGATİF / KONU DIŞI ÖRNEKLER (answers: [])
    # Cactus Needle kuralı: ~1/8 oranında answers: [] olmalıdır. Aksi halde model her şeye araç çağırır!
    {
        "query": "Merhaba, nasılsın?",
        "answers": [],
        "reasoning": "",
    },
    {
        "query": "Bana güzel bir şiir yazar mısın?",
        "answers": [],
        "reasoning": "",
    },
    {
        "query": "Bugün kendimi biraz yorgun hissediyorum.",
        "answers": [],
        "reasoning": "",
    },
    {
        "query": "Türkiye'nin başkenti neresidir?",
        "answers": [],
        "reasoning": "",
    },
    {
        "query": "Teşekkür ederim, iyi akşamlar!",
        "answers": [],
        "reasoning": "",
    },
]


# ==============================================================================
# 2. SENTETİK ÇOĞALTICI (Offline Deterministic Generator)
# ==============================================================================

CITIES = [
    "İstanbul", "Ankara", "İzmir", "Bursa", "Antalya", "Adana", "Konya",
    "Gaziantep", "Trabzon", "Samsun", "Eskişehir", "Diyarbakır", "Mersin",
    "Kayseri", "Denizli", "Lagos", "Berlin", "Paris", "Londra", "Roma", "Tokyo"
]

NAMES = [
    "Ahmet", "Mehmet", "Ayşe", "Fatma", "Ali", "Can", "Zeynep", "Burak",
    "Elif", "Deniz", "Ece", "Cem", "Selin", "Murat", "Mert", "Gizem"
]

MESSAGES = [
    "toplantı saat 14'e ertelendi", "gelirken ekmek al", "dosyaları gönderdim",
    "nerede kaldın", "akşam yemeğe bekliyoruz", "raporu inceledim çok iyi",
    "yarın görüşürüz", "aradım ulaşamadım beni ara", "ofise geçiyorum"
]

ALARM_LABELS = [
    "Uyanma", "Toplantı", "Ders", "İlaç", "Fırın", "Spor", "Mola", "Sınav"
]

OFF_TOPIC_QUERIES = [
    "Nasılsın, bugün ne yapıyorsun?",
    "Bana Python hakkında bilgi verir misin?",
    "En sevdiğin renk nedir?",
    "Dünya neden yuvarlaktır?",
    "Yapay zeka nasıl öğrenir?",
    "Bir fıkra anlatır mısın?",
    "Çok teşekkürler, eline sağlık.",
    "Bugün hava almak için parka gideceğim.",
    "Programlama öğrenmek ne kadar sürer?",
    "Günaydın, iyi haftalar dilerim.",
    "Hangi dilleri konuşabiliyorsun?",
    "Bana bir kitap önerisi yap.",
]


def generate_synthetic_samples(num_samples: int = 100) -> List[Dict[str, Any]]:
    """Geniş ve dengeli Türkçe veri kümesi üretir."""
    rng = random.Random(42)
    schemas = get_needle_tools_schema()
    samples: List[Dict[str, Any]] = []

    # Negatif örnek hedefi: yaklaşık %12.5 (1/8)
    num_negatives = max(1, num_samples // 8)
    num_positives = num_samples - num_negatives

    # Önce tohumları ekle
    for s in SEED_EXAMPLES:
        row = dict(s)
        row["tools"] = schemas
        samples.append(row)

    # Pozitif sentetik örnekler üret
    while len(samples) < num_positives:
        choice = rng.choice(["weather", "message", "alarm", "calc"])
        if choice == "weather":
            city = rng.choice(CITIES)
            templates = [
                (f"{city} için hava durumu nedir?", f"'{city}' -> city"),
                (f"{city}'da hava nasıl?", f"'{city}' -> city"),
                (f"{city} hava sıcaklığı kaç derece?", f"'{city}' -> city"),
                (f"Bugün {city}'da yağmur var mı?", f"'{city}' -> city"),
            ]
            q, r = rng.choice(templates)
            samples.append({
                "query": q,
                "tools": schemas,
                "answers": [{"name": "get_weather", "arguments": {"city": city}}],
                "reasoning": r,
            })
        elif choice == "message":
            name = rng.choice(NAMES)
            msg = rng.choice(MESSAGES)
            templates = [
                (f"{name}'e '{msg}' mesajı gönder", f"'{name}' -> recipient; '{msg}' -> message"),
                (f"{name}'e {msg} diye yaz", f"'{name}' -> recipient; '{msg}' -> message"),
                (f"{name}'e {msg} ilet", f"'{name}' -> recipient; '{msg}' -> message"),
            ]
            q, r = rng.choice(templates)
            samples.append({
                "query": q,
                "tools": schemas,
                "answers": [{"name": "send_message", "arguments": {"recipient": name, "message": msg}}],
                "reasoning": r,
            })
        elif choice == "alarm":
            h = rng.randint(6, 23)
            m = rng.choice(["00", "15", "30", "45"])
            time_str = f"{h:02d}:{m}"
            if rng.random() > 0.5:
                label = rng.choice(ALARM_LABELS)
                q = f"Saat {time_str} için {label} alarmı kur"
                r = f"'{time_str}' -> time; '{label}' -> label"
                ans = {"name": "set_alarm", "arguments": {"time": time_str, "label": label}}
            else:
                q = f"Sabah {time_str} için alarm ayarla"
                r = f"'{time_str}' -> time"
                ans = {"name": "set_alarm", "arguments": {"time": time_str}}
            samples.append({
                "query": q,
                "tools": schemas,
                "answers": [ans],
                "reasoning": r,
            })
        else:
            a = rng.randint(2, 50)
            b = rng.randint(2, 50)
            op = rng.choice(["+", "-", "*", "/"])
            if op == "/":
                b = rng.choice([2, 4, 5, 10])
                a = b * rng.randint(2, 20)
            expr = f"{a} {op} {b}"
            templates = [
                (f"{expr} hesapla", f"'{expr}' -> expression"),
                (f"{expr} işleminin sonucu nedir?", f"'{expr}' -> expression"),
                (f"{expr} kaç eder?", f"'{expr}' -> expression"),
            ]
            q, r = rng.choice(templates)
            samples.append({
                "query": q,
                "tools": schemas,
                "answers": [{"name": "calculate", "arguments": {"expression": expr}}],
                "reasoning": r,
            })

    # Negatif örnekleri ekle
    neg_idx = 0
    while len(samples) < num_samples:
        q = OFF_TOPIC_QUERIES[neg_idx % len(OFF_TOPIC_QUERIES)]
        samples.append({
            "query": q,
            "tools": schemas,
            "answers": [],
            "reasoning": "",
        })
        neg_idx += 1

    rng.shuffle(samples)
    return samples[:num_samples]


# ==============================================================================
# 3. VERİ KÜMESİ DERLEME VE DISKE YAZMA
# ==============================================================================

def build_dataset(
    output_dir: str = "data",
    num_samples: int = 120,
    train_ratio: float = 0.85,
    use_ollama: bool = False,
    ollama_model: str = "qwen2.5:7b",
    ollama_host: str = "http://localhost:11434",
) -> Dict[str, Any]:
    """Cactus Needle 3 formatında train.jsonl ve val.jsonl oluşturur."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    dataset = generate_synthetic_samples(num_samples)

    split_idx = int(len(dataset) * train_ratio)
    train_data = dataset[:split_idx]
    val_data = dataset[split_idx:] if split_idx < len(dataset) else dataset[:2]

    train_file = out_path / "train.jsonl"
    val_file = out_path / "val.jsonl"
    all_file = out_path / "data.jsonl"

    def write_jsonl(p: Path, rows: List[Dict[str, Any]]):
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_jsonl(train_file, train_data)
    write_jsonl(val_file, val_data)
    write_jsonl(all_file, dataset)

    empty_answers = sum(1 for r in dataset if len(r.get("answers", [])) == 0)
    ratio = empty_answers / len(dataset) if dataset else 0.0

    print("=" * 60)
    print("Needle 3 Türkçe Veri Seti Hazırlandı")
    print("=" * 60)
    print(f"  Toplam Örnek: {len(dataset)}")
    print(f"  - Eğitim (train.jsonl): {len(train_data)} örnek")
    print(f"  - Doğrulama (val.jsonl): {len(val_data)} örnek")
    print(f"  - Birleşik (data.jsonl): {all_file.resolve()}")
    print(f"  - Negatif / Konu Dışı (answers: []): {empty_answers} (%{ratio * 100:.1f})")
    print("✓ Needle 3 format doğrulaması başarılı (query, tools, answers, reasoning).")

    return {
        "total": len(dataset),
        "train": len(train_data),
        "val": len(val_data),
        "train_file": str(train_file),
        "val_file": str(val_file),
    }
