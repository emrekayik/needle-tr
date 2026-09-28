"""Ollama tabanlı sentetik Türkçe veri seti üreticisi.

Bu modül:
1. Ollama API'si (ör. llama3, qwen2.5, mistral) üzerinden Türkçe kullanıcı sorguları
   ve karşılık gelen araç çağrılarını (function calling) sentetik olarak üretir.
2. Ollama çalışmıyorken veya çevrimdışı kullanımda zengin tohum (seed) veri kümesini
   kullanarak hemen train.jsonl / val.jsonl dosyalarını oluşturabilir.
3. Üretilen verileri ChatML ve OpenAI Tool Calling formatlarında kaydeder.
"""

from __future__ import annotations

import json
import os
import random
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from .tools import ACTIVE_TOOLS, get_tools_schema


SEED_DATA: List[Dict[str, Any]] = [
    # calculate (Hesaplama)
    {
        "query": "25 * 4 hesapla",
        "tool": "calculate",
        "arguments": {"expression": "25 * 4"},
    },
    {
        "query": "15 ile 48'i topla",
        "tool": "calculate",
        "arguments": {"expression": "15 + 48"},
    },
    {
        "query": "120 bölü 6 kaç eder?",
        "tool": "calculate",
        "arguments": {"expression": "120 / 6"},
    },
    {
        "query": "7'nin karesini al",
        "tool": "calculate",
        "arguments": {"expression": "7 ** 2"},
    },
    {
        "query": "250'den 85 çıkar",
        "tool": "calculate",
        "arguments": {"expression": "250 - 85"},
    },
    {
        "query": "(10 + 5) * 3 işleminin sonucu nedir?",
        "tool": "calculate",
        "arguments": {"expression": "(10 + 5) * 3"},
    },
    {
        "query": "500 * 0.18 hesaplar mısın?",
        "tool": "calculate",
        "arguments": {"expression": "500 * 0.18"},
    },
    # get_weather (Hava Durumu)
    {
        "query": "Lagos'ta hava nasıl?",
        "tool": "get_weather",
        "arguments": {"city": "Lagos"},
    },
    {
        "query": "İstanbul için hava durumunu göster",
        "tool": "get_weather",
        "arguments": {"city": "İstanbul"},
    },
    {
        "query": "Ankara'da hava durumu nedir?",
        "tool": "get_weather",
        "arguments": {"city": "Ankara"},
    },
    {
        "query": "İzmir'de şu an hava kaç derece?",
        "tool": "get_weather",
        "arguments": {"city": "İzmir"},
    },
    {
        "query": "Londra hava durumu nasıl?",
        "tool": "get_weather",
        "arguments": {"city": "Londra"},
    },
    {
        "query": "Berlin için hava raporunu al",
        "tool": "get_weather",
        "arguments": {"city": "Berlin"},
    },
    # send_message (Mesaj Gönderme)
    {
        "query": "Ahmet'e 'Toplantı başladı' mesajı gönder",
        "tool": "send_message",
        "arguments": {"recipient": "Ahmet", "message": "Toplantı başladı"},
    },
    {
        "query": "Mehmet'e yarın saat 10'da ofisteyim diye yaz",
        "tool": "send_message",
        "arguments": {"recipient": "Mehmet", "message": "Yarın saat 10'da ofisteyim"},
    },
    {
        "query": "Ayşe'ye mesaj at: Sunum hazır",
        "tool": "send_message",
        "arguments": {"recipient": "Ayşe", "message": "Sunum hazır"},
    },
    {
        "query": "Ali'ye 'Gelirken ekmek al' iletisi gönder",
        "tool": "send_message",
        "arguments": {"recipient": "Ali", "message": "Gelirken ekmek al"},
    },
    {
        "query": "Zeynep'e tebrikler mesajı yolla",
        "tool": "send_message",
        "arguments": {"recipient": "Zeynep", "message": "Tebrikler"},
    },
    # set_alarm (Alarm Kurma)
    {
        "query": "Saat 07:30'a alarm kur",
        "tool": "set_alarm",
        "arguments": {"time": "07:30", "label": "Alarm"},
    },
    {
        "query": "Sabah 08:00 için Uyanış alarmı ayarla",
        "tool": "set_alarm",
        "arguments": {"time": "08:00", "label": "Uyanış"},
    },
    {
        "query": "14:15'e Toplantı alarmı kur",
        "tool": "set_alarm",
        "arguments": {"time": "14:15", "label": "Toplantı"},
    },
    {
        "query": "22:00'ye İlaç Zamanı hatırlatıcısı kur",
        "tool": "set_alarm",
        "arguments": {"time": "22:00", "label": "İlaç Zamanı"},
    },
    {
        "query": "06:45 için alarm oluştur",
        "tool": "set_alarm",
        "arguments": {"time": "06:45", "label": "Alarm"},
    },
]


def check_ollama_available(host: str = "http://localhost:11434") -> bool:
    """Ollama sunucusunun çalışıp çalışmadığını kontrol eder (asla kendisi başlatmaz)."""
    try:
        req = urllib.request.Request(f"{host}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            return resp.status == 200
    except Exception:
        return False


def call_ollama_generate(
    prompt: str,
    model: str = "qwen2.5:7b",
    host: str = "http://localhost:11434",
    timeout: int = 30,
) -> Optional[str]:
    """Ollama REST API'sine istek gönderir (Ollama çalışmıyorsa None döner)."""
    url = f"{host}/api/generate"
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
    }).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data.get("response")
    except Exception as exc:
        print(f"[UYARI] Ollama'ya ulaşılamadı: {exc}", file=sys.stderr)
        return None


def generate_synthetic_samples_with_ollama(
    model: str = "qwen2.5:7b",
    count_per_tool: int = 5,
    host: str = "http://localhost:11434",
) -> List[Dict[str, Any]]:

    """Ollama üzerinden araçlar için sentetik Türkçe örnekler üretir."""
    if not check_ollama_available(host):
        print(f"[BİLGİ] Ollama sunucusu ({host}) aktif değil. Tohum (seed) veri seti kullanılacak.")
        return []

    schemas = get_tools_schema()
    samples: List[Dict[str, Any]] = []

    prompt = f"""
Sen Türkçe fonksiyon çağırma (tool calling) veri kümesi üreten bir asistansın.
Aşağıda kullanabileceğin araçların şemaları verilmiştir:
{json.dumps(schemas, ensure_ascii=False, indent=2)}

Lütfen her bir araç için kullanıcıların sorabileceği {count_per_tool} farklı doğal Türkçe kullanıcı sorgusu üret.
Çıktıyı kesinlikle şu JSON formatında bir liste olarak ver:
[
  {{
    "query": "kullanıcının Türkçe cümlesi",
    "tool": "çağrılacak_fonksiyon_adı",
    "arguments": {{"parametre_adı": "değeri"}}
  }}
]
"""
    print(f"[Ollama] {model} modeli kullanılarak sentetik veri üretiliyor...")
    res = call_ollama_generate(prompt, model=model, host=host)
    if not res:
        return []

    try:
        parsed = json.loads(res)
        if isinstance(parsed, list):
            for item in parsed:
                if "query" in item and "tool" in item and "arguments" in item:
                    samples.append(item)
    except json.JSONDecodeError:
        print("[HATA] Ollama çıktısı JSON olarak ayrıştırılamadı.", file=sys.stderr)

    return samples


def to_chatml_format(sample: Dict[str, Any], system_prompt: str) -> Dict[str, Any]:
    """Örneği standart SFT/ChatML eğitim formatına dönüştürür."""
    tool_call = {
        "name": sample["tool"],
        "arguments": sample["arguments"],
    }
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": sample["query"]},
            {
                "role": "assistant",
                "content": f"```json\n{json.dumps(tool_call, ensure_ascii=False)}\n```",
            },
        ]
    }


def build_dataset(
    output_dir: str = "data",
    use_ollama: bool = False,
    ollama_model: str = "qwen2.5:7b",
    ollama_host: str = "http://localhost:11434",
    train_ratio: float = 0.85,
) -> Dict[str, int]:
    """Veri setini oluşturur ve data/train.jsonl ile data/val.jsonl olarak kaydeder."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_samples = list(SEED_DATA)

    if use_ollama:
        ollama_samples = generate_synthetic_samples_with_ollama(
            model=ollama_model, host=ollama_host
        )
        if ollama_samples:
            all_samples.extend(ollama_samples)

    # Çeşitlilik için karıştır
    random.seed(42)
    random.shuffle(all_samples)

    system_prompt = (
        "Sen Türkçe komutları uygun araçlara dönüştüren bir yapay zeka asistanısın. "
        "Kullanıcı isteklerine uygun olan aracı ve parametrelerini JSON formatında çağır."
    )

    formatted_dataset = [to_chatml_format(s, system_prompt) for s in all_samples]

    split_idx = int(len(formatted_dataset) * train_ratio)
    train_data = formatted_dataset[:split_idx]
    val_data = formatted_dataset[split_idx:] if split_idx < len(formatted_dataset) else formatted_dataset[:2]

    train_file = out_path / "train.jsonl"
    val_file = out_path / "val.jsonl"

    with open(train_file, "w", encoding="utf-8") as f:
        for row in train_data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(val_file, "w", encoding="utf-8") as f:
        for row in val_data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[Veri Seti] Toplam {len(formatted_dataset)} örnek hazırlandı:")
    print(f"  - Eğitim: {train_file} ({len(train_data)} örnek)")
    print(f"  - Doğrulama: {val_file} ({len(val_data)} örnek)")

    return {"total": len(formatted_dataset), "train": len(train_data), "val": len(val_data)}
