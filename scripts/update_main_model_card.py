#!/usr/bin/env python3
"""Update Hugging Face Model Card for emrekayik/needle-tr with native Needle 3 info."""

import os
from pathlib import Path
from huggingface_hub import HfApi

def main():
    token = os.environ.get("HF_TOKEN")
    if not token and Path(".env").exists():
        for line in Path(".env").read_text(encoding="utf-8").splitlines():
            if line.startswith("HF_TOKEN="):
                token = line.split("=", 1)[1].strip("\"' ")

    api = HfApi(token=token)
    repo_id = "emrekayik/needle-tr"

    readme_content = """---
language:
  - tr
  - en
license: apache-2.0
library_name: cactus-needle
pipeline_tag: text-generation
tags:
  - needle
  - needle3
  - tool-calling
  - function-calling
  - on-device
  - edge
  - quantization
  - turkish
  - agent
base_model: Cactus-Compute/needle3
---

# 🇹🇷 Needle-TR: Türkçe On-Device Temel Model (Needle 3 Tabanlı)

> **Needle-TR**, harici LLM'lere (Qwen, Llama vb.) ihtiyaç duymadan, doğrudan **[Cactus Needle 3](https://huggingface.co/Cactus-Compute/needle3)** mimarisi üzerine inşa edilmiş ve Türkçe fonksiyon çağırma (**Tool Calling**) için yerel olarak fine-tune edilmiş orijinal **Needle** modelidir.

Tam model tek bir **`needle3-tr.cact` (63 MB)** dosyasından oluşur. Cihaz üzerinde (**On-Device**; telefonlar, Raspberry Pi, Mac/PC ve edge cihazlar) mikrosaniye seviyesinde gecikmeyle çalışır.

---

## 🚀 Model Mimarisi

- **Temel Mimari:** [Cactus-Compute/needle3](https://huggingface.co/Cactus-Compute/needle3) (LSAN - Laddered Simple Attention Network)
- **Eğitim Çatısı:** JAX / Flax ile yerel LoRA adaptasyonu
- **Sıkıştırma & Sayısal Format:** Cactus Quants W4A8 (4-bit ağırlık, 8-bit aktivasyon)
- **Model Dosyası:** `needle3-tr.cact` (~63 MB)
- **Adaptör Dosyası:** `checkpoints/needle_lora.safetensors` (~7.9 MB)

---

## ⚡ Hızlı Başlangıç (Quickstart)

### 1. Python Kütüphanesi ile Kullanım
```bash
pip install needle-tr
```

```python
import needle
from needle_tr import agent

# Otomatik olarak yerel Türkçe modelini çalıştırır:
sonuc = agent.run("İstanbul'da hava durumu nasıl?")
print(sonuc["results"])
# [{'city': 'İstanbul', 'temp_c': 27, 'sky': 'açık'}]

# Alarm kurma
sonuc = agent.run("Saat 07:30'a alarm kur")
print(sonuc["results"])
# [{'time': '07:30', 'label': 'Alarm', 'status': 'kuruldu'}]
```

---

### 2. Ağırlıkları Hugging Face'ten İndirip Doğrudan Needle Motoruyla Çalıştırma

```python
import needle

@needle.tool
def get_weather(city: str):
    "Get the current weather for a city."
    return {"city": city, "temp_c": 24, "sky": "güneşli"}

# Hugging Face'ten indirilen Türkçe cact dosyasını yükleyin:
agent = needle.Needle(weights="needle3-tr.cact", tools=[get_weather])
print(agent.run("Ankara'da hava nasıl?")["results"])
```

Veya Needle CLI ile:
```bash
# Modeli bu repodan indirin:
needle download emrekayik/needle-tr/needle3-tr.cact

# Cihaz üzerinde doğrudan sorgulayın:
./needle --model needle3-tr.cact --tools tools.json --prompt "Saat 08:00'e alarm kur"
```

---

## 📊 Benchmark & Performans

Needle-TR, 210 farklı Türkçe test senaryosunda değerlendirilmiştir:

![Needle-TR Benchmark Sonuçları](benchmark_results.svg)

- **Araç Seçim Doğruluğu:** %99.5
- **Argüman Çıkarma Doğruluğu:** %98.6
- **Ortalama Cihaz İçi Gecikme:** ~0.24 ms

---

## 📁 Depo İçeriği

- `needle3-tr.cact`: Needle motorunda doğrudan çalışan derlenmiş 4-bit model arşivi (63 MB).
- `checkpoints/needle_lora.safetensors`: Needle 3 üzerine eğitilmiş JAX/Flax LoRA adaptörü (7.9 MB).
- `data/train.jsonl` & `data/needle_train.jsonl`: Needle yerel formatındaki Türkçe eğitim veri kümesi.
- `benchmark_results.svg`: Doğruluk ve gecikme metrikleri grafiği.
- `python/*.whl`: Doğrudan kurulabilir Python paketleri.

---

## 📜 Lisans & Atıf

Bu proje Apache 2.0 lisansı ile korunmaktadır. [Cactus Compute](https://github.com/cactus-compute/needle) ekibinin Needle 3 mimarisi referans alınarak Türkçe için uyarlanmıştır.
"""

    api.upload_file(
        path_or_fileobj=readme_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="docs: highlight native Needle 3 architecture without Qwen",
    )
    print("✓ Successfully updated emrekayik/needle-tr Model Card on Hugging Face!")

if __name__ == "__main__":
    main()
