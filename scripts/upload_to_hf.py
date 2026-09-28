#!/usr/bin/env python3
"""Upload Needle-TR model, Modelfile, datasets, benchmarks, and package to Hugging Face."""

import os
import shutil
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

    print(f"Checking repository '{repo_id}'...")
    api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")

    staging_dir = Path("scratch_hf_repo")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    # 1. Modelfile
    if Path("Modelfile").exists():
        shutil.copy("Modelfile", staging_dir / "Modelfile")

    # 2. Benchmark results SVG
    if Path("benchmark_results.svg").exists():
        shutil.copy("benchmark_results.svg", staging_dir / "benchmark_results.svg")

    # 3. Data files
    (staging_dir / "data").mkdir(exist_ok=True)
    if Path("data/train.jsonl").exists():
        shutil.copy("data/train.jsonl", staging_dir / "data/train.jsonl")
    if Path("data/val.jsonl").exists():
        shutil.copy("data/val.jsonl", staging_dir / "data/val.jsonl")

    # 4. Built wheels
    (staging_dir / "python").mkdir(exist_ok=True)
    for whl in Path("dist").glob("*.whl"):
        shutil.copy(whl, staging_dir / "python" / whl.name)

    # 5. LoRA weights & tokenizer
    lora_dir = Path("models/needle_lora")
    if lora_dir.exists():
        for f in lora_dir.iterdir():
            if f.is_file() and f.name != "README.md":
                shutil.copy(f, staging_dir / f.name)

    # 6. config.json
    config_content = """{
  "model_type": "needle-tr",
  "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
  "framework": "cactus-needle",
  "languages": ["tr", "en"],
  "task": "tool-calling",
  "active_tools": ["get_weather", "send_message", "set_alarm", "calculate"],
  "version": "0.1.5"
}"""
    (staging_dir / "config.json").write_text(config_content, encoding="utf-8")

    # 7. Model Card README.md
    readme_content = """---
language:
  - tr
  - en
license: apache-2.0
library_name: cactus-needle
pipeline_tag: text-generation
tags:
  - needle
  - tool-calling
  - function-calling
  - on-device
  - edge
  - turkish
  - agent
  - ollama
base_model: Qwen/Qwen2.5-0.5B-Instruct
---

# 🇹🇷 Needle-TR: Türkçe Araç Çağırıcı ve On-Device Ajan Modeli

> **Needle-TR**, Türkçe komutları ve doğal dil ifadelerini anında yerel fonksiyon çağrılarına (**Tool Calling / Function Calling**) dönüştüren, [Cactus Needle](https://huggingface.co/Cactus-Compute/needle3) tabanlı ultra hafif ve yüksek doğruluklu bir yapay zeka sistemidir.

Harici sunuculara veya 15+ GB devasa modellere bağımlı olmadan, cihaz üzerinde (**On-Device**) doğrudan çalışır.

---

## 📊 Benchmark & Performans

Needle-TR, 210 farklı Türkçe test senaryosunda kapsamlı şekilde değerlendirilmiştir:

![Needle-TR Benchmark Sonuçları](benchmark_results.svg)

- **Araç Seçim Doğruluğu:** %99.5
- **Argüman Çıkarma Doğruluğu:** %98.6
- **Ortalama Gecikme:** ~0.24 ms (Cihaz içi çalıştırma)

---

## ⚡ Hızlı Başlangıç (Quickstart)

### 1. Python Kütüphanesi ile Kullanım
```bash
pip install needle-tr
```

```python
from needle_tr import agent

# Türkçe hava durumu sorgusu
sonuc = agent.run("İstanbul'da hava durumu nasıl?")
print(sonuc["results"])
# [{'city': 'İstanbul', 'temp_c': 22, 'condition': 'Güneşli'}]

# Alarm kurma
sonuc = agent.run("Yarın sabah saat 07:30'a alarm kur")
print(sonuc["results"])
# [{'time': '07:30', 'label': 'Sabah Alarmı', 'status': 'kuruldu'}]

# Matematiksel hesaplama
sonuc = agent.run("45 ile 12'yi çarp")
print(sonuc["results"])
# [{'operation': 'multiply', 'a': 45, 'b': 12, 'result': 540}]
```

---

### 2. Ollama ile Yerel LLM Olarak Çalıştırma

Modeli Ollama içerisine aktararak bağımsız bir yerel model olarak çalıştırabilirsiniz:

```bash
# Bu repodan Modelfile dosyasını indirin:
ollama create needle-tr -f Modelfile

# Doğrudan Türkçe komut verin:
ollama run needle-tr "Mehmet'e 'Yarınki toplantıyı 14:00'e erteledik' mesajı at"
```

---

### 3. Hugging Face Transformers & PEFT ile Kullanım

Bu repodaki LoRA adaptörlerini doğrudan Python transformers ortamınıza yükleyebilirsiniz:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
repo_id = "emrekayik/needle-tr"

tokenizer = AutoTokenizer.from_pretrained(repo_id)
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto")
model = PeftModel.from_pretrained(base_model, repo_id)
```

---

## 🛠️ Desteklenen Yerel Araçlar (Active Tools)

| Fonksiyon Adı | Açıklama | Örnek Türkçe İfade |
| :--- | :--- | :--- |
| `get_weather(city)` | Verilen şehir için hava durumu verisi döner. | *"İzmir'de bugün hava nasıl?"* |
| `send_message(recipient, message)` | Belirtilen kişiye mesaj/ileti gönderir. | *"Ahmet'e 'Dosyaları ilettim' yaz."* |
| `set_alarm(time, label)` | İstenilen saate ve etikete göre alarm kurar. | *"Saat 08:00'e toplantı alarmı kur."* |
| `calculate(expression)` | Matematiksel işlemleri güvenli şekilde hesaplar. | *"125 çarpı 8 kaç eder?"* |

---

## 📁 Depo İçeriği (Repository Structure)

- `adapter_model.safetensors`: Needle-TR LoRA adaptör ağırlıkları (~8.6 MB).
- `adapter_config.json`: Peft / LoRA hiperparametreleri.
- `Modelfile`: Ollama için hazır model tanım dosyası.
- `benchmark_results.svg`: 210 testlik doğruluk ve gecikme performans grafiği.
- `tokenizer.json` & `tokenizer_config.json`: Model tokenizer dosyaları.
- `data/train.jsonl` & `data/val.jsonl`: Türkçe sentetik eğitim ve doğrulama veri kümesi.
- `python/*.whl`: Doğrudan kurulabilir Python paketi.

---

## 📜 Lisans & Atıf

Bu proje Apache 2.0 lisansı ile korunmaktadır. [Cactus Needle](https://huggingface.co/Cactus-Compute/needle3) mimarisi üzerine inşa edilmiştir.

```bibtex
@misc{needle_tr_2026,
  title        = {Needle-TR: Turkish On-Device Tool-Calling Foundation Model},
  author       = {Emre Kayik},
  year         = {2026},
  howpublished = {https://huggingface.co/emrekayik/needle-tr}
}
```
"""
    (staging_dir / "README.md").write_text(readme_content, encoding="utf-8")

    print(f"Uploading files to Hugging Face model repository '{repo_id}'...")
    api.upload_folder(
        folder_path=str(staging_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Release Needle-TR: Model weights, Modelfile, benchmarks, whl, and datasets"
    )
    shutil.rmtree(staging_dir, ignore_errors=True)
    print(f"✨ [Başarılı] Needle-TR modeli Hugging Face üzerinde yayında: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    main()
