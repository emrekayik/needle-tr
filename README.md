# needle-tr

Türkçe Araç Çağırıcı (Tool Calling) ve Fine-Tuning Sistemi.

`needle-tr`, Türkçe dilinde fonksiyon / araç çağırma (function calling) yeteneklerine sahip yapay zeka modelleri geliştirmek, sentetik veri üretmek, LoRA ile fine-tuning yapmak ve Ollama üzerinde dağıtmak için tasarlanmış uçtan uca bir kütüphane ve CLI aracıdır.

Proje, [Cactus Compute](https://github.com/cactus-compute/needle) tarafından geliştirilen hafif ve cihaz üstü (on-device) araç çağırma modeli **[Needle](https://github.com/cactus-compute/needle)** (`cactus-needle`) altyapısını temel alarak Türkçe dil desteği ve fine-tuning araçlarıyla genişletilmiştir.

---

## 🚀 Özellikler

- **Yerleşik Türkçe Araçlar:** Hava durumu sorgulama, mesaj gönderme, alarm kurma, matematiksel hesaplama gibi şablon araçlar.
- **Sentetik Veri Üretici:** Tohum (seed) veriler ve isteğe bağlı yerel Ollama (`qwen2.5:7b` vb.) desteğiyle Türkçe veri seti genişletme (`data/train.jsonl`, `data/val.jsonl`).
- **LoRA Fine-Tuning Desteği:** Hugging Face `transformers`, `peft` ve `trl` entegrasyonu ile hedef modelleri Türkçe araç çağırma formatına uyarlama.
- **Ollama Dağıtımı:** Eğitilen veya mevcut modeller için otomatik `Modelfile` üretimi ve tek tıkla Ollama ortamına aktarım.

---

## 📦 Kurulum

Bu proje [uv](https://github.com/astral-sh/uv) paket yöneticisi ile yönetilmektedir:

```bash
# Bağımlılıkları senkronize edin
uv sync
```

---

## 🛠️ CLI Kullanımı

Tüm süreç `needle-tr` komut satırı arayüzü ile yönetilebilir:

### 1. Sentetik Veri Üretimi

```bash
# Yerel tohum verileri kullanarak
uv run needle-tr generate-data

# Ollama ile zenginleştirilmiş veri üretimi
uv run needle-tr generate-data --use-ollama --ollama-model qwen2.5:7b
```

### 2. Fine-Tuning

```bash
# Doğrulama (Dry-run)
uv run needle-tr train --dry-run

# Eğitim başlatma
uv run needle-tr train --base-model Qwen/Qwen2.5-7B-Instruct --epochs 3
```

### 3. Ollama Modelfile Dışa Aktarma

```bash
uv run needle-tr export-ollama --base-model qwen2.5:7b --model-name needle-tr
```

Ardından Ollama üzerinde modeli oluşturun ve çalıştırın:

```bash
ollama create needle-tr -f Modelfile
ollama run needle-tr "25 * 4 hesapla"
```

---

## 🐍 Python Ajan Kullanımı

Cihaz üstü yerel ajan çalıştırma, [Cactus Compute Needle](https://github.com/cactus-compute/needle) (`cactus-needle`) kütüphanesini kullanır:

```bash
uv run needle-agent
```

Veya Python kodunuz içerisinde:

```python
from needle_tr import agent

response = agent.run("İstanbul'da hava durumu nasıl?")
print(response)
```

---

## 📚 Detaylı Rehber

Ayrıntılı adımlar ve mimari açıklamaları için [FINETUNE_GUIDE.md](FINETUNE_GUIDE.md) dosyasına göz atabilirsiniz.

---

## 🙏 Referanslar & Teşekkürler

- **[Cactus Compute - Needle](https://github.com/cactus-compute/needle):** Cihaz üzerinde (on-device) çalışan hafif ve verimli fonksiyon / araç çağırma kütüphanesi (`needle`).
