# needle-tr

<p align="center">
  <b>Turkish Tool Calling & Fine-Tuning System | Türkçe Araç Çağırıcı ve İnce Ayar Sistemi</b><br>
  Built on <a href="https://github.com/cactus-compute/needle">Cactus Compute Needle</a> (<code>cactus-needle</code>)
</p>

<p align="center">
  <a href="#-english"><b>English</b></a> &nbsp;|&nbsp; <a href="#-türkçe"><b>Türkçe</b></a>
</p>

---

<a name="-english"></a>
## 🌐 English

### Overview

`needle-tr` is an end-to-end Python framework and CLI tool designed to develop AI models with Turkish **tool calling** (function calling) capabilities, synthesize Turkish datasets, perform **fine-tuning** with LoRA (PEFT), and deploy models to Ollama and edge devices.

The project builds upon and extends the lightweight on-device foundation model **[Needle](https://github.com/cactus-compute/needle)** (`cactus-needle`) by [Cactus Compute](https://github.com/cactus-compute/needle), bringing specialized Turkish language understanding, dataset pipelines, and training utilities.

---

### 🚀 Key Features

- **Built-in Tools:** Predefined functions for weather queries, sending messages, setting alarms, and mathematical calculations.
- **Synthetic Data Generator:** Dataset synthesis (`data/train.jsonl`, `data/val.jsonl`) using seed templates and local Ollama LLMs (`qwen2.5:7b`, etc.).
- **LoRA Fine-Tuning Pipeline:** Direct integration with Hugging Face `transformers`, `peft`, and `trl` to train open-weights models (e.g., Qwen 2.5 Instruct) on Turkish tool-calling schemas.
- **Ollama Modelfile Export:** Automatic generation of optimized `Modelfile` configs ready for `ollama create`.
- **On-Device Agent:** Native Python agent execution powered by `cactus-needle` for fast, lightweight local inference.
- **Automated Benchmark & Charts:** Built-in benchmarking suite measuring tool accuracy, argument matching, JSON validity, and latency, generating vector SVG reports.
- **Release Automation:** Single-command SemVer bumping, git tagging, and GitHub Actions publishing to PyPI.

---

### 📦 Installation

Standard lightweight installation (agent and CLI only, ~20 MB):
```bash
pip install needle-tr
```

For **fine-tuning** and **Hugging Face** training dependencies:
```bash
pip install "needle-tr[train]"
```

Or for development with [uv](https://github.com/astral-sh/uv):
```bash
git clone https://github.com/emrekayik/needle-tr.git
cd needle-tr
uv sync --extra train
```

---

### 🛠️ CLI Usage

All operations are accessible through the `needle-tr` CLI:

#### 1. Synthetic Data Generation
```bash
# Using local seed data
uv run needle-tr generate-data

# Augmented data generation using a local Ollama model
uv run needle-tr generate-data --use-ollama --ollama-model qwen2.5:7b
```

#### 2. Fine-Tuning & Hugging Face Upload
```bash
# Dry-run validation (checks dataset and GPU settings without training)
uv run needle-tr train --dry-run

# Run LoRA training
uv run needle-tr train --base-model Qwen/Qwen2.5-7B-Instruct --epochs 3

# Train and automatically publish LoRA adapter to Hugging Face
uv run needle-tr train --base-model Qwen/Qwen2.5-7B-Instruct --push-to-hub --hub-model-id username/needle-tr-lora

# Push existing trained model or dataset to Hugging Face Hub anytime
uv run needle-tr push-hub username/needle-tr-lora --model-dir models/needle_lora
uv run needle-tr push-hub username/needle-tr-dataset --dataset
```

#### 3. Export to Ollama
```bash
# Generate Modelfile
uv run needle-tr export-ollama --base-model qwen2.5:7b --model-name needle-tr

# Register and test in Ollama
ollama create needle-tr -f Modelfile
ollama run needle-tr "25 * 4 hesapla"
```

#### 4. Automated Benchmark
```bash
# Evaluate on-device Cactus Needle agent
uv run needle-tr benchmark --suite

# Evaluate Ollama model
uv run needle-tr benchmark --target ollama --ollama-model needle-tr
```

#### 5. Release & Publishing
```bash
# All-in-one release command: bumps version, tags git, cleans dist, builds & publishes to PyPI
./release.sh          # Patch bump (e.g., 0.1.5 -> 0.1.6)
./release.sh minor    # Minor bump (e.g., 0.1.5 -> 0.2.0)

# Or via Python CLI:
uv run needle-tr release patch --push
```

---

### 🐍 Python Agent Usage

Run on-device inference using the Python library:

```python
from needle_tr import agent

# Turkish weather inquiry
result = agent.run("İstanbul'da hava durumu nasıl?")
print("Weather Result:", result["results"])

# Mathematical computation
calc_result = agent.run("25 * 4 hesapla")
print("Calculation:", calc_result["results"])
```

Or run the pre-configured CLI agent:
```bash
uv run needle-agent
```

---

<!-- BENCHMARK_START -->
### 📊 Benchmark & Evaluation Results

Latest benchmark results measured with the built-in `needle-tr benchmark` tool:

![Needle-TR Benchmark Results](https://raw.githubusercontent.com/emrekayik/needle-tr/main/benchmark_results.svg)

| Metric | Value | Description |
| :--- | :--- | :--- |
| **Target Tested** | `Cactus Needle (On-Device Agent)` | Tested environment (Agent / Model) |
| **Total Test Samples** | **210** | Number of Turkish test queries |
| **Tool Selection Accuracy** | **%75.7** (159/210) | Correct tool selected rate |
| **Argument Match Accuracy** | **%66.2** (139/210) | Exact/semantic parameter match rate |
| **Valid Format Rate** | **%99.5** (209/210) | Cleanly parsed structured output rate |
| **Average Latency** | **197.8 ms** | Min: 51.5 ms, Max: 1014.2 ms |
| **Last Updated** | `2026-09-28 19:14:06` | Benchmark execution timestamp |

| Tool | Tool Accuracy | Argument Match | Samples | Average Latency |
| :--- | :---: | :---: | :---: | :---: |
| `calculate` | %90.9 | %67.3 | 55 | 126.6 ms |
| `get_weather` | %92.7 | %81.8 | 55 | 253.5 ms |
| `send_message` | %16.0 | %14.0 | 50 | 164.2 ms |
| `set_alarm` | %100.0 | %100.0 | 50 | 248.5 ms |
<!-- BENCHMARK_END -->

---

### 📚 Detailed Documentation

For step-by-step training architecture and dataset formatting, see [FINETUNE_GUIDE.md](FINETUNE_GUIDE.md).

---

### 🙏 References & Acknowledgements

- **[Cactus Compute - Needle](https://github.com/cactus-compute/needle):** Lightweight on-device function and tool calling foundation library (`cactus-needle`).

<br>

---
---

<a name="-türkçe"></a>
## 🇹🇷 Türkçe

### Genel Bakış

`needle-tr`, Türkçe dilinde **tool calling** (araç çağırma) ve **function calling** (fonksiyon çağırma) yeteneklerine sahip yapay zeka modelleri geliştirmek, **synthetic dataset** (sentetik veri kümesi) üretmek, LoRA yöntemiyle **fine-tuning** (modele ince ayar yapma) gerçekleştirmek ve modelleri Ollama üzerinde **deploy** etmek (dağıtmak/yayınlamak) için tasarlanmış uçtan uca bir Python kütüphanesi ve CLI (komut satırı) aracıdır.

Proje, [Cactus Compute](https://github.com/cactus-compute/needle) tarafından geliştirilen ve doğrudan yerel donanımda çalışan **on-device** (cihaz üstü) araç çağırma modeli **[Needle](https://github.com/cactus-compute/needle)** (`cactus-needle`) altyapısını temel alarak Türkçe dil desteği, veri üretim bantları ve eğitim araçlarıyla genişletilmiştir.

---

### 🚀 Öne Çıkan Özellikler

- **Yerleşik Türkçe Araçlar (`Built-in Tools`):** Hava durumu sorgulama, mesaj gönderme, alarm kurma ve matematiksel işlem şablon fonksiyonları.
- **Sentetik Veri Üretici (`Synthetic Dataset Generator`):** **Seed** (tohum / başlangıç) veriler ve yerel Ollama modelleri (`qwen2.5:7b` vb.) desteğiyle Türkçe veri seti çoğaltma (`data/train.jsonl`, `data/val.jsonl`).
- **LoRA İnce Ayar Desteği (`LoRA Fine-Tuning`):** Hugging Face `transformers`, `peft` ve `trl` kütüphaneleriyle açık ağırlıklı modelleri Türkçe araç şemasına uyarlama.
- **Ollama Dağıtımı (`Ollama Deployment`):** Eğitilen modeller için optimize edilmiş `Modelfile` şablonunu otomatik üretme.
- **Cihaz Üstü Ajan (`On-Device Agent`):** `cactus-needle` ile buluta bağımlı olmadan hızlı ve yerel **inference** (model çıkarımı / tahmin yürütme).
- **Otomatik Başarım Testi (`Benchmark Suite`):** Araç seçim doğruluğu (`tool accuracy`), parametre eşleşmesi (`argument match`) ve **latency** (yanıt gecikme süresi) ölçen, vektörel SVG grafik üreten yerleşik test aracı.
- **Sürüm Yönetimi (`Release Management`):** Tek komutla sürüm artırma (`version bump`), Git etiketi (`tag`) ve GitHub Actions üzerinden PyPI'ye otomatik **release** (sürüm yayınlama).

---

### 📦 Kurulum

Standart hafif kurulum (yalnızca yerel ajan ve CLI, ~20 MB):
```bash
pip install needle-tr
```

**Fine-tuning** (ince ayar) ve **Hugging Face** eğitim kütüphaneleriyle birlikte kurulum:
```bash
pip install "needle-tr[train]"
```

Geliştirici ortamı için [uv](https://github.com/astral-sh/uv) ile senkronize edin:
```bash
git clone https://github.com/emrekayik/needle-tr.git
cd needle-tr
uv sync --extra train
```

---

### 🛠️ CLI (Komut Satırı) Kullanımı

Tüm işlemler `needle-tr` komut arayüzü ile yönetilebilir:

#### 1. Sentetik Veri Üretimi (`Dataset Generation`)
```bash
# Yerel tohum verileri kullanarak temel veri seti üret
uv run needle-tr generate-data

# Yerel Ollama modeli kullanarak verileri çeşitlendir ve çoğalt
uv run needle-tr generate-data --use-ollama --ollama-model qwen2.5:7b
```

#### 2. İnce Ayar Eğitimi & Hugging Face Dağıtımı (`Fine-Tuning & Hub Push`)
```bash
# Dry-run (eğitimi başlatmadan veri setini ve GPU ayarlarını ön doğrulama)
uv run needle-tr train --dry-run

# LoRA eğitimini başlat
uv run needle-tr train --base-model Qwen/Qwen2.5-7B-Instruct --epochs 3

# Modeli eğitip otomatik olarak Hugging Face Hub'a yükle
uv run needle-tr train --base-model Qwen/Qwen2.5-7B-Instruct --push-to-hub --hub-model-id kullanici_adi/needle-tr-lora

# Eğitilmiş modeli veya veri setini dilediğiniz zaman Hugging Face'e aktarın
uv run needle-tr push-hub kullanici_adi/needle-tr-lora --model-dir models/needle_lora
uv run needle-tr push-hub kullanici_adi/needle-tr-dataset --dataset
```

#### 3. Ollama Ortamına Aktarım (`Ollama Export`)
```bash
# Modelfile dosyasını oluştur
uv run needle-tr export-ollama --base-model qwen2.5:7b --model-name needle-tr

# Ollama üzerinde modeli derle ve çalıştır
ollama create needle-tr -f Modelfile
ollama run needle-tr "25 * 4 hesapla"
```

#### 4. Başarım ve Hız Testi (`Benchmark`)
```bash
# Cihaz üstü (on-device) Cactus Needle ajanını test et ve grafiği güncelle
uv run needle-tr benchmark --suite

# Ollama modelini test et
uv run needle-tr benchmark --target ollama --ollama-model needle-tr
```

#### 5. Sürüm Yayınlama (`Release & Publish`)
```bash
# Tek komutla sürüm artırma, git etiketleme, paket derleme ve PyPI yayını:
./release.sh          # Patch artırma (örn: 0.1.5 -> 0.1.6)
./release.sh minor    # Minor artırma (örn: 0.1.5 -> 0.2.0)

# Veya Python CLI üzerinden:
uv run needle-tr release patch --push
```

---

### 🐍 Python Ajanı Kullanımı (`On-Device Inference`)

Python kodunuz içerisinden yerel **tool calling** (araç çağırma) ajanını çalıştırma:

```python
from needle_tr import agent

# Türkçe hava durumu sorgusu
cevap = agent.run("İstanbul'da hava durumu nasıl?")
print("Hava Durumu:", cevap["results"])

# Matematiksel hesaplama sorgusu
hesap = agent.run("25 * 4 hesapla")
print("Hesaplama Sonucu:", hesap["results"])
```

Hazır terminal ajanıyla denemek için:
```bash
uv run needle-agent
```

---

<!-- BENCHMARK_TR_START -->
### 📊 Benchmark (Başarım Testi) Sonuçları

`needle-tr benchmark` aracıyla otomatik üretilen son performans raporu:

![Needle-TR Benchmark Sonuçları](https://raw.githubusercontent.com/emrekayik/needle-tr/main/benchmark_results.svg)

| Metrik | Değer | Açıklama |
| :--- | :--- | :--- |
| **Test Edilen Hedef** | `Cactus Needle (On-Device Agent)` | Test edilen ortam / ajan modeli |
| **Toplam Test Sorgusu** | **210** | Değerlendirilen Türkçe komut sayısı |
| **Araç Seçim Doğruluğu** | **%75.7** (159/210) | Doğru fonksiyonu seçme başarısı |
| **Argüman Doğruluğu** | **%66.2** (139/210) | Parametreleri eksiksiz ayrıştırma oranı |
| **Geçerli Yanıt Oranı** | **%100.0** (209/210) | Hatasız parse edilen yapısal çıktı oranı |
| **Ortalama Gecikme (Latency)** | **197.8 ms** | İstek başına ortalama yanıt süresi |
| **Son Güncelleme** | `2026-09-28 19:14:06` | Testin çalıştırıldığı zaman |

| Araç (`Tool`) | Araç Doğruluğu | Argüman Eşleşmesi | Test Sayısı | Ortalama Gecikme |
| :--- | :---: | :---: | :---: | :---: |
| `calculate` | %90.9 | %67.3 | 55 | 126.6 ms |
| `get_weather` | %92.7 | %81.8 | 55 | 253.5 ms |
| `send_message` | %16.0 | %14.0 | 50 | 164.2 ms |
| `set_alarm` | %100.0 | %100.0 | 50 | 248.5 ms |
<!-- BENCHMARK_TR_END -->

---

### 📚 Detaylı Rehber

Model eğitimi mimarisi, veri formatları ve hiperparametreler için [FINETUNE_GUIDE.md](FINETUNE_GUIDE.md) dokümanını inceleyebilirsiniz.

---

### 🙏 Referanslar & Teşekkürler

- **[Cactus Compute - Needle](https://github.com/cactus-compute/needle):** Cihaz üzerinde (**on-device**) çalışan hafif ve verimli fonksiyon / araç çağırma (**tool calling**) temel kütüphanesi (`cactus-needle`).
