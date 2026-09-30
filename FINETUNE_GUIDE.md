# Needle-TR: Cactus Needle 3 Fine-Tuning Rehberi

Bu proje, [Cactus Compute Needle 3](https://cactuscompute.com/blog/finetuning-needle) modelini  
Türkçe araç çağırma (tool calling) görevine özel olarak fine-tune etmek için tasarlanmıştır.

---

## 🏗️ Mimari ve Çalışma Prensibi

```
1. Araçlar (tools.py)
   ├── get_weather, send_message, set_alarm, calculate
   └── Needle 3 uyumlu JSON şemaları
       │
       ▼
2. Veri Üretimi (generate-data)
   ├── Yerleşik Türkçe tohum (seed) örnekler  ← çevrimdışı, API gerekmez
   ├── Deterministic sentetik çoğaltıcı
   └── (Opsiyonel) Ollama ile ekstra varyasyon
       │
       ▼
3. Veri Kümeleri — Needle 3 formatı
   ├── data/train.jsonl  { query, tools, answers, reasoning }
   └── data/val.jsonl
       │
       ▼
4. Fine-Tuning  (needle finetune / trainer.py)
   ├── JAX / Flax tabanlı, Needle 3 base ağırlıklar üzerinde LoRA
   ├── Checkpoint: checkpoints/needle3.safetensors  (otomatik indirilir)
   └── Çıktı: models/adapter.safetensors
       │
       ▼
5. Derleme  (needle build / cmd_build)
   ├── LoRA adaptörü + taban model → .cact arşivi
   └── Çıktı: models/needle3-tr.cact   (~30–65 MB)
       │
       ▼
6. Çalıştırma
   ├── needle_tr.create_agent(weights="models/needle3-tr.cact")
   └── (Opsiyonel) Ollama Modelfile ile dağıtım
```

---

## 🚀 Adım Adım Kullanım

### 1. Veri Seti Oluşturma

Yerleşik Türkçe veri (API gerekmez):

```bash
uv run needle-tr generate-data --num-samples 120
```

Ollama üzerinden ek varyasyon üret (opsiyonel, Ollama çalışıyor olmalı):

```bash
uv run needle-tr generate-data --use-ollama --ollama-model qwen2.5:7b
```

### 2. Doğrulama (Dry-run)

Eğitim başlatmadan veri ve parametreleri kontrol et:

```bash
uv run needle-tr train --dry-run
```

### 3. Fine-Tuning

Needle 3 base ağırlıklar (121M parametre, ~29 MB) üzerinde LoRA eğitimi:

```bash
uv run needle-tr train --epochs 10
```

Hızlı bir deneme için tüm eğitim ayarlarını tek seçenekle küçültebilirsiniz:

```bash
uv run needle-tr train --quick --no-build
```

`--quick`; epoch sayısını 1'e, maksimum sekans uzunluğunu 512'ye, LoRA rank'ini
8'e ve derleme katmanı sayısını 2'ye indirir. Açıkça verdiğiniz değerler bu
profilde önceliklidir. Gerçek model çıktısı gerektiğinde `--no-build` seçeneğini
kaldırın.

Tüm parametreler:

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `--epochs` | 10 | Epoch sayısı (küçük veri için 10–30 önerilir) |
| `--batch-size` | 16 | Batch boyutu |
| `--lr` | 1e-4 | Öğrenme oranı |
| `--lora-rank` | 16 | LoRA rank |
| `--lora-alpha` | 32.0 | LoRA alpha |
| `--max-len` | 1024 | Max token uzunluğu |
| `--layers` | 20 | Hedef katman sayısı (2–20) |
| `--checkpoint` | None | Özel base checkpoint yolu (None = otomatik) |
| `--output-dir` | models | Model kayıt dizini |

### 4. Model Derleme (.cact)

Fine-tuning sonrası **otomatik olarak** çalışır. Manuel çalıştırmak için:

```bash
uv run needle-tr build --lora models/adapter.safetensors --out models/needle3-tr.cact
```

### 5. Modeli Test Etme

```bash
uv run needle-tr run "Lagos'ta hava nasıl?"
uv run needle-tr run "Ahmet'e 'Toplantı başladı' mesajı gönder"
uv run needle-tr run "Saat 07:30'a alarm kur"
```

Python'dan:

```python
from needle_tr import create_agent

agent = create_agent(weights="models/needle3-tr.cact")
print(agent.run("25 * 4 hesapla")["results"])
```

### 6. Benchmark

```bash
uv run needle-tr benchmark --target agent
```

---

## 🦙 Ollama Entegrasyonu (Opsiyonel)

Needle 3 modeli Ollama gerektirmez. Ancak Ollama tabanlı dağıtım istiyorsanız:

```bash
# Modelfile oluştur
uv run needle-tr export-ollama --base-model qwen2.5:7b --model-name needle-tr

# Ollama'ya kaydet
ollama create needle-tr -f Modelfile
ollama run needle-tr "25 * 4 hesapla"
```

---

## 📊 Veri Formatı (Needle 3)

Her satır tek bir JSON nesnesi:

```json
{
  "query": "İstanbul için hava durumu nedir?",
  "tools": [{ "name": "get_weather", "description": "...", "parameters": {...} }],
  "answers": [{ "name": "get_weather", "arguments": { "city": "İstanbul" } }],
  "reasoning": "'İstanbul' -> city"
}
```

**Önemli kurallar:**
- `reasoning`: Sorgu içindeki değer → argüman eşlemesini açıklar (parametre grounding)
- Negatif örnekler (`answers: []`): ~1/8 oranında olmalı (model her şeye araç çağırmasın)
- `tools`: Her satıra tam şema eklenir

---

## 🔧 Platform Fine-Tuning (Cactus GPU)

Cactus Platform'da GPU ile eğitmek için (API anahtarı gerekir):

```bash
uv run needle-tr platform finetune data/train.jsonl
```

Daha fazla bilgi: [cactuscompute.com](https://cactuscompute.com)
