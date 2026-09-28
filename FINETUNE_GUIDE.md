# Needleeee Türkçe Araç Çağırıcı Fine-Tuning ve Ollama Rehberi

Bu sistem, Ollama'yı öğretmen model (sentetik veri üretici) ve hedef çalışma ortamı (Modelfile / GGUF dağıtımı) olarak kullanan uçtan uca bir fine-tuning ve adaptasyon mimarisidir.

---

## 🏗️ Mimari ve Çalışma Prensibi

```
1. Araçlar (tools.py)
   ├── get_weather, send_message, set_alarm, calculate
   └── JSON Şemaları (OpenAI/Ollama uyumlu)
       │
       ▼
2. Sentetik Veri Üretimi (dataset_generator.py)
   ├── Tohum (Seed) Türkçe veri kümesi
   └── Ollama REST API (llama3 / qwen2.5) üzerinden Türkçe veri zenginleştirme
       │
       ▼
3. Veri Kümeleri (data/train.jsonl & data/val.jsonl)
   └── Standart ChatML ve Tool Calling formatı
       │
       ▼
4. Fine-Tuning (trainer.py)
   ├── LoRA (Low-Rank Adaptation) eğitimi
   └── Çıktı: models/needle_lora
       │
       ▼
5. Ollama Dağıtımı (Modelfile & export_ollama.py)
   └── `ollama create needle-turkish -f Modelfile`
```

---

## 🚀 Komut Satırı Kullanımı (CLI)

Tüm adımlar `needleeee-ft` komutu üzerinden yönetilebilir:

### 1. Sentetik Türkçe Veri Seti Oluşturma
Yerel tohum (seed) verilerini kullanarak `data/train.jsonl` ve `data/val.jsonl` oluşturur:
```bash
uv run needleeee-ft generate-data
```

Ollama sunucunuz aktif olduğunda, yerel `qwen2.5:7b` modelinizden yararlanarak yüzlerce yeni Türkçe varyasyon üretmek için:
```bash
uv run needleeee-ft generate-data --use-ollama --ollama-model qwen2.5:7b
```

### 2. Fine-Tuning Yapılandırması ve Eğitimi
Verileri ve eğitim parametrelerini doğrulamak için (Dry-run):
```bash
uv run needleeee-ft train --dry-run
```

Tam eğitim başlatmak istediğinizde gerekli bağımlılıkları ekleyin:
```bash
uv add torch transformers peft trl datasets
uv run needleeee-ft train --base-model Qwen/Qwen2.5-7B-Instruct --epochs 3
```

### 3. Ollama Modelfile Oluşturma / Güncelleme
Mevcut araç şemalarını ve Türkçe sistem talimatlarını içeren `Modelfile` dosyasını `qwen2.5:7b` tabanlı üretmek için:
```bash
uv run needleeee-ft export-ollama --base-model qwen2.5:7b --model-name needle-turkish
```


---

## 🦙 Ollama Üzerinde Modeli Kaydetme (Gerektiğinde)

Ollama uygulamanızı başlattıktan sonra, oluşturulan Modelfile ile özel modelinizi tek komutla oluşturabilirsiniz:
```bash
ollama create needle-turkish -f Modelfile
ollama run needle-turkish "25 * 4 hesapla"
```
