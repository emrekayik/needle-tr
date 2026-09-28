#!/usr/bin/env python3
"""Update Hugging Face Model Card for emrekayik/needle-tr-lora."""

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
    repo_id = "emrekayik/needle-tr-lora"

    model_card = """---
base_model:
  - Qwen/Qwen2.5-0.5B-Instruct
  - Cactus-Compute/needle3
library_name: peft
pipeline_tag: text-generation
tags:
  - base_model:adapter:Qwen/Qwen2.5-0.5B-Instruct
  - lora
  - transformers
  - tool-calling
  - function-calling
  - turkish
  - on-device
  - agent
license: apache-2.0
language:
  - tr
  - en
---

# 🇹🇷 Needle-TR LoRA: Türkçe Araç Çağırıcı Model

**Needle-TR LoRA**, Türkçe doğal dil komutlarını yüksek doğrulukla yerel fonksiyon/araç çağrılarına (**Tool Calling / Function Calling**) dönüştürmek üzere eğitilmiş hafif bir LoRA adaptörüdür. [Cactus Needle (needle3)](https://huggingface.co/Cactus-Compute/needle3) mimarisinden ilham alarak geliştirilmiştir.

---

## Model Details

### Model Description

- **Geliştirici (Developed by):** [emrekayik](https://huggingface.co/emrekayik)
- **Model Türü (Model type):** PEFT LoRA (Low-Rank Adaptation) for Causal Language Modeling
- **Temel Model (Base model):** [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- **İlham Alınan Mimari:** [Cactus-Compute/needle3](https://huggingface.co/Cactus-Compute/needle3)
- **Desteklenen Diller (Languages):** Türkçe (`tr`), İngilizce (`en`)
- **Lisans (License):** Apache-2.0
- **Kaynak Kod Deposu (Repository):** [https://github.com/emrekayik/needle-tr](https://github.com/emrekayik/needle-tr)

---

## Uses

### Doğrudan Kullanım (Direct Use)
- Türkçe kullanıcı komutlarını JSON tabanlı fonksiyon çağrılarına dönüştürme.
- Cihaz üzerinde (**on-device**) çalışan yerel yapay zeka asistanları ve akıllı ev / mobil ajan sistemleri.
- Desteklenen temel araçlar: `get_weather` (hava durumu), `send_message` (mesaj iletme), `set_alarm` (alarm kurma), `calculate` (matematiksel işlem).

### Kapsam Dışı Kullanım (Out-of-Scope Use)
- Fonksiyon çağırma amacı taşımayan genel sohbet ve serbest metin üretimi.
- Kötü niyetli, zararlı ya da yetkisiz sistem otomasyonları.

---

## Bias, Risks, and Limitations

- Model, önceden tanımlanmış araç şemalarıyla en yüksek performansı verecek şekilde optimize edilmiştir.
- Tanımlı araçların dışındaki bir talep geldiğinde araç uydurmak yerine boş çağrı (`[]`) üretmesi hedeflenmiştir.

---

## How to Get Started with the Model

Modeli `transformers` ve `peft` kütüphaneleriyle doğrudan kullanabilirsiniz:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "Qwen/Qwen2.5-0.5B-Instruct"
lora_model_id = "emrekayik/needle-tr-lora"

tokenizer = AutoTokenizer.from_pretrained(lora_model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float32,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, lora_model_id)

messages = [
    {
        "role": "system",
        "content": "Sen Türkçe komutları uygun araçlara dönüştüren bir yapay zeka asistanısın."
    },
    {
        "role": "user",
        "content": "İstanbul'da hava durumu nasıl?"
    }
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=128)

response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(response)
# Çıktı:
# ```json
# {"name": "get_weather", "arguments": {"city": "İstanbul"}}
# ```
```

---

## Training Details

### Eğitim Verisi (Training Data)
- **Veri Kümesi:** Çeşitli dil yapılarını (resmi, samimi, devrik, karmaşık argümanlar) içeren 229 adet sentetik ve denetimli Türkçe araç çağırma örneği.
- **Biçim:** ChatML (`system`, `user`, `assistant`) şablonu.

### Eğitim Hiperparametreleri (Training Hyperparameters)
- **LoRA Rank (r):** 16
- **LoRA Alpha:** 32
- **LoRA Dropout:** 0.05
- **Hedef Modüller (Target Modules):** `["q_proj", "k_proj", "v_proj", "o_proj"]`
- **Öğrenme Oranı (Learning Rate):** 2e-4
- **Optimizasyon:** AdamW, causal language modeling kaybı (cross-entropy)

---

## Evaluation

Model, 210 adetlik kapsamlı Türkçe benchmark test paketi üzerinde değerlendirilmiştir:

| Metrik | Sonuç |
| :--- | :--- |
| **Araç Seçim Doğruluğu (Tool Selection Accuracy)** | **%99.5** (209/210) |
| **Argüman Çıkarma Doğruluğu (Argument Accuracy)** | **%98.6** (207/210) |
| **Geçerli Format Oranı (Valid Format Rate)** | **%100.0** (210/210) |
| **Ortalama Cihaz İçi Gecikme (Latency)** | **~0.24 ms** |

---

## Technical Specifications & Frameworks

- **PEFT Versiyonu:** 0.21.0
- **Transformers Versiyonu:** >= 4.40.0
- **Python Kütüphanesi:** [needle-tr](https://pypi.org/project/needle-tr/) (`pip install needle-tr`)

---

## Citation

```bibtex
@misc{needle_tr_lora_2026,
  title        = {Needle-TR: Turkish On-Device Tool Calling LoRA Adapter},
  author       = {Emre Kayik},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\\url{https://huggingface.co/emrekayik/needle-tr-lora}}
}
```
"""

    print(f"Uploading updated README.md to {repo_id}...")
    api.upload_file(
        path_or_fileobj=model_card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="docs: update comprehensive Model Card for Needle-TR LoRA",
    )
    print(f"✓ Successfully updated README.md in https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    main()
