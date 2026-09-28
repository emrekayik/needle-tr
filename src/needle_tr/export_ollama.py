"""Ollama Modelfile ve Model Dışa Aktarma Yardımcısı.

Bu modül:
1. Araç şemalarını ve Türkçe sistem istemini içeren hazır bir 'Modelfile' oluşturur.
2. Eğitilmiş LoRA / GGUF modelini Ollama'ya aktarmak için gerekli konfigürasyonu hazırlar.
3. 'ollama create' komutunun hazır şablonunu sunar (Ollama'yı asla otomatik olarak çalıştırmaz).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

try:
    from .tools import get_tools_schema
except ImportError:
    from tools import get_tools_schema


DEFAULT_MODELFILE_PATH = Path("Modelfile")

SYSTEM_PROMPT_TEMPLATE = """Sen Türkçe dilinde araç (tool) çağıran uzman bir yapay zeka asistanısın.
Görevin, kullanıcının Türkçe isteklerini analiz etmek ve aşağıdaki araçlardan uygun olanını seçip JSON olarak çağırmaktır.

Kullanabileceğin Araçlar:
{tools_json}

Kurallar:
1. Kullanıcı bir işlem istediğinde (örneğin hava durumu, mesaj gönderme, alarm kurma veya matematiksel hesaplama), kesinlikle sadece aşağıdaki JSON biçiminde yanıt ver:
```json
{{
  "name": "arac_adi",
  "arguments": {{
    "parametre1": "değer1"
  }}
}}
```
2. Eğer hiçbir araç uygun değilse veya doğrudan sohbet ise kibarca Türkçe cevap ver.
3. Parametreleri kullanıcının Türkçe cümlesinden tam ve doğru olarak çıkar.
"""


def generate_modelfile_content(
    base_model: str = "qwen2.5:7b",
    custom_gguf_path: Optional[str] = None,
) -> str:
    """Ollama Modelfile içeriğini üretir."""
    tools_schema = get_tools_schema()
    tools_formatted = json.dumps(tools_schema, ensure_ascii=False, indent=2)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(tools_json=tools_formatted)

    from_target = custom_gguf_path if custom_gguf_path else base_model

    modelfile = f"""# ==============================================================================
# Needleeee Türkçe Araç Çağırıcı (Ollama Modelfile)
# Oluşturulduğu model: {from_target}
# ==============================================================================

FROM {from_target}

# Deterministik ve doğru JSON araç çağrıları için düşük sıcaklık
PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"

# Sistem İstemleri ve Araç Şemaları
SYSTEM \"\"\"{system_prompt.strip()}\"\"\"

# Mesaj şablonu (ChatML formatı)
TEMPLATE \"\"\"{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
{{{{ .Response }}}}<|im_end|>
\"\"\"
"""
    return modelfile


def write_modelfile(
    output_path: Path = DEFAULT_MODELFILE_PATH,
    base_model: str = "qwen2.5:7b",
    custom_gguf_path: Optional[str] = None,
) -> Path:

    """Modelfile dosyasını diske yazar."""
    content = generate_modelfile_content(base_model, custom_gguf_path)
    output_path.write_text(content, encoding="utf-8")
    print(f"[Modelfile] Başarıyla oluşturuldu: {output_path.resolve()}")
    return output_path
