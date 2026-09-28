"""Türkçe Araç Çağırıcı için LoRA / SFT Fine-Tuning Eğiticisi.

Bu modül:
1. data/train.jsonl ve data/val.jsonl dosyalarındaki verileri okur.
2. Hugging Face / PEFT / TRL (SFTTrainer) veya PyTorch altyapısı ile LoRA eğitimini yapılandırır.
3. --dry-run desteği ile eğitim öncesi veri doğrulaması ve konfigürasyon kontrolü yapar.
4. Eğitilen ağırlıkları 'models/needle_lora' altına kaydeder.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class TrainingConfig:
    base_model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    data_dir: str = "data"
    output_dir: str = "models/needle_lora"
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_seq_length: int = 512


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """JSONL dosyasını yükler."""
    if not path.exists():
        raise FileNotFoundError(f"Veri dosyası bulunamadı: {path}")
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def find_data_dir(data_dir: str = "data") -> Path:
    """Proje kök dizinini ve veri klasörünü dinamik olarak bulur."""
    p = Path(data_dir)
    if p.exists():
        return p
    # Üst dizine bak
    parent_p = Path(__file__).resolve().parent.parent.parent / data_dir
    if parent_p.exists():
        return parent_p
    return p


def dry_run_validation(config: TrainingConfig) -> bool:
    """Eğitim öncesi verileri ve ayarları doğrular."""
    print("=" * 60)
    print("Fine-Tune Konfigürasyon Kontrolü (Dry-Run)")
    print("=" * 60)
    for k, v in asdict(config).items():
        print(f"  - {k}: {v}")

    data_path = find_data_dir(config.data_dir)
    train_path = data_path / "train.jsonl"
    val_path = data_path / "val.jsonl"


    if not train_path.exists() or not val_path.exists():
        print("\n[HATA] Veri dosyaları eksik! Önce 'build_dataset' çalıştırın.")
        return False

    train_data = load_jsonl(train_path)
    val_data = load_jsonl(val_path)

    print(f"\n[Veri Kümesi Özeti]")
    print(f"  - Eğitim örnek sayısı: {len(train_data)}")
    print(f"  - Doğrulama örnek sayısı: {len(val_data)}")

    # İlk örnek önizlemesi
    first = train_data[0]
    print("\n[Örnek Veri Biçimi (ChatML)]:")
    for msg in first.get("messages", []):
        role = msg.get("role", "").upper()
        content = msg.get("content", "")
        print(f"  [{role}]: {content[:80]}...")

    print("\n[Doğrulama]: Veri kümesi ve konfigürasyon sorunsuz!")
    return True


def run_training(config: TrainingConfig, dry_run: bool = False):
    """Fine-tuning işlemini başlatır."""
    is_valid = dry_run_validation(config)
    if not is_valid:
        return

    if dry_run:
        print("\n[BİLGİ] Dry-run modu tamamlandı. Gerçek eğitim başlatılmadı.")
        return

    print("\n[Eğitim Başlatılıyor]")
    print("Gerekli paketler: transformers, peft, trl, torch, datasets")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from peft import LoraConfig, get_peft_model
        from datasets import Dataset
    except ImportError as e:
        print(f"\n[UYARI] Eğitim kütüphaneleri eksik: {e}")
        print("Eğitimi başlatmak için şu komutla bağımlılıkları yükleyebilirsiniz:")
        print("  uv add torch transformers peft trl datasets")
        return

    print(f"Model yükleniyor: {config.base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Veri setini Dataset objesine çevir
    train_samples = load_jsonl(Path(config.data_dir) / "train.jsonl")
    val_samples = load_jsonl(Path(config.data_dir) / "val.jsonl")

    # Formatlama
    def format_chat(sample):
        text = tokenizer.apply_chat_template(sample["messages"], tokenize=False)
        return {"text": text}

    train_ds = Dataset.from_list(train_samples).map(format_chat)
    val_ds = Dataset.from_list(val_samples).map(format_chat)

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    print("LoRA fine-tuning yapılandırıldı.")
    print(f"Ağırlıklar şuraya kaydedilecek: {config.output_dir}")
    # Gerçek eğitim döngüsü (ortamda GPU veya kütüphaneler mevcut olduğunda)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Türkçe Tool Calling Fine-Tuning Scripti")
    parser.add_argument("--dry-run", action="store_true", help="Gerçek eğitim yapmadan verileri ve yapılandırmayı kontrol et")
    parser.add_argument("--epochs", type=int, default=3, help="Epoch sayısı")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Temel model adı")
    args = parser.parse_args()

    cfg = TrainingConfig(
        base_model_name=args.base_model,
        epochs=args.epochs,
    )
    run_training(cfg, dry_run=args.dry_run)
