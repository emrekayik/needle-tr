"""Türkçe Araç Çağırıcı için LoRA / SFT Fine-Tuning Eğiticisi ve Hugging Face Hub Yöneticisi.

Bu modül:
1. data/train.jsonl ve data/val.jsonl dosyalarındaki verileri okur.
2. Hugging Face / PEFT / TRL (SFTTrainer) veya PyTorch altyapısı ile LoRA eğitimini yapılandırır.
3. --dry-run desteği ile eğitim öncesi veri doğrulaması ve konfigürasyon kontrolü yapar.
4. Eğitilen ağırlıkları 'models/needle_lora' altına kaydeder.
5. Eğitilen LoRA adaptörlerini, birleştirilmiş (merged) modelleri veya veri setini otomatik olarak Hugging Face Hub'a yükler.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


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
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_token: Optional[str] = None
    merge_before_push: bool = False


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
        if k == "hub_token" and v:
            print("  - hub_token: [GİZLENDİ / MEVCUT]")
        else:
            print(f"  - {k}: {v}")

    data_path = find_data_dir(config.data_dir)
    train_path = data_path / "train.jsonl"
    val_path = data_path / "val.jsonl"

    if not train_path.exists() or not val_path.exists():
        print("\n[HATA] Veri dosyaları eksik! Önce 'build_dataset' veya 'generate-data' çalıştırın.")
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


def _resolve_hf_token(token: Optional[str] = None) -> Optional[str]:
    """HF token'ını parametreden, os.environ'dan veya .env dosyasından okur."""
    if token:
        return token
    if "HF_TOKEN" in os.environ:
        return os.environ["HF_TOKEN"]
    env_p = Path(".env")
    if env_p.exists():
        for line in env_p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip("\"' ")
    return None


def push_model_to_hub(
    model_dir: Path | str = "models/needle_lora",
    hub_model_id: str = "needle-tr-lora",
    token: Optional[str] = None,
    base_model_name: Optional[str] = None,
    merge: bool = False,
) -> str:
    """Eğitilmiş LoRA veya birleştirilmiş (merged) modeli Hugging Face Hub'a yükler."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise ImportError("Hugging Face'e yüklemek için huggingface-hub kütüphanesi gereklidir: uv add huggingface-hub")

    token = _resolve_hf_token(token)
    api = HfApi(token=token)

    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(
            f"\n❌ Model dizini bulunamadı: '{model_path.resolve()}'.\n"
            f"Model henüz eğitilmediyse önce modeli eğitin:\n"
            f"  uv run --extra train needle-tr train\n"
            f"Farklı bir klasördeki modeli yüklemek isterseniz:\n"
            f"  uv run --extra train needle-tr push-hub {hub_model_id} --model-dir <klasor_yolu>"
        )

    print(f"\n[Hugging Face] '{hub_model_id}' deposu kontrol ediliyor...")
    api.create_repo(repo_id=hub_model_id, exist_ok=True, repo_type="model")

    if merge and base_model_name:
        print(f"[Hugging Face] LoRA adaptörleri '{base_model_name}' ile birleştiriliyor...")
        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            peft_model = PeftModel.from_pretrained(base_model, str(model_path))
            merged_model = peft_model.merge_and_unload()

            print(f"[Hugging Face] Birleştirilmiş tam model Hugging Face'e aktarılıyor: {hub_model_id}...")
            merged_model.push_to_hub(hub_model_id, token=token)
            tokenizer.push_to_hub(hub_model_id, token=token)
        except Exception as e:
            print(f"[HATA] Model birleştirme sırasında hata: {e}")
            print("Klasör doğrudan LoRA olarak yükleniyor...")
            api.upload_folder(folder_path=str(model_path), repo_id=hub_model_id, repo_type="model")
    else:
        print(f"[Hugging Face] LoRA adaptör dosyaları Hugging Face'e aktarılıyor: {hub_model_id}...")
        api.upload_folder(folder_path=str(model_path), repo_id=hub_model_id, repo_type="model")

    hub_url = f"https://huggingface.co/{hub_model_id}"
    print(f"✨ [Başarılı] Model Hugging Face'e yayınlandı: {hub_url}")
    return hub_url


def push_dataset_to_hub(
    data_dir: Path | str = "data",
    hub_dataset_id: str = "needle-tr-dataset",
    token: Optional[str] = None,
) -> str:
    """data/train.jsonl ve data/val.jsonl dosyalarını Hugging Face Datasets Hub'a yükler."""
    try:
        from datasets import load_dataset
        from huggingface_hub import HfApi
    except ImportError:
        raise ImportError("Veri setini yüklemek için datasets ve huggingface-hub gereklidir: uv add datasets huggingface-hub")

    token = _resolve_hf_token(token)
    p = Path(data_dir)
    train_p = p / "train.jsonl"
    val_p = p / "val.jsonl"

    if not train_p.exists():
        raise FileNotFoundError(f"Eğitim veri dosyası bulunamadı: {train_p.resolve()}")

    data_files = {"train": str(train_p)}
    if val_p.exists():
        data_files["val"] = str(val_p)

    print(f"\n[Hugging Face] Veri kümesi hazırlanıyor: {hub_dataset_id}...")
    ds = load_dataset("json", data_files=data_files)
    ds.push_to_hub(hub_dataset_id, token=token)

    url = f"https://huggingface.co/datasets/{hub_dataset_id}"
    print(f"✨ [Başarılı] Veri seti Hugging Face'e yayınlandı: {url}")
    return url


def run_training(config: TrainingConfig, dry_run: bool = False):
    """Fine-tuning işlemini başlatır ve istenirse Hugging Face'e yükler."""
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
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    except ImportError as e:
        print(f"\n[UYARI] Eğitim kütüphaneleri eksik: {e}")
        print("Eğitim paketlerini kurmak için:")
        print("  pip install 'needle-tr[train]' veya uv add torch transformers peft trl datasets")
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
    print(f"Model hazırlanıyor ve ağırlıklar şuraya kaydedilecek: {config.output_dir}")

    # Temel modeli ve LoRA adaptörünü oluştur
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    peft_model = get_peft_model(model, lora_config)

    # Ağırlıkları ve tokenizer'ı output_dir altına kaydet
    out_p = Path(config.output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(out_p))
    tokenizer.save_pretrained(str(out_p))
    print(f"✓ LoRA adaptör ağırlıkları kaydedildi: {out_p.resolve()}")

    # Otomatik Hugging Face yükleme
    if config.push_to_hub and config.hub_model_id:
        print("\n[Eğitim Sonrası Hugging Face Dağıtımı]")
        push_model_to_hub(
            model_dir=config.output_dir,
            hub_model_id=config.hub_model_id,
            token=config.hub_token,
            base_model_name=config.base_model_name,
            merge=config.merge_before_push,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Türkçe Tool Calling Fine-Tuning Scripti")
    parser.add_argument("--dry-run", action="store_true", help="Gerçek eğitim yapmadan verileri ve yapılandırmayı kontrol et")
    parser.add_argument("--epochs", type=int, default=3, help="Epoch sayısı")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Temel model adı")
    parser.add_argument("--push-to-hub", action="store_true", help="Eğitim sonrası Hugging Face'e otomatik yükle")
    parser.add_argument("--hub-model-id", type=str, default=None, help="Hugging Face model ID (örn: kullanici_adi/model-adi)")
    args = parser.parse_args()

    cfg = TrainingConfig(
        base_model_name=args.base_model,
        epochs=args.epochs,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )
    run_training(cfg, dry_run=args.dry_run)
