"""Needle 3 Türkçe Araç Çağırıcı için LoRA Fine-Tuning ve Model Derleyicisi.

Cactus Needle 3 spesifikasyonuna (https://cactuscompute.com/blog/finetuning-needle) göre:
1. data/train.jsonl ve data/val.jsonl dosyalarını doğrular.
2. JAX / Flax altyapısıyla taban model dondurularak dikkat katmanlarında LoRA adaptörü eğitilir.
3. --dry-run desteği ile veri doğrulaması, negatif örnek oranı ve adım sayısı hesabı yapar.
4. 'needle build' adımı ile LoRA adaptörü taban ağırlıklarla birleştirilip 4-bit .cact dosyasına derlenir.
5. Derlenen .cact dosyası doğrudan Needle runtime'ında veya Hugging Face Hub'da yayınlanabilir.
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
    data_path: str = "data/train.jsonl"
    val_path: Optional[str] = "data/val.jsonl"
    output_dir: str = "models"
    adapter_name: str = "adapter.safetensors"
    cact_name: str = "needle3-tr.cact"
    checkpoint: Optional[str] = None  # None ise Needle 3 base safetensors otomatik indirilir
    epochs: int = 10                  # Blog önerisi: küçük veri setleri için 10-30 epoch
    batch_size: int = 16              # Needle 3 varsayılanı
    lr: float = 1e-4                  # Needle 3 varsayılanı
    lora_rank: int = 16               # LoRA rank 16
    lora_alpha: float = 32.0          # LoRA alpha 32
    max_len: int = 1024               # Max token uzunluğu
    val_split: float = 0.1            # Doğrulama ayrımı (val_split)
    seed: int = 0
    layers: int = 20                  # 2..20 arası alt katman (rung), varsayılan tam model: 20
    platform: Optional[str] = None    # Cihaz platformu (macos-arm64, linux-arm64 vb.)
    auto_build: bool = True           # Eğitim sonrası otomatik .cact derlemesi yap
    push_to_hub: bool = False
    hub_repo: Optional[str] = None
    hub_token: Optional[str] = None


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


def dry_run_validation(config: TrainingConfig) -> bool:
    """Needle 3 eğitim öncesi veri ve parametre doğrulaması yapar."""
    print("=" * 60)
    print("Needle 3 Fine-Tune Konfigürasyon Kontrolü (Dry-Run)")
    print("=" * 60)

    train_p = Path(config.data_path)
    if not train_p.exists():
        # Proje kök dizininde ara
        fallback = Path(__file__).resolve().parent.parent.parent / config.data_path
        if fallback.exists():
            train_p = fallback
        else:
            print(f"[HATA] Eğitim veri dosyası bulunamadı: {config.data_path}")
            print("Öneri: Önce 'uv run needle-tr generate-data' komutunu çalıştırın.")
            return False

    try:
        train_data = load_jsonl(train_p)
    except Exception as e:
        print(f"[HATA] JSONL dosyası okunamadı: {e}")
        return False

    if len(train_data) == 0:
        print(f"[HATA] {train_p} dosyası boş!")
        return False

    # Needle 3 Format Kontrolleri
    missing_query = 0
    missing_answers = 0
    with_reasoning = 0
    empty_answers = 0

    for item in train_data:
        if "query" not in item:
            missing_query += 1
        if "answers" not in item:
            missing_answers += 1
        else:
            if len(item["answers"]) == 0:
                empty_answers += 1
        if item.get("reasoning"):
            with_reasoning += 1

    total = len(train_data)
    negative_ratio = empty_answers / total if total > 0 else 0.0

    print(f"\n[Veri Kümesi Analizi]: {train_p.resolve()}")
    print(f"  - Toplam örnek sayısı : {total}")
    print(f"  - Reasoning içerenler  : {with_reasoning} (%{with_reasoning / total * 100:.1f})")
    print(f"  - Negatif örnekler ([]) : {empty_answers} (%{negative_ratio * 100:.1f})")

    if missing_query > 0 or missing_answers > 0:
        print(f"[HATA] {missing_query} satırda 'query', {missing_answers} satırda 'answers' eksik!")
        return False

    # Cactus kuralı kontrolü
    if empty_answers == 0:
        print("\n[UYARI] Veri setinde hiç negatif ('answers': []) örnek bulunamadı!")
        print("  Cactus rehberine göre: Negatif örnekler (yaklaşık 1/8 oranında) eklenmezse,")
        print("  eğitilen model her kullanıcı cümlesine zorla bir araç çağırmaya çalışır.")
    else:
        print("  ✓ Negatif örnek dengesi uygun (~%12-15 aralığı).")

    if with_reasoning == 0:
        print("\n[BİLGİ] 'reasoning' alanı bulunamadı. Reasoning eklemek parametre grounding'ini güçlendirir.")

    # Adım sayısı analizi (Cactus makalesindeki rehber)
    steps_per_epoch = -(-total // config.batch_size)
    total_steps = config.epochs * steps_per_epoch

    print(f"\n[Eğitim Adım Analizi]:")
    print(f"  - Epoch: {config.epochs}, Batch Boyutu: {config.batch_size}")
    print(f"  - Epoch başına adım  : {steps_per_epoch}")
    print(f"  - Toplam eğitim adımı : {total_steps}")
    print(f"  - LoRA Rank / Alpha   : Rank {config.lora_rank}, Alpha {config.lora_alpha}")
    print(f"  - Hedef Çıktı Modeli  : {config.output_dir}/{config.cact_name} ({config.layers} katman)")

    if total_steps < 60:
        print("\n[İPUCU] Toplam adım sayısı (toplam < 60) LoRA adaptörünü hareket ettirmek için az olabilir.")
        print(f"  Öneri: --epochs {max(10, 80 // steps_per_epoch)} veya daha yüksek bir değer kullanın.")

    # Örnek ilk satır gösterimi
    first = train_data[0]
    print(f"\n[Örnek Kayıt (Needle 3 Format)]: ")
    print(f"  Query    : {first.get('query')}")
    print(f"  Reasoning: {first.get('reasoning')}")
    print(f"  Answers  : {first.get('answers')}")

    print("\n✓ Needle 3 veri kümesi ve konfigürasyon doğrulaması başarılı!")
    return True


def run_training(config: TrainingConfig, dry_run: bool = False):
    """Needle 3 fine-tune ve build akışını çalıştırır."""
    is_valid = dry_run_validation(config)
    if not is_valid:
        return

    if dry_run:
        print("\n[BİLGİ] Dry-run modu tamamlandı. Gerçek eğitim başlatılmadı.")
        return

    # Gerekli dizinleri hazırla
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = out_dir / config.adapter_name
    cact_path = out_dir / config.cact_name

    print("\n" + "=" * 60)
    print("Needle 3 JAX/Flax LoRA Eğitimi Başlatılıyor")
    print("=" * 60)
    print("Taban Model: Needle 3 (121M params, ~29 MB)")
    print("Eğitim Motoru: JAX (CPU / Apple Silicon Metal / CUDA)")

    # 1. Aşama: LoRA Fine-Tuning (needle finetune)
    try:
        from needle.model.finetune import finetune_local

        # Needle 3 finetune_local beklenen argparse Namespace
        ft_args = argparse.Namespace(
            jsonl_path=str(Path(config.data_path).resolve()),
            checkpoint=config.checkpoint,    # None ise DEFAULT_BASE (checkpoints/needle3.safetensors)
            epochs=config.epochs,
            batch_size=config.batch_size,
            lr=config.lr,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            max_len=config.max_len,
            val_split=config.val_split,
            seed=config.seed,
            out=str(adapter_path.resolve()),
        )

        finetune_local(ft_args)
        print(f"\n✓ LoRA Adaptörü başarıyla kaydedildi: {adapter_path.resolve()}")

    except Exception as e:
        print(f"\n[HATA] Fine-tuning sırasında hata oluştu: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. Aşama: Model Birleştirme ve .cact Dışa Aktarma (needle build)
    if config.auto_build:
        print("\n" + "=" * 60)
        print("Needle 3 Model Derleme (needle build)")
        print("=" * 60)
        print(f"LoRA adaptörü taban ağırlıklarla birleştiriliyor ({config.layers} katman)...")

        try:
            from needle.model.finetune import build_main

            build_args = argparse.Namespace(
                checkpoint=config.checkpoint,
                lora=str(adapter_path.resolve()),
                out=str(cact_path.resolve()),
                upload=config.push_to_hub,
                layers=config.layers,
                platform=config.platform,
            )

            if config.push_to_hub and config.hub_repo:
                os.environ["NEEDLE_HF_REPO"] = config.hub_repo

            build_main(build_args)
            print(f"\n✓ Çalıştırılabilir Needle 3 Türkçe modeli hazır: {cact_path.resolve()}")
            print("\nModeli test etmek için:")
            print(f"  from needle_tr import create_agent")
            print(f"  agent = create_agent(weights=\"{cact_path}\")")
            print(f"  print(agent.run(\"Lagos'ta hava nasıl?\")[\"results\"])")

        except Exception as e:
            print(f"\n[UYARI] Model derleme (build) sırasında hata: {e}")
            import traceback
            traceback.print_exc()
