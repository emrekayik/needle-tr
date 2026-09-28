"""Needle-TR Fine-Tuning ve Ollama Yönetim Arayüzü (CLI)."""

import argparse
import sys
from pathlib import Path

from .dataset_generator import build_dataset, check_ollama_available
from .export_ollama import write_modelfile
from .trainer import TrainingConfig, run_training


def cmd_generate_data(args):
    """Veri seti üretimi komutu."""
    use_ollama = args.use_ollama
    if use_ollama:
        if not check_ollama_available(args.ollama_host):
            print(
                f"[BİLGİ] Ollama sunucusu ({args.ollama_host}) şu an çalışmıyor.\n"
                "Ollama'yı çalıştırmadan devam ediliyor; yerel tohum (seed) veri kümesi kullanılacak.\n"
                "(Ollama hazır olduğunda 'ollama serve' veya Ollama uygulamasını başlatıp bu komutu tekrar çalıştırabilirsiniz.)"
            )
            use_ollama = False

    build_dataset(
        output_dir=args.output_dir,
        use_ollama=use_ollama,
        ollama_model=args.ollama_model,
        ollama_host=args.ollama_host,
    )


def cmd_train(args):
    """Fine-tuning başlatma komutu."""
    cfg = TrainingConfig(
        base_model_name=args.base_model,
        epochs=args.epochs,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )
    run_training(cfg, dry_run=args.dry_run)


def cmd_export_ollama(args):
    """Ollama Modelfile oluşturma komutu."""
    target_path = Path(args.output)
    write_modelfile(
        output_path=target_path,
        base_model=args.base_model,
        custom_gguf_path=args.gguf_path,
    )
    print("\n[Sonraki Adım]:")
    print(f"Ollama'yı başlattıktan sonra modeli kaydetmek için terminalde şunu çalıştırabilirsiniz:")
    print(f"  ollama create {args.model_name} -f {target_path}")


def main():
    parser = argparse.ArgumentParser(
        prog="needle-tr",
        description="Needle-TR: Türkçe Araç Çağırıcı Fine-Tuning ve Ollama Sistemi",
    )
    subparsers = parser.add_subparsers(dest="command", help="Alt komutlar")

    # generate-data
    p_gen = subparsers.add_parser("generate-data", help="Sentetik Türkçe veri seti üret")
    p_gen.add_argument("--use-ollama", action="store_true", help="Ollama API'sini kullanarak veri çoğalt")
    p_gen.add_argument("--ollama-model", default="qwen2.5:7b", help="Kullanılacak Ollama modeli")
    p_gen.add_argument("--ollama-host", default="http://localhost:11434", help="Ollama API adresi")
    p_gen.add_argument("--output-dir", default="data", help="Verilerin kaydedileceği dizin")
    p_gen.set_defaults(func=cmd_generate_data)

    # train
    p_train = subparsers.add_parser("train", help="LoRA Fine-tuning başlat")
    p_train.add_argument("--dry-run", action="store_true", help="Eğitimi başlatmadan veri ve ayarları doğrula")
    p_train.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct", help="Temel model adı")
    p_train.add_argument("--epochs", type=int, default=3, help="Epoch sayısı")
    p_train.add_argument("--data-dir", default="data", help="Eğitim verisi dizini")
    p_train.add_argument("--output-dir", default="models/needle_lora", help="Model kayıt dizini")
    p_train.set_defaults(func=cmd_train)

    # export-ollama
    p_export = subparsers.add_parser("export-ollama", help="Ollama için Modelfile hazırla")
    p_export.add_argument("--output", default="Modelfile", help="Modelfile dosya yolu")
    p_export.add_argument("--base-model", default="qwen2.5:7b", help="Ollama temel modeli")
    p_export.add_argument("--model-name", default="needle-tr", help="Oluşturulacak model adı")
    p_export.add_argument("--gguf-path", default=None, help="Özel GGUF model dosya yolu (opsiyonel)")
    p_export.set_defaults(func=cmd_export_ollama)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
