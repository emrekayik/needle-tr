"""Needle 3 Türkçe Araç Çağırıcı ve Fine-Tuning Yönetim Arayüzü (CLI)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .benchmark import generate_benchmark_svg, run_benchmark, update_readme_benchmark
from .dataset_generator import build_dataset
from .export_ollama import write_modelfile
from .release import run_release
from .trainer import TrainingConfig, dry_run_validation, run_training


def cmd_generate_data(args):
    """Cactus Needle 3 formatında Türkçe veri seti üretir."""
    build_dataset(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        use_ollama=getattr(args, "use_ollama", False),
        ollama_model=getattr(args, "ollama_model", "qwen2.5:7b"),
        ollama_host=getattr(args, "ollama_host", "http://localhost:11434"),
    )


def cmd_train(args):
    """Needle 3 LoRA Fine-Tuning ve model derleme akışını çalıştırır."""
    if args.quick:
        # Fast smoke-test profile; explicit CLI values still take precedence.
        args.epochs = 1 if args.epochs == 10 else args.epochs
        args.batch_size = 32 if args.batch_size == 16 else args.batch_size
        args.max_len = 512 if args.max_len == 1024 else args.max_len
        args.lora_rank = 8 if args.lora_rank == 16 else args.lora_rank
        args.layers = 2 if args.layers == 20 else args.layers

    cfg = TrainingConfig(
        data_path=args.data,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        max_len=args.max_len,
        val_split=args.val_split,
        seed=args.seed,
        layers=args.layers,
        platform=args.platform,
        auto_build=not args.no_build,
        push_to_hub=args.push_to_hub,
        hub_repo=args.hub_repo,
    )
    run_training(cfg, dry_run=args.dry_run)


def cmd_build(args):
    """LoRA adaptörünü taban modelle birleştirip .cact arşivi oluşturur."""
    from needle.model.finetune import build_main

    build_args = argparse.Namespace(
        checkpoint=args.checkpoint,
        lora=args.lora,
        out=args.out,
        upload=args.upload,
        layers=args.layers,
        platform=args.platform,
    )
    build_main(build_args)


def cmd_run(args):
    """Bir Türkçe kullanıcı sorgusunu Needle ajanında çalıştırır."""
    from . import create_agent

    agent = create_agent(weights=args.weights)
    print(f"\n[Sorgu]: {args.query}")
    res = agent.run(args.query)
    print(f"[Sonuç]: {res.get('results', res)}")


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


def cmd_benchmark(args):
    """Model veya ajan için benchmark çalıştırır."""
    dataset_path = Path(args.dataset) if args.dataset else None
    print(f"\n[1/3] 🧪 Benchmark başlatılıyor (Hedef: {args.target})...")
    summary = run_benchmark(
        target=args.target,
        dataset_path=dataset_path,
        use_builtin_suite=args.suite,
        ollama_model=args.ollama_model,
        ollama_host=args.ollama_host,
    )

    print(f"\n{'='*55}")
    print(f"  ⚡ NEEDLE-TR BENCHMARK SONUÇLARI")
    print(f"{'='*55}")
    print(f" Hedef:                 {summary.target_name}")
    print(f" Toplam Test:           {summary.total_tests}")
    print(f" Araç Seçim Doğruluğu:  %{summary.tool_accuracy:.1f} ({summary.tool_correct_count}/{summary.total_tests})")
    print(f" Argüman Doğruluğu:     %{summary.args_accuracy:.1f} ({summary.args_correct_count}/{summary.total_tests})")
    print(f" Geçerli Format Oranı:  %{summary.valid_format_rate:.1f} ({summary.valid_format_count}/{summary.total_tests})")
    print(f" Ortalama Gecikme:      {summary.avg_latency_ms:.1f} ms (min: {summary.min_latency_ms:.1f}, max: {summary.max_latency_ms:.1f})")
    print(f"{'-'*55}")
    print(f" Araç Bazında Başarım:")
    for t_name, stats in summary.per_tool_stats.items():
        print(f"   • {t_name:<14} -> Araç: %{stats['tool_accuracy']:<5.1f} | Arg: %{stats['args_accuracy']:<5.1f} | Gecikme: {stats['avg_latency_ms']:.1f}ms")
    print(f"{'='*55}\n")

    chart_path = Path(args.chart)
    print(f"[2/3] 📊 Performans grafiği oluşturuluyor: {chart_path}...")
    generate_benchmark_svg(summary, chart_path)
    print(f"      Grafik başarıyla kaydedildi: {chart_path.resolve()}")

    if not args.no_readme:
        readme_path = Path(args.readme)
        print(f"[3/3] 📝 Sonuçlar ve grafik {readme_path} dosyasına yazılıyor...")
        chart_url = (
            "https://raw.githubusercontent.com/emrekayik/needle-tr/main/benchmark_results.svg"
            if args.chart == "benchmark_results.svg"
            else args.chart
        )
        update_readme_benchmark(summary, svg_relative_path=chart_url, readme_path=readme_path)
        print(f"      {readme_path} başarıyla güncellendi.")

    print("\n✨ Benchmark tamamlandı!")


def cmd_platform(args):
    """Cactus Platform komutlarını çalıştırır (cactuscompute.com)."""
    from needle.cli import main as needle_cli_main

    # needle platform <verb> alt argümanlarını needle cli'ya aktar
    sys.argv = ["needle", "platform"] + sys.argv[2:]
    needle_cli_main()


def cmd_release(args):
    """Sürüm yükseltme ve tag oluşturma komutu."""
    run_release(
        bump_type=args.bump,
        push=args.push,
        publish=getattr(args, "publish", False),
        custom_message=args.message,
    )


def main():
    parser = argparse.ArgumentParser(
        prog="needle-tr",
        description="Needle-TR: Needle 3 Türkçe Araç Çağırıcı ve Fine-Tuning Sistemi (Cactus Compute)",
    )
    subparsers = parser.add_subparsers(dest="command", help="Alt komutlar")

    # generate-data
    p_gen = subparsers.add_parser("generate-data", help="Needle 3 formatında Türkçe veri seti üret")
    p_gen.add_argument("--num-samples", type=int, default=120, help="Üretilecek örnek sayısı (varsayılan: 120)")
    p_gen.add_argument("--output-dir", default="data", help="Verilerin kaydedileceği dizin")
    p_gen.add_argument("--use-ollama", action="store_true", help="Ollama API'si ile ek sentetik veri çoğalt")
    p_gen.add_argument("--ollama-model", default="qwen2.5:7b", help="Kullanılacak Ollama modeli")
    p_gen.add_argument("--ollama-host", default="http://localhost:11434", help="Ollama API adresi")
    p_gen.set_defaults(func=cmd_generate_data)

    # train / finetune
    for name in ["train", "finetune"]:
        p_train = subparsers.add_parser(name, help="Needle 3 LoRA Fine-tuning başlat")
        p_train.add_argument("--data", default="data/train.jsonl", help="Eğitim verisi dosya yolu")
        p_train.add_argument("--checkpoint", default=None, help="Özel taban model checkpoint yolu (None ise auto-download)")
        p_train.add_argument("--epochs", type=int, default=10, help="Epoch sayısı (varsayılan: 10)")
        p_train.add_argument("--batch-size", type=int, default=16, help="Batch boyutu (varsayılan: 16)")
        p_train.add_argument("--lr", type=float, default=1e-4, help="Öğrenme oranı (varsayılan: 1e-4)")
        p_train.add_argument("--lora-rank", type=int, default=16, help="LoRA rank (varsayılan: 16)")
        p_train.add_argument("--lora-alpha", type=float, default=32.0, help="LoRA alpha (varsayılan: 32.0)")
        p_train.add_argument("--max-len", type=int, default=1024, help="Maksimum sekans uzunluğu")
        p_train.add_argument("--val-split", type=float, default=0.1, help="Doğrulama oranı")
        p_train.add_argument("--seed", type=int, default=0, help="Rastgelelik tohumu")
        p_train.add_argument("--layers", type=int, default=20, help="Alt ağ katman sayısı (2..20, varsayılan: 20)")
        p_train.add_argument("--platform", default=None, help="Hedef platform (macos-arm64, linux-arm64 vb.)")
        p_train.add_argument("--output-dir", default="models", help="Model kayıt dizini")
        p_train.add_argument(
            "--quick",
            action="store_true",
            help="Hızlı deneme profili: 1 epoch, daha kısa sekans ve daha küçük LoRA",
        )
        p_train.add_argument("--dry-run", action="store_true", help="Eğitimi başlatmadan veri ve ayarları doğrula")
        p_train.add_argument("--no-build", action="store_true", help="Eğitim sonrası otomatik .cact derlemesini atla")
        p_train.add_argument("--push-to-hub", action="store_true", help="Eğitim sonrası modeli Hugging Face Hub'a yükle")
        p_train.add_argument("--hub-repo", default=None, help="Hugging Face repo (örn: emrekayik/needle-tr)")
        p_train.set_defaults(func=cmd_train)

    # build
    p_build = subparsers.add_parser("build", help="LoRA adaptörünü taban modelle birleştirip .cact arşivi oluştur")
    p_build.add_argument("checkpoint", nargs="?", default=None, help="Taban checkpoint (varsayılan: Needle 3)")
    p_build.add_argument("--lora", default="models/adapter.safetensors", help="Birleştirilecek LoRA adaptör yolu")
    p_build.add_argument("--out", default="models/needle3-tr.cact", help="Çıktı .cact dosya yolu")
    p_build.add_argument("--layers", type=int, default=20, help="Katman sayısı (2..20)")
    p_build.add_argument("--platform", default=None, help="Cihaz platformu")
    p_build.add_argument("--upload", action="store_true", help="NEEDLE_HF_REPO adresine yükle")
    p_build.set_defaults(func=cmd_build)

    # run
    p_run = subparsers.add_parser("run", help="Bir Türkçe kullanıcı sorgusunu Needle ajanında çalıştır")
    p_run.add_argument("query", help="Kullanıcı cümlesi (örn: 'Lagos hava durumu nasıl?')")
    p_run.add_argument("--weights", default=None, help="Kullanılacak .cact model ağırlık yolu")
    p_run.set_defaults(func=cmd_run)

    # benchmark
    p_bench = subparsers.add_parser("benchmark", help="Doğruluk ve gecikme benchmark'ı çalıştırıp grafik üret")
    p_bench.add_argument("--target", choices=["agent", "ollama"], default="agent", help="Test edilecek hedef (varsayılan: agent)")
    p_bench.add_argument("--dataset", default="data/val.jsonl", help="Doğrulama veri seti dosya yolu")
    p_bench.add_argument("--suite", action="store_true", help="Yerleşik kapsamlı benchmark test paketini kullan")
    p_bench.add_argument("--ollama-model", default="needle-tr", help="Ollama model adı (target=ollama için)")
    p_bench.add_argument("--ollama-host", default="http://localhost:11434", help="Ollama API adresi")
    p_bench.add_argument("--chart", default="benchmark_results.svg", help="Üretilecek SVG grafik dosya adı")
    p_bench.add_argument("--readme", default="README.md", help="Güncellenecek README dosya yolu")
    p_bench.add_argument("--no-readme", action="store_true", help="README.md güncellemesini atla")
    p_bench.set_defaults(func=cmd_benchmark)

    # platform
    p_plat = subparsers.add_parser("platform", help="Cactus Platform komutları (cactuscompute.com)")
    p_plat.set_defaults(func=cmd_platform)

    # export-ollama
    p_export = subparsers.add_parser("export-ollama", help="Ollama için Modelfile hazırla")
    p_export.add_argument("--output", default="Modelfile", help="Modelfile dosya yolu")
    p_export.add_argument("--base-model", default="qwen2.5:7b", help="Ollama temel modeli")
    p_export.add_argument("--model-name", default="needle-tr", help="Oluşturulacak model adı")
    p_export.add_argument("--gguf-path", default=None, help="Özel GGUF model dosya yolu (opsiyonel)")
    p_export.set_defaults(func=cmd_export_ollama)

    # release
    p_release = subparsers.add_parser("release", help="Sürümü artır, git commit ve tag oluştur")
    p_release.add_argument("bump", nargs="?", default="patch", help="patch, minor, major")
    p_release.add_argument("--push", action="store_true", help="Git commit ve etiketlerini gönder")
    p_release.add_argument("-m", "--message", default=None, help="Özel commit mesajı")
    p_release.add_argument("--publish", action="store_true", help="PyPI'ye yükle")
    p_release.set_defaults(func=cmd_release)

    args, unknown = parser.parse_known_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
