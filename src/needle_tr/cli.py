"""Needle-TR Fine-Tuning ve Ollama Yönetim Arayüzü (CLI)."""

import argparse
import sys
from pathlib import Path

from .benchmark import generate_benchmark_svg, run_benchmark, update_readme_benchmark
from .dataset_generator import build_dataset, check_ollama_available
from .export_ollama import write_modelfile
from .release import run_release
from .trainer import (
    TrainingConfig,
    push_dataset_to_hub,
    push_model_to_hub,
    run_training,
)


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
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_token=args.hub_token,
        merge_before_push=args.merge_before_push,
    )
    run_training(cfg, dry_run=args.dry_run)


def cmd_push_hub(args):
    """Hugging Face Hub'a model veya veri seti yükleme komutu."""
    if args.dataset:
        push_dataset_to_hub(
            data_dir=args.data_dir,
            hub_dataset_id=args.repo_id,
            token=args.token,
        )
    else:
        push_model_to_hub(
            model_dir=args.model_dir,
            hub_model_id=args.repo_id,
            token=args.token,
            base_model_name=args.base_model,
            merge=args.merge,
        )


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
    p_train.add_argument("--push-to-hub", action="store_true", help="Eğitim sonrası modeli Hugging Face Hub'a otomatik yükle")
    p_train.add_argument("--hub-model-id", default=None, help="Hugging Face model ID (örn: kullanici_adi/model-adi)")
    p_train.add_argument("--hub-token", default=None, help="Hugging Face API token (HF_TOKEN)")
    p_train.add_argument("--merge-before-push", action="store_true", help="Yüklemeden önce LoRA'yı temel modelle birleştir")
    p_train.set_defaults(func=cmd_train)

    # push-hub
    p_push = subparsers.add_parser("push-hub", help="Modeli veya veri kümesini Hugging Face Hub'a yükle")
    p_push.add_argument("repo_id", help="Hugging Face repo ID (örn: kullanici_adi/needle-tr-lora)")
    p_push.add_argument("--model-dir", default="models/needle_lora", help="Yüklenecek model klasörü")
    p_push.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct", help="Birleştirme için temel model")
    p_push.add_argument("--merge", action="store_true", help="LoRA adaptörünü temel modelle birleştirip tam model yükle")
    p_push.add_argument("--dataset", action="store_true", help="Model yerine data/ veri setini yükle")
    p_push.add_argument("--data-dir", default="data", help="Veri seti klasörü (dataset yükleme için)")
    p_push.add_argument("--token", default=None, help="Hugging Face API token (varsayılan: HF_TOKEN env)")
    p_push.set_defaults(func=cmd_push_hub)

    # export-ollama
    p_export = subparsers.add_parser("export-ollama", help="Ollama için Modelfile hazırla")
    p_export.add_argument("--output", default="Modelfile", help="Modelfile dosya yolu")
    p_export.add_argument("--base-model", default="qwen2.5:7b", help="Ollama temel modeli")
    p_export.add_argument("--model-name", default="needle-tr", help="Oluşturulacak model adı")
    p_export.add_argument("--gguf-path", default=None, help="Özel GGUF model dosya yolu (opsiyonel)")
    p_export.set_defaults(func=cmd_export_ollama)

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

    # release
    p_release = subparsers.add_parser("release", help="Sürümü artır, git commit ve tag oluştur")
    p_release.add_argument(
        "bump",
        nargs="?",
        default="patch",
        help="Sürüm artırma türü: 'patch', 'minor', 'major' veya doğrudan '0.2.0' (varsayılan: patch)",
    )
    p_release.add_argument(
        "--push",
        action="store_true",
        help="Commit ve etiketleri otomatik olarak GitHub'a gönder (git push origin <branch> --tags)",
    )
    p_release.add_argument(
        "-m",
        "--message",
        default=None,
        help="Özel commit mesajı (varsayılan: 'chore: release vX.Y.Z')",
    )
    p_release.add_argument(
        "--publish",
        action="store_true",
        help="Sürüm yükseltildikten sonra paketi derle ve .env dosyasındaki token ile PyPI'ye yükle",
    )
    p_release.set_defaults(func=cmd_release)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
