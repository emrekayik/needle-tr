"""Needle-TR Benchmark ve Başarım Ölçüm Sistemi.

Bu modül:
1. Modelin veya ajanın (Cactus Needle / Ollama) Türkçe araç çağırma doğruluğunu ve gecikmesini test eder.
2. Doğruluk (Tool & Argüman Eşleşmesi), geçerli JSON oranı ve gecikme (ms) metriklerini hesaplar.
3. Otomatik olarak yüksek çözünürlüklü ve estetik bir SVG performans grafiği üretir.
4. Sonuçları ve grafiği doğrudan README.md dosyasına işler.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_BENCHMARK_DATA = [
    # calculate
    {
        "query": "25 * 4 hesapla",
        "expected_tool": "calculate",
        "expected_args": {"expression": "25 * 4"},
    },
    {
        "query": "15 ile 48'i topla",
        "expected_tool": "calculate",
        "expected_args": {"expression": "15 + 48"},
    },
    {
        "query": "120 bölü 6 kaç eder?",
        "expected_tool": "calculate",
        "expected_args": {"expression": "120 / 6"},
    },
    {
        "query": "7'nin karesini al",
        "expected_tool": "calculate",
        "expected_args": {"expression": "7 ** 2"},
    },
    # get_weather
    {
        "query": "İstanbul'da hava durumu nasıl?",
        "expected_tool": "get_weather",
        "expected_args": {"city": "İstanbul"},
    },
    {
        "query": "Ankara için hava durumunu göster",
        "expected_tool": "get_weather",
        "expected_args": {"city": "Ankara"},
    },
    {
        "query": "İzmir hava durumu nedir?",
        "expected_tool": "get_weather",
        "expected_args": {"city": "İzmir"},
    },
    {
        "query": "Lagos'ta hava nasıl?",
        "expected_tool": "get_weather",
        "expected_args": {"city": "Lagos"},
    },
    # send_message
    {
        "query": "Ahmet'e 'Toplantı başladı' mesajı gönder",
        "expected_tool": "send_message",
        "expected_args": {"recipient": "Ahmet", "message": "Toplantı başladı"},
    },
    {
        "query": "Mehmet'e 'Yarın buluşuyoruz' diye mesaj ilet",
        "expected_tool": "send_message",
        "expected_args": {"recipient": "Mehmet", "message": "Yarın buluşuyoruz"},
    },
    {
        "query": "Ayşe'ye 'Dosyayı gönderdim' mesajı at",
        "expected_tool": "send_message",
        "expected_args": {"recipient": "Ayşe", "message": "Dosyayı gönderdim"},
    },
    # set_alarm
    {
        "query": "Saat 07:30'a alarm kur",
        "expected_tool": "set_alarm",
        "expected_args": {"time": "07:30"},
    },
    {
        "query": "14:15'e Toplantı alarmı kur",
        "expected_tool": "set_alarm",
        "expected_args": {"time": "14:15", "label": "Toplantı"},
    },
    {
        "query": "Sabah 08:00'e Uyanış alarmı ayarla",
        "expected_tool": "set_alarm",
        "expected_args": {"time": "08:00", "label": "Uyanış"},
    },
]


@dataclass
class TestCaseResult:
    query: str
    expected_tool: str
    expected_args: Dict[str, Any]
    predicted_tool: Optional[str]
    predicted_args: Dict[str, Any]
    tool_correct: bool
    args_correct: bool
    valid_format: bool
    latency_ms: float
    raw_response: Any = None


@dataclass
class BenchmarkSummary:
    target_name: str
    total_tests: int
    tool_correct_count: int
    tool_accuracy: float
    args_correct_count: int
    args_accuracy: float
    valid_format_count: int
    valid_format_rate: float
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    per_tool_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    details: List[TestCaseResult] = field(default_factory=list)


def load_test_cases(dataset_path: Optional[Path] = None, use_builtin_suite: bool = False) -> List[Dict[str, Any]]:
    """Test veri setini dosyadan (JSONL) veya yerleşik şablondan yükler."""
    if use_builtin_suite:
        return DEFAULT_BENCHMARK_DATA

    if dataset_path and dataset_path.exists():
        cases = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "messages" in obj:
                        msgs = obj["messages"]
                        user_msg = next((m["content"] for m in msgs if m.get("role") == "user"), "")
                        asst_msg = next((m["content"] for m in msgs if m.get("role") == "assistant"), "")

                        # JSON bloğunu ayıkla
                        json_match = re.search(r"```json\s*(.*?)\s*```", asst_msg, re.DOTALL)
                        json_str = json_match.group(1) if json_match else asst_msg.strip()
                        parsed = json.loads(json_str)

                        cases.append({
                            "query": user_msg,
                            "expected_tool": parsed.get("name", ""),
                            "expected_args": parsed.get("arguments", {}),
                        })
                    elif "query" in obj and "tool" in obj:
                        cases.append({
                            "query": obj["query"],
                            "expected_tool": obj["tool"],
                            "expected_args": obj.get("arguments", {}),
                        })
                except Exception:
                    continue
        if cases:
            return cases

    return DEFAULT_BENCHMARK_DATA


def _parse_agent_response(res: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any], bool]:
    """Cactus Needle agent yanıtından çağrılan aracı ve argümanları çıkarır."""
    if not isinstance(res, dict):
        return None, {}, False

    # 1. results listesinden çıkarım
    results = res.get("results", [])
    if results and isinstance(results, list):
        first_res = results[0]
        if isinstance(first_res, dict):
            if "result" in first_res or "expression" in first_res:
                return "calculate", {"expression": str(first_res.get("expression", ""))}, True
            if "temp_c" in first_res or "sky" in first_res:
                return "get_weather", {"city": str(first_res.get("city", ""))}, True
            if "status" in first_res and "recipient" in first_res:
                return "send_message", {
                    "recipient": str(first_res.get("recipient", "")),
                    "message": str(first_res.get("message", "")),
                }, True
            if "status" in first_res and ("time" in first_res or "label" in first_res):
                args = {"time": str(first_res.get("time", ""))}
                if "label" in first_res:
                    args["label"] = str(first_res.get("label", ""))
                return "set_alarm", args, True

    # 2. function_calls listesinden çıkarım
    calls = res.get("function_calls", [])
    if calls and isinstance(calls, list):
        c = calls[0]
        if isinstance(c, dict):
            return c.get("name"), c.get("arguments", {}), True

    # 3. suppressed_calls listesinden çıkarım
    suppressed = res.get("suppressed_calls", [])
    if suppressed and isinstance(suppressed, list):
        c = suppressed[0]
        if isinstance(c, dict):
            return c.get("name"), c.get("arguments", {}), True

    # Araç çağrısı bulunamadı
    return None, {}, bool(res.get("success", False))


def _query_ollama(
    query: str,
    model: str = "needle-tr",
    host: str = "http://localhost:11434",
    timeout: int = 15,
) -> Tuple[Optional[str], Dict[str, Any], bool, float]:
    """Ollama API'sine istek gönderip dönen yanıtı parse eder."""
    url = f"{host.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": query}],
        "stream": False,
        "options": {"temperature": 0.1},
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            latency_ms = (time.perf_counter() - t0) * 1000
            content = data.get("message", {}).get("content", "").strip()

            # Markdown JSON ayıkla
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
            raw_json = match.group(1).strip() if match else content

            parsed = json.loads(raw_json)
            pred_tool = parsed.get("name")
            pred_args = parsed.get("arguments", {})
            return pred_tool, pred_args, True, latency_ms
    except Exception:
        latency_ms = (time.perf_counter() - t0) * 1000
        return None, {}, False, latency_ms


def _args_match(expected: Dict[str, Any], predicted: Dict[str, Any]) -> bool:
    """Argümanların anlamsal/içeriksel eşleşmesini kontrol eder."""
    if not expected and not predicted:
        return True
    for k, exp_val in expected.items():
        if k not in predicted:
            return False
        pred_val = predicted[k]
        # Küçük harf ve boşluk normalizasyonu
        if isinstance(exp_val, str) and isinstance(pred_val, str):
            if exp_val.strip().lower() != pred_val.strip().lower():
                # Kısmi eşleşme kontrolü (örn. "İstanbul için" vs "İstanbul")
                if exp_val.strip().lower() not in pred_val.strip().lower() and pred_val.strip().lower() not in exp_val.strip().lower():
                    return False
        elif exp_val != pred_val:
            return False
    return True


def run_benchmark(
    target: str = "agent",
    dataset_path: Optional[Path] = None,
    use_builtin_suite: bool = False,
    ollama_model: str = "needle-tr",
    ollama_host: str = "http://localhost:11434",
) -> BenchmarkSummary:
    """Belirtilen hedef üzerinde benchmark çalıştırır ve özet metrikleri döner."""
    test_cases = load_test_cases(dataset_path, use_builtin_suite=use_builtin_suite)

    # Hedef model/ajan hazırlığı
    agent_instance = None
    if target == "agent":
        from . import agent as default_agent
        agent_instance = default_agent
        target_display = "Cactus Needle (On-Device Agent)"
    else:
        target_display = f"Ollama ({ollama_model})"

    results: List[TestCaseResult] = []
    per_tool: Dict[str, Dict[str, Any]] = {}

    for case in test_cases:
        query = case["query"]
        expected_tool = case["expected_tool"]
        expected_args = case.get("expected_args", {})

        if expected_tool not in per_tool:
            per_tool[expected_tool] = {
                "total": 0,
                "tool_correct": 0,
                "args_correct": 0,
                "latencies": [],
            }
        per_tool[expected_tool]["total"] += 1

        if target == "agent":
            t0 = time.perf_counter()
            try:
                res = agent_instance.run(query)
                latency_ms = (time.perf_counter() - t0) * 1000
                pred_tool, pred_args, valid = _parse_agent_response(res)
            except Exception as e:
                latency_ms = (time.perf_counter() - t0) * 1000
                pred_tool, pred_args, valid, res = None, {}, False, str(e)
        else:
            pred_tool, pred_args, valid, latency_ms = _query_ollama(
                query=query, model=ollama_model, host=ollama_host
            )
            res = None

        tool_ok = bool(pred_tool and pred_tool.lower() == expected_tool.lower())
        args_ok = tool_ok and _args_match(expected_args, pred_args)

        if tool_ok:
            per_tool[expected_tool]["tool_correct"] += 1
        if args_ok:
            per_tool[expected_tool]["args_correct"] += 1
        per_tool[expected_tool]["latencies"].append(latency_ms)

        results.append(
            TestCaseResult(
                query=query,
                expected_tool=expected_tool,
                expected_args=expected_args,
                predicted_tool=pred_tool,
                predicted_args=pred_args,
                tool_correct=tool_ok,
                args_correct=args_ok,
                valid_format=valid,
                latency_ms=latency_ms,
                raw_response=res,
            )
        )

    # Genel metrikler
    total = len(results)
    tool_correct_cnt = sum(1 for r in results if r.tool_correct)
    args_correct_cnt = sum(1 for r in results if r.args_correct)
    valid_format_cnt = sum(1 for r in results if r.valid_format)
    latencies = [r.latency_ms for r in results]

    per_tool_final: Dict[str, Dict[str, Any]] = {}
    for t_name, data in per_tool.items():
        t_total = data["total"]
        t_latencies = data["latencies"]
        per_tool_final[t_name] = {
            "total": t_total,
            "tool_correct": data["tool_correct"],
            "tool_accuracy": (data["tool_correct"] / t_total * 100) if t_total else 0.0,
            "args_correct": data["args_correct"],
            "args_accuracy": (data["args_correct"] / t_total * 100) if t_total else 0.0,
            "avg_latency_ms": sum(t_latencies) / len(t_latencies) if t_latencies else 0.0,
        }

    return BenchmarkSummary(
        target_name=target_display,
        total_tests=total,
        tool_correct_count=tool_correct_cnt,
        tool_accuracy=(tool_correct_cnt / total * 100) if total else 0.0,
        args_correct_count=args_correct_cnt,
        args_accuracy=(args_correct_cnt / total * 100) if total else 0.0,
        valid_format_count=valid_format_cnt,
        valid_format_rate=(valid_format_cnt / total * 100) if total else 0.0,
        avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
        min_latency_ms=min(latencies) if latencies else 0.0,
        max_latency_ms=max(latencies) if latencies else 0.0,
        per_tool_stats=per_tool_final,
        details=results,
    )


def generate_benchmark_svg(summary: BenchmarkSummary, output_path: Path) -> Path:
    """Modern, şık ve karanlık temalı vektörel SVG performans kartı oluşturur."""
    width = 820
    tool_count = len(summary.per_tool_stats)
    height = 420 + max(0, (tool_count - 4) * 45)

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="100%" height="100%">',
        "  <defs>",
        "    <!-- Arka plan ve kart gradyanları -->",
        '    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">',
        '      <stop offset="0%" stop-color="#0f172a" />',
        '      <stop offset="100%" stop-color="#1e293b" />',
        "    </linearGradient>",
        '    <linearGradient id="cardGrad" x1="0%" y1="0%" x2="0%" y2="100%">',
        '      <stop offset="0%" stop-color="#1e293b" stop-opacity="0.8" />',
        '      <stop offset="100%" stop-color="#0f172a" stop-opacity="0.9" />',
        "    </linearGradient>",
        '    <linearGradient id="barGradCyan" x1="0%" y1="0%" x2="100%" y2="0%">',
        '      <stop offset="0%" stop-color="#38bdf8" />',
        '      <stop offset="100%" stop-color="#818cf8" />',
        "    </linearGradient>",
        '    <linearGradient id="barGradGreen" x1="0%" y1="0%" x2="100%" y2="0%">',
        '      <stop offset="0%" stop-color="#34d399" />',
        '      <stop offset="100%" stop-color="#10b981" />',
        "    </linearGradient>",
        '    <filter id="shadow" x="-5%" y="-5%" width="110%" height="110%">',
        '      <feDropShadow dx="0" dy="4" stdDeviation="6" flood-color="#000000" flood-opacity="0.4" />',
        "    </filter>",
        "  </defs>",
        "",
        '  <style>',
        '    .title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 20px; font-weight: 700; fill: #f8fafc; }',
        '    .subtitle { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 13px; fill: #94a3b8; }',
        '    .kpi-title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; fill: #94a3b8; font-weight: 600; }',
        '    .kpi-value { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 24px; font-weight: 800; }',
        '    .kpi-sub { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 11px; fill: #64748b; }',
        '    .section-title { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 14px; font-weight: 600; fill: #cbd5e1; }',
        '    .tool-label { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace; font-size: 13px; fill: #e2e8f0; }',
        '    .tool-stat { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 12px; fill: #94a3b8; }',
        "  </style>",
        "",
        f'  <!-- Arka Plan -->',
        f'  <rect width="{width}" height="{height}" rx="16" fill="url(#bgGrad)" stroke="#334155" stroke-width="1" />',
        "",
        "  <!-- Başlık ve Meta -->",
        '  <text x="32" y="44" class="title">⚡ Needle-TR Benchmark Raporu</text>',
        f'  <text x="32" y="68" class="subtitle">Hedef: <tspan fill="#38bdf8" font-weight="600">{summary.target_name}</tspan> | Test Sayısı: {summary.total_tests} | Tarih: {summary.timestamp}</text>',
        "",
        "  <!-- KPI Kartları -->",
    ]

    kpis = [
        ("Araç Doğruluğu", f"%{summary.tool_accuracy:.1f}", f"{summary.tool_correct_count}/{summary.total_tests} başarılı", "#38bdf8"),
        ("Argüman Doğruluğu", f"%{summary.args_accuracy:.1f}", f"{summary.args_correct_count}/{summary.total_tests} eşleşme", "#818cf8"),
        ("Geçerli Çağrı Oranı", f"%{summary.valid_format_rate:.1f}", f"{summary.valid_format_count}/{summary.total_tests} parse edildi", "#34d399"),
        ("Ortalama Gecikme", f"{summary.avg_latency_ms:.1f} ms", f"min: {summary.min_latency_ms:.1f} / max: {summary.max_latency_ms:.1f}", "#f472b6"),
    ]

    card_w = 175
    card_h = 86
    card_gap = 16
    start_x = 32
    start_y = 96

    for i, (ktitle, kval, ksub, color) in enumerate(kpis):
        cx = start_x + i * (card_w + card_gap)
        svg_lines.extend([
            f'  <g transform="translate({cx}, {start_y})" filter="url(#shadow)">',
            f'    <rect width="{card_w}" height="{card_h}" rx="12" fill="url(#cardGrad)" stroke="#334155" stroke-width="1" />',
            f'    <text x="14" y="24" class="kpi-title">{ktitle}</text>',
            f'    <text x="14" y="54" class="kpi-value" fill="{color}">{kval}</text>',
            f'    <text x="14" y="72" class="kpi-sub">{ksub}</text>',
            "  </g>",
        ])

    # Araç Başarı Dağılımı Çubuk Grafiği
    bar_y_start = 226
    svg_lines.extend([
        "",
        f'  <!-- Araç Bazında Başarım Dağılımı -->',
        f'  <text x="32" y="{bar_y_start}" class="section-title">📊 Araç Bazında Doğruluk Dağılımı</text>',
        f'  <rect x="32" y="{bar_y_start + 12}" width="{width - 64}" height="{height - bar_y_start - 36}" rx="12" fill="url(#cardGrad)" stroke="#334155" stroke-width="1" filter="url(#shadow)" />',
    ])

    item_y = bar_y_start + 44
    max_bar_width = 380

    for tool_name, stats in summary.per_tool_stats.items():
        t_acc = stats["tool_accuracy"]
        a_acc = stats["args_accuracy"]
        bar_len = int((t_acc / 100.0) * max_bar_width)

        svg_lines.extend([
            f'  <!-- Tool: {tool_name} -->',
            f'  <text x="52" y="{item_y + 14}" class="tool-label">{tool_name}()</text>',
            f'  <!-- Background bar -->',
            f'  <rect x="220" y="{item_y}" width="{max_bar_width}" height="18" rx="9" fill="#1e293b" stroke="#334155" stroke-width="1" />',
            f'  <!-- Fill bar -->',
            f'  <rect x="220" y="{item_y}" width="{max(bar_len, 6)}" height="18" rx="9" fill="url(#barGradCyan)" />',
            f'  <!-- Percent & Details -->',
            f'  <text x="{220 + max_bar_width + 16}" y="{item_y + 14}" class="tool-stat"><tspan fill="#38bdf8" font-weight="700">%{t_acc:.1f}</tspan> (Araç) | <tspan fill="#818cf8">%{a_acc:.1f}</tspan> (Arg) | <tspan fill="#94a3b8">{stats["avg_latency_ms"]:.1f}ms</tspan></text>',
        ])
        item_y += 38

    svg_lines.append("</svg>")

    content = "\n".join(svg_lines)
    output_path.write_text(content, encoding="utf-8")
    return output_path


def update_readme_benchmark(
    summary: BenchmarkSummary,
    svg_relative_path: str = "https://raw.githubusercontent.com/emrekayik/needle-tr/main/benchmark_results.svg",
    readme_path: Path = Path("README.md"),
) -> None:
    """README.md dosyasındaki benchmark tablosunu ve görsel referansını günceller."""
    if not readme_path.exists():
        return

    content = readme_path.read_text(encoding="utf-8")

    # Tablo satırları oluştur
    table_rows = []
    for tool_name, stats in summary.per_tool_stats.items():
        table_rows.append(
            f"| `{tool_name}` | %{stats['tool_accuracy']:.1f} | %{stats['args_accuracy']:.1f} | {stats['total']} | {stats['avg_latency_ms']:.1f} ms |"
        )
    rows_str = "\n".join(table_rows)

    benchmark_section = f"""## 📊 Benchmark ve Başarım Sonuçları

`needle-tr` yerleşik benchmark aracı ile ölçülen son başarım sonuçları:

![Needle-TR Benchmark Sonuçları]({svg_relative_path})

### 📈 Özet Performans Metrikleri

| Metrik | Değer | Açıklama |
| :--- | :--- | :--- |
| **Test Edilen Hedef** | `{summary.target_name}` | Test edilen ortam (Ajan / Model) |
| **Toplam Test Sayısı** | **{summary.total_tests}** | Doğrulama sorgusu sayısı |
| **Araç Seçim Doğruluğu** | **%{summary.tool_accuracy:.1f}** ({summary.tool_correct_count}/{summary.total_tests}) | Doğru fonksiyonun seçilme oranı |
| **Argüman Doğruluğu** | **%{summary.args_accuracy:.1f}** ({summary.args_correct_count}/{summary.total_tests}) | Parametrelerin eksiksiz eşleşme oranı |
| **Geçerli Yanıt / JSON** | **%{summary.valid_format_rate:.1f}** ({summary.valid_format_count}/{summary.total_tests}) | Bozulma olmadan parse edilen çağrılar |
| **Ortalama Gecikme** | **{summary.avg_latency_ms:.1f} ms** | Min: {summary.min_latency_ms:.1f} ms, Max: {summary.max_latency_ms:.1f} ms |
| **Son Güncelleme** | `{summary.timestamp}` | Otomatik benchmark çalıştırma zamanı |

### 🔍 Araç Bazında Detay Dağılımı

| Araç (Tool) | Araç Doğruluğu | Argüman Eşleşmesi | Test Sayısı | Ortalama Gecikme |
| :--- | :---: | :---: | :---: | :---: |
{rows_str}

Benchmark testlerini kendiniz çalıştırmak ve grafiği güncellemek için:

```bash
# Cactus Needle (On-device) Ajanı için
uv run needle-tr benchmark --target agent

# Ollama Modeli için
uv run needle-tr benchmark --target ollama --ollama-model needle-tr
```
"""

    pattern = r"## 📊 Benchmark ve Başarım Sonuçları[\s\S]*?(?=\n## |\Z)"
    if re.search(pattern, content):
        new_content = re.sub(pattern, benchmark_section.strip() + "\n", content)
    else:
        # Ajan kullanımından hemen sonrasına veya referansların öncesine ekle
        ref_pattern = r"(## 🙏 Referanslar & Teşekkürler)"
        if re.search(ref_pattern, content):
            new_content = re.sub(ref_pattern, benchmark_section.strip() + "\n\n---\n\n\\1", content)
        else:
            new_content = content.rstrip() + "\n\n---\n\n" + benchmark_section.strip() + "\n"

    readme_path.write_text(new_content, encoding="utf-8")
