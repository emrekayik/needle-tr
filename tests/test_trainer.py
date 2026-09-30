"""trainer.py için birim testler.

Test kapsamı:
- TrainingConfig varsayılan değerleri
- load_jsonl() dosya okuma
- dry_run_validation() başarı / hata senaryoları
- Negatif örnek uyarısı
- Reasoning yokken bilgi mesajı
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from needle_tr.trainer import (
    TrainingConfig,
    dry_run_validation,
    load_jsonl,
)


# ==============================================================================
# TrainingConfig — varsayılan değerler
# ==============================================================================

class TestTrainingConfig:
    def test_default_epochs(self):
        cfg = TrainingConfig()
        assert cfg.epochs == 10

    def test_default_batch_size(self):
        assert TrainingConfig().batch_size == 16

    def test_default_lora_rank_and_alpha(self):
        cfg = TrainingConfig()
        assert cfg.lora_rank == 16
        assert cfg.lora_alpha == 32.0

    def test_default_layers(self):
        assert TrainingConfig().layers == 20

    def test_auto_build_enabled_by_default(self):
        assert TrainingConfig().auto_build is True

    def test_custom_values(self):
        cfg = TrainingConfig(epochs=5, batch_size=8, lr=2e-4)
        assert cfg.epochs == 5
        assert cfg.batch_size == 8
        assert cfg.lr == pytest.approx(2e-4)


# ==============================================================================
# load_jsonl
# ==============================================================================

class TestLoadJsonl:
    def _write_jsonl(self, path: Path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def test_loads_valid_file(self, tmp_path):
        p = tmp_path / "test.jsonl"
        rows = [{"query": "test", "answers": []}]
        self._write_jsonl(p, rows)
        result = load_jsonl(p)
        assert result == rows

    def test_skips_empty_lines(self, tmp_path):
        p = tmp_path / "test.jsonl"
        p.write_text('{"a": 1}\n\n{"b": 2}\n', encoding="utf-8")
        result = load_jsonl(p)
        assert len(result) == 2

    def test_raises_if_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_jsonl(tmp_path / "nonexistent.jsonl")

    def test_multiple_rows(self, tmp_path):
        p = tmp_path / "test.jsonl"
        rows = [{"query": f"q{i}", "answers": []} for i in range(10)]
        self._write_jsonl(p, rows)
        assert len(load_jsonl(p)) == 10


# ==============================================================================
# dry_run_validation
# ==============================================================================

def _make_config(tmp_path: Path, rows: list) -> TrainingConfig:
    """Geçici bir JSONL dosyasıyla TrainingConfig oluşturur."""
    data_file = tmp_path / "train.jsonl"
    with open(data_file, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return TrainingConfig(data_path=str(data_file))


def _positive(query="test query", tool="get_weather", args=None):
    return {
        "query": query,
        "tools": [],
        "answers": [{"name": tool, "arguments": args or {"city": "Test"}}],
        "reasoning": "'Test' -> city",
    }


def _negative(query="konu dışı soru"):
    return {"query": query, "tools": [], "answers": [], "reasoning": ""}


class TestDryRunValidation:
    def test_valid_data_returns_true(self, tmp_path):
        rows = [_positive() for _ in range(7)] + [_negative()]
        cfg = _make_config(tmp_path, rows)
        assert dry_run_validation(cfg) is True

    def test_missing_file_returns_false(self, tmp_path):
        cfg = TrainingConfig(data_path=str(tmp_path / "missing.jsonl"))
        assert dry_run_validation(cfg) is False

    def test_empty_file_returns_false(self, tmp_path):
        data_file = tmp_path / "train.jsonl"
        data_file.write_text("", encoding="utf-8")
        cfg = TrainingConfig(data_path=str(data_file))
        assert dry_run_validation(cfg) is False

    def test_missing_query_field_returns_false(self, tmp_path):
        rows = [{"answers": [], "reasoning": ""} for _ in range(5)]
        cfg = _make_config(tmp_path, rows)
        assert dry_run_validation(cfg) is False

    def test_missing_answers_field_returns_false(self, tmp_path):
        rows = [{"query": "test", "reasoning": ""} for _ in range(5)]
        cfg = _make_config(tmp_path, rows)
        assert dry_run_validation(cfg) is False

    def test_no_negative_examples_still_valid(self, tmp_path, capsys):
        """Negatif örnek yoksa uyarı verir ama True döner."""
        rows = [_positive() for _ in range(8)]
        cfg = _make_config(tmp_path, rows)
        result = dry_run_validation(cfg)
        captured = capsys.readouterr()
        assert result is True
        assert "negatif" in captured.out.lower() or "UYARI" in captured.out

    def test_no_reasoning_shows_info(self, tmp_path, capsys):
        """reasoning yoksa bilgi mesajı görünmeli."""
        rows = [
            {"query": "test", "tools": [], "answers": [{"name": "get_weather", "arguments": {"city": "X"}}], "reasoning": ""}
            for _ in range(8)
        ]
        cfg = _make_config(tmp_path, rows)
        dry_run_validation(cfg)
        captured = capsys.readouterr()
        assert "reasoning" in captured.out.lower() or "BİLGİ" in captured.out

    def test_step_count_analysis_shown(self, tmp_path, capsys):
        rows = [_positive() for _ in range(7)] + [_negative()]
        cfg = _make_config(tmp_path, rows)
        dry_run_validation(cfg)
        captured = capsys.readouterr()
        assert "adım" in captured.out.lower() or "step" in captured.out.lower()
