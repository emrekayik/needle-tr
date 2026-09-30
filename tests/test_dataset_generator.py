"""dataset_generator.py için birim testler.

Test kapsamı:
- generate_synthetic_samples() çıktı yapısı ve dengesi
- SEED_EXAMPLES Needle 3 format doğrulaması
- build_dataset() dosya oluşturma ve JSONL formatı
- Negatif örnek oranı (~%12.5)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from needle_tr.dataset_generator import (
    SEED_EXAMPLES,
    build_dataset,
    generate_synthetic_samples,
)


# ==============================================================================
# SEED_EXAMPLES format kontrolü
# ==============================================================================

class TestSeedExamples:
    def test_all_have_required_keys(self):
        for i, ex in enumerate(SEED_EXAMPLES):
            assert "query" in ex, f"SEED_EXAMPLES[{i}] 'query' eksik"
            assert "answers" in ex, f"SEED_EXAMPLES[{i}] 'answers' eksik"
            assert "reasoning" in ex, f"SEED_EXAMPLES[{i}] 'reasoning' eksik"

    def test_positive_examples_have_non_empty_answers(self):
        positives = [e for e in SEED_EXAMPLES if e["answers"]]
        assert len(positives) > 0

    def test_negative_examples_have_empty_answers(self):
        negatives = [e for e in SEED_EXAMPLES if not e["answers"]]
        assert len(negatives) > 0, "En az bir negatif (answers: []) örnek olmalı"

    def test_positive_answers_have_name_and_arguments(self):
        for ex in SEED_EXAMPLES:
            for ans in ex["answers"]:
                assert "name" in ans
                assert "arguments" in ans

    def test_known_tool_names_only(self):
        allowed = {"get_weather", "send_message", "set_alarm", "calculate"}
        for ex in SEED_EXAMPLES:
            for ans in ex["answers"]:
                assert ans["name"] in allowed, f"Bilinmeyen araç: {ans['name']}"


# ==============================================================================
# generate_synthetic_samples
# ==============================================================================

class TestGenerateSyntheticSamples:
    @pytest.mark.parametrize("n", [20, 50, 120])
    def test_returns_requested_count(self, n):
        samples = generate_synthetic_samples(n)
        assert len(samples) == n

    def test_each_sample_has_required_fields(self):
        samples = generate_synthetic_samples(30)
        for s in samples:
            assert "query" in s
            assert "tools" in s
            assert "answers" in s
            assert "reasoning" in s

    def test_tools_field_is_list(self):
        samples = generate_synthetic_samples(20)
        for s in samples:
            assert isinstance(s["tools"], list)
            assert len(s["tools"]) == 4  # 4 aktif araç

    def test_negative_ratio_approx_12_5_pct(self):
        """Cactus kuralı: ~%12.5 negatif örnek."""
        samples = generate_synthetic_samples(120)
        negatives = [s for s in samples if len(s["answers"]) == 0]
        ratio = len(negatives) / len(samples)
        assert 0.08 <= ratio <= 0.20, f"Negatif oranı beklenenden farklı: %{ratio*100:.1f}"

    def test_deterministic_with_same_seed(self):
        """Aynı seed → aynı çıktı (deterministik üretim)."""
        a = generate_synthetic_samples(30)
        b = generate_synthetic_samples(30)
        assert [s["query"] for s in a] == [s["query"] for s in b]

    def test_all_tool_types_represented(self):
        samples = generate_synthetic_samples(120)
        tool_names = {
            ans["name"]
            for s in samples
            for ans in s["answers"]
        }
        assert "get_weather" in tool_names
        assert "send_message" in tool_names
        assert "set_alarm" in tool_names
        assert "calculate" in tool_names


# ==============================================================================
# build_dataset — dosya oluşturma
# ==============================================================================

class TestBuildDataset:
    def test_creates_train_and_val_jsonl(self, tmp_path):
        result = build_dataset(output_dir=str(tmp_path), num_samples=30)
        assert Path(result["train_file"]).exists()
        assert Path(result["val_file"]).exists()

    def test_train_jsonl_is_valid(self, tmp_path):
        result = build_dataset(output_dir=str(tmp_path), num_samples=30)
        with open(result["train_file"], encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) > 0
        for row in lines:
            assert "query" in row
            assert "answers" in row

    def test_val_jsonl_is_valid(self, tmp_path):
        result = build_dataset(output_dir=str(tmp_path), num_samples=30)
        with open(result["val_file"], encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        assert len(lines) > 0

    def test_train_plus_val_equals_total(self, tmp_path):
        result = build_dataset(output_dir=str(tmp_path), num_samples=40, train_ratio=0.8)
        assert result["train"] + result["val"] == result["total"]

    def test_total_matches_num_samples(self, tmp_path):
        result = build_dataset(output_dir=str(tmp_path), num_samples=40)
        assert result["total"] == 40

    def test_combined_data_jsonl_created(self, tmp_path):
        build_dataset(output_dir=str(tmp_path), num_samples=20)
        assert (tmp_path / "data.jsonl").exists()
