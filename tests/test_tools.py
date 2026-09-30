"""tools.py için birim testler.

Test kapsamı:
- Her araç fonksiyonunun doğru çıktı döndürmesi
- calculate() güvenli eval sandbox
- get_needle_tools_schema() şema yapısı
- get_tools_schema() OpenAI / Needle formatları
"""

from __future__ import annotations

import pytest
from needle_tr.tools import (
    ACTIVE_TOOLS,
    calculate,
    get_needle_tools_schema,
    get_tools_schema,
    get_weather,
    send_message,
    set_alarm,
)


# ==============================================================================
# get_weather
# ==============================================================================

class TestGetWeather:
    def test_returns_city(self):
        result = get_weather("İstanbul")
        assert result["city"] == "İstanbul"

    def test_has_temp_and_sky(self):
        result = get_weather("Ankara")
        assert "temp_c" in result
        assert "sky" in result

    def test_any_city_name(self):
        for city in ["Lagos", "Berlin", "Tokyo"]:
            r = get_weather(city)
            assert r["city"] == city


# ==============================================================================
# send_message
# ==============================================================================

class TestSendMessage:
    def test_recipient_and_message_echoed(self):
        r = send_message("Ahmet", "Toplantı başladı")
        assert r["recipient"] == "Ahmet"
        assert r["message"] == "Toplantı başladı"

    def test_status_field(self):
        r = send_message("Zeynep", "Merhaba")
        assert "status" in r

    def test_unicode_message(self):
        r = send_message("Ali", "Gelirken iki ekmek al 🍞")
        assert r["message"] == "Gelirken iki ekmek al 🍞"


# ==============================================================================
# set_alarm
# ==============================================================================

class TestSetAlarm:
    def test_basic_time(self):
        r = set_alarm("07:30")
        assert r["time"] == "07:30"
        assert r["status"] == "kuruldu"

    def test_with_label(self):
        r = set_alarm("14:15", "Toplantı")
        assert r["label"] == "Toplantı"

    def test_default_label(self):
        r = set_alarm("06:00")
        assert r["label"] == "Alarm"  # varsayılan

    def test_various_times(self):
        for t in ["00:00", "12:00", "23:59"]:
            assert set_alarm(t)["time"] == t


# ==============================================================================
# calculate
# ==============================================================================

class TestCalculate:
    @pytest.mark.parametrize("expr,expected", [
        ("25 * 4", 100),
        ("15 + 48", 63),
        ("120 / 6", 20.0),
        ("7 ** 2", 49),
        ("250 - 85", 165),
        ("(10 + 5) * 3", 45),
        ("500 * 0.18", 90.0),
    ])
    def test_valid_expressions(self, expr, expected):
        r = calculate(expr)
        assert r["expression"] == expr
        assert r["result"] == pytest.approx(expected)

    def test_invalid_expression_returns_none(self):
        r = calculate("invalid_python")
        assert r["result"] is None

    def test_division_by_zero_returns_none(self):
        r = calculate("1 / 0")
        assert r["result"] is None

    def test_no_builtins_sandbox(self):
        """Tehlikeli built-in'lerin sandbox'ta çalışmaması gerekir."""
        r = calculate("__import__('os').getcwd()")
        assert r["result"] is None


# ==============================================================================
# ACTIVE_TOOLS
# ==============================================================================

class TestActiveTools:
    def test_has_four_tools(self):
        assert len(ACTIVE_TOOLS) == 4

    def test_all_callable(self):
        for fn in ACTIVE_TOOLS:
            assert callable(fn)

    def test_expected_names(self):
        names = {fn.__name__ for fn in ACTIVE_TOOLS}
        assert names == {"get_weather", "send_message", "set_alarm", "calculate"}


# ==============================================================================
# get_needle_tools_schema
# ==============================================================================

class TestGetNeedleToolsSchema:
    def setup_method(self):
        self.schemas = get_needle_tools_schema()

    def test_returns_list(self):
        assert isinstance(self.schemas, list)

    def test_four_schemas(self):
        assert len(self.schemas) == 4

    def test_each_has_name_description_parameters(self):
        for s in self.schemas:
            assert "name" in s, f"{s} 'name' alanı eksik"
            assert "description" in s, f"{s} 'description' alanı eksik"
            assert "parameters" in s, f"{s} 'parameters' alanı eksik"

    def test_parameters_structure(self):
        for s in self.schemas:
            params = s["parameters"]
            assert params["type"] == "object"
            assert "properties" in params
            assert "required" in params

    def test_get_weather_required_city(self):
        schema = next(s for s in self.schemas if s["name"] == "get_weather")
        assert "city" in schema["parameters"]["required"]
        assert "city" in schema["parameters"]["properties"]

    def test_send_message_required_fields(self):
        schema = next(s for s in self.schemas if s["name"] == "send_message")
        required = schema["parameters"]["required"]
        assert "recipient" in required
        assert "message" in required

    def test_set_alarm_optional_label(self):
        schema = next(s for s in self.schemas if s["name"] == "set_alarm")
        required = schema["parameters"]["required"]
        assert "time" in required
        assert "label" not in required  # label opsiyonel

    def test_custom_tools_subset(self):
        schemas = get_needle_tools_schema([get_weather, calculate])
        assert len(schemas) == 2
        names = {s["name"] for s in schemas}
        assert names == {"get_weather", "calculate"}


# ==============================================================================
# get_tools_schema — format seçenekleri
# ==============================================================================

class TestGetToolsSchema:
    def test_needle_format_default(self):
        schemas = get_tools_schema()
        assert len(schemas) == 4
        # Needle formatı doğrudan obje (type wrapper yok)
        assert "name" in schemas[0]

    def test_openai_format_wraps_with_type_function(self):
        schemas = get_tools_schema(format="openai")
        for s in schemas:
            assert s["type"] == "function"
            assert "function" in s
            assert "name" in s["function"]
