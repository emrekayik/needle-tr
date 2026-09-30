"""Needle 3 Türkçe Araç (Tool) tanımları ve şema yardımcıları."""

from __future__ import annotations

import inspect
import re
from typing import Any, Callable, Dict, List, Optional


def get_weather(city: str) -> Dict[str, Any]:
    """Get the current weather for a city."""
    return {"city": city, "temp_c": 27, "sky": "açık"}


def send_message(recipient: str, message: str) -> Dict[str, Any]:
    """Send a text message to a recipient."""
    return {"recipient": recipient, "message": message, "status": "iletildi"}


def set_alarm(time: str, label: str = "Alarm") -> Dict[str, Any]:
    """Set an alarm for a time."""
    return {"time": time, "label": label, "status": "kuruldu"}


def calculate(expression: str) -> Dict[str, Any]:
    """Calculate a mathematical expression."""
    expression = re.sub(r"\s*(?:hesapla|hesaplayabilir misin|kaç eder)\s*[?.!]*$", "", expression.strip(), flags=re.IGNORECASE)
    try:
        result = eval(expression, {"__builtins__": None}, {})
    except Exception:
        result = None
    return {"expression": expression, "result": result}


ACTIVE_TOOLS: List[Callable[..., Any]] = [
    get_weather,
    send_message,
    set_alarm,
    calculate,
]


def get_needle_tools_schema(tools: Optional[List[Callable[..., Any]]] = None) -> List[Dict[str, Any]]:
    """Needle 3 modelinin doğrudan kabul ettiği araç şemalarını üretir."""
    target_tools = tools if tools is not None else ACTIVE_TOOLS
    try:
        import needle
        return [needle.build_schema(fn) for fn in target_tools]
    except Exception:
        # needle yüklü değilse fallback şema oluşturucu
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
        }
        schemas = []
        for fn in target_tools:
            sig = inspect.signature(fn)
            props: Dict[str, Any] = {}
            required: List[str] = []
            for name, param in sig.parameters.items():
                param_type = type_map.get(param.annotation, "string")
                props[name] = {"type": param_type}
                if param.default is inspect.Parameter.empty:
                    required.append(name)
            schemas.append({
                "name": fn.__name__,
                "description": (fn.__doc__ or "").strip(),
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": required,
                },
            })
        return schemas


def get_tools_schema(
    tools: Optional[List[Callable[..., Any]]] = None,
    format: str = "needle",
) -> List[Dict[str, Any]]:
    """Tüm aktif araçların şemalarını üretir (Needle veya OpenAI formatında)."""
    needle_schemas = get_needle_tools_schema(tools)
    if format == "openai":
        return [
            {
                "type": "function",
                "function": s,
            }
            for s in needle_schemas
        ]
    return needle_schemas
