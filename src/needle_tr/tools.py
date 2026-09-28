"""Araç (Tool) tanımları ve şema yardımcıları."""

import inspect
import json
from typing import Any, Callable, Dict, List


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


def get_tools_schema() -> List[Dict[str, Any]]:
    """Tüm aktif araçların OpenAI/Ollama uyumlu JSON şemalarını üretir."""
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
    }
    schemas = []
    for fn in ACTIVE_TOOLS:
        sig = inspect.signature(fn)
        props: Dict[str, Any] = {}
        required: List[str] = []

        for name, param in sig.parameters.items():
            param_type = type_map.get(param.annotation, "string")
            props[name] = {
                "type": param_type,
                "description": f"{name} parametresi",
            }
            if param.default is inspect.Parameter.empty:
                required.append(name)

        schemas.append({
            "type": "function",
            "function": {
                "name": fn.__name__,
                "description": (fn.__doc__ or "").strip(),
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": required,
                },
            },
        })
    return schemas
