"""OpenAI 兼容模型客户端。

当前业务要求：风险识别必须依赖大模型。
因此没有配置模型或模型不可连通时，Agent 不能产出风险结论。
"""

from __future__ import annotations

import base64
import json
import re
from typing import Any

from .config import get_effective_settings


def _client():
    """创建 OpenAI 兼容客户端。失败时返回 None。"""

    runtime_settings = get_effective_settings()
    if not runtime_settings.openai_api_key or not runtime_settings.openai_base_url:
        return None
    try:
        from openai import OpenAI
    except Exception:
        return None
    return OpenAI(api_key=runtime_settings.openai_api_key, base_url=runtime_settings.openai_base_url)


def is_model_configured() -> bool:
    """判断文本模型所需配置是否齐全。"""

    runtime_settings = get_effective_settings()
    return bool(runtime_settings.openai_base_url and runtime_settings.openai_api_key and runtime_settings.text_model)


def require_model_ready() -> None:
    """确认模型已配置且可调用。

    每次开始识别前执行一次，避免未连接模型时产生任何风险识别结果。
    """

    if not is_model_configured():
        raise RuntimeError("请先在模型配置中填写 Base URL、API Key 和文本模型；未连接大模型时不能识别风险物料。")
    text = call_text("请回复 OK，用于确认模型连通性。", temperature=0)
    if not text:
        raise RuntimeError("大模型未返回结果，请检查 Base URL、API Key 和模型名称。")
    if text.startswith("模型调用失败:"):
        raise RuntimeError(text)


def call_text(prompt: str, temperature: float = 0.1) -> str | None:
    """调用文本模型，返回纯文本。"""

    client = _client()
    if client is None:
        return None
    runtime_settings = get_effective_settings()
    try:
        response = client.chat.completions.create(
            model=runtime_settings.text_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        return f"模型调用失败: {exc}"


def call_json(prompt: str) -> Any | None:
    """调用文本模型并尝试解析 JSON。"""

    text = call_text(prompt)
    if not text:
        return None
    if text.startswith("模型调用失败:"):
        raise RuntimeError(text)
    json_text = _extract_json_text(text)
    try:
        return json.loads(json_text)
    except Exception:
        return None


def call_json_structured(prompt: str, schema: dict[str, Any], schema_name: str) -> Any | None:
    """优先使用 OpenAI 兼容 Structured Outputs 调用文本模型。

    很多公司内网模型只兼容基础 chat.completions，不支持 response_format=json_schema。
    因此这里失败时返回 None，由调用方回退到普通 JSON prompt。
    """

    client = _client()
    if client is None:
        return None
    runtime_settings = get_effective_settings()
    try:
        response = client.chat.completions.create(
            model=runtime_settings.text_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                },
            },
        )
        text = response.choices[0].message.content or ""
        return json.loads(_extract_json_text(text))
    except Exception:
        return None


def call_json_required(prompt: str, schema: dict[str, Any] | None = None, schema_name: str = "structured_output") -> Any:
    """调用文本模型并要求返回可解析 JSON。

    如果传入 JSON Schema，先尝试 Structured Outputs；不支持时自动回退。
    """

    if schema:
        structured_data = call_json_structured(prompt, schema, schema_name)
        if structured_data is not None:
            return structured_data

    data = call_json(prompt)
    if data is None:
        raise RuntimeError("大模型没有返回合法 JSON，无法生成风险物料结果。请检查模型能力或提示词兼容性。")
    return data


def _extract_json_text(text: str) -> str:
    """从模型回复中提取 JSON。

    兼容 ```json 代码块和前后带解释文字的情况。
    """

    match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
    if match:
        return match.group(1).strip()
    start_positions = [pos for pos in [text.find("["), text.find("{")] if pos >= 0]
    if not start_positions:
        return text.strip()
    start = min(start_positions)
    end = max(text.rfind("]"), text.rfind("}"))
    if end > start:
        return text[start : end + 1].strip()
    return text.strip()


def call_vision(prompt: str, image_bytes: bytes, mime_type: str = "image/png") -> str | None:
    """调用视觉模型识别图纸页面。

    使用 chat.completions 的常见 OpenAI 兼容格式，便于替换公司内网模型。
    """

    client = _client()
    if client is None:
        return None
    runtime_settings = get_effective_settings()
    data_url = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
    try:
        response = client.chat.completions.create(
            model=runtime_settings.vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        return f"视觉模型调用失败: {exc}"
