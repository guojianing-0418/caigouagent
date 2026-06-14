"""应用配置。

本文件只读取环境变量，不在代码里写死任何公司模型地址或密钥。
IT 部署时只需要复制 `.env.example`，并在启动前设置对应环境变量即可。
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass
class Settings:
    """后端运行配置。

    这里没有使用复杂配置框架，方便后续维护人员直接阅读和修改。
    """

    openai_base_url: str | None
    openai_api_key: str | None
    text_model: str
    vision_model: str
    data_dir: Path


def load_settings() -> Settings:
    """从环境变量读取配置，并给出本地开发默认值。"""

    data_dir = Path(os.getenv("DATA_DIR", "data")).resolve()
    return Settings(
        openai_base_url=os.getenv("OPENAI_BASE_URL"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        text_model=os.getenv("TEXT_MODEL", "gpt-4.1-mini"),
        vision_model=os.getenv("VISION_MODEL", os.getenv("TEXT_MODEL", "gpt-4.1-mini")),
        data_dir=data_dir,
    )


settings = load_settings()


def ensure_data_dirs() -> None:
    """确保本地单机运行需要的目录都存在。"""

    for name in ["uploads", "exports", "cache", "projects", "templates", "config"]:
        (settings.data_dir / name).mkdir(parents=True, exist_ok=True)


def model_config_path() -> Path:
    """返回 WebUI 保存的模型配置文件路径。"""

    ensure_data_dirs()
    return settings.data_dir / "config" / "model_settings.json"


def load_saved_model_config() -> dict:
    """读取 WebUI 保存的模型配置。

    本地单机部署时，配置文件保存在 data/config 下，不进入代码仓库。
    """

    path = model_config_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_model_config(
    *,
    openai_base_url: str | None,
    openai_api_key: str | None,
    text_model: str | None,
    vision_model: str | None,
    keep_existing_key: bool = True,
) -> Settings:
    """保存 WebUI 提交的模型配置。

    如果 keep_existing_key=True 且 api_key 为空，则沿用旧密钥，避免前端必须重复输入。
    """

    current = load_saved_model_config()
    api_key = openai_api_key
    if keep_existing_key and not api_key:
        api_key = current.get("openai_api_key") or settings.openai_api_key

    payload = {
        "openai_base_url": openai_base_url or "",
        "openai_api_key": api_key or "",
        "text_model": text_model or "",
        "vision_model": vision_model or "",
    }
    model_config_path().write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return get_effective_settings()


def get_effective_settings() -> Settings:
    """获取最终生效配置。

    WebUI 保存的配置优先级高于环境变量；没有保存时使用环境变量。
    """

    saved = load_saved_model_config()
    return replace(
        settings,
        openai_base_url=saved.get("openai_base_url") or settings.openai_base_url,
        openai_api_key=saved.get("openai_api_key") or settings.openai_api_key,
        text_model=saved.get("text_model") or settings.text_model,
        vision_model=saved.get("vision_model") or settings.vision_model,
    )


def mask_api_key(api_key: str | None) -> str:
    """返回密钥掩码，避免接口把 API key 原样回显给前端。"""

    if not api_key:
        return ""
    if len(api_key) <= 8:
        return "*" * len(api_key)
    return f"{api_key[:4]}****{api_key[-4:]}"
