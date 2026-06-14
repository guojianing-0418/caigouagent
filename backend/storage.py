"""本地单机存储。

第一版不做用户体系和权限隔离，项目状态存 JSON，飞书消息缓存存 SQLite。
这样既容易调试，也能避免重复抓取大量历史消息。
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .config import ensure_data_dirs, settings
from .models import LarkMessage, ProjectState, ProjectSummary


def _json_default(value):
    """给 json.dump 使用的兜底序列化函数。"""

    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"不支持序列化: {type(value)!r}")


def project_dir(project_id: str) -> Path:
    """返回单个项目的本地目录。"""

    ensure_data_dirs()
    path = settings.data_dir / "projects" / project_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def project_state_path(project_id: str) -> Path:
    """返回项目状态 JSON 路径。"""

    return project_dir(project_id) / "state.json"


def save_project(state: ProjectState) -> None:
    """保存项目状态。"""

    state.updated_at = datetime.now().isoformat(timespec="seconds")
    path = project_state_path(state.id)
    with path.open("w", encoding="utf-8") as f:
        json.dump(state.model_dump(), f, ensure_ascii=False, indent=2, default=_json_default)


def load_project(project_id: str) -> ProjectState:
    """读取项目状态。"""

    path = project_state_path(project_id)
    with path.open("r", encoding="utf-8") as f:
        return ProjectState.model_validate(json.load(f))


def list_projects() -> list[ProjectSummary]:
    """读取本机历史项目列表。

    只返回摘要，避免历史列表接口一次性返回过大的日志和证据内容。
    """

    ensure_data_dirs()
    summaries: list[ProjectSummary] = []
    projects_root = settings.data_dir / "projects"
    for path in projects_root.glob("*/state.json"):
        try:
            state = ProjectState.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
        pending_questions = [q for q in state.questions if q.status == "pending"]
        pending_required_questions = [q for q in pending_questions if q.required]
        summaries.append(
            ProjectSummary(
                id=state.id,
                project_name=state.config.project_name,
                status=state.status,
                current_step=state.current_step,
                created_at=state.created_at,
                updated_at=state.updated_at,
                risk_count=len(state.risks),
                question_count=len(pending_questions),
                required_question_count=len(pending_required_questions),
                export_path=state.export_path,
            )
        )
    return sorted(summaries, key=lambda item: item.updated_at, reverse=True)


def append_log(state: ProjectState, message: str) -> None:
    """追加一条运行日志。"""

    stamp = datetime.now().strftime("%H:%M:%S")
    state.logs.append(f"[{stamp}] {message}")
    save_project(state)


def db_path() -> Path:
    """返回 SQLite 数据库路径。"""

    ensure_data_dirs()
    return settings.data_dir / "cache" / "risk_agent.sqlite3"


def init_db() -> None:
    """初始化 SQLite 表结构。"""

    with sqlite3.connect(db_path()) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS lark_messages (
                project_id TEXT NOT NULL,
                chat_id TEXT NOT NULL,
                message_id TEXT NOT NULL,
                sender TEXT,
                create_time TEXT,
                content TEXT,
                raw_json TEXT,
                PRIMARY KEY (project_id, message_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_cache (
                project_id TEXT NOT NULL,
                source_name TEXT NOT NULL,
                chunk_hash TEXT NOT NULL,
                model_name TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                raw_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (project_id, source_name, chunk_hash, model_name, prompt_version)
            )
            """
        )
        conn.commit()


def save_lark_messages(project_id: str, messages: Iterable[LarkMessage]) -> None:
    """缓存飞书消息，重复 message_id 会自动覆盖。"""

    init_db()
    with sqlite3.connect(db_path()) as conn:
        for msg in messages:
            conn.execute(
                """
                INSERT OR REPLACE INTO lark_messages
                (project_id, chat_id, message_id, sender, create_time, content, raw_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    msg.chat_id,
                    msg.message_id,
                    msg.sender,
                    msg.create_time,
                    msg.content,
                    json.dumps(msg.raw, ensure_ascii=False),
                ),
            )
        conn.commit()


def load_lark_messages(project_id: str) -> list[LarkMessage]:
    """读取某项目缓存过的飞书消息。"""

    init_db()
    with sqlite3.connect(db_path()) as conn:
        rows = conn.execute(
            """
            SELECT chat_id, message_id, sender, create_time, content, raw_json
            FROM lark_messages
            WHERE project_id = ?
            ORDER BY create_time ASC
            """,
            (project_id,),
        ).fetchall()
    messages: list[LarkMessage] = []
    for chat_id, message_id, sender, create_time, content, raw_json in rows:
        raw = json.loads(raw_json) if raw_json else {}
        messages.append(
            LarkMessage(
                chat_id=chat_id,
                message_id=message_id,
                sender=sender or "",
                create_time=create_time or "",
                content=content or "",
                raw=raw,
            )
        )
    return messages


def load_llm_cache(
    *,
    project_id: str,
    source_name: str,
    chunk_hash: str,
    model_name: str,
    prompt_version: str,
) -> object | None:
    """读取某个来源分块的大模型原始 JSON 输出缓存。"""

    init_db()
    with sqlite3.connect(db_path()) as conn:
        row = conn.execute(
            """
            SELECT raw_json
            FROM llm_cache
            WHERE project_id = ?
              AND source_name = ?
              AND chunk_hash = ?
              AND model_name = ?
              AND prompt_version = ?
            """,
            (project_id, source_name, chunk_hash, model_name, prompt_version),
        ).fetchone()
    if not row:
        return None
    try:
        return json.loads(row[0])
    except Exception:
        return None


def save_llm_cache(
    *,
    project_id: str,
    source_name: str,
    chunk_hash: str,
    model_name: str,
    prompt_version: str,
    raw_json: object,
) -> None:
    """保存某个来源分块的大模型原始 JSON 输出缓存。"""

    init_db()
    with sqlite3.connect(db_path()) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO llm_cache
            (project_id, source_name, chunk_hash, model_name, prompt_version, raw_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                project_id,
                source_name,
                chunk_hash,
                model_name,
                prompt_version,
                json.dumps(raw_json, ensure_ascii=False),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        conn.commit()
