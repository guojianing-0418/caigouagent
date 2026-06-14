"""飞书群聊读取。

第一版通过 lark-cli 使用 user 身份读取当前用户可见的群聊。
如果未安装 lark-cli 或未授权，系统会记录日志并继续处理其他输入。
"""

from __future__ import annotations

import json
import subprocess
from typing import Any

from .models import LarkMessage, Question
from .question_engine import create_question


def search_chats(query: str) -> tuple[list[dict[str, Any]], str | None]:
    """按群名搜索飞书群聊。"""

    if not query:
        return [], None
    code, stdout, stderr = _run_lark(["im", "+chat-search", "--query", query, "--as", "user", "--json"])
    if code != 0:
        return [], stderr or stdout
    data = _safe_json(stdout)
    return _find_items(data, ["items", "chats", "data"]), None


def fetch_messages_by_chat(project_id: str, chat_ref: str) -> tuple[list[LarkMessage], list[Question], list[str]]:
    """读取某个群从第一条到最新的历史消息。

    chat_ref 可以是 chat_id，也可以是群名。群名匹配多个结果时返回选择问题。
    """

    logs: list[str] = []
    questions: list[Question] = []
    if not chat_ref:
        return [], questions, ["未指定飞书项目群，跳过群聊风险来源。"]

    chat_id = chat_ref.strip()
    if not chat_id.startswith("oc_"):
        chats, error = search_chats(chat_ref)
        if error:
            return [], questions, [f"飞书群搜索失败：{error}"]
        if not chats:
            return [], questions, [f"未搜索到飞书群：{chat_ref}"]
        if len(chats) > 1:
            options = []
            for chat in chats[:10]:
                name = chat.get("name") or chat.get("chat_name") or chat.get("title") or "未命名群"
                cid = chat.get("chat_id") or chat.get("open_chat_id") or chat.get("id") or ""
                options.append(f"{name} | {cid}")
            questions.append(
                create_question(
                    question_kind="lark_chat_selection",
                    input_type="single_select",
                    title="请选择飞书项目群",
                    message=f"群名“{chat_ref}”匹配到多个群聊，请选择本项目实际使用的群。",
                    reason="同一个关键词命中了多个飞书群，需要人工选择后才能继续抓取历史消息。",
                    options=options,
                    blocking=True,
                    required=True,
                    permission="ask",
                    context={"query": chat_ref, "source": "飞书群搜索"},
                    allow_custom=False,
                )
            )
            return [], questions, ["飞书群名匹配多个结果，等待人工选择。"]
        chat_id = chats[0].get("chat_id") or chats[0].get("open_chat_id") or chats[0].get("id") or chat_id

    # page-all 是 lark-cli im +chat-messages-list 的自动分页参数；如果 CLI 版本不支持，会在 stderr 中提示。
    args = ["im", "+chat-messages-list", "--chat-id", chat_id, "--as", "user", "--page-all", "--json"]
    code, stdout, stderr = _run_lark(args)
    if code != 0:
        return [], questions, [f"飞书消息抓取失败：{stderr or stdout}"]

    raw = _safe_json(stdout)
    items = _find_items(raw, ["items", "messages", "data"])
    messages = [_message_from_raw(project_id, chat_id, item) for item in items]
    logs.append(f"已读取飞书群 {chat_id} 历史消息 {len(messages)} 条。")
    return messages, questions, logs


def _run_lark(args: list[str]) -> tuple[int, str, str]:
    """运行 lark-cli，统一捕获错误。"""

    try:
        completed = subprocess.run(
            ["lark-cli", *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
        )
        return completed.returncode, completed.stdout, completed.stderr
    except FileNotFoundError:
        return 127, "", "未找到 lark-cli，请先安装并完成飞书授权。"
    except subprocess.TimeoutExpired:
        return 124, "", "lark-cli 执行超时。"


def _safe_json(text: str) -> Any:
    """宽松解析 JSON 输出。"""

    try:
        return json.loads(text)
    except Exception:
        return {}


def _find_items(data: Any, preferred_keys: list[str]) -> list[dict[str, Any]]:
    """从不同 lark-cli 输出结构中寻找列表数据。"""

    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if not isinstance(data, dict):
        return []
    for key in preferred_keys:
        value = data.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            nested = _find_items(value, preferred_keys)
            if nested:
                return nested
    for value in data.values():
        nested = _find_items(value, preferred_keys)
        if nested:
            return nested
    return []


def _message_from_raw(project_id: str, chat_id: str, raw: dict[str, Any]) -> LarkMessage:
    """把 lark-cli 原始消息转换成系统内部消息结构。"""

    content = raw.get("content") or raw.get("text") or raw.get("body") or ""
    if isinstance(content, dict):
        content = json.dumps(content, ensure_ascii=False)
    sender = raw.get("sender_name") or raw.get("sender") or raw.get("sender_id") or ""
    create_time = raw.get("create_time") or raw.get("created_at") or raw.get("time") or ""
    return LarkMessage(
        chat_id=chat_id,
        message_id=raw.get("message_id") or raw.get("id") or f"{project_id}-{len(json.dumps(raw, ensure_ascii=False))}",
        sender=str(sender),
        create_time=str(create_time),
        content=str(content),
        raw=raw,
    )
