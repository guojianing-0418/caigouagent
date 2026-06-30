"""飞书群聊读取。

第一版通过 lark-cli 使用 user 身份读取当前用户可见的群聊。
如果未安装 lark-cli 或未授权，系统会记录日志并继续处理其他输入。
"""

from __future__ import annotations

import html
import json
import mimetypes
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

from .llm_client import call_vision
from .models import LarkMessage, Question
from .parsers.document_ingestor import ingest_document
from .question_engine import create_question
from .storage import project_dir


LARK_MESSAGE_PAGE_SIZE = 50
LARK_MESSAGE_MAX_PAGES = 20
LARK_LINK_MAX_PER_PROJECT = 40
LARK_RESOURCE_MAX_PER_PROJECT = 80
LARK_ENRICHMENT_MAX_CHARS_PER_ITEM = 4000
LARK_ENRICHMENT_MAX_CHARS_PER_MESSAGE = 16000
LARK_SHEET_MAX_SHEETS = 5
LARK_SHEET_MAX_ROWS_PER_SHEET = 80
LARK_SHEET_MAX_COLUMNS_PER_SHEET = 40
LARK_CLI_NOT_FOUND_MESSAGE = (
    "未找到 lark-cli。后端启动环境未能在 LARK_CLI_PATH、PATH 或 Windows npm 全局目录中找到 "
    "lark-cli；这通常是启动后端的终端没有继承 npm 全局路径，不代表飞书授权失败。"
)
LARK_URL_PATTERN = re.compile(r"https?://[^\s<>'\"）)】\]}]+", flags=re.I)
LARK_DOC_PATH_PATTERN = re.compile(
    r"/(?P<kind>docx|doc|wiki|sheets|base|bitable|slides|drive/file)/(?P<token>[A-Za-z0-9_\-]+)",
    flags=re.I,
)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DOCUMENT_SUFFIXES = {".xlsx", ".xlsm", ".pdf", ".txt", ".md", ".csv"}
HTML_SUFFIXES = {".html", ".htm"}
PREVIEW_TYPE_PRIORITY = ["text", "html", "pdf", "image"]
LARK_IMAGE_PROMPT = """你是 IPD 计划阶段采购风险识别助手。
请读取这张飞书群聊或文档中的图片，重点提取与采购风险相关的信息：
1. 物料、规格、供应商、交期、成本、认证、工艺、质量或验证要求；
2. 表格、截图、周报、会议纪要中的风险描述和待确认事项；
3. 如果只是无关图片，请简短说明未发现采购风险线索。
请用简短中文要点输出，不要编造看不到的信息。"""


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

    resource_dir = _lark_resource_dir(project_id)
    items, message_logs, error = _fetch_chat_message_pages(chat_id, resource_dir)
    logs.extend(message_logs)
    if error:
        return [], questions, [f"飞书消息抓取失败：{error}"]

    enrichment_stats = _new_enrichment_stats()
    messages = [_message_from_raw(project_id, chat_id, item, resource_dir, logs, enrichment_stats) for item in items]
    logs.append(f"已读取飞书群 {chat_id} 历史消息 {len(messages)} 条。")
    if _has_enrichment(enrichment_stats):
        logs.append(
            "已展开飞书群关联资料："
            f"链接 {enrichment_stats['links']} 个，"
            f"正文读取 {enrichment_stats['read_success']} 个，"
            f"预览读取 {enrichment_stats['preview_success']} 个，"
            f"可读附件 {enrichment_stats['resources']} 个，"
            f"图片识别 {enrichment_stats['images']} 张，"
            f"无预览跳过 {enrichment_stats['skipped_no_readable_preview']} 个，"
            f"权限不足未申请 {enrichment_stats['permission_denied_no_request']} 个，"
            f"其他失败 {enrichment_stats['failures']} 个。"
        )
    return messages, questions, logs


def _fetch_chat_message_pages(chat_id: str, resource_dir: Path) -> tuple[list[dict[str, Any]], list[str], str | None]:
    """按 page_token 手动分页读取群消息。"""

    items: list[dict[str, Any]] = []
    logs: list[str] = []
    page_token = ""
    for page_index in range(LARK_MESSAGE_MAX_PAGES):
        args = [
            "im",
            "+chat-messages-list",
            "--chat-id",
            chat_id,
            "--as",
            "user",
            "--page-size",
            str(LARK_MESSAGE_PAGE_SIZE),
            "--order",
            "asc",
            "--no-reactions",
            "--json",
        ]
        if page_token:
            args.extend(["--page-token", page_token])

        code, stdout, stderr = _run_lark(args, cwd=resource_dir)
        if code != 0:
            return items, logs, stderr or stdout

        raw = _safe_json(stdout)
        data = _extract_data(raw)
        page_items = _find_items(data, ["messages", "items", "data"])
        items.extend(page_items)
        has_more = bool(data.get("has_more")) if isinstance(data, dict) else False
        page_token = str(data.get("page_token") or "") if isinstance(data, dict) else ""
        if not has_more or not page_token:
            return items, logs, None

    logs.append(f"飞书群消息超过 {LARK_MESSAGE_MAX_PAGES * LARK_MESSAGE_PAGE_SIZE} 条，本次仅读取前 {len(items)} 条。")
    return items, logs, None


def _run_lark(args: list[str], cwd: Path | None = None) -> tuple[int, str, str]:
    """运行 lark-cli，统一捕获错误。"""

    command = _resolve_lark_cli_command()
    if not command:
        return 127, "", LARK_CLI_NOT_FOUND_MESSAGE

    try:
        completed = subprocess.run(
            [*command, *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(cwd) if cwd else None,
            timeout=180,
        )
        return completed.returncode, completed.stdout, completed.stderr
    except FileNotFoundError:
        return 127, "", LARK_CLI_NOT_FOUND_MESSAGE
    except subprocess.TimeoutExpired:
        return 124, "", "lark-cli 执行超时。"


def _resolve_lark_cli_command() -> list[str] | None:
    """Resolve lark-cli across PATH, explicit config, and npm global installs."""

    for candidate in _lark_cli_candidates():
        path = _normalize_lark_cli_candidate(candidate)
        if not path:
            continue
        if path.suffix.lower() == ".ps1":
            return ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(path)]
        return [str(path)]
    return None


def _lark_cli_candidates() -> list[str]:
    candidates: list[str] = []
    env_path = os.environ.get("LARK_CLI_PATH")
    if env_path:
        candidates.append(env_path)

    for name in ("lark-cli", "lark-cli.cmd", "lark-cli.exe", "lark-cli.ps1"):
        found = shutil.which(name)
        if found:
            candidates.append(found)

    appdata = os.environ.get("APPDATA")
    if appdata:
        npm_dir = Path(appdata) / "npm"
        candidates.extend(str(npm_dir / name) for name in ("lark-cli.cmd", "lark-cli.exe", "lark-cli.ps1", "lark-cli"))

    return candidates


def _normalize_lark_cli_candidate(candidate: str) -> Path | None:
    path = Path(candidate).expanduser()
    if path.is_file():
        return path

    resolved = shutil.which(candidate)
    if resolved:
        resolved_path = Path(resolved)
        if resolved_path.is_file():
            return resolved_path

    return None


def _safe_json(text: str) -> Any:
    """宽松解析 JSON 输出。"""

    try:
        return json.loads(text)
    except Exception:
        return {}


def _extract_data(data: Any) -> Any:
    """兼容 lark-cli 的顶层 envelope。"""

    if isinstance(data, dict) and isinstance(data.get("data"), dict):
        return data["data"]
    return data


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


def _message_from_raw(
    project_id: str,
    chat_id: str,
    raw: dict[str, Any],
    resource_dir: Path,
    logs: list[str],
    stats: dict[str, int],
) -> LarkMessage:
    """把 lark-cli 原始消息转换成系统内部消息结构。"""

    content = _message_content_text(raw)
    enrichment = _enrichment_from_raw_message(raw, resource_dir, logs, stats)
    if enrichment:
        content = _clip_text(
            f"{content}\n\n[飞书关联资料]\n{enrichment}" if content else f"[飞书关联资料]\n{enrichment}",
            LARK_ENRICHMENT_MAX_CHARS_PER_MESSAGE,
        )
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


def _message_content_text(raw: dict[str, Any]) -> str:
    """提取消息正文；复杂结构保留为 JSON 文本，便于后续模型理解。"""

    content = raw.get("content") or raw.get("text") or raw.get("body") or ""
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def _enrichment_from_raw_message(
    raw: dict[str, Any],
    resource_dir: Path,
    logs: list[str],
    stats: dict[str, int],
    depth: int = 0,
) -> str:
    """展开消息中的附件、图片、飞书链接和线程回复。"""

    sections: list[str] = []
    seen_urls: set[str] = set()

    for resource in _resources_from_raw(raw):
        if stats["resources"] >= LARK_RESOURCE_MAX_PER_PROJECT:
            stats["failures"] += 1
            break
        text = _text_from_message_resource(resource, resource_dir, logs, stats)
        if text:
            sections.append(text)

    for url in _extract_lark_urls_from_raw(raw):
        if url in seen_urls:
            continue
        seen_urls.add(url)
        if stats["links"] >= LARK_LINK_MAX_PER_PROJECT:
            stats["failures"] += 1
            break
        text = _text_from_lark_url(url, resource_dir, logs, stats)
        if text:
            sections.append(text)

    if depth < 2:
        for reply in raw.get("thread_replies") or []:
            if not isinstance(reply, dict):
                continue
            reply_text = _message_content_text(reply)
            reply_enrichment = _enrichment_from_raw_message(reply, resource_dir, logs, stats, depth=depth + 1)
            if reply_text or reply_enrichment:
                sender = reply.get("sender_name") or reply.get("sender") or reply.get("sender_id") or "线程回复"
                sections.append(_clip_text(f"线程回复/{sender}: {reply_text}\n{reply_enrichment}".strip(), 2000))

    return "\n\n".join(section for section in sections if section)


def _resources_from_raw(raw: dict[str, Any]) -> list[dict[str, Any]]:
    resources = raw.get("resources")
    if not isinstance(resources, list):
        return []
    return [resource for resource in resources if isinstance(resource, dict)]


def _text_from_message_resource(
    resource: dict[str, Any],
    resource_dir: Path,
    logs: list[str],
    stats: dict[str, int],
) -> str:
    """只处理历史缓存中已经存在的资源块；新消息不主动下载附件原文件。"""

    if resource.get("error"):
        stats["failures"] += 1
        return ""
    local_path = str(resource.get("local_path") or "").strip()
    if not local_path:
        stats["failures"] += 1
        logs.append("飞书群消息包含附件资源，但当前只读策略不下载原文件；已保留消息文本和链接继续分析。")
        return ""
    path = _resolve_lark_local_path(local_path, resource_dir)
    if not path.exists():
        stats["failures"] += 1
        logs.append(f"skipped_no_readable_preview: 飞书群附件无可读预览或本地缓存不存在，已跳过：{local_path}")
        return ""

    stats["resources"] += 1
    label = f"飞书群附件/{path.name}"
    text = _text_from_preview_file(path, label, logs, stats)
    if not text:
        stats["failures"] += 1
    return text


def _text_from_lark_url(url: str, resource_dir: Path, logs: list[str], stats: dict[str, int]) -> str:
    """按飞书 URL 类型读取在线文档、表格、云盘文件或幻灯片。"""

    stats["links"] += 1
    inspected = _inspect_lark_url(url, resource_dir)
    if not inspected:
        inspected = _guess_lark_url(url)
    if not inspected:
        stats["failures"] += 1
        logs.append(f"无法识别飞书链接类型，已跳过：{url}")
        return ""

    doc_type = str(inspected.get("type") or "").lower()
    token = str(inspected.get("token") or "").strip()
    title = str(inspected.get("title") or "").strip() or token or "飞书链接"
    try:
        if doc_type in {"doc", "docx", "wiki"}:
            return _text_from_lark_doc(url, title, resource_dir, logs, stats)
        if doc_type == "sheet":
            return _text_from_lark_sheet(url, token, title, resource_dir, logs, stats)
        if doc_type == "file":
            return _text_from_drive_preview(token, title, resource_dir, logs, stats)
        if doc_type == "slides":
            return _text_from_lark_slides(token, title, resource_dir, logs, stats)
        if doc_type == "bitable":
            logs.append(f"飞书多维表格链接暂未自动展开，已记录链接标题：{title}")
            return f"飞书多维表格链接：{title} {url}"
    except Exception as exc:
        stats["failures"] += 1
        logs.append(f"飞书链接展开失败：{title}，原因：{exc}")
        return ""

    stats["failures"] += 1
    logs.append(f"暂不支持自动展开的飞书链接类型：{doc_type or '未知'}，标题：{title}")
    return ""


def _inspect_lark_url(url: str, cwd: Path) -> dict[str, Any] | None:
    code, stdout, stderr = _run_lark(["drive", "+inspect", "--url", url, "--as", "user", "--json"], cwd=cwd)
    if code != 0:
        return None
    raw = _safe_json(stdout)
    if _is_lark_error(raw):
        return None
    data = _extract_data(raw)
    return data if isinstance(data, dict) else None


def _guess_lark_url(url: str) -> dict[str, str] | None:
    match = LARK_DOC_PATH_PATTERN.search(url)
    if not match:
        return None
    kind = match.group("kind").lower()
    token = match.group("token")
    mapping = {
        "doc": "doc",
        "docx": "docx",
        "wiki": "wiki",
        "sheets": "sheet",
        "base": "bitable",
        "bitable": "bitable",
        "slides": "slides",
        "drive/file": "file",
    }
    return {"type": mapping.get(kind, kind), "token": token, "title": token, "url": url}


def _text_from_lark_doc(
    url: str,
    title: str,
    resource_dir: Path,
    logs: list[str],
    stats: dict[str, int],
) -> str:
    code, stdout, stderr = _run_lark(
        ["docs", "+fetch", "--api-version", "v2", "--doc", url, "--doc-format", "xml", "--as", "user", "--json"],
        cwd=resource_dir,
    )
    if code != 0:
        _record_lark_error("飞书文档读取", title, stdout, stderr, logs, stats)
        return ""
    raw = _safe_json(stdout)
    if _is_lark_error(raw):
        _record_lark_error("飞书文档读取", title, stdout, stderr, logs, stats)
        return ""

    content = _find_first_value(raw, "content")
    if not content:
        _record_skipped_preview("飞书文档读取", title, "未读取到正文", logs, stats)
        return f"飞书文档/{title}: 未读取到正文。"
    xml_text = str(content)
    stats["read_success"] += 1
    logs.append(f"read_success: 飞书文档读取成功：{title}")
    sections = [f"飞书文档/{title}:\n{_clip_text(_xml_to_text(xml_text), LARK_ENRICHMENT_MAX_CHARS_PER_ITEM)}"]

    for index, media in enumerate(_media_refs_from_doc_xml(xml_text), start=1):
        if stats["resources"] >= LARK_RESOURCE_MAX_PER_PROJECT:
            break
        media_text = _text_from_doc_media(media, title, index, resource_dir, logs, stats)
        if media_text:
            sections.append(media_text)

    for sheet_token in _sheet_tokens_from_doc_xml(xml_text):
        if stats["links"] >= LARK_LINK_MAX_PER_PROJECT:
            break
        stats["links"] += 1
        sheet_text = _text_from_lark_sheet("", sheet_token, f"{title}-嵌入表格", resource_dir, logs, stats)
        if sheet_text:
            sections.append(sheet_text)

    return "\n\n".join(sections)


def _text_from_doc_media(
    media: dict[str, str],
    title: str,
    index: int,
    resource_dir: Path,
    logs: list[str],
    stats: dict[str, int],
) -> str:
    token = media.get("token", "")
    if not token:
        return ""
    if media.get("type") == "whiteboard":
        _record_skipped_preview("飞书文档素材预览", title, "画板素材不支持预览读取，且当前策略不下载原文件", logs, stats)
        return ""
    media_dir = resource_dir / "doc_media"
    media_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_file_stem(f"{title}-{index}-{media.get('name') or media.get('type') or 'media'}")
    output_name = str(Path("doc_media") / stem)
    code, stdout, stderr = _run_lark(
        ["docs", "+media-preview", "--token", token, "--output", output_name, "--overwrite", "--as", "user", "--json"],
        cwd=resource_dir,
    )
    if code != 0:
        _record_lark_error("飞书文档素材预览", f"{title} / {media.get('name') or token}", stdout, stderr, logs, stats)
        return ""

    raw = _safe_json(stdout)
    if _is_lark_error(raw):
        _record_lark_error("飞书文档素材预览", f"{title} / {media.get('name') or token}", stdout, stderr, logs, stats)
        return ""
    path = _path_from_lark_output(raw, resource_dir) or _latest_matching_file(media_dir, stem)
    if not path or not path.exists():
        _record_skipped_preview("飞书文档素材预览", title, f"预览成功但未找到本地预览文件：{media.get('name') or token}", logs, stats)
        return ""

    stats["resources"] += 1
    stats["preview_success"] += 1
    logs.append(f"preview_success: 飞书文档素材预览成功：{title} / {path.name}")
    return _text_from_preview_file(path, f"飞书文档素材/{title}/{path.name}", logs, stats)


def _text_from_lark_sheet(
    url: str,
    token: str,
    title: str,
    resource_dir: Path,
    logs: list[str],
    stats: dict[str, int],
) -> str:
    """通过只读表格 API 分 sheet 读取可见单元格内容。"""

    locator = _sheet_locator_args(url, token)
    if not locator:
        _record_skipped_preview("飞书表格读取", title, "缺少 spreadsheet token", logs, stats)
        return ""

    code, stdout, stderr = _run_lark(["sheets", "+workbook-info", *locator, "--as", "user", "--json"], cwd=resource_dir)
    if code != 0:
        _record_lark_error("飞书表格读取", title, stdout, stderr, logs, stats)
        return ""
    raw = _safe_json(stdout)
    if _is_lark_error(raw):
        _record_lark_error("飞书表格读取", title, stdout, stderr, logs, stats)
        return ""

    sheets = [sheet for sheet in _sheets_from_workbook(raw) if not _as_bool(sheet.get("is_hidden"))]
    if not sheets:
        _record_skipped_preview("飞书表格读取", title, "没有可见工作表", logs, stats)
        return ""

    sections: list[str] = []
    for sheet in sheets[:LARK_SHEET_MAX_SHEETS]:
        sheet_id = str(sheet.get("sheet_id") or sheet.get("id") or "").strip()
        sheet_name = str(sheet.get("title") or sheet.get("sheet_name") or sheet.get("name") or sheet_id or "Sheet").strip()
        sheet_locator = ["--sheet-id", sheet_id] if sheet_id else ["--sheet-name", sheet_name]
        row_count = _as_int(sheet.get("row_count") or sheet.get("rowCount"), LARK_SHEET_MAX_ROWS_PER_SHEET)
        col_count = _as_int(sheet.get("column_count") or sheet.get("col_count") or sheet.get("columnCount"), LARK_SHEET_MAX_COLUMNS_PER_SHEET)
        read_rows = max(1, min(row_count, LARK_SHEET_MAX_ROWS_PER_SHEET))
        read_cols = max(1, min(col_count, LARK_SHEET_MAX_COLUMNS_PER_SHEET))
        read_range = f"A1:{_excel_column_name(read_cols)}{read_rows}"
        csv_args = [
            "sheets",
            "+csv-get",
            *locator,
            *sheet_locator,
            "--range",
            read_range,
            "--skip-hidden",
            "true",
            "--as",
            "user",
            "--json",
        ]
        code, stdout, stderr = _run_lark(csv_args, cwd=resource_dir)
        if code != 0:
            _record_lark_error("飞书表格工作表读取", f"{title}/{sheet_name}", stdout, stderr, logs, stats)
            continue
        sheet_raw = _safe_json(stdout)
        if _is_lark_error(sheet_raw):
            _record_lark_error("飞书表格工作表读取", f"{title}/{sheet_name}", stdout, stderr, logs, stats)
            continue
        csv_text = str(_find_first_value(sheet_raw, "annotated_csv") or "").strip()
        if not csv_text:
            continue
        current_region = _find_first_value(sheet_raw, "current_region") or read_range
        sections.append(
            f"工作表 {sheet_name}（读取范围 {read_range}，当前区域 {current_region}）:\n"
            f"{_clip_text(csv_text, LARK_ENRICHMENT_MAX_CHARS_PER_ITEM)}"
        )

    if not sections:
        _record_skipped_preview("飞书表格读取", title, "未读取到可见单元格内容", logs, stats)
        return ""

    stats["read_success"] += 1
    logs.append(f"read_success: 飞书表格读取成功：{title}")
    omitted = ""
    if len(sheets) > LARK_SHEET_MAX_SHEETS:
        omitted = f"\n...（还有 {len(sheets) - LARK_SHEET_MAX_SHEETS} 个工作表未展开）"
    return f"飞书表格/{title}:\n" + "\n\n".join(sections) + omitted


def _text_from_lark_slides(
    token: str,
    title: str,
    resource_dir: Path,
    logs: list[str],
    stats: dict[str, int],
) -> str:
    """优先用 slides 只读 XML 接口读取 PPT 文本，失败后尝试 Drive 预览。"""

    if not token:
        _record_skipped_preview("飞书幻灯片读取", title, "缺少 slides token", logs, stats)
        return ""
    code, stdout, stderr = _run_lark(
        ["slides", "xml_presentations", "get", "--xml-presentation-id", token, "--revision-id", "-1", "--as", "user", "--json"],
        cwd=resource_dir,
    )
    raw = _safe_json(stdout)
    if code == 0 and not _is_lark_error(raw):
        content = str(_find_first_value(raw, "content") or "").strip()
        if content:
            stats["read_success"] += 1
            logs.append(f"read_success: 飞书幻灯片 XML 读取成功：{title}")
            return f"飞书幻灯片/{title}:\n{_clip_text(_xml_to_text(content), LARK_ENRICHMENT_MAX_CHARS_PER_ITEM)}"

    logs.append(f"飞书幻灯片 XML 读取未成功，尝试读取预览：{title}")
    preview_text = _text_from_drive_preview(token, title, resource_dir, logs, stats, source_label="飞书幻灯片预览")
    if preview_text:
        return preview_text
    if code != 0 or _is_lark_error(raw):
        _record_lark_error("飞书幻灯片读取", title, stdout, stderr, logs, stats)
    return ""


def _text_from_drive_preview(
    token: str,
    title: str,
    resource_dir: Path,
    logs: list[str],
    stats: dict[str, int],
    *,
    source_label: str = "飞书云盘文件预览",
) -> str:
    """读取 Drive 文件预览产物，不下载原始文件。"""

    if not token:
        _record_skipped_preview(source_label, title, "缺少 file token", logs, stats)
        return ""

    code, stdout, stderr = _run_lark(["drive", "+preview", "--file-token", token, "--list-only", "--as", "user", "--json"], cwd=resource_dir)
    if code != 0:
        _record_lark_error(source_label, title, stdout, stderr, logs, stats)
        return ""
    raw = _safe_json(stdout)
    if _is_lark_error(raw):
        _record_lark_error(source_label, title, stdout, stderr, logs, stats)
        return ""

    candidate = _choose_preview_candidate(_preview_candidates(raw))
    if not candidate:
        _record_skipped_preview(source_label, title, "没有 text/html/pdf/image 可读预览候选项", logs, stats)
        return ""

    preview_type = _preview_download_type(candidate)
    if not preview_type:
        _record_skipped_preview(source_label, title, "预览候选项不是可读取类型", logs, stats)
        return ""

    preview_dir = resource_dir / "drive_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_file_stem(f"{title}-{preview_type}")
    output_name = str(Path("drive_previews") / stem)
    code, stdout, stderr = _run_lark(
        [
            "drive",
            "+preview",
            "--file-token",
            token,
            "--type",
            preview_type,
            "--output",
            output_name,
            "--if-exists",
            "overwrite",
            "--as",
            "user",
            "--json",
        ],
        cwd=resource_dir,
    )
    if code != 0:
        _record_lark_error(source_label, title, stdout, stderr, logs, stats)
        return ""
    preview_raw = _safe_json(stdout)
    if _is_lark_error(preview_raw):
        _record_lark_error(source_label, title, stdout, stderr, logs, stats)
        return ""

    path = _path_from_lark_output(preview_raw, resource_dir) or _latest_matching_file(preview_dir, stem)
    if not path or not path.exists():
        _record_skipped_preview(source_label, title, "预览生成成功但未找到本地预览文件", logs, stats)
        return ""

    stats["resources"] += 1
    stats["preview_success"] += 1
    logs.append(f"preview_success: {source_label}读取成功：{title} / {path.name}")
    return _text_from_preview_file(path, f"{source_label}/{title}/{path.name}", logs, stats)


def _text_from_preview_file(path: Path, source_name: str, logs: list[str], stats: dict[str, int]) -> str:
    """把只读/预览产物转成风险识别可读文本。"""

    suffix = path.suffix.lower()
    if suffix in DOCUMENT_SUFFIXES:
        result = ingest_document(path, source_name=source_name, max_units=240)
        lines = result.lines(max_lines=120)
        if result.diagnostics.warnings:
            logs.append(f"{source_name} 解析提示：{'；'.join(result.diagnostics.warnings[:2])}")
        if not lines:
            return f"{source_name}: 预览文件已读取，但未抽取到可读文本。"
        return f"{source_name} 解析内容:\n{_clip_text(chr(10).join(lines), LARK_ENRICHMENT_MAX_CHARS_PER_ITEM)}"

    if suffix in HTML_SUFFIXES:
        text = _html_to_text(path.read_text(encoding="utf-8", errors="ignore"))
        if text:
            return f"{source_name} HTML预览内容:\n{_clip_text(text, LARK_ENRICHMENT_MAX_CHARS_PER_ITEM)}"
        return f"{source_name}: HTML 预览文件已读取，但未抽取到可读文本。"

    if suffix in IMAGE_SUFFIXES:
        mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
        vision_text = call_vision(LARK_IMAGE_PROMPT, path.read_bytes(), mime_type=mime_type)
        if vision_text and not vision_text.startswith("视觉模型调用失败:"):
            stats["images"] += 1
            return f"{source_name} 图片识别:\n{_clip_text(vision_text, LARK_ENRICHMENT_MAX_CHARS_PER_ITEM)}"
        logs.append(f"{source_name} 图片未能完成视觉识别。")
        return ""

    logs.append(f"{source_name} 预览格式暂不支持自动解析：{path.suffix or '无扩展名'}")
    return f"{source_name}: 预览文件格式 {path.suffix or '未知'} 暂未自动解析。"


def _extract_lark_urls_from_raw(raw: dict[str, Any]) -> list[str]:
    shallow = {key: value for key, value in raw.items() if key not in {"thread_replies", "resources", "reactions"}}
    text = " ".join([_message_content_text(raw), json.dumps(shallow, ensure_ascii=False)])
    urls: list[str] = []
    seen: set[str] = set()
    for match in LARK_URL_PATTERN.finditer(text):
        url = _clean_url(match.group(0))
        if not _is_supported_lark_url(url) or url in seen:
            continue
        seen.add(url)
        urls.append(url)
    return urls


def _clean_url(url: str) -> str:
    clean = html.unescape(str(url).replace("\\/", "/").replace("\\u0026", "&")).strip()
    return clean.rstrip(".,;:!?，。；：！？")


def _is_supported_lark_url(url: str) -> bool:
    lower = url.lower()
    if not any(domain in lower for domain in ["feishu.cn", "larksuite.com", "doubao.com"]):
        return False
    return bool(LARK_DOC_PATH_PATTERN.search(lower))


def _media_refs_from_doc_xml(content: str) -> list[dict[str, str]]:
    refs: list[dict[str, str]] = []
    for tag in re.findall(r"<(?:img|source|whiteboard)\b[^>]*>", content, flags=re.I):
        attrs = _xml_attrs(tag)
        token = attrs.get("token") or attrs.get("file-token") or attrs.get("id")
        if not token:
            continue
        media_type = "whiteboard" if tag.lower().startswith("<whiteboard") else "media"
        refs.append({"token": token, "name": attrs.get("name", ""), "type": media_type})
    return refs[:20]


def _sheet_tokens_from_doc_xml(content: str) -> list[str]:
    tokens: list[str] = []
    for tag in re.findall(r"<(?:sheet|cite)\b[^>]*>", content, flags=re.I):
        attrs = _xml_attrs(tag)
        file_type = attrs.get("file-type", "")
        token = attrs.get("token", "")
        if token and (tag.lower().startswith("<sheet") or file_type == "sheets"):
            tokens.append(token)
    return _dedupe_strings(tokens)[:5]


def _xml_attrs(tag: str) -> dict[str, str]:
    return {key: html.unescape(value) for key, value in re.findall(r"([\w:-]+)=\"([^\"]*)\"", tag)}


def _xml_to_text(content: str) -> str:
    text = re.sub(r"</?(?:title|h[1-6]|p|li|tr|table|blockquote|ul|ol)\b[^>]*>", "\n", content, flags=re.I)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def _html_to_text(content: str) -> str:
    text = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", content)
    text = re.sub(r"(?i)</?(?:p|div|li|tr|table|br|h[1-6]|section|article)\b[^>]*>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return "\n".join(" ".join(line.split()) for line in text.splitlines() if line.strip())


def _sheet_locator_args(url: str, token: str) -> list[str]:
    if token:
        return ["--spreadsheet-token", token]
    if url:
        return ["--url", url]
    return []


def _sheets_from_workbook(raw: Any) -> list[dict[str, Any]]:
    candidates = _find_items(_extract_data(raw), ["sheets", "items", "data"])
    return [item for item in candidates if item.get("sheet_id") or item.get("title") or item.get("sheet_name")]


def _excel_column_name(index: int) -> str:
    index = max(1, index)
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def _preview_candidates(raw: Any) -> list[dict[str, Any]]:
    return _find_items(_extract_data(raw), ["candidates", "items", "data"])


def _choose_preview_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    ready = [candidate for candidate in candidates if _preview_candidate_ready(candidate)]
    for preview_type in PREVIEW_TYPE_PRIORITY:
        for candidate in ready:
            if _preview_download_type(candidate) == preview_type:
                return candidate
    return None


def _preview_candidate_ready(candidate: dict[str, Any]) -> bool:
    status = str(candidate.get("status") or "").lower()
    downloadable = candidate.get("downloadable")
    if downloadable is False or str(downloadable).lower() == "false":
        return False
    return not status or status in {"ready", "ok", "success", "available"}


def _preview_download_type(candidate: dict[str, Any]) -> str:
    value = str(candidate.get("type") or "").lower()
    if value in {"txt", "text/plain"}:
        return "text"
    if value in {"htm", "html", "text/html"}:
        return "html"
    if value in {"pdf", "application/pdf"}:
        return "pdf"
    if value in {"png", "jpg", "jpeg", "image"} or value.startswith("image/"):
        return "image"
    if value in PREVIEW_TYPE_PRIORITY:
        return value
    return ""


def _path_from_lark_output(raw: Any, base_dir: Path) -> Path | None:
    for value in _walk_json_values(raw):
        if not isinstance(value, str):
            continue
        if not any(key in value.lower() for key in [".xlsx", ".xlsm", ".pdf", ".txt", ".md", ".csv", ".png", ".jpg", ".jpeg", ".webp", ".html", ".htm", ".pptx"]):
            continue
        path = _resolve_lark_local_path(value, base_dir)
        if path.exists():
            return path
    return None


def _walk_json_values(value: Any):
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk_json_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_json_values(item)
    else:
        yield value


def _resolve_lark_local_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return base_dir / path


def _latest_matching_file(folder: Path, stem: str) -> Path | None:
    matches = sorted(folder.glob(f"{stem}*"), key=lambda item: item.stat().st_mtime if item.exists() else 0, reverse=True)
    return matches[0] if matches else None


def _find_first_value(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for item in value.values():
            found = _find_first_value(item, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_first_value(item, key)
            if found is not None:
                return found
    return None


def _is_lark_error(raw: Any) -> bool:
    return isinstance(raw, dict) and raw.get("ok") is False


def _lark_error_message(raw: Any) -> str:
    if not isinstance(raw, dict):
        return ""
    error = raw.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or error.get("type") or error)
    return str(error or raw)


def _record_lark_error(
    action: str,
    title: str,
    stdout: str,
    stderr: str,
    logs: list[str],
    stats: dict[str, int],
) -> None:
    raw = _safe_json(stdout) or _safe_json(stderr)
    message = _lark_error_message(raw) if raw else (stderr or stdout or "未知错误")
    if _looks_permission_denied(message):
        stats["permission_denied_no_request"] += 1
        logs.append(f"permission_denied_no_request: {action}失败：{title}。仅阅读/预览策略不申请权限，原因：{message}")
    elif _looks_preview_unavailable(message):
        _record_skipped_preview(action, title, message, logs, stats)
    else:
        stats["failures"] += 1
        logs.append(f"{action}失败：{title}，原因：{message}")


def _record_skipped_preview(
    action: str,
    title: str,
    reason: str,
    logs: list[str],
    stats: dict[str, int],
) -> None:
    stats["skipped_no_readable_preview"] += 1
    logs.append(f"skipped_no_readable_preview: {action}跳过：{title}，原因：{reason}")


def _looks_permission_denied(message: str) -> bool:
    text = str(message or "").lower()
    return any(keyword in text for keyword in ["permission denied", "no permission", "forbidden", "unauthorized", "无权限", "权限不足", "没有权限"])


def _looks_preview_unavailable(message: str) -> bool:
    text = str(message or "").lower()
    return any(keyword in text for keyword in ["no support", "unsupported", "not support", "preview", "预览", "不支持", "生成中", "processing"])


def _safe_file_stem(value: str) -> str:
    text = re.sub(r"[\\/:*?\"<>|\s]+", "_", str(value or "").strip()).strip("._")
    return (text or "lark_resource")[:80]


def _clip_text(value: str, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n...（内容过长，已截断）"


def _dedupe_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _lark_resource_dir(project_id: str) -> Path:
    path = project_dir(project_id) / "lark_resources"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _new_enrichment_stats() -> dict[str, int]:
    return {
        "resources": 0,
        "links": 0,
        "images": 0,
        "read_success": 0,
        "preview_success": 0,
        "skipped_no_readable_preview": 0,
        "permission_denied_no_request": 0,
        "failures": 0,
    }


def _has_enrichment(stats: dict[str, int]) -> bool:
    return any(stats.get(key, 0) for key in stats)
