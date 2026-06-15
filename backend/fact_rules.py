"""PRD / 规格书产品事实抽取。

事实层用于把文档中的产品要求先结构化，再交给风险识别引用。
"""

from __future__ import annotations

import hashlib
from typing import Any

from .config import get_effective_settings
from .llm_client import call_json_required
from .models import FACT_TYPES, DocumentSourceRef, DocumentTextUnit, FactItem
from .question_engine import build_human_answer_context
from .storage import load_llm_cache, save_llm_cache


FACT_PROMPT_VERSION = "plan-fact-v1"


FACT_OUTPUT_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["facts"],
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["fact_type", "subject", "value", "source_basis", "source_refs"],
                "properties": {
                    "fact_type": {"type": "string", "enum": FACT_TYPES},
                    "subject": {"type": "string"},
                    "value": {"type": "string"},
                    "source_basis": {"type": "string"},
                    "source_refs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["source_name", "file_name", "sheet", "row_number", "page_number", "parser", "excerpt"],
                            "properties": {
                                "source_name": {"type": "string"},
                                "file_name": {"type": "string"},
                                "sheet": {"type": "string"},
                                "row_number": {"type": "integer"},
                                "page_number": {"type": "integer"},
                                "parser": {"type": "string"},
                                "excerpt": {"type": "string"},
                            },
                        },
                    },
                },
            },
        }
    },
}


def extract_document_facts(
    units: list[DocumentTextUnit],
    *,
    source_name: str,
    project_id: str = "",
    human_context: str = "",
    max_chars: int = 14000,
) -> list[FactItem]:
    """从文档文本单元中抽取产品事实。"""

    clean_units = [unit for unit in units if unit.text.strip()]
    if not clean_units:
        return []

    facts: list[FactItem] = []
    model_name = get_effective_settings().text_model
    human_context = human_context.strip()
    for chunk_index, chunk in enumerate(_chunk_units(clean_units, max_chars=max_chars), start=1):
        source_context = "\n".join(unit.as_line() for unit in chunk)
        prompt = _build_fact_prompt(source_name, chunk_index, source_context, human_context=human_context)
        chunk_hash = _fact_chunk_hash(source_name, source_context, human_context)
        data = load_llm_cache(
            project_id=project_id or "no-project",
            source_name=f"{source_name}-事实",
            chunk_hash=chunk_hash,
            model_name=model_name,
            prompt_version=FACT_PROMPT_VERSION,
        )
        if data is None:
            data = call_json_required(prompt, schema=FACT_OUTPUT_JSON_SCHEMA, schema_name="product_fact_extraction")
            save_llm_cache(
                project_id=project_id or "no-project",
                source_name=f"{source_name}-事实",
                chunk_hash=chunk_hash,
                model_name=model_name,
                prompt_version=FACT_PROMPT_VERSION,
                raw_json=data,
            )
        facts.extend(_facts_from_model(data, chunk))
    return _dedupe_facts(facts)


def format_facts_for_prompt(facts: list[FactItem], limit: int = 120) -> str:
    """把事实列表渲染成风险识别 prompt 可读上下文。"""

    rows = []
    for fact in facts[:limit]:
        refs = "；".join(_format_ref(ref) for ref in fact.source_refs[:3])
        rows.append(f"- [{fact.fact_type}] {fact.subject}: {fact.value} | 依据:{fact.source_basis}" + (f" | 来源:{refs}" if refs else ""))
    if len(facts) > limit:
        rows.append(f"- 其余 {len(facts) - limit} 条事实略")
    return "\n".join(rows)


def _build_fact_prompt(source_name: str, chunk_index: int, source_context: str, human_context: str = "") -> str:
    """生成事实抽取提示词。"""

    human_section = f"""
已处理的人工确认/补充信息：
{human_context}

""" if human_context else ""
    fact_types = "、".join(FACT_TYPES)
    return f"""你是 IPD 计划阶段的产品事实抽取助手。

请从【{source_name}】第 {chunk_index} 段中抽取会影响采购风险识别的产品事实。

要求：
- fact_type 只能从这些类型中选择：{fact_types}；
- 只抽取文档或人工补充中明确出现的信息，不要推测；
- source_basis 必须引用关键原文；
- source_refs 尽量填写来源位置，可从输入行前缀中提取 sheet/row/page 信息；
- 如果没有可用事实，返回 {{"facts": []}}。

{human_section}待抽取内容：
{source_context}

请只返回 JSON 对象，格式为：
{{
  "facts": [
    {{
      "fact_type": "性能指标",
      "subject": "功率精度",
      "value": "±1.0%",
      "source_basis": "PRD R38: 功率精度 | ±1.0%",
      "source_refs": [{{"source_name": "PRD", "file_name": "", "sheet": "", "row_number": 38, "page_number": 0, "parser": "", "excerpt": "功率精度 | ±1.0%"}}]
    }}
  ]
}}
"""


def _chunk_units(units: list[DocumentTextUnit], max_chars: int) -> list[list[DocumentTextUnit]]:
    """按字符数分块文档单元。"""

    chunks: list[list[DocumentTextUnit]] = []
    current: list[DocumentTextUnit] = []
    current_len = 0
    for unit in units:
        line = unit.as_line()
        extra = len(line) + 1
        if current and current_len + extra > max_chars:
            chunks.append(current)
            current = []
            current_len = 0
        current.append(unit)
        current_len += extra
    if current:
        chunks.append(current)
    return chunks


def _fact_chunk_hash(source_name: str, source_context: str, human_context: str) -> str:
    """计算事实抽取缓存 hash。"""

    payload = "\n".join([FACT_PROMPT_VERSION, source_name, human_context.strip(), source_context])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _facts_from_model(data: Any, source_units: list[DocumentTextUnit]) -> list[FactItem]:
    """校验并转换模型事实输出。"""

    raw_facts = data.get("facts", []) if isinstance(data, dict) else []
    if not isinstance(raw_facts, list):
        raise RuntimeError("大模型返回 JSON 中 facts 不是数组，无法生成产品事实。")

    facts: list[FactItem] = []
    fallback_refs = [_ref_from_unit(unit) for unit in source_units[:3]]
    for row in raw_facts:
        if not isinstance(row, dict):
            continue
        fact_type = str(row.get("fact_type") or "").strip()
        subject = str(row.get("subject") or "").strip()
        value = str(row.get("value") or "").strip()
        source_basis = str(row.get("source_basis") or "").strip()
        if fact_type not in FACT_TYPES or not subject or not value or not source_basis:
            continue
        refs = _refs_from_model(row.get("source_refs"))
        facts.append(
            FactItem(
                fact_type=fact_type,
                subject=subject,
                value=value,
                source_basis=source_basis,
                source_refs=refs or fallback_refs,
            )
        )
    return facts


def _refs_from_model(raw_refs: Any) -> list[DocumentSourceRef]:
    """转换模型返回的来源引用。"""

    if not isinstance(raw_refs, list):
        return []
    refs: list[DocumentSourceRef] = []
    for row in raw_refs:
        if not isinstance(row, dict):
            continue
        refs.append(
            DocumentSourceRef(
                source_name=str(row.get("source_name") or "").strip(),
                file_name=str(row.get("file_name") or "").strip(),
                sheet=str(row.get("sheet") or "").strip(),
                row_number=_as_int(row.get("row_number")),
                page_number=_as_int(row.get("page_number")),
                parser=str(row.get("parser") or "").strip(),
                excerpt=str(row.get("excerpt") or "").strip(),
            )
        )
    return refs


def _ref_from_unit(unit: DocumentTextUnit) -> DocumentSourceRef:
    """从摄取文本单元生成来源引用。"""

    return DocumentSourceRef(
        source_name=unit.source_name,
        file_name=unit.file_name,
        sheet=unit.sheet,
        row_number=unit.row_number,
        page_number=unit.page_number,
        parser=unit.parser,
        excerpt=unit.text,
    )


def _format_ref(ref: DocumentSourceRef) -> str:
    """渲染来源引用。"""

    if ref.sheet and ref.row_number:
        return f"{ref.file_name}/{ref.sheet} R{ref.row_number}"
    if ref.page_number:
        return f"{ref.file_name} P{ref.page_number}"
    return ref.file_name or ref.source_name


def _dedupe_facts(facts: list[FactItem]) -> list[FactItem]:
    """按类型、主体和值去重。"""

    result: list[FactItem] = []
    seen: set[tuple[str, str, str]] = set()
    for fact in facts:
        key = (fact.fact_type, fact.subject, fact.value)
        if key in seen:
            continue
        seen.add(key)
        result.append(fact)
    return result


def _as_int(value: Any) -> int:
    """安全转 int。"""

    try:
        return int(value or 0)
    except Exception:
        return 0
