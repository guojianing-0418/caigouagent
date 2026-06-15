"""大模型风险识别与结果整理。

重要原则：
1. 风险物料识别必须由大模型完成；
2. 本文件不保留关键词兜底，不根据固定词表直接产生风险结论；
3. 代码只负责准备上下文、校验模型输出、合并去重和生成人工确认问题。
"""

from __future__ import annotations

import difflib
import hashlib
import re
from collections import defaultdict
from typing import Any

from .config import get_effective_settings
from .llm_client import call_json_required
from .models import LarkMessage, MaterialRecord, Question, RISK_TYPES, RiskItem
from .question_engine import create_question
from .storage import load_llm_cache, save_llm_cache


PROMPT_VERSION = "plan-risk-v3-human-answer-context"


RISK_OUTPUT_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["risks", "questions"],
    "properties": {
        "risks": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["material_name", "module", "risk_type", "risk_reason", "source_basis"],
                "properties": {
                    "material_name": {"type": "string"},
                    "module": {"type": "string"},
                    "risk_type": {"type": "string", "enum": RISK_TYPES},
                    "risk_reason": {"type": "string"},
                    "source_basis": {"type": "string"},
                },
            },
        },
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "question_kind",
                    "input_type",
                    "title",
                    "message",
                    "reason",
                    "options",
                    "blocking",
                    "required",
                    "permission",
                    "context",
                    "related_risk_ids",
                    "allow_custom",
                ],
                "properties": {
                    "question_kind": {"type": "string"},
                    "input_type": {"type": "string", "enum": ["single_select", "multi_select", "boolean", "text", "textarea"]},
                    "title": {"type": "string"},
                    "message": {"type": "string"},
                    "reason": {"type": "string"},
                    "options": {"type": "array", "items": {"type": "string"}},
                    "blocking": {"type": "boolean"},
                    "required": {"type": "boolean"},
                    "permission": {"type": "string", "enum": ["allow", "ask", "deny"]},
                    "context": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "source_excerpt": {"type": "string"},
                            "source_name": {"type": "string"},
                            "effect": {"type": "string"},
                            "risk_id": {"type": "string"},
                            "left_material": {"type": "string"},
                            "right_material": {"type": "string"},
                        },
                    },
                    "related_risk_ids": {"type": "array", "items": {"type": "string"}},
                    "allow_custom": {"type": "boolean"},
                },
            },
        },
    },
}


RISK_JSON_SCHEMA = """
请只返回 JSON 对象，不要返回 Markdown，不要解释。对象格式如下：
{
  "risks": [
    {
      "material_name": "风险物料名称",
      "module": "所属模块，可为空",
      "risk_type": "必须从指定风险类型中选择",
      "risk_reason": "面向采购的风险原因，一句话到两句话",
      "source_basis": "来源与依据，说明来自哪个文件/消息/行，并引用关键依据"
    }
  ],
  "questions": [
    {
      "question_kind": "procurement_confirmation 或 risk_material_mapping 等业务分类",
      "input_type": "single_select/multi_select/boolean/text/textarea",
      "title": "给采购看的问题标题",
      "message": "具体需要采购确认的问题",
      "reason": "为什么需要人工确认",
      "options": ["可选答案1", "可选答案2"],
      "blocking": true,
      "required": true,
      "permission": "ask",
      "context": {"source_excerpt": "触发该问题的依据摘要"}
    }
  ]
}
如果没有足够依据识别风险物料，返回 {"risks": [], "questions": []}。
如模型只支持旧格式，也可以返回风险数组，系统会自动兼容。

questions 的 blocking / required 必须按以下规则判断：
- blocking=true 只用于“采购不立即回答，Agent 就无法继续正确执行”的问题。
- blocking=true 的典型场景：
  1. 无法继续读取来源，例如必须选择正确飞书群；
  2. 风险无法绑定到具体 BOM 物料，且会影响后续识别、合并或归类；
  3. 输入证据互相冲突，不确认就会导致后续大面积错配。
- blocking=false 用于“可以先完成识别，后续由采购确认结果”的问题。
- blocking=false 的典型场景：
  1. 某条风险是否保留；
  2. 两个风险物料是否合并；
  3. 风险类型是否调整；
  4. 证据较弱但可先进入候选清单；
  5. 采购希望补充说明但不影响 Agent 继续运行。
- required=true 表示该问题最终导出正式 Excel 前建议必须处理。
- required=false 表示采购可跳过，不影响最终输出。
- 如果不能确定是否必须立即阻塞，优先设置 blocking=false。
"""


def normalize_name(name: str) -> str:
    """物料名称归一化，只用于去重，不用于风险判断。"""

    return re.sub(r"[\s\-_/（）()，,。.:：]+", "", name).lower()


ExtractionBundle = tuple[list[RiskItem], list[Question]]


def extract_bom_risks(materials: list[MaterialRecord], project_id: str = "", human_context: str = "") -> ExtractionBundle:
    """用大模型从 BOM 识别计划阶段采购风险物料。"""

    lines = [
        f"sheet={item.sheet}; row={item.row_number}; module={item.module}; level={item.level}; "
        f"name={item.name}; spec={item.spec}; material={item.material}; quantity={item.quantity}"
        for item in materials
    ]
    return _extract_with_llm("草 BOM", lines, materials, project_id=project_id, human_context=human_context)


def extract_document_risks(
    lines: list[str],
    materials: list[MaterialRecord],
    source_name: str,
    project_id: str = "",
    human_context: str = "",
) -> ExtractionBundle:
    """用大模型从 PRD、规格书或 PDF 图纸线索中识别风险物料。"""

    return _extract_with_llm(source_name, lines, materials, project_id=project_id, human_context=human_context)


def extract_rd_risks(
    records: list[dict[str, str]],
    materials: list[MaterialRecord],
    project_id: str = "",
    human_context: str = "",
) -> ExtractionBundle:
    """用大模型理解研发自提风险并归入统一输出结构。"""

    lines = [
        f"sheet={record.get('sheet', '')}; 风险物料名称={record.get('material_name', '')}; 原因={record.get('reason', '')}"
        for record in records
    ]
    return _extract_with_llm("研发自提风险", lines, materials, project_id=project_id, human_context=human_context)


def extract_lark_risks(
    messages: list[LarkMessage],
    materials: list[MaterialRecord],
    project_id: str = "",
    human_context: str = "",
) -> ExtractionBundle:
    """用大模型从飞书群历史消息中识别风险物料。"""

    lines = [
        f"time={msg.create_time}; sender={msg.sender}; message_id={msg.message_id}; content={msg.content}"
        for msg in messages
        if msg.content.strip()
    ]
    return _extract_with_llm("飞书项目群历史消息", lines, materials, max_chars=10000, project_id=project_id, human_context=human_context)


def merge_risks(risks: list[RiskItem]) -> list[RiskItem]:
    """合并同一物料同一风险类型的多来源证据。"""

    grouped: dict[tuple[str, str], list[RiskItem]] = defaultdict(list)
    for risk in risks:
        grouped[(normalize_name(risk.material_name), risk.risk_type)].append(risk)

    merged: list[RiskItem] = []
    for items in grouped.values():
        first = items[0]
        reasons = _dedupe_text([item.risk_reason for item in items])
        bases = _dedupe_text([item.source_basis for item in items])
        merged.append(
            RiskItem(
                id=first.id,
                material_name=first.material_name,
                module=first.module,
                risk_type=first.risk_type,
                risk_reason="；".join(reasons),
                source_basis="；".join(bases),
            )
        )
    return merged


def build_questions(risks: list[RiskItem]) -> list[Question]:
    """生成需要人工确认的问题。

    风险识别本身由大模型完成；这里仅处理模型输出后的人工确认。
    """

    questions: list[Question] = []
    for risk in risks:
        if risk.material_name in ["待确认物料", "未知物料", ""]:
            questions.append(
                Question(
                    question_kind="risk_keep_review",
                    input_type="boolean",
                    title="这条风险是否保留？",
                    message=f"物料“{risk.material_name or '待确认物料'}”的风险依据较弱，是否保留在采购风险清单中？",
                    reason="模型没有明确关联到 BOM 中的具体物料，需要采购确认。",
                    options=["保留", "删除"],
                    blocking=False,
                    required=False,
                    context={
                        "risk_id": risk.id,
                        "source_excerpt": risk.source_basis,
                        "effect": "remove_related_risks_when_false",
                    },
                    related_risk_ids=[risk.id],
                )
            )

    for i, left in enumerate(risks):
        for right in risks[i + 1 :]:
            if left.id == right.id or left.risk_type != right.risk_type:
                continue
            ratio = difflib.SequenceMatcher(None, normalize_name(left.material_name), normalize_name(right.material_name)).ratio()
            if 0.86 <= ratio < 1:
                questions.append(
                    Question(
                        question_kind="risk_merge_review",
                        input_type="boolean",
                        title="以下两个风险物料是否合并？",
                        message=f"“{left.material_name}”和“{right.material_name}”名称相近，风险类型相同，是否合并？",
                        reason="物料名称相似但不能完全确定是否同一物料。",
                        options=["合并", "不合并"],
                        blocking=False,
                        required=False,
                        context={
                            "left_material": left.material_name,
                            "right_material": right.material_name,
                            "source_excerpt": f"{left.source_basis}；{right.source_basis}",
                        },
                        related_risk_ids=[left.id, right.id],
                    )
                )
    return questions


def apply_question_answer(risks: list[RiskItem], question: Question) -> list[RiskItem]:
    """根据人工答案调整风险清单。

    兼容旧调用入口；主流程已迁移到 question_engine.apply_question_effect。
    """

    if question.question_kind == "risk_keep_review" and question.answer in [False, "删除"]:
        return [risk for risk in risks if risk.id not in question.related_risk_ids]

    if question.question_kind == "risk_merge_review" and question.answer in [True, "合并"] and len(question.related_risk_ids) == 2:
        left_id, right_id = question.related_risk_ids
        left = next((risk for risk in risks if risk.id == left_id), None)
        right = next((risk for risk in risks if risk.id == right_id), None)
        if left and right:
            left.risk_reason = "；".join(_dedupe_text([left.risk_reason, right.risk_reason]))
            left.source_basis = "；".join(_dedupe_text([left.source_basis, right.source_basis]))
            return [risk for risk in risks if risk.id != right_id]
    return risks


def _extract_with_llm(
    source_name: str,
    source_lines: list[str],
    materials: list[MaterialRecord],
    max_chars: int = 14000,
    project_id: str = "",
    human_context: str = "",
) -> ExtractionBundle:
    """把某个来源分块交给大模型识别风险物料。"""

    clean_lines = [line.strip() for line in source_lines if line and line.strip()]
    if not clean_lines:
        return [], []

    risks: list[RiskItem] = []
    questions: list[Question] = []
    material_context = _material_context(materials)
    model_name = get_effective_settings().text_model
    human_context = human_context.strip()
    for chunk_index, chunk in enumerate(_chunk_lines(clean_lines, max_chars=max_chars), start=1):
        prompt = _build_prompt(source_name, chunk_index, material_context, chunk, human_context=human_context)
        chunk_hash = _chunk_hash(source_name, material_context, chunk, human_context=human_context)
        cache_hit = True
        data = load_llm_cache(
            project_id=project_id or "no-project",
            source_name=source_name,
            chunk_hash=chunk_hash,
            model_name=model_name,
            prompt_version=PROMPT_VERSION,
        )
        if data is None:
            cache_hit = False
            data = call_json_required(
                prompt,
                schema=RISK_OUTPUT_JSON_SCHEMA,
                schema_name="procurement_risk_extraction",
            )
        chunk_risks, chunk_questions = _items_and_questions_from_model(data, source_name, materials)
        if not cache_hit:
            save_llm_cache(
                project_id=project_id or "no-project",
                source_name=source_name,
                chunk_hash=chunk_hash,
                model_name=model_name,
                prompt_version=PROMPT_VERSION,
                raw_json=data,
            )
        risks.extend(chunk_risks)
        questions.extend(chunk_questions)
    return risks, questions


def _build_prompt(
    source_name: str,
    chunk_index: int,
    material_context: str,
    source_context: str,
    human_context: str = "",
) -> str:
    """生成风险识别提示词。"""

    risk_types = "、".join(RISK_TYPES)
    human_section = ""
    if human_context:
        human_section = f"""
已处理的人工确认/补充信息：
{human_context}

"""
    return f"""你是 IPD 计划阶段的采购风险物料识别 Agent，输出对象只给采购查看。

请基于【{source_name}】第 {chunk_index} 段内容，识别计划阶段需要采购提前关注的风险物料。

要求：
- 风险定义偏采购视角，但要包含研发、结构、工艺导致的采购风险；
- 不输出风险等级、建议动作、责任人、状态；
- 规格在计划阶段天然不明确，不要把“规格不明确”作为泛化风险；
- 风险类型只能从以下 8 类选择：{risk_types}；
- 优先关联到 BOM 中真实物料名称；确实无法关联时，material_name 可写“待确认物料”；
- 来源与依据必须能追溯到输入内容，不要编造没有出现的信息。
- 如果存在“已处理的人工确认/补充信息”，必须把它作为本项目的补充上下文参与判断。
- 人工回答中的“不确定、无法确认、待确认、需要后续补充”等表达不能视为风险解除；应保留对应候选风险或生成非阻塞必答问题。

可参考的 BOM 物料清单：
{material_context}

{human_section}待识别来源内容：
{source_context}

{RISK_JSON_SCHEMA}
"""


def _material_context(materials: list[MaterialRecord], limit: int = 180) -> str:
    """生成给大模型参考的 BOM 物料清单。"""

    rows = []
    for item in materials[:limit]:
        rows.append(f"- {item.name} | 模块:{item.module} | 规格:{item.spec} | 材质:{item.material}")
    if len(materials) > limit:
        rows.append(f"- 其余 {len(materials) - limit} 条物料略")
    return "\n".join(rows) or "无 BOM 物料清单"


def _chunk_lines(lines: list[str], max_chars: int) -> list[str]:
    """按字符数分块，避免单次提示词过长。"""

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in lines:
        extra = len(line) + 1
        if current and current_len + extra > max_chars:
            chunks.append("\n".join(current))
            current = []
            current_len = 0
        current.append(line)
        current_len += extra
    if current:
        chunks.append("\n".join(current))
    return chunks


def _chunk_hash(source_name: str, material_context: str, chunk: str, human_context: str = "") -> str:
    """计算模型输入分块 hash，用于 LLM 输出缓存。"""

    payload = "\n".join([PROMPT_VERSION, source_name, material_context, human_context.strip(), chunk])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _items_and_questions_from_model(
    data: Any,
    source_name: str,
    materials: list[MaterialRecord],
) -> ExtractionBundle:
    """校验并转换模型返回的风险和问题 JSON。"""

    if isinstance(data, dict):
        raw_risks = data.get("risks", [])
        raw_questions = data.get("questions", [])
    else:
        raw_risks = data
        raw_questions = []
    if not isinstance(raw_risks, list):
        raise RuntimeError("大模型返回 JSON 中 risks 不是数组，无法生成风险物料结果。")

    risks: list[RiskItem] = []
    for row in raw_risks:
        if not isinstance(row, dict):
            continue
        material_name = str(row.get("material_name") or "").strip()
        risk_type = str(row.get("risk_type") or "").strip()
        risk_reason = str(row.get("risk_reason") or "").strip()
        source_basis = str(row.get("source_basis") or "").strip()
        if not material_name or not risk_reason or not source_basis:
            continue
        if risk_type not in RISK_TYPES:
            raise RuntimeError(f"大模型返回了非法风险类型：{risk_type}。请检查模型输出或提示词。")
        risks.append(
            RiskItem(
                material_name=material_name,
                module=str(row.get("module") or _module_for_material(material_name, materials)),
                risk_type=risk_type,
                risk_reason=risk_reason,
                source_basis=source_basis if source_name in source_basis else f"{source_name}：{source_basis}",
            )
        )
    return risks, _questions_from_model(raw_questions, source_name)


def _questions_from_model(raw_questions: Any, source_name: str) -> list[Question]:
    """把大模型输出的问题转成通用 Question。"""

    if not isinstance(raw_questions, list):
        return []

    questions: list[Question] = []
    for row in raw_questions:
        if not isinstance(row, dict):
            continue
        input_type = str(row.get("input_type") or "boolean").strip()
        if input_type not in {"single_select", "multi_select", "boolean", "text", "textarea"}:
            continue
        permission = str(row.get("permission") or "ask").strip()
        if permission not in {"allow", "ask", "deny"}:
            permission = "ask"
        question_kind = str(row.get("question_kind") or "procurement_confirmation").strip()
        # 模型未明确给出 blocking 时，默认不打断流程；只有模型明确判断为 true 才弹窗阻塞。
        blocking = _bool_from_model(row.get("blocking"), default=False)
        required = _bool_from_model(row.get("required"), default=blocking)
        context = row.get("context") if isinstance(row.get("context"), dict) else {}
        context = {"source_name": source_name, **context}
        related_ids = row.get("related_risk_ids")
        questions.append(
            create_question(
                question_kind=question_kind,
                input_type=input_type,  # type: ignore[arg-type]
                title=str(row.get("title") or "需要采购确认").strip(),
                message=str(row.get("message") or row.get("title") or "").strip(),
                reason=str(row.get("reason") or "").strip(),
                options=[str(option) for option in row.get("options", [])] if isinstance(row.get("options"), list) else [],
                default_value=row.get("default_value"),
                blocking=blocking,
                required=required,
                permission=permission,  # type: ignore[arg-type]
                context=context,
                related_risk_ids=[str(item) for item in related_ids] if isinstance(related_ids, list) else [],
                allow_custom=_bool_from_model(row.get("allow_custom"), default=True),
            )
        )
    return questions


def _bool_from_model(value: Any, default: bool = False) -> bool:
    """把模型输出的布尔值安全转成 Python bool。

    大模型有时会输出字符串 "true" / "false"，不能直接用 bool(value)。
    """

    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "yes", "y", "1", "是", "需要", "阻塞"}:
        return True
    if text in {"false", "no", "n", "0", "否", "不需要", "非阻塞"}:
        return False
    return default


def _module_for_material(material_name: str, materials: list[MaterialRecord]) -> str:
    """按物料名查找所属模块，只用于补全模型漏填的模块。"""

    normalized = normalize_name(material_name)
    for item in materials:
        if normalize_name(item.name) == normalized:
            return item.module
    for item in materials:
        item_name = normalize_name(item.name)
        if item_name and (item_name in normalized or normalized in item_name):
            return item.module
    return ""


def _dedupe_text(values: list[str]) -> list[str]:
    """保序去重，并丢弃空文本。"""

    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = value.strip()
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result
