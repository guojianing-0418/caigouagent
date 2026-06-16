"""人工回答决策账本。

本模块把自由文本答案转成稳定业务决策，用于后续去重、导出门禁和模型上下文。
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from .llm_client import call_json_required
from .models import HumanDecision, ProjectState, Question

DecisionClassifier = Callable[[Question, str], dict[str, Any] | None]

_DISPOSITIONS = {"confirmed", "unresolved", "pending_external_confirmation", "out_of_scope", "ignored", "custom_note"}
_RISK_EFFECTS = {"keep_risk", "remove_risk", "reduce_risk", "none"}

DECISION_OUTPUT_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["disposition", "owner", "risk_effect", "ask_again", "export_blocking"],
    "properties": {
        "disposition": {
            "type": "string",
            "enum": ["confirmed", "unresolved", "pending_external_confirmation", "out_of_scope", "ignored", "custom_note"],
        },
        "owner": {"type": "string"},
        "risk_effect": {"type": "string", "enum": ["keep_risk", "remove_risk", "reduce_risk", "none"]},
        "ask_again": {"type": "boolean"},
        "export_blocking": {"type": "boolean"},
    },
}


def sync_question_keys(question: Question) -> None:
    """给问题补充稳定主题和意图 key。"""

    question.topic_key = question.topic_key or question_topic_key(question)
    question.intent_key = question.intent_key or question_intent_key(question)


def sync_all_question_keys(state: ProjectState) -> None:
    """给项目中所有问题补齐 key。"""

    for question in state.questions:
        sync_question_keys(question)


def record_decision_for_question(
    state: ProjectState,
    question: Question,
    *,
    allow_llm: bool = True,
    llm_decision_fn: DecisionClassifier | None = None,
) -> HumanDecision:
    """根据已处理问题生成或更新决策。"""

    sync_question_keys(question)
    answer_text = answer_to_text(question.answer)
    decision = parse_human_decision(question, answer_text, allow_llm=allow_llm, llm_decision_fn=llm_decision_fn)
    existing_index = next((index for index, item in enumerate(state.decisions) if item.question_id == question.id), None)
    if existing_index is None:
        state.decisions.append(decision)
    else:
        state.decisions[existing_index] = decision
    apply_decision_effect(state, decision)
    return decision


def ensure_decisions_from_completed_questions(state: ProjectState, *, allow_llm: bool = False) -> int:
    """为历史已处理问题懒生成决策。"""

    sync_all_question_keys(state)
    existing_question_ids = {decision.question_id for decision in state.decisions}
    created = 0
    for question in state.questions:
        if question.status not in {"answered", "skipped", "rejected"} or question.id in existing_question_ids:
            continue
        record_decision_for_question(state, question, allow_llm=allow_llm)
        created += 1
    return created


def parse_human_decision(
    question: Question,
    answer_text: str,
    *,
    allow_llm: bool = True,
    llm_decision_fn: DecisionClassifier | None = None,
) -> HumanDecision:
    """模型优先、规则兜底地解析人工回答。"""

    base = {
        "question_id": question.id,
        "topic_key": question.topic_key or question_topic_key(question),
        "intent_key": question.intent_key or question_intent_key(question),
        "answer_text": answer_text,
        "related_risk_ids": question.related_risk_ids,
    }
    classifier = llm_decision_fn or _llm_decision
    llm = _normalize_decision_payload(classifier(question, answer_text)) if allow_llm else None
    if llm:
        return HumanDecision(**base, **llm)
    rule = _rule_decision(answer_text)
    if rule:
        return HumanDecision(**base, **rule)
    return HumanDecision(
        **base,
        disposition="custom_note",
        risk_effect="keep_risk",
        ask_again=False,
        export_blocking=False,
    )


def apply_decision_effect(state: ProjectState, decision: HumanDecision) -> None:
    """把决策影响同步到风险清单。"""

    if decision.risk_effect == "remove_risk":
        state.risks = [risk for risk in state.risks if risk.id not in decision.related_risk_ids]
        return

    if decision.risk_effect == "keep_risk":
        note = _decision_unresolved_note(decision)
        if not note:
            return
        for risk in state.risks:
            if decision.related_risk_ids and risk.id not in decision.related_risk_ids:
                continue
            if _topic_overlaps_risk(decision, risk) and note not in risk.unresolved_questions:
                risk.unresolved_questions.append(note)


def suppress_questions_by_decisions(state: ProjectState) -> int:
    """删除已由决策处理的 pending 问题。"""

    sync_all_question_keys(state)
    active_decisions = [decision for decision in state.decisions if not decision.ask_again]
    if not active_decisions:
        return 0

    kept: list[Question] = []
    removed = 0
    for question in state.questions:
        if question.status != "pending":
            kept.append(question)
            continue
        decision = matching_decision_for_question(active_decisions, question)
        if decision:
            question.suppressed_by_decision_id = decision.id
            removed += 1
            continue
        kept.append(question)
    if removed:
        state.questions = kept
    return removed


def matching_decision_for_question(decisions: list[HumanDecision], question: Question) -> HumanDecision | None:
    """查找可覆盖该问题的历史决策。"""

    sync_question_keys(question)
    if not question.topic_key:
        return None
    for decision in decisions:
        if decision.ask_again or not decision.topic_key:
            continue
        if decision.topic_key != question.topic_key:
            continue
        if not decision.intent_key or not question.intent_key or decision.intent_key == question.intent_key:
            return decision
        if _intent_compatible(decision.intent_key, question.intent_key):
            return decision
    return None


def build_decision_context(state: ProjectState, max_items: int = 40) -> str:
    """把决策账本渲染给模型。"""

    if not state.decisions:
        return ""
    lines = [
        "以下是本项目已处理的人工决策，后续识别必须遵守。",
        "ask_again=false 的主题不得再次向采购重复提问；未决类答案应保留风险并写入未决点。",
    ]
    selected = state.decisions[-max_items:]
    omitted = len(state.decisions) - len(selected)
    if omitted > 0:
        lines.append(f"前面还有 {omitted} 条较早决策已省略。")
    for index, decision in enumerate(selected, start=1):
        owner = f"；责任方:{decision.owner}" if decision.owner else ""
        lines.append(
            f"{index}. topic={decision.topic_key}; intent={decision.intent_key}; "
            f"disposition={decision.disposition}; effect={decision.risk_effect}; "
            f"ask_again={str(decision.ask_again).lower()}; export_blocking={str(decision.export_blocking).lower()}{owner}"
        )
        if decision.answer_text:
            lines.append(f"   人工回答: {_clip(decision.answer_text, 500)}")
    return "\n".join(lines)


def answer_to_text(answer: Any) -> str:
    """把答案转成文本。"""

    if isinstance(answer, list):
        return "；".join(str(item).strip() for item in answer if str(item).strip())
    if isinstance(answer, bool):
        return "是" if answer else "否"
    if answer is None:
        return ""
    return str(answer).strip()


def question_topic_key(question: Question) -> str:
    """生成跨改写稳定的主题 key。"""

    context = question.context or {}
    text = _normalize(
        " ".join(
            [
                question.title,
                question.message,
                question.reason,
                str(context.get("material_name") or ""),
                str(context.get("left_material") or ""),
                str(context.get("right_material") or ""),
                str(context.get("source_excerpt") or ""),
                str(context.get("query") or ""),
            ]
        )
    )
    special = _special_topic_key(text)
    if special:
        return special
    terms = _matched_terms(text, _TOPIC_TERMS)
    if terms:
        return "|".join(terms)
    return text[:80]


def question_intent_key(question: Question) -> str:
    """生成问题意图 key。"""

    text = _normalize(" ".join([question.title, question.message, question.reason]))
    if question.question_kind == "risk_merge_review":
        return "merge"
    if question.question_kind == "risk_keep_review":
        return "risk_keep"
    intents = []
    for intent, terms in _INTENT_TERMS.items():
        if _matched_terms(text, terms):
            intents.append(intent)
    if "supplier" in intents:
        return "supplier"
    if "bom_mapping" in intents:
        return "bom_mapping"
    if "spec" in intents:
        return "spec"
    return "|".join(sorted(intents)) or question.question_kind


def _rule_decision(answer_text: str) -> dict[str, Any] | None:
    text = _normalize(answer_text)
    if not text:
        return {
            "disposition": "custom_note",
            "risk_effect": "keep_risk",
            "ask_again": False,
            "export_blocking": False,
        }
    if any(term in text for term in ["研发确认", "问研发", "找研发", "研发评估", "研发待确认"]):
        return {
            "disposition": "pending_external_confirmation",
            "owner": "研发",
            "risk_effect": "keep_risk",
            "ask_again": False,
            "export_blocking": False,
        }
    if any(term in text for term in ["待确认", "不确定", "待定", "先保留", "保留为风险", "暂时保留"]):
        return {
            "disposition": "unresolved",
            "risk_effect": "keep_risk",
            "ask_again": False,
            "export_blocking": False,
        }
    if any(term in text for term in ["不属于采购", "不是采购", "采购不负责", "不用关联风险", "不用关联", "不纳入采购"]):
        return {
            "disposition": "out_of_scope",
            "risk_effect": "remove_risk",
            "ask_again": False,
            "export_blocking": False,
        }
    if any(term in text for term in ["不用管", "暂不处理", "先不用管", "忽略"]):
        return {
            "disposition": "ignored",
            "risk_effect": "none",
            "ask_again": False,
            "export_blocking": False,
        }
    if any(term in text for term in ["已有", "已定点", "已锁定", "已确认", "已确定", "有候选", "有意向"]):
        return {
            "disposition": "confirmed",
            "risk_effect": "reduce_risk",
            "ask_again": False,
            "export_blocking": False,
        }
    return None


def _llm_decision(question: Question, answer_text: str) -> dict[str, Any] | None:
    if not answer_text:
        return None
    prompt = f"""你是采购风险 Agent 的人工回答分类器。

请把用户对确认问题的回答分类成固定 JSON。不要解释。

问题标题：{question.title}
问题内容：{question.message}
用户回答：{answer_text}

分类规则：
- 必须结合问题标题、问题内容和用户回答整体判断，不要只按单个关键词分类。
- 如果回答里有否定、转折或多个结论并存，按更保守的采购风险处理：需要外部角色确认优先于已确认，不确定/保留风险优先于已确认，不应把“不是不属于采购范围”判成 out_of_scope。
- confirmed：用户给出明确确认、已有资源、已定点、已锁定等结论。
- unresolved：用户表示待确认、不确定、先保留。
- pending_external_confirmation：用户表示需要其他角色确认，例如研发、质量、客户。
- out_of_scope：用户表示不属于采购范围或不应纳入采购风险。
- ignored：用户表示不用管、暂不处理。
- custom_note：无法归类但仍应视为已处理的备注。
- ask_again 通常为 false；只有用户明确要求继续追问采购才为 true。
- export_blocking 对 unresolved / pending_external_confirmation 必须为 false。
"""
    try:
        data = call_json_required(prompt, schema=DECISION_OUTPUT_JSON_SCHEMA, schema_name="human_answer_decision")
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return _normalize_decision_payload(data)


def _normalize_decision_payload(data: dict[str, Any] | None) -> dict[str, Any] | None:
    """归一化模型或测试替身返回的决策载荷。"""

    if not isinstance(data, dict):
        return None
    disposition = str(data.get("disposition") or "custom_note")
    if disposition not in _DISPOSITIONS:
        disposition = "custom_note"
    risk_effect = str(data.get("risk_effect") or "keep_risk")
    if risk_effect not in _RISK_EFFECTS:
        risk_effect = "keep_risk"
    return {
        "disposition": disposition,
        "owner": str(data.get("owner") or "").strip(),
        "risk_effect": risk_effect,
        "ask_again": _as_bool(data.get("ask_again"), default=False),
        "export_blocking": _as_bool(data.get("export_blocking"), default=False),
    }


def _as_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "是", "需要", "继续"}:
            return True
        if normalized in {"false", "0", "no", "n", "否", "不", "不用", "无需"}:
            return False
    return bool(value)


def _decision_unresolved_note(decision: HumanDecision) -> str:
    if decision.disposition == "pending_external_confirmation":
        owner = decision.owner or "外部角色"
        return f"人工确认：需{owner}进一步确认。"
    if decision.disposition == "unresolved":
        return f"人工确认：{decision.answer_text or '待确认，风险先保留'}。"
    if decision.disposition == "custom_note":
        return f"人工备注：{decision.answer_text}。"
    return ""


def _topic_overlaps_risk(decision: HumanDecision, risk: Any) -> bool:
    if decision.related_risk_ids:
        return True
    risk_text = _normalize(" ".join([getattr(risk, "material_name", ""), getattr(risk, "risk_reason", ""), getattr(risk, "source_basis", "")]))
    return bool(decision.topic_key and decision.topic_key.replace("|", "")[:4] in risk_text)


def _intent_compatible(left: str, right: str) -> bool:
    compatible = {
        ("bom_mapping", "risk_keep"),
        ("risk_keep", "bom_mapping"),
        ("supplier", "process"),
        ("process", "supplier"),
    }
    return (left, right) in compatible


def _special_topic_key(text: str) -> str:
    if "泥浆设备" in text:
        return "泥浆设备"
    if "踏板供应商" in text or ("踏板" in text and any(term in text for term in ["供应商", "定厂", "定点", "寻源"])):
        return "踏板供应商"
    if "电池" in text and any(term in text for term in ["连接", "焊线", "fpc", "连接器"]):
        return "电池连接方案"
    if "钛合金" in text and "轴心" in text:
        return "钛合金轴心"
    return ""


def _normalize(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", value).lower()


def _matched_terms(text: str, terms: set[str]) -> list[str]:
    matched = set()
    for term in terms:
        normalized = _normalize(term)
        if normalized and normalized in text:
            matched.add(term)
    return sorted(matched)


def _clip(value: str, limit: int) -> str:
    clean = " ".join(str(value).split())
    if len(clean) <= limit:
        return clean
    return clean[:limit].rstrip() + "..."


_TOPIC_TERMS = {
    "15-5ph",
    "7075t6",
    "ant",
    "bluetooth",
    "fpc",
    "keo",
    "m7",
    "m18",
    "pcba",
    "spd",
    "spd-sl",
    "不锈钢",
    "主板pcba",
    "低压注塑",
    "低压注塑硬胶",
    "供应商",
    "充电接口",
    "充电线",
    "天线",
    "尾端顶紧",
    "应变片",
    "接口",
    "电池",
    "电阻应变片",
    "螺丝",
    "螺钉",
    "碳纤维",
    "脚踏",
    "脚踏本体",
    "踏板",
    "踏板本体",
    "轴心",
    "钛合金",
    "锂电池",
    "长碳纤维",
    "锁片",
}


_INTENT_TERMS = {
    "bom_mapping": {"bom", "对应", "关联", "归属", "纳入", "物料清单", "单独采购", "独立bom"},
    "supplier": {"供应商", "定点", "定厂", "寻源", "资源", "储备", "备选", "候选"},
    "spec": {"规格", "参数", "尺寸", "型号", "容量", "材质", "材料", "牌号", "方案"},
    "certification": {"认证", "合规", "法规", "ce", "fcc", "rohs", "un38.3", "iec62133"},
    "process": {"工艺", "模具", "注塑", "贴片", "盐雾", "量产"},
    "schedule": {"交期", "周期", "延期", "时间节点", "开发进展"},
    "cost": {"成本", "价格", "目标成本"},
}
