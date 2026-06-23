"""通用问题引擎。

本模块集中处理 Agent 的人机交互问题：
1. 统一创建问题；
2. 统一校验和保存答案；
3. 根据答案对项目状态或风险清单产生影响。

它对应 opencode-style 的 Question Tool 思路：问题不是日志文本，
而是执行过程中的结构化能力调用。
"""

from __future__ import annotations

import re
from typing import Any

from .decision_engine import build_decision_context
from .models import ProjectState, Question, QuestionAction, QuestionInputType, QuestionPermission, normalize_question_input_type
from .risk_classification import merge_classification_fields


VALID_INPUT_TYPES = {"single_select", "multi_select", "boolean", "text", "textarea"}
VALID_PERMISSIONS = {"allow", "ask", "deny"}


def create_question(
    *,
    question_kind: str,
    input_type: QuestionInputType,
    title: str,
    message: str = "",
    reason: str = "",
    options: list[str] | None = None,
    default_value: Any | None = None,
    blocking: bool = False,
    required: bool = True,
    permission: QuestionPermission = "ask",
    context: dict[str, Any] | None = None,
    related_risk_ids: list[str] | None = None,
    allow_custom: bool = True,
) -> Question:
    """创建一个通用 Question，并做最基础的字段校验。"""

    normalized_options = [str(option).strip() for option in (options or []) if str(option).strip()]
    input_type = normalize_question_input_type(input_type, normalized_options)  # type: ignore[assignment]
    if input_type not in VALID_INPUT_TYPES:
        raise ValueError(f"不支持的问题输入类型：{input_type}")
    if permission not in VALID_PERMISSIONS:
        raise ValueError(f"不支持的问题权限：{permission}")

    return Question(
        question_kind=question_kind,
        input_type=input_type,
        title=title.strip() or "需要人工确认",
        message=(message or title).strip(),
        reason=reason.strip(),
        options=normalized_options,
        default_value=default_value,
        blocking=blocking,
        required=required,
        permission=permission,
        context=context or {},
        related_risk_ids=related_risk_ids or [],
        allow_custom=allow_custom,
    )


def answer_question(question: Question, answer: Any, action: QuestionAction = "submit") -> Question:
    """校验并保存用户答案。

    action=submit 时会按 input_type 校验答案；
    action=skip/reject 用于非必答问题或后续扩展的拒绝场景。
    """

    if action == "skip":
        if question.required:
            raise ValueError("该问题为必答问题，不能跳过。")
        question.answer = answer
        question.status = "skipped"
        return question

    if action == "reject":
        question.answer = answer
        question.status = "rejected"
        return question

    question.answer = _validate_and_normalize_answer(question, answer)
    question.status = "answered"
    return question


def apply_question_effect(state: ProjectState, question: Question) -> ProjectState:
    """根据 question_kind 和答案更新项目状态或风险清单。

    第一版内置飞书群选择、风险保留、风险合并和物料映射。
    采购确认类问题还可以通过 context.effect 做简单可配置动作。
    """

    if question.status not in {"answered", "rejected", "skipped"}:
        return state

    if question.question_kind == "lark_chat_selection" and question.status == "answered":
        selected = _as_text(question.answer)
        # 选项格式通常为“群名 | oc_xxx”，这里取最后一段作为 chat_id。
        state.config.lark_chat = selected.split("|")[-1].strip()
        return state

    if question.question_kind == "risk_keep_review" and question.status == "answered":
        if not _answer_is_positive(question.answer):
            state.risks = [risk for risk in state.risks if risk.id not in question.related_risk_ids]
        return state

    if question.question_kind == "risk_merge_review" and question.status == "answered":
        if _answer_is_positive(question.answer):
            _merge_related_risks(state, question.related_risk_ids)
        return state

    if question.question_kind == "risk_material_mapping" and question.status == "answered":
        target_name = _material_name_from_answer(question.answer)
        if target_name:
            for risk in state.risks:
                if risk.id in question.related_risk_ids:
                    risk.material_name = target_name
        return state

    if question.question_kind == "procurement_confirmation" and question.status == "answered":
        _apply_context_effect(state, question)
        return state

    return state


def active_question(state: ProjectState) -> Question | None:
    """返回当前优先级最高的阻塞问题。"""

    if state.active_question_id:
        active = next((question for question in state.questions if question.id == state.active_question_id and question.status == "pending"), None)
        if active:
            return active
    pending = pending_blocking_questions(state)
    return pending[0] if pending else None


def pending_blocking_questions(state: ProjectState) -> list[Question]:
    """找出会暂停 Agent 执行的 pending 问题。"""

    return [
        question
        for question in state.questions
        if question.status == "pending" and question.blocking and question.permission == "ask"
    ]


def pending_required_questions(state: ProjectState) -> list[Question]:
    """找出正式导出前建议必须处理的 pending 问题。"""

    return [question for question in state.questions if question.status == "pending" and question.required]


def build_human_answer_context(state: ProjectState, max_questions: int = 40, max_chars: int = 8000) -> str:
    """把已处理的人工确认问题整理成可回流给大模型的上下文。

    人工答案不是普通日志。Agent resume 后，后续模型调用必须看到这些答案，
    才能真正按用户补充的信息继续识别风险。
    """

    completed = [question for question in state.questions if question.status in {"answered", "skipped", "rejected"}]
    decision_context = build_decision_context(state)
    if not completed and not decision_context:
        return ""

    selected = completed[-max_questions:]
    lines = []
    if decision_context:
        lines.append(decision_context)
        lines.append("")
    lines.extend(
        [
            "以下是本项目最近的人工确认原文，仅作补充参考。",
            "如果结构化决策和原文有冲突，以结构化决策为准。",
        ]
    )
    omitted_count = len(completed) - len(selected)
    if omitted_count > 0:
        lines.append(f"前面还有 {omitted_count} 条较早的人工确认已省略，仅保留最近 {len(selected)} 条。")

    for index, question in enumerate(selected, start=1):
        source_name = str(question.context.get("source_name") or "").strip()
        source_excerpt = str(question.context.get("source_excerpt") or "").strip()
        status_label = _question_status_label(question.status)
        lines.append(f"{index}. [{status_label}] {question.title}")
        lines.append(f"   分类: {question.question_kind}" + (f"；来源: {source_name}" if source_name else ""))
        if question.message:
            lines.append(f"   原问题: {_clip_text(question.message, 500)}")
        if question.reason:
            lines.append(f"   提问原因: {_clip_text(question.reason, 400)}")
        if source_excerpt:
            lines.append(f"   原始依据: {_clip_text(source_excerpt, 500)}")
        if question.status == "answered":
            lines.append(f"   人工回答: {_clip_text(_answer_to_text(question.answer), 800)}")
        elif question.status == "skipped":
            lines.append("   人工处理: 已跳过；不能视为已确认事实。")
        else:
            lines.append(f"   人工处理: 已拒绝/驳回；反馈: {_clip_text(_answer_to_text(question.answer), 800)}")

    text = "\n".join(lines)
    if len(text) > max_chars:
        return text[:max_chars].rstrip() + "\n...（人工确认上下文过长，已截断）"
    return text


def _validate_and_normalize_answer(question: Question, answer: Any) -> Any:
    """按控件类型校验答案，并返回便于后端处理的值。"""

    if question.input_type == "single_select":
        value = _as_text(answer)
        if question.required and not value:
            raise ValueError("请选择或填写一个答案。")
        if value and question.options and not question.allow_custom and value not in question.options:
            raise ValueError("答案不在允许选项中。")
        return value

    if question.input_type == "multi_select":
        values = _as_list(answer)
        if question.required and not values:
            raise ValueError("请至少选择或填写一个答案。")
        if question.options and not question.allow_custom:
            invalid = [value for value in values if value not in question.options]
            if invalid:
                raise ValueError(f"存在不允许的选项：{', '.join(invalid)}")
        return values

    if question.input_type == "boolean":
        normalized_type = normalize_question_input_type(question.input_type, question.options)
        if normalized_type == "single_select":
            value = _as_text(answer)
            if question.required and not value:
                raise ValueError("请选择或填写一个答案。")
            if value and question.options and not question.allow_custom and value not in question.options:
                raise ValueError("答案不在允许选项中。")
            question.input_type = "single_select"
            return value
        if isinstance(answer, bool):
            return answer
        value = _as_text(answer)
        if not value and not question.required:
            return None
        positive = {"是", "确认", "保留", "合并", "通过", "true", "yes", "y", "1", "allow"}
        negative = {"否", "不", "删除", "不合并", "不保留", "驳回", "false", "no", "n", "0", "deny"}
        if value.lower() in positive:
            return True
        if value.lower() in negative:
            return False
        if value in positive:
            return True
        if value in negative:
            return False
        raise ValueError("请提交是/否类答案。")

    if question.input_type in {"text", "textarea"}:
        value = _as_text(answer)
        if question.required and not value:
            raise ValueError("请填写答案。")
        return value

    raise ValueError(f"不支持的问题输入类型：{question.input_type}")


def _apply_context_effect(state: ProjectState, question: Question) -> None:
    """执行 context 中声明的轻量后续动作。

    这样大模型或后续规则可以把采购确认问题表达得更灵活，
    但仍避免在代码里不断新增硬编码 question_kind。
    """

    effect = str(question.context.get("effect") or "").strip()
    is_positive = _answer_is_positive(question.answer)

    if effect == "remove_related_risks_when_false" and not is_positive:
        state.risks = [risk for risk in state.risks if risk.id not in question.related_risk_ids]
        return

    if effect == "remove_related_risks_when_true" and is_positive:
        state.risks = [risk for risk in state.risks if risk.id not in question.related_risk_ids]
        return

    if effect == "update_material_name":
        target_name = _material_name_from_answer(question.answer)
        if target_name:
            for risk in state.risks:
                if risk.id in question.related_risk_ids:
                    risk.material_name = target_name


def _merge_related_risks(state: ProjectState, related_risk_ids: list[str]) -> None:
    """把两个相关风险合并到第一条风险上。"""

    if len(related_risk_ids) != 2:
        return
    left_id, right_id = related_risk_ids
    left = next((risk for risk in state.risks if risk.id == left_id), None)
    right = next((risk for risk in state.risks if risk.id == right_id), None)
    if not left or not right:
        return
    left.risk_reason = "；".join(_dedupe_text([left.risk_reason, right.risk_reason]))
    left.source_basis = "；".join(_dedupe_text([left.source_basis, right.source_basis]))
    left.unresolved_questions = _dedupe_text([*left.unresolved_questions, *right.unresolved_questions])
    merge_classification_fields(left, right)
    state.risks = [risk for risk in state.risks if risk.id != right_id]


def _answer_is_positive(answer: Any) -> bool:
    """把布尔或中文按钮答案统一判断为肯定。"""

    if isinstance(answer, bool):
        return answer
    value = _as_text(answer).lower()
    return value in {"是", "确认", "保留", "合并", "通过", "true", "yes", "y", "1", "allow"}


def _material_name_from_answer(answer: Any) -> str:
    """从单选答案中提取物料名。"""

    value = _as_text(answer)
    if not value:
        return ""
    # 常见选项格式：“物料名称 | 模块 | 行号”，取第一段作为物料名。
    return value.split("|")[0].strip()


def _as_text(value: Any) -> str:
    """把前端传入的答案安全转为字符串。"""

    if value is None:
        return ""
    return str(value).strip()


def _as_list(value: Any) -> list[str]:
    """把多选答案转为字符串列表。"""

    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    return [item.strip() for item in re.split(r"[,，;；\n]+", str(value)) if item.strip()]


def _answer_to_text(answer: Any) -> str:
    """把人工答案转成适合放进模型上下文的文本。"""

    if isinstance(answer, list):
        return "；".join(str(item).strip() for item in answer if str(item).strip())
    if isinstance(answer, bool):
        return "是" if answer else "否"
    return _as_text(answer) or "（空）"


def _question_status_label(status: str) -> str:
    """把问题状态转成给模型看的中文标签。"""

    return {
        "answered": "已回答",
        "skipped": "已跳过",
        "rejected": "已拒绝",
    }.get(status, status)


def _clip_text(text: str, limit: int) -> str:
    """限制单条上下文长度，避免人工回答把提示词撑爆。"""

    clean = " ".join(str(text).split())
    if len(clean) <= limit:
        return clean
    return clean[:limit].rstrip() + "..."


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
