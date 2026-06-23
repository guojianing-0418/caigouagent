"""问题控量规则。

风险识别可以尽量完整，但界面待确认问题必须少而关键。
本模块只治理 Question 数量，不改变风险物料本身。
"""

from __future__ import annotations

from .decision_engine import sync_question_keys
from .models import ProjectState, Question


MAX_UI_PENDING_QUESTIONS = 5
MAX_SOURCE_NONBLOCKING_QUESTIONS = 2
MAX_MERGE_REVIEW_QUESTIONS = 3


def limit_source_questions(questions: list[Question], *, max_nonblocking: int = MAX_SOURCE_NONBLOCKING_QUESTIONS) -> list[Question]:
    """限制单个来源由模型直接生成的问题数量。"""

    if not questions:
        return []
    blocking = [question for question in questions if question.blocking and question.permission == "ask"]
    nonblocking = [question for question in questions if question not in blocking]
    for question in nonblocking:
        question.required = False
    ranked = sorted(enumerate(nonblocking), key=lambda item: (-_question_priority(item[1]), item[0]))
    selected_nonblocking = [question for _, question in ranked[:max_nonblocking]]
    return [*blocking, *selected_nonblocking]


def compact_project_questions(project: ProjectState, *, max_pending: int = MAX_UI_PENDING_QUESTIONS) -> int:
    """把项目中的 pending 问题压缩到少量重点问题。"""

    if not project.questions:
        return 0

    kept_completed: list[Question] = []
    pending: list[Question] = []
    for question in project.questions:
        sync_question_keys(question)
        if question.status != "pending":
            kept_completed.append(question)
            continue
        if not question.blocking:
            question.required = False
        pending.append(question)

    blocking = [question for question in pending if question.blocking and question.permission == "ask"]
    nonblocking = [question for question in pending if question not in blocking]
    budget = max(max_pending - len(blocking), 0)
    selected_nonblocking: list[Question] = []
    removed_count = 0
    seen_signatures: set[tuple[str, str, str]] = set()
    merge_count = 0

    ranked = sorted(enumerate(nonblocking), key=lambda item: (-_question_priority(item[1]), item[0]))
    for _, question in ranked:
        signature = _compact_signature(question)
        if signature in seen_signatures:
            removed_count += 1
            continue
        if question.question_kind == "risk_merge_review":
            if merge_count >= MAX_MERGE_REVIEW_QUESTIONS:
                removed_count += 1
                continue
            merge_count += 1
        if len(selected_nonblocking) >= budget:
            removed_count += 1
            continue
        seen_signatures.add(signature)
        selected_nonblocking.append(question)

    if removed_count:
        project.questions = [*kept_completed, *blocking, *selected_nonblocking]
    else:
        project.questions = [*kept_completed, *blocking, *selected_nonblocking]
    return removed_count


def _question_priority(question: Question) -> int:
    """越高越应保留在界面上。"""

    if question.blocking:
        return 1000
    text = f"{question.title} {question.message} {question.reason}"
    score = 0
    if question.question_kind == "risk_material_mapping":
        score += 90
    elif question.question_kind == "risk_keep_review":
        score += 80
    elif question.question_kind == "risk_merge_review":
        score += 70
    elif question.question_kind == "procurement_confirmation":
        score += 40
    if any(term in text for term in ("对应BOM", "BOM", "物料确认", "无法绑定", "无法对应")):
        score += 20
    if any(term in text for term in ("冲突", "矛盾", "不一致", "错配")):
        score += 15
    if question.related_risk_ids:
        score += min(len(question.related_risk_ids), 5)
    if question.required:
        score += 5
    return score


def _compact_signature(question: Question) -> tuple[str, str, str]:
    kind = "human_confirmation" if question.question_kind in {"procurement_confirmation", "risk_material_mapping"} else question.question_kind
    return (
        kind,
        question.topic_key or "",
        question.intent_key or "",
    )
