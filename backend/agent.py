"""LangGraph 计划阶段风险识别 Agent。

为了方便维护，节点函数都保持普通 Python 函数形态。
如果环境里没有安装 langgraph，系统会自动使用同样节点的顺序执行版本。
"""

from __future__ import annotations

from typing import Any, TypedDict

from .exporter import export_risks
from .lark_client import fetch_messages_by_chat
from .llm_client import require_model_ready
from .models import MaterialRecord, ProjectState, Question, RiskItem
from .fact_rules import extract_document_facts, format_facts_for_prompt
from .parsers.bom_parser import parse_bom
from .parsers.document_ingestor import ingest_document
from .parsers.drawing_parser import parse_drawing_folder
from .parsers.rd_risk_parser import parse_rd_risk_excel
from .risk_rules import (
    build_questions,
    extract_bom_risks,
    extract_document_risks,
    extract_lark_risks,
    extract_rd_risks,
    merge_risks,
)
from .question_engine import active_question, build_human_answer_context
from .storage import append_log, load_project, save_lark_messages, save_project


class AgentState(TypedDict, total=False):
    """LangGraph 节点之间传递的状态。"""

    project: ProjectState
    materials: list[MaterialRecord]
    candidate_risks: list[RiskItem]
    questions: list[Question]


def run_plan_stage(state: ProjectState) -> ProjectState:
    """运行计划阶段风险识别。"""

    if active_question(state):
        state.status = "waiting"
        state.current_step = "等待人工确认"
        append_log(state, "存在阻塞型人工确认问题，Agent 暂停等待采购处理。")
        save_project(state)
        return state

    state.status = "running"
    state.error = None
    state.risks = []
    state.facts = []
    state.export_path = None
    # 重新运行时保留全部问题；新生成问题会按签名 upsert，避免未回答问题消失或重复。
    state.questions = [q for q in state.questions if q.status in {"answered", "skipped", "rejected", "pending"}]
    save_project(state)

    try:
        require_model_ready()
        append_log(state, "大模型配置和连通性检查通过，开始识别。")
        graph_input: AgentState = {"project": state, "candidate_risks": [], "questions": []}
        result = _run_with_langgraph_if_available(graph_input)
        final_state = result["project"]
        if active_question(final_state):
            final_state.status = "waiting"
            final_state.current_step = "等待人工确认"
            append_log(final_state, "Agent 已暂停，等待采购处理阻塞型问题后继续运行。")
            save_project(final_state)
            return final_state
        final_state.status = "done"
        final_state.current_step = "已完成"
        final_state.export_path = str(export_risks(final_state.id, final_state.config.project_name, final_state.risks))
        append_log(final_state, f"已导出风险物料清单：{final_state.export_path}")
        save_project(final_state)
        return final_state
    except Exception as exc:
        state.status = "error"
        state.error = str(exc)
        append_log(state, f"运行失败：{exc}")
        save_project(state)
        return state


def _run_with_langgraph_if_available(initial: AgentState) -> AgentState:
    """优先使用 LangGraph；没有依赖时按固定顺序执行节点。"""

    try:
        from langgraph.graph import END, StateGraph
    except Exception:
        current = initial
        for node in [_node_parse_inputs, _node_fetch_lark, _node_extract_facts, _node_extract_risks, _node_merge_and_question]:
            current = node(current)
        return current

    graph = StateGraph(AgentState)
    graph.add_node("parse_inputs", _node_parse_inputs)
    graph.add_node("fetch_lark", _node_fetch_lark)
    graph.add_node("extract_facts", _node_extract_facts)
    graph.add_node("extract_risks", _node_extract_risks)
    graph.add_node("merge_and_question", _node_merge_and_question)
    graph.set_entry_point("parse_inputs")
    graph.add_edge("parse_inputs", "fetch_lark")
    graph.add_edge("fetch_lark", "extract_facts")
    graph.add_edge("extract_facts", "extract_risks")
    graph.add_edge("extract_risks", "merge_and_question")
    graph.add_edge("merge_and_question", END)
    return graph.compile().invoke(initial)


def _node_parse_inputs(state: AgentState) -> AgentState:
    """节点：解析 BOM、PRD、规格书、图纸和研发自提文件。"""

    project = state["project"]
    project.current_step = "解析输入文件"
    save_project(project)

    materials = parse_bom(project.config.bom_path)
    append_log(project, f"BOM 解析完成，识别物料 {len(materials)} 条。")

    prd_result = ingest_document(project.config.prd_path, source_name="PRD", max_units=800)
    prd_lines = prd_result.lines(max_lines=800)
    append_log(project, f"PRD 文本线索 {len(prd_lines)} 条。")
    if prd_result.diagnostics.warnings:
        append_log(project, f"PRD 解析提示：{'；'.join(prd_result.diagnostics.warnings[:3])}")

    spec_result = ingest_document(project.config.spec_path, source_name="规格书", max_units=800)
    spec_lines = spec_result.lines(max_lines=800)
    if project.config.spec_path:
        append_log(project, f"规格书文本线索 {len(spec_lines)} 条。")
        if spec_result.diagnostics.warnings:
            append_log(project, f"规格书解析提示：{'；'.join(spec_result.diagnostics.warnings[:3])}")

    drawing_lines = parse_drawing_folder(project.config.drawing_dir)
    if project.config.drawing_dir:
        append_log(project, f"PDF 图纸线索 {len(drawing_lines)} 条。")

    rd_records = parse_rd_risk_excel(project.config.rd_risk_path)
    if project.config.rd_risk_path:
        append_log(project, f"研发自提风险 {len(rd_records)} 条。")

    project.__dict__["_prd_lines"] = prd_lines
    project.__dict__["_prd_units"] = prd_result.units
    project.__dict__["_spec_lines"] = spec_lines
    project.__dict__["_spec_units"] = spec_result.units
    project.__dict__["_drawing_lines"] = drawing_lines
    project.__dict__["_rd_records"] = rd_records
    return {**state, "project": project, "materials": materials}


def _node_extract_facts(state: AgentState) -> AgentState:
    """节点：从 PRD / 规格书抽取产品事实。"""

    project = state["project"]
    if active_question(project):
        project.current_step = "等待人工确认"
        save_project(project)
        return {**state, "project": project}

    project.current_step = "抽取产品事实"
    save_project(project)

    human_context = build_human_answer_context(project)
    facts = []
    prd_units = project.__dict__.get("_prd_units", [])
    if prd_units:
        try:
            facts.extend(extract_document_facts(prd_units, source_name="PRD", project_id=project.id, human_context=human_context))
        except Exception as exc:
            append_log(project, f"PRD 产品事实抽取失败，继续执行风险识别：{exc}")
    spec_units = project.__dict__.get("_spec_units", [])
    if spec_units:
        try:
            facts.extend(extract_document_facts(spec_units, source_name="规格书", project_id=project.id, human_context=human_context))
        except Exception as exc:
            append_log(project, f"规格书产品事实抽取失败，继续执行风险识别：{exc}")

    project.facts = facts
    append_log(project, f"产品事实 {len(facts)} 条。")
    save_project(project)
    return {**state, "project": project}


def _node_fetch_lark(state: AgentState) -> AgentState:
    """节点：读取飞书群历史消息。"""

    project = state["project"]
    project.current_step = "读取飞书群聊"
    save_project(project)

    messages, questions, logs = fetch_messages_by_chat(project.id, project.config.lark_chat or "")
    for log in logs:
        append_log(project, log)
    if messages:
        save_lark_messages(project.id, messages)
    project.__dict__["_lark_messages"] = messages
    upsert_result = _upsert_questions(project, questions)
    _append_question_upsert_log(project, upsert_result, "飞书群聊")
    return {**state, "project": project, "questions": project.questions}


def _node_extract_risks(state: AgentState) -> AgentState:
    """节点：各来源独立抽取候选风险。"""

    project = state["project"]
    materials = state.get("materials", [])
    if active_question(project):
        project.current_step = "等待人工确认"
        save_project(project)
        return {**state, "project": project, "candidate_risks": [], "questions": project.questions}

    project.current_step = "抽取候选风险物料"
    save_project(project)

    risks: list[RiskItem] = []
    generated_questions: list[Question] = []
    human_context = build_human_answer_context(project)
    fact_context = format_facts_for_prompt(project.facts)
    if human_context:
        append_log(project, "已加载人工确认上下文，后续模型识别会参考已提交答案。")
    if fact_context:
        append_log(project, "已加载产品事实上下文，后续风险识别会参考事实层。")
    # 按来源顺序抽取；若某个来源产生阻塞问题，立即暂停，等待采购回答。
    extractors = [
        lambda: extract_bom_risks(materials, project.id, human_context=human_context, fact_context=fact_context),
        lambda: extract_document_risks(project.__dict__.get("_prd_lines", []), materials, "PRD", project.id, human_context=human_context, fact_context=fact_context),
        lambda: extract_document_risks(project.__dict__.get("_spec_lines", []), materials, "规格书", project.id, human_context=human_context, fact_context=fact_context),
        lambda: extract_document_risks(project.__dict__.get("_drawing_lines", []), materials, "PDF图纸", project.id, human_context=human_context, fact_context=fact_context),
        lambda: extract_rd_risks(project.__dict__.get("_rd_records", []), materials, project.id, human_context=human_context, fact_context=fact_context),
        lambda: extract_lark_risks(project.__dict__.get("_lark_messages", []), materials, project.id, human_context=human_context, fact_context=fact_context),
    ]
    for extractor in extractors:
        source_risks, source_questions = extractor()
        risks.extend(source_risks)
        upsert_result = _upsert_questions(project, source_questions)
        generated_questions.extend(upsert_result["added"])
        generated_questions.extend(upsert_result["refreshed"])
        if active_question(project):
            append_log(project, f"候选风险物料已暂存 {len(risks)} 条，等待采购回答阻塞型问题。")
            return {**state, "project": project, "candidate_risks": risks, "questions": project.questions}

    append_log(project, f"候选风险物料 {len(risks)} 条。")
    if generated_questions:
        append_log(project, f"大模型生成采购确认问题 {len(generated_questions)} 个。")
    return {**state, "project": project, "candidate_risks": risks, "questions": project.questions}


def _node_merge_and_question(state: AgentState) -> AgentState:
    """节点：合并去重并生成待确认问题。"""

    project = state["project"]
    if active_question(project):
        project.current_step = "等待人工确认"
        save_project(project)
        return {**state, "project": project, "candidate_risks": state.get("candidate_risks", [])}

    project.current_step = "合并风险物料"
    save_project(project)

    merged = merge_risks(state.get("candidate_risks", []))
    questions = build_questions(merged)
    project.risks = merged
    upsert_result = _upsert_questions(project, questions)
    cleaned_count = _remove_stale_risk_action_questions(project)
    _append_question_upsert_log(project, upsert_result, "风险合并")
    if cleaned_count:
        append_log(project, f"已清理过期风险确认问题 {cleaned_count} 个。")
    append_log(project, f"合并后风险物料 {len(merged)} 条，待人工确认问题 {len(project.questions)} 个。")
    save_project(project)
    return {**state, "project": project, "candidate_risks": merged, "questions": project.questions}


class QuestionUpsertResult(TypedDict):
    """问题 upsert 结果，用于日志和评测。"""

    added: list[Question]
    refreshed: list[Question]
    ignored: list[Question]


def _upsert_questions(project: ProjectState, questions: list[Question]) -> QuestionUpsertResult:
    """按问题签名新增或刷新 pending 问题，避免重跑后重复或丢失。"""

    _sync_completed_questions_from_latest(project)
    answered_signatures = {
        _question_signature(question)
        for question in project.questions
        if question.status in {"answered", "skipped", "rejected"}
    }
    pending_by_signature = {
        _question_signature(question): question
        for question in project.questions
        if question.status == "pending"
    }
    added: list[Question] = []
    refreshed: list[Question] = []
    ignored: list[Question] = []
    for question in questions:
        signature = _question_signature(question)
        if signature in answered_signatures:
            ignored.append(question)
            continue
        existing = pending_by_signature.get(signature)
        if existing:
            _refresh_pending_question(existing, question)
            refreshed.append(existing)
            continue
        project.questions.append(question)
        pending_by_signature[signature] = question
        added.append(question)
    return {"added": added, "refreshed": refreshed, "ignored": ignored}


def _sync_completed_questions_from_latest(project: ProjectState) -> None:
    """同步运行期间已提交的人工答案，避免后台旧 state 覆盖用户操作。"""

    try:
        latest = load_project(project.id)
    except FileNotFoundError:
        return
    completed_by_id = {
        question.id: question
        for question in latest.questions
        if question.status in {"answered", "skipped", "rejected"}
    }
    if not completed_by_id:
        return
    existing_ids = {question.id for question in project.questions}
    for index, question in enumerate(project.questions):
        completed = completed_by_id.get(question.id)
        if completed:
            project.questions[index] = completed
    for question_id, completed in completed_by_id.items():
        if question_id not in existing_ids:
            project.questions.append(completed)


def _refresh_pending_question(existing: Question, incoming: Question) -> None:
    """用新生成问题刷新旧 pending 问题，同时保留旧 id/status/answer。"""

    existing.type = incoming.type
    existing.question_kind = incoming.question_kind
    existing.input_type = incoming.input_type
    existing.title = incoming.title
    existing.message = incoming.message
    existing.reason = incoming.reason
    existing.options = incoming.options
    existing.default_value = incoming.default_value
    existing.blocking = incoming.blocking
    existing.required = incoming.required
    existing.permission = incoming.permission
    existing.context = incoming.context
    existing.allow_custom = incoming.allow_custom
    existing.related_risk_ids = incoming.related_risk_ids


def _remove_stale_risk_action_questions(project: ProjectState) -> int:
    """清理当前风险已不存在的风险保留/合并 pending 问题。"""

    active_risk_ids = {risk.id for risk in project.risks}
    kept: list[Question] = []
    removed_count = 0
    for question in project.questions:
        if _is_stale_risk_action_question(question, active_risk_ids):
            removed_count += 1
            continue
        kept.append(question)
    project.questions = kept
    return removed_count


def _is_stale_risk_action_question(question: Question, active_risk_ids: set[str]) -> bool:
    """判断风险操作类 pending 问题是否已失效。"""

    return (
        question.status == "pending"
        and question.question_kind in {"risk_keep_review", "risk_merge_review"}
        and bool(question.related_risk_ids)
        and all(risk_id not in active_risk_ids for risk_id in question.related_risk_ids)
    )


def _append_question_upsert_log(project: ProjectState, result: QuestionUpsertResult, source_label: str) -> None:
    """记录问题新增/刷新情况。"""

    added_count = len(result["added"])
    refreshed_count = len(result["refreshed"])
    if added_count:
        append_log(project, f"{source_label}新增确认问题 {added_count} 个。")
    if refreshed_count:
        append_log(project, f"{source_label}刷新已有确认问题 {refreshed_count} 个。")


def _question_signature(question: Question) -> tuple[str, str, str, str]:
    """生成问题去重签名。"""

    return (
        question.question_kind,
        question.title.strip(),
        question.message.strip(),
        str(question.context.get("source_excerpt") or "").strip(),
    )
