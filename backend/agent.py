"""LangGraph 计划阶段风险识别 Agent。

为了方便维护，节点函数都保持普通 Python 函数形态。
如果环境里没有安装 langgraph，系统会自动使用同样节点的顺序执行版本。
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any, TypedDict

from .config import settings
from .decision_engine import (
    ensure_decisions_from_completed_questions,
    matching_decision_for_question,
    suppress_questions_by_decisions,
    sync_all_question_keys,
    sync_question_keys,
)
from .exporter import export_risks
from .lark_client import fetch_messages_by_chat
from .llm_client import require_model_ready
from .models import DocumentTextUnit, LarkMessage, MaterialRecord, ProjectState, Question, RiskItem
from .fact_rules import extract_document_facts, format_facts_for_prompt
from .parsers.bom_parser import parse_bom
from .parsers.document_ingestor import ingest_document
from .parsers.drawing_parser import parse_drawing_folder
from .parsers.rd_risk_parser import parse_rd_risk_excel
from .risk_classification import classify_risks, ensure_risk_classification
from .risk_rules import (
    build_questions,
    extract_bom_risks,
    extract_document_risks,
    extract_lark_risks,
    extract_rd_risks,
    merge_risks,
)
from .question_engine import active_question, build_human_answer_context, pending_required_questions
from .storage import append_log, load_project, save_lark_messages, save_project


class AgentState(TypedDict, total=False):
    """LangGraph 节点之间传递的状态。"""

    project: dict[str, Any]
    materials: list[dict[str, Any]]
    candidate_risks: list[dict[str, Any]]
    questions: list[dict[str, Any]]
    prd_lines: list[str]
    prd_units: list[dict[str, Any]]
    spec_lines: list[str]
    spec_units: list[dict[str, Any]]
    drawing_lines: list[str]
    rd_records: list[dict[str, str]]
    lark_messages: list[dict[str, Any]]


def run_plan_stage(state: ProjectState) -> ProjectState:
    """运行计划阶段风险识别。"""

    if active_question(state):
        state.status = "waiting"
        state.current_step = "等待人工确认"
        state.active_question_id = active_question(state).id if active_question(state) else None
        append_log(state, "存在阻塞型人工确认问题，Agent 暂停等待采购处理。")
        save_project(state)
        return state

    state.run_revision += 1
    state.status = "running"
    state.error = None
    state.risks = []
    state.facts = []
    state.export_path = None
    state.active_question_id = None
    # 重新运行时保留全部问题；新生成问题会按签名 upsert，避免未回答问题消失或重复。
    state.questions = [q for q in state.questions if q.status in {"answered", "skipped", "rejected", "pending"}]
    _prepare_question_state(state)
    removed_duplicates = _dedupe_existing_questions(state)
    if removed_duplicates:
        append_log(state, f"已合并重复确认问题 {removed_duplicates} 个。")
    save_project(state)

    try:
        require_model_ready()
        append_log(state, "大模型配置和连通性检查通过，开始识别。")
        graph_input: AgentState = {"project": state.model_dump(mode="json"), "candidate_risks": [], "questions": []}
        result = _run_with_langgraph_if_available(graph_input, thread_id=_thread_id(state))
        if "__interrupt__" in result:
            latest = load_project(state.id)
            latest.status = "waiting"
            latest.current_step = "等待人工确认"
            save_project(latest)
            return latest
        final_state = _coerce_project(result["project"])
        if active_question(final_state):
            final_state.status = "waiting"
            final_state.current_step = "等待人工确认"
            final_state.active_question_id = active_question(final_state).id if active_question(final_state) else None
            append_log(final_state, "Agent 已暂停，等待采购处理阻塞型问题后继续运行。")
            save_project(final_state)
            return final_state
        final_state.status = "done"
        final_state.current_step = "已完成"
        final_state.active_question_id = None
        final_state.risks = ensure_risk_classification(final_state.risks)
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


def resume_plan_stage(project_id: str, answer_payload: dict[str, Any] | None = None) -> ProjectState:
    """从 LangGraph checkpoint 继续运行。"""

    state = load_project(project_id)
    state.status = "running"
    state.current_step = "继续识别"
    state.active_question_id = None
    save_project(state)
    try:
        if not _can_use_langgraph_checkpoint():
            return run_plan_stage(state)

        from langgraph.types import Command

        result = _run_with_langgraph_if_available(Command(resume=answer_payload or {}), thread_id=_thread_id(state))
        if "__interrupt__" in result:
            latest = load_project(project_id)
            latest.status = "waiting"
            latest.current_step = "等待人工确认"
            save_project(latest)
            return latest
        final_state = _coerce_project(result["project"])
        if active_question(final_state):
            final_state.status = "waiting"
            final_state.current_step = "等待人工确认"
            final_state.active_question_id = active_question(final_state).id if active_question(final_state) else None
            save_project(final_state)
            return final_state
        final_state.status = "done"
        final_state.current_step = "已完成"
        final_state.active_question_id = None
        final_state.risks = ensure_risk_classification(final_state.risks)
        final_state.export_path = str(export_risks(final_state.id, final_state.config.project_name, final_state.risks))
        append_log(final_state, f"已导出风险物料清单：{final_state.export_path}")
        save_project(final_state)
        return final_state
    except Exception as exc:
        state = load_project(project_id)
        state.status = "error"
        state.error = str(exc)
        state.active_question_id = None
        append_log(state, f"继续运行失败：{exc}")
        save_project(state)
        return state


def _run_with_langgraph_if_available(initial: AgentState | Any, thread_id: str | None = None) -> AgentState:
    """优先使用 LangGraph；没有依赖时按固定顺序执行节点。"""

    if not _can_use_langgraph_checkpoint():
        return _run_sequential(initial)

    try:
        from langgraph.graph import END, StateGraph
        from langgraph.checkpoint.sqlite import SqliteSaver
    except Exception:
        return _run_sequential(initial)

    graph = StateGraph(AgentState)
    graph.add_node("parse_inputs", _node_parse_inputs)
    graph.add_node("fetch_lark", _node_fetch_lark)
    graph.add_node("gate_after_lark", _node_question_gate)
    graph.add_node("extract_facts", _node_extract_facts)
    graph.add_node("gate_after_facts", _node_question_gate)
    graph.add_node("extract_risks", _node_extract_risks)
    graph.add_node("gate_after_risks", _node_question_gate)
    graph.add_node("merge_risks", _node_merge_risks)
    graph.add_node("classify_risks", _node_classify_risks)
    graph.add_node("build_risk_questions", _node_build_risk_questions)
    graph.add_node("gate_after_questions", _node_question_gate)
    graph.set_entry_point("parse_inputs")
    graph.add_edge("parse_inputs", "fetch_lark")
    graph.add_edge("fetch_lark", "gate_after_lark")
    graph.add_edge("gate_after_lark", "extract_facts")
    graph.add_edge("extract_facts", "gate_after_facts")
    graph.add_edge("gate_after_facts", "extract_risks")
    graph.add_edge("extract_risks", "gate_after_risks")
    graph.add_edge("gate_after_risks", "merge_risks")
    graph.add_edge("merge_risks", "classify_risks")
    graph.add_edge("classify_risks", "build_risk_questions")
    graph.add_edge("build_risk_questions", "gate_after_questions")
    graph.add_edge("gate_after_questions", END)
    checkpoint_path = settings.data_dir / "cache" / "langgraph_checkpoints.sqlite3"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(checkpoint_path, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    compiled = graph.compile(checkpointer=saver)
    return compiled.invoke(initial, config={"configurable": {"thread_id": thread_id or "default"}})


def _run_sequential(initial: AgentState) -> AgentState:
    """Run the agent nodes without LangGraph checkpoint/interrupt support."""

    current = initial
    for node in [
        _node_parse_inputs,
        _node_fetch_lark,
        _node_question_gate_without_interrupt,
        _node_extract_facts,
        _node_question_gate_without_interrupt,
        _node_extract_risks,
        _node_question_gate_without_interrupt,
        _node_merge_risks,
        _node_classify_risks,
        _node_build_risk_questions,
        _node_question_gate_without_interrupt,
    ]:
        current = node(current)
    return current


def _can_use_langgraph_checkpoint() -> bool:
    """Return whether the installed LangGraph supports checkpoint resume."""

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: F401
        from langgraph.graph import StateGraph  # noqa: F401
        from langgraph.types import Command  # noqa: F401
    except Exception:
        return False
    return True


def _node_parse_inputs(state: AgentState) -> AgentState:
    """节点：解析 BOM、PRD、规格书、图纸和研发自提文件。"""

    project = _coerce_project(state["project"])
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

    return {
        **state,
        "project": project.model_dump(mode="json"),
        "materials": [item.model_dump(mode="json") for item in materials],
        "prd_lines": prd_lines,
        "prd_units": [unit.model_dump(mode="json") for unit in prd_result.units],
        "spec_lines": spec_lines,
        "spec_units": [unit.model_dump(mode="json") for unit in spec_result.units],
        "drawing_lines": drawing_lines,
        "rd_records": rd_records,
    }


def _node_extract_facts(state: AgentState) -> AgentState:
    """节点：从 PRD / 规格书抽取产品事实。"""

    project = _coerce_project(state["project"])
    if active_question(project):
        project.current_step = "等待人工确认"
        save_project(project)
        return {**state, "project": project.model_dump(mode="json")}

    project.current_step = "抽取产品事实"
    save_project(project)

    # 事实层只抽取文档明确事实，避免人工回答改变事实缓存 hash。
    human_context = ""
    facts = []
    prd_units = [_coerce_text_unit(item) for item in state.get("prd_units", [])]
    if prd_units:
        try:
            facts.extend(extract_document_facts(prd_units, source_name="PRD", project_id=project.id, human_context=human_context))
        except Exception as exc:
            append_log(project, f"PRD 产品事实抽取失败，继续执行风险识别：{exc}")
    spec_units = [_coerce_text_unit(item) for item in state.get("spec_units", [])]
    if spec_units:
        try:
            facts.extend(extract_document_facts(spec_units, source_name="规格书", project_id=project.id, human_context=human_context))
        except Exception as exc:
            append_log(project, f"规格书产品事实抽取失败，继续执行风险识别：{exc}")

    project.facts = facts
    append_log(project, f"产品事实 {len(facts)} 条。")
    save_project(project)
    return {**state, "project": project.model_dump(mode="json")}


def _node_fetch_lark(state: AgentState) -> AgentState:
    """节点：读取飞书群历史消息。"""

    project = _coerce_project(state["project"])
    project.current_step = "读取飞书群聊"
    save_project(project)

    messages, questions, logs = fetch_messages_by_chat(project.id, project.config.lark_chat or "")
    for log in logs:
        append_log(project, log)
    if messages:
        save_lark_messages(project.id, messages)
    upsert_result = _upsert_questions(project, questions)
    _append_question_upsert_log(project, upsert_result, "飞书群聊")
    return {
        **state,
        "project": project.model_dump(mode="json"),
        "questions": [question.model_dump(mode="json") for question in project.questions],
        "lark_messages": [message.model_dump(mode="json") for message in messages],
    }


def _node_extract_risks(state: AgentState) -> AgentState:
    """节点：各来源独立抽取候选风险。"""

    project = _coerce_project(state["project"])
    materials = [_coerce_material(item) for item in state.get("materials", [])]
    if active_question(project):
        project.current_step = "等待人工确认"
        save_project(project)
        return {
            **state,
            "project": project.model_dump(mode="json"),
            "candidate_risks": [],
            "questions": [question.model_dump(mode="json") for question in project.questions],
        }

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
        lambda: extract_document_risks(state.get("prd_lines", []), materials, "PRD", project.id, human_context=human_context, fact_context=fact_context),
        lambda: extract_document_risks(state.get("spec_lines", []), materials, "规格书", project.id, human_context=human_context, fact_context=fact_context),
        lambda: extract_document_risks(state.get("drawing_lines", []), materials, "PDF图纸", project.id, human_context=human_context, fact_context=fact_context),
        lambda: extract_rd_risks(state.get("rd_records", []), materials, project.id, human_context=human_context, fact_context=fact_context),
        lambda: extract_lark_risks([_coerce_lark_message(item) for item in state.get("lark_messages", [])], materials, project.id, human_context=human_context, fact_context=fact_context),
    ]
    for extractor in extractors:
        source_risks, source_questions = extractor()
        risks.extend(source_risks)
        upsert_result = _upsert_questions(project, source_questions)
        generated_questions.extend(upsert_result["added"])
        generated_questions.extend(upsert_result["refreshed"])
        if active_question(project):
            append_log(project, f"候选风险物料已暂存 {len(risks)} 条，等待采购回答阻塞型问题。")
            return {
                **state,
                "project": project.model_dump(mode="json"),
                "candidate_risks": [risk.model_dump(mode="json") for risk in risks],
                "questions": [question.model_dump(mode="json") for question in project.questions],
            }

    append_log(project, f"候选风险物料 {len(risks)} 条。")
    if generated_questions:
        append_log(project, f"大模型生成采购确认问题 {len(generated_questions)} 个。")
    return {
        **state,
        "project": project.model_dump(mode="json"),
        "candidate_risks": [risk.model_dump(mode="json") for risk in risks],
        "questions": [question.model_dump(mode="json") for question in project.questions],
    }


def _node_merge_risks(state: AgentState) -> AgentState:
    """节点：合并同物料、同风险类型的候选风险。"""

    project = _coerce_project(state["project"])
    if active_question(project):
        project.current_step = "等待人工确认"
        save_project(project)
        return {**state, "project": project.model_dump(mode="json"), "candidate_risks": state.get("candidate_risks", [])}

    project.current_step = "合并风险物料"
    save_project(project)

    merged = merge_risks([_coerce_risk(item) for item in state.get("candidate_risks", [])])
    project.risks = merged
    append_log(project, f"合并后风险物料 {len(merged)} 条。")
    save_project(project)
    return {
        **state,
        "project": project.model_dump(mode="json"),
        "candidate_risks": [risk.model_dump(mode="json") for risk in merged],
        "questions": [question.model_dump(mode="json") for question in project.questions],
    }


def _node_classify_risks(state: AgentState) -> AgentState:
    """节点：为风险补充主归口、物料属性、标签和可发群问题。"""

    project = _coerce_project(state["project"])
    if active_question(project):
        project.current_step = "等待人工确认"
        save_project(project)
        return {**state, "project": project.model_dump(mode="json"), "candidate_risks": state.get("candidate_risks", [])}

    project.current_step = "风险分类与问题清单"
    save_project(project)

    materials = [_coerce_material(item) for item in state.get("materials", [])]
    classified = classify_risks([_coerce_risk(item) for item in state.get("candidate_risks", [])], materials)
    project.risks = classified
    append_log(project, f"已完成风险分类、标签和采购前置问题生成 {len(classified)} 条。")
    save_project(project)
    return {
        **state,
        "project": project.model_dump(mode="json"),
        "candidate_risks": [risk.model_dump(mode="json") for risk in classified],
        "questions": [question.model_dump(mode="json") for question in project.questions],
    }


def _node_build_risk_questions(state: AgentState) -> AgentState:
    """节点：基于风险结果生成现有人工确认问题。"""

    project = _coerce_project(state["project"])
    if active_question(project):
        project.current_step = "等待人工确认"
        save_project(project)
        return {**state, "project": project.model_dump(mode="json"), "candidate_risks": state.get("candidate_risks", [])}

    project.current_step = "生成风险确认问题"
    save_project(project)

    risks = [_coerce_risk(item) for item in state.get("candidate_risks", [])]
    questions = build_questions(risks)
    project.risks = risks
    upsert_result = _upsert_questions(project, questions)
    cleaned_count = _remove_stale_risk_action_questions(project)
    _append_question_upsert_log(project, upsert_result, "风险确认")
    if cleaned_count:
        append_log(project, f"已清理过期风险确认问题 {cleaned_count} 个。")
    append_log(project, f"待人工确认问题 {len(project.questions)} 个。")
    save_project(project)
    return {
        **state,
        "project": project.model_dump(mode="json"),
        "candidate_risks": [risk.model_dump(mode="json") for risk in risks],
        "questions": [question.model_dump(mode="json") for question in project.questions],
    }


def _node_question_gate(state: AgentState) -> AgentState:
    """节点：遇到 pending required 问题时 interrupt 暂停。"""

    try:
        from langgraph.types import interrupt
    except Exception:
        return state

    project = _coerce_project(state["project"])
    try:
        project = load_project(project.id)
    except FileNotFoundError:
        pass
    _prepare_question_state(project)
    pending = pending_required_questions(project)
    if not pending:
        project.active_question_id = None
        if project.status == "waiting":
            project.status = "running"
            project.current_step = "继续识别"
        save_project(project)
        return {**state, "project": project.model_dump(mode="json"), "questions": [q.model_dump(mode="json") for q in project.questions]}

    question = pending[0]
    project.status = "waiting"
    project.current_step = "等待人工确认"
    project.active_question_id = question.id
    save_project(project)
    interrupt({"project_id": project.id, "question": question.model_dump(mode="json")})
    latest = load_project(project.id)
    latest.status = "running"
    latest.current_step = "继续识别"
    latest.active_question_id = None
    save_project(latest)
    return {**state, "project": latest.model_dump(mode="json"), "questions": [q.model_dump(mode="json") for q in latest.questions]}


def _node_question_gate_without_interrupt(state: AgentState) -> AgentState:
    """顺序执行模式的问题门禁，不能调用 LangGraph interrupt。"""

    project = _coerce_project(state["project"])
    try:
        project = load_project(project.id)
    except FileNotFoundError:
        pass
    _prepare_question_state(project)
    pending = pending_required_questions(project)
    if not pending:
        project.active_question_id = None
        if project.status == "waiting":
            project.status = "running"
            project.current_step = "继续识别"
        save_project(project)
        return {**state, "project": project.model_dump(mode="json"), "questions": [q.model_dump(mode="json") for q in project.questions]}

    question = pending[0]
    project.status = "waiting"
    project.current_step = "等待人工确认"
    project.active_question_id = question.id
    save_project(project)
    return {**state, "project": project.model_dump(mode="json"), "questions": [q.model_dump(mode="json") for q in project.questions], "__interrupt__": True}


def _prepare_question_state(project: ProjectState) -> None:
    """同步问题 key、历史决策和决策抑制。"""

    sync_all_question_keys(project)
    ensure_decisions_from_completed_questions(project)
    suppress_questions_by_decisions(project)


def _thread_id(project: ProjectState) -> str:
    """生成 LangGraph checkpoint thread id。"""

    return f"{project.id}:{project.run_revision}"


def _coerce_project(value: ProjectState | dict[str, Any]) -> ProjectState:
    """把 checkpoint 中的 project 转回 ProjectState。"""

    if isinstance(value, ProjectState):
        return value
    return ProjectState.model_validate(value)


def _coerce_material(value: MaterialRecord | dict[str, Any]) -> MaterialRecord:
    if isinstance(value, MaterialRecord):
        return value
    return MaterialRecord.model_validate(value)


def _coerce_risk(value: RiskItem | dict[str, Any]) -> RiskItem:
    if isinstance(value, RiskItem):
        return value
    return RiskItem.model_validate(value)


def _coerce_text_unit(value: DocumentTextUnit | dict[str, Any]) -> DocumentTextUnit:
    if isinstance(value, DocumentTextUnit):
        return value
    return DocumentTextUnit.model_validate(value)


def _coerce_lark_message(value: LarkMessage | dict[str, Any]) -> LarkMessage:
    if isinstance(value, LarkMessage):
        return value
    return LarkMessage.model_validate(value)


class QuestionUpsertResult(TypedDict):
    """问题 upsert 结果，用于日志和评测。"""

    added: list[Question]
    refreshed: list[Question]
    ignored: list[Question]


def _upsert_questions(project: ProjectState, questions: list[Question]) -> QuestionUpsertResult:
    """按问题签名新增或刷新 pending 问题，避免重跑后重复或丢失。"""

    _prepare_question_state(project)
    _sync_completed_questions_from_latest(project)
    _dedupe_existing_questions(project)
    active_decisions = [decision for decision in project.decisions if not decision.ask_again]
    answered_exact_signatures = {
        _question_exact_signature(question)
        for question in project.questions
        if question.status in {"answered", "skipped", "rejected"}
    }
    answered_stable_signatures = {
        _question_signature(question)
        for question in project.questions
        if question.status in {"answered", "skipped", "rejected"}
    }
    pending_by_exact_signature = {
        _question_exact_signature(question): question
        for question in project.questions
        if question.status == "pending"
    }
    pending_by_stable_signature: dict[tuple[str, str, str], Question] = {}
    for question in project.questions:
        if question.status == "pending":
            pending_by_stable_signature.setdefault(_question_signature(question), question)
    added: list[Question] = []
    refreshed: list[Question] = []
    ignored: list[Question] = []
    for question in questions:
        sync_question_keys(question)
        if matching_decision_for_question(active_decisions, question):
            ignored.append(question)
            continue
        exact_signature = _question_exact_signature(question)
        stable_signature = _question_signature(question)
        if exact_signature in answered_exact_signatures or stable_signature in answered_stable_signatures:
            ignored.append(question)
            continue
        existing = pending_by_exact_signature.get(exact_signature) or pending_by_stable_signature.get(stable_signature)
        if existing:
            _refresh_pending_question(existing, question)
            refreshed.append(existing)
            pending_by_exact_signature[_question_exact_signature(existing)] = existing
            pending_by_stable_signature[_question_signature(existing)] = existing
            continue
        project.questions.append(question)
        pending_by_exact_signature[exact_signature] = question
        pending_by_stable_signature[stable_signature] = question
        added.append(question)
    return {"added": added, "refreshed": refreshed, "ignored": ignored}


def _dedupe_existing_questions(project: ProjectState) -> int:
    """合并历史遗留的重复 pending 问题，已处理问题优先保留。"""

    sync_all_question_keys(project)
    removed_by_decision = suppress_questions_by_decisions(project)
    completed_exact_signatures = {
        _question_exact_signature(question)
        for question in project.questions
        if question.status in {"answered", "skipped", "rejected"}
    }
    completed_stable_signatures = {
        _question_signature(question)
        for question in project.questions
        if question.status in {"answered", "skipped", "rejected"}
    }
    pending_by_exact_signature: dict[tuple[str, str, str, str], Question] = {}
    pending_by_stable_signature: dict[tuple[str, str, str], Question] = {}
    kept: list[Question] = []
    removed_count = removed_by_decision

    for question in project.questions:
        exact_signature = _question_exact_signature(question)
        stable_signature = _question_signature(question)
        if question.status in {"answered", "skipped", "rejected"}:
            kept.append(question)
            continue

        if question.status != "pending":
            kept.append(question)
            continue

        if exact_signature in completed_exact_signatures or stable_signature in completed_stable_signatures:
            removed_count += 1
            continue
        if exact_signature in pending_by_exact_signature or stable_signature in pending_by_stable_signature:
            removed_count += 1
            continue

        pending_by_exact_signature[exact_signature] = question
        pending_by_stable_signature[stable_signature] = question
        kept.append(question)

    if removed_count:
        project.questions = kept
    return removed_count


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
    existing.topic_key = incoming.topic_key
    existing.intent_key = incoming.intent_key
    existing.suppressed_by_decision_id = incoming.suppressed_by_decision_id


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


def _question_signature(question: Question) -> tuple[str, str, str]:
    """生成问题去重签名，优先使用稳定业务指纹。"""

    business_signature = _question_business_signature(question)
    if business_signature:
        return business_signature

    return (
        _question_kind_group(question),
        _question_stable_subject(question),
        _question_stable_source(question),
    )


def _question_exact_signature(question: Question) -> tuple[str, str, str, str]:
    """生成严格签名，用于完全相同问题的优先匹配。"""

    return (
        question.question_kind,
        question.title.strip(),
        question.message.strip(),
        str(question.context.get("source_excerpt") or "").strip(),
    )


def _question_stable_subject(question: Question) -> str:
    """从问题文本中提取稳定主题，降低模型改写导致的重复提问。"""

    text = _normalize_question_text(" ".join([question.title, question.message, question.reason]))
    keywords = _matched_terms(text, _IMPORTANT_QUESTION_KEYWORDS)
    if keywords:
        return "|".join(keywords)
    return text[:120]


def _question_stable_source(question: Question) -> str:
    """提取问题来源里的稳定锚点，如 PRD 行号、BOM 行号或相关风险 ID。"""

    context = question.context or {}
    source_excerpt = _normalize_question_text(str(context.get("source_excerpt") or ""))
    source_text = _normalize_question_text(
        " ".join(
            [
                str(context.get("source_name") or ""),
                source_excerpt,
                str(context.get("risk_id") or ""),
                str(context.get("left_material") or ""),
                str(context.get("right_material") or ""),
            ]
        )
    )
    anchors = re.findall(r"(?:PRD|BOM|R|row|行)\s*[:=]?\s*\d+", source_text, flags=re.IGNORECASE)
    keywords = _matched_terms(source_text, _IMPORTANT_QUESTION_KEYWORDS)
    if anchors or keywords:
        return "|".join(sorted(set(anchors)) + keywords)
    if source_excerpt:
        return source_excerpt[:120]
    related_ids = [risk_id for risk_id in question.related_risk_ids if risk_id]
    if related_ids:
        return "risk:" + "|".join(sorted(related_ids))
    return source_text[:120]


def _normalize_question_text(value: str) -> str:
    """压缩问题文本，去掉标点和空白，保留中英文数字关键词。"""

    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", value).lower()


def _question_business_signature(question: Question) -> tuple[str, str, str] | None:
    """生成面向业务对象和确认意图的稳定指纹。"""

    if question.question_kind == "risk_merge_review":
        pair = _merge_material_pair_key(question)
        if pair:
            return ("risk_merge_review", pair, "merge")

    if question.question_kind == "risk_keep_review":
        subject = _risk_keep_subject_key(question)
        if subject:
            return ("risk_keep_review", subject, _evidence_anchor_key(question))

    if _question_kind_group(question) == "human_confirmation":
        subject = _question_object_key(question)
        intent = _question_intent_key(question)
        if subject and intent:
            return ("human_confirmation", subject, intent)
        if subject:
            return ("human_confirmation", subject, _question_stable_source(question))

    return None


def _question_kind_group(question: Question) -> str:
    """把模型偶发混用的业务分类归到同一去重组。"""

    if question.question_kind in {"procurement_confirmation", "risk_material_mapping"}:
        return "human_confirmation"
    return question.question_kind


def _merge_material_pair_key(question: Question) -> str:
    """提取风险合并问题里的无序物料对。"""

    context = question.context or {}
    left = str(context.get("left_material") or "").strip()
    right = str(context.get("right_material") or "").strip()
    if not left or not right:
        quoted = re.findall(r"[“\"]([^”\"]+)[”\"]", question.message)
        if len(quoted) >= 2:
            left, right = quoted[0], quoted[1]
    materials = [_normalize_material_key(value) for value in [left, right] if value]
    if len(materials) < 2:
        return ""
    return "|".join(sorted(materials[:2]))


def _risk_keep_subject_key(question: Question) -> str:
    """提取风险保留问题的物料主体。"""

    context = question.context or {}
    material_name = str(context.get("material_name") or "").strip()
    if not material_name:
        match = re.search(r"物料[“\"]([^”\"]+)[”\"]", question.message)
        if match:
            material_name = match.group(1)
    material_key = _normalize_material_key(material_name)
    risk_type = _normalize_question_text(str(context.get("risk_type") or ""))
    subject = "|".join(part for part in [material_key, risk_type] if part)
    return subject or _question_object_key(question)


def _question_object_key(question: Question) -> str:
    """按业务对象生成问题主体 key。"""

    context = question.context or {}
    context_text = " ".join(
        str(context.get(key) or "")
        for key in [
            "material_name",
            "left_material",
            "right_material",
            "source_name",
            "source_excerpt",
            "query",
        ]
    )
    text = _normalize_question_text(" ".join([question.title, question.message, question.reason, context_text]))
    return "|".join(_matched_terms(text, _QUESTION_OBJECT_KEYWORDS))


def _question_intent_key(question: Question) -> str:
    """按确认意图生成问题动作 key。"""

    text = _normalize_question_text(" ".join([question.title, question.message, question.reason]))
    intents: list[str] = []
    for intent, keywords in _QUESTION_INTENT_KEYWORDS.items():
        if _matched_terms(text, keywords):
            intents.append(intent)
    if "supplier" in intents:
        return "supplier"
    if "bom_mapping" in intents:
        return "bom_mapping"
    return "|".join(sorted(intents))


def _evidence_anchor_key(question: Question) -> str:
    """提取弱证据问题的稳定来源锚点。"""

    source_key = _question_stable_source(question)
    if source_key:
        return source_key
    return _question_stable_subject(question)


def _normalize_material_key(value: str) -> str:
    """把物料名归一到可跨轮次比较的形式。"""

    text = _normalize_question_text(value)
    text = re.sub(r"(左侧|右侧|左轴|右轴|左|右)$", "", text)
    text = re.sub(r"(left|right)$", "", text)
    return text[:120]


def _matched_terms(text: str, terms: set[str]) -> list[str]:
    """查找已归一化文本中的业务关键词。"""

    matched: set[str] = set()
    for term in terms:
        normalized = _normalize_question_text(term)
        if normalized and normalized in text:
            matched.add(term)
    return sorted(matched)


_IMPORTANT_QUESTION_KEYWORDS = {
    "15-5ph",
    "7075t6",
    "ant",
    "bluetooth",
    "keo",
    "m7",
    "m18",
    "pcba",
    "spd",
    "spd-sl",
    "spdsl",
    "不锈钢",
    "主板pcba",
    "低压注塑硬胶",
    "磁吸",
    "充电",
    "充电接口",
    "充电线",
    "充电柱",
    "圆形天线",
    "天线",
    "尾端顶紧",
    "注塑",
    "应变片",
    "fpc",
    "接口",
    "材料",
    "材质",
    "欧盟",
    "电池",
    "电阻应变片",
    "认证",
    "螺丝",
    "螺钉",
    "碳纤维",
    "脚踏",
    "脚踏本体",
    "踏板",
    "踏板本体",
    "锁片",
    "轴心",
    "锂电池",
    "长碳纤维",
    "阻值",
    "供应商",
    "定点",
}


_QUESTION_OBJECT_KEYWORDS = _IMPORTANT_QUESTION_KEYWORDS - {
    "材料",
    "材质",
    "认证",
    "欧盟",
    "供应商",
    "定点",
}


_QUESTION_INTENT_KEYWORDS = {
    "bom_mapping": {
        "bom",
        "对应",
        "归属",
        "纳入",
        "物料清单",
        "单独采购",
        "采购管理",
        "独立bom",
    },
    "certification": {
        "认证",
        "合规",
        "法规",
        "ce",
        "fcc",
        "rohs",
        "un38.3",
        "iec62133",
        "欧盟",
    },
    "cost": {
        "成本",
        "价格",
        "目标成本",
        "售价",
    },
    "process": {
        "工艺",
        "模具",
        "注塑",
        "贴片",
        "盐雾",
        "处理方案",
        "量产经验",
        "量产能力",
    },
    "schedule": {
        "交期",
        "周期",
        "时间节点",
        "开发阶段",
        "开发进展",
    },
    "spec": {
        "参数",
        "尺寸",
        "型号",
        "品牌",
        "容量",
        "技术规格",
        "材质",
        "材料标准",
        "规格",
        "牌号",
        "阻值",
    },
    "supplier": {
        "供应商",
        "定点",
        "寻源",
        "资源",
        "储备",
        "备选",
        "开发供应商",
        "成熟供应商",
        "合格供应商",
    },
}
