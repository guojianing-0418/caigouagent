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
from .parsers.bom_parser import parse_bom
from .parsers.document_parser import parse_document_lines
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
from .storage import append_log, save_lark_messages, save_project


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
    state.export_path = None
    # 重新运行时保留已回答问题作为去重依据，清掉旧的非阻塞 pending 问题，避免重复生成。
    state.questions = [
        q
        for q in state.questions
        if q.status in {"answered", "skipped", "rejected"} or (q.status == "pending" and q.blocking)
    ]
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
        for node in [_node_parse_inputs, _node_fetch_lark, _node_extract_risks, _node_merge_and_question]:
            current = node(current)
        return current

    graph = StateGraph(AgentState)
    graph.add_node("parse_inputs", _node_parse_inputs)
    graph.add_node("fetch_lark", _node_fetch_lark)
    graph.add_node("extract_risks", _node_extract_risks)
    graph.add_node("merge_and_question", _node_merge_and_question)
    graph.set_entry_point("parse_inputs")
    graph.add_edge("parse_inputs", "fetch_lark")
    graph.add_edge("fetch_lark", "extract_risks")
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

    prd_lines = parse_document_lines(project.config.prd_path)
    append_log(project, f"PRD 文本线索 {len(prd_lines)} 条。")

    spec_lines = parse_document_lines(project.config.spec_path)
    if project.config.spec_path:
        append_log(project, f"规格书文本线索 {len(spec_lines)} 条。")

    drawing_lines = parse_drawing_folder(project.config.drawing_dir)
    if project.config.drawing_dir:
        append_log(project, f"PDF 图纸线索 {len(drawing_lines)} 条。")

    rd_records = parse_rd_risk_excel(project.config.rd_risk_path)
    if project.config.rd_risk_path:
        append_log(project, f"研发自提风险 {len(rd_records)} 条。")

    project.__dict__["_prd_lines"] = prd_lines
    project.__dict__["_spec_lines"] = spec_lines
    project.__dict__["_drawing_lines"] = drawing_lines
    project.__dict__["_rd_records"] = rd_records
    return {**state, "project": project, "materials": materials}


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
    project.questions.extend(questions)
    return {**state, "project": project, "questions": state.get("questions", []) + questions}


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
    if human_context:
        append_log(project, "已加载人工确认上下文，后续模型识别会参考已提交答案。")
    # 按来源顺序抽取；若某个来源产生阻塞问题，立即暂停，等待采购回答。
    extractors = [
        lambda: extract_bom_risks(materials, project.id, human_context=human_context),
        lambda: extract_document_risks(project.__dict__.get("_prd_lines", []), materials, "PRD", project.id, human_context=human_context),
        lambda: extract_document_risks(project.__dict__.get("_spec_lines", []), materials, "规格书", project.id, human_context=human_context),
        lambda: extract_document_risks(project.__dict__.get("_drawing_lines", []), materials, "PDF图纸", project.id, human_context=human_context),
        lambda: extract_rd_risks(project.__dict__.get("_rd_records", []), materials, project.id, human_context=human_context),
        lambda: extract_lark_risks(project.__dict__.get("_lark_messages", []), materials, project.id, human_context=human_context),
    ]
    for extractor in extractors:
        source_risks, source_questions = extractor()
        source_questions = _filter_repeated_questions(project, source_questions)
        risks.extend(source_risks)
        generated_questions.extend(source_questions)
        project.questions.extend(source_questions)
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
    project.questions.extend(questions)
    append_log(project, f"合并后风险物料 {len(merged)} 条，待人工确认问题 {len(project.questions)} 个。")
    save_project(project)
    return {**state, "project": project, "candidate_risks": merged, "questions": project.questions}


def _filter_repeated_questions(project: ProjectState, questions: list[Question]) -> list[Question]:
    """过滤已经回答过的同类问题，避免 resume 后重复打断采购。"""

    answered_signatures = {
        _question_signature(question)
        for question in project.questions
        if question.status in {"answered", "skipped", "rejected"}
    }
    fresh_questions: list[Question] = []
    for question in questions:
        signature = _question_signature(question)
        if signature in answered_signatures:
            continue
        fresh_questions.append(question)
    return fresh_questions


def _question_signature(question: Question) -> tuple[str, str, str, str]:
    """生成问题去重签名。"""

    return (
        question.question_kind,
        question.title.strip(),
        question.message.strip(),
        str(question.context.get("source_excerpt") or "").strip(),
    )
