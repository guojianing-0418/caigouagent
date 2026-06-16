"""FastAPI 入口。

运行方式：
    python -m backend.main

前端默认通过 Vite 代理访问 /api。
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Annotated

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .agent import _dedupe_existing_questions, resume_plan_stage, run_plan_stage
from .config import ensure_data_dirs, get_effective_settings, mask_api_key, save_model_config, settings
from .exporter import create_rd_risk_template, export_risks
from .models import (
    AnswerRequest,
    ExportCheckQuestion,
    ExportCheckResponse,
    ModelSettingsRequest,
    ModelSettingsResponse,
    ProjectConfig,
    ProjectState,
    ProjectSummary,
    Question,
)
from .decision_engine import ensure_decisions_from_completed_questions, record_decision_for_question, suppress_questions_by_decisions, sync_all_question_keys
from .question_engine import (
    active_question,
    answer_question as record_question_answer,
    apply_question_effect,
    pending_required_questions,
)
from .storage import append_log, list_projects, load_project, project_dir, save_project


app = FastAPI(title="计划阶段采购风险物料识别 Agent", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    """启动时准备本地目录和模板文件。"""

    ensure_data_dirs()
    create_rd_risk_template()


@app.get("/api/health")
def health() -> dict[str, str]:
    """健康检查接口。"""

    return {"status": "ok"}


@app.get("/api/settings/model", response_model=ModelSettingsResponse)
def get_model_settings() -> ModelSettingsResponse:
    """读取当前生效的模型配置。

    API key 只返回掩码，不返回明文。
    """

    runtime = get_effective_settings()
    return ModelSettingsResponse(
        openai_base_url=runtime.openai_base_url,
        text_model=runtime.text_model,
        vision_model=runtime.vision_model,
        has_api_key=bool(runtime.openai_api_key),
        api_key_masked=mask_api_key(runtime.openai_api_key),
    )


@app.post("/api/settings/model", response_model=ModelSettingsResponse)
def update_model_settings(payload: ModelSettingsRequest) -> ModelSettingsResponse:
    """保存模型配置。

    如果前端不传 API key 或传空字符串，则保留已有密钥。
    """

    runtime = save_model_config(
        openai_base_url=(payload.openai_base_url or "").strip(),
        openai_api_key=(payload.openai_api_key or "").strip(),
        text_model=(payload.text_model or "").strip(),
        vision_model=(payload.vision_model or payload.text_model or "").strip(),
        keep_existing_key=True,
    )
    return ModelSettingsResponse(
        openai_base_url=runtime.openai_base_url,
        text_model=runtime.text_model,
        vision_model=runtime.vision_model,
        has_api_key=bool(runtime.openai_api_key),
        api_key_masked=mask_api_key(runtime.openai_api_key),
    )


@app.post("/api/projects")
async def create_project(
    project_name: Annotated[str, Form()],
    lark_chat: Annotated[str | None, Form()] = None,
    bom_file: Annotated[UploadFile, File()] = None,
    prd_file: Annotated[UploadFile | None, File()] = None,
    spec_file: Annotated[UploadFile | None, File()] = None,
    rd_risk_file: Annotated[UploadFile | None, File()] = None,
    drawing_files: Annotated[list[UploadFile] | None, File()] = None,
) -> ProjectState:
    """创建项目并保存上传文件。"""

    if not bom_file:
        raise HTTPException(status_code=400, detail="草 BOM 为必填文件。")

    state = ProjectState(
        config=ProjectConfig(
            project_name=project_name,
            bom_path="",
            prd_path=None,
            spec_path=None,
            drawing_dir=None,
            rd_risk_path=None,
            lark_chat=lark_chat,
        )
    )
    input_dir = project_dir(state.id) / "inputs"
    drawing_dir = input_dir / "drawings"
    input_dir.mkdir(parents=True, exist_ok=True)
    drawing_dir.mkdir(parents=True, exist_ok=True)

    state.config.bom_path = str(await _save_upload(bom_file, input_dir))
    if prd_file:
        state.config.prd_path = str(await _save_upload(prd_file, input_dir))
    if spec_file:
        state.config.spec_path = str(await _save_upload(spec_file, input_dir))
    if rd_risk_file:
        state.config.rd_risk_path = str(await _save_upload(rd_risk_file, input_dir))
    if drawing_files:
        saved_count = 0
        for file in drawing_files:
            if file.filename and file.filename.lower().endswith(".pdf"):
                await _save_upload(file, drawing_dir)
                saved_count += 1
        if saved_count:
            state.config.drawing_dir = str(drawing_dir)

    append_log(state, "项目已创建，等待启动识别。")
    save_project(state)
    return state


@app.get("/api/projects", response_model=list[ProjectSummary])
def get_project_history() -> list[ProjectSummary]:
    """获取本机历史识别项目列表。"""

    return list_projects()


@app.get("/api/projects/{project_id}")
def get_project(project_id: str) -> ProjectState:
    """查询项目状态。"""

    state = _load_or_404(project_id)
    _migrate_question_decisions_if_needed(state)
    _cleanup_duplicate_questions_if_needed(state)
    return state


@app.post("/api/projects/{project_id}/run")
def run_project(project_id: str, background_tasks: BackgroundTasks) -> dict[str, str]:
    """启动计划阶段识别。

    使用后台任务避免前端等待长时间 HTTP 请求。
    """

    state = _load_or_404(project_id)
    _migrate_question_decisions_if_needed(state)
    if state.status == "running":
        return {"status": "running"}
    background_tasks.add_task(run_plan_stage, state)
    return {"status": "started"}


@app.get("/api/projects/{project_id}/questions")
def get_questions(project_id: str):
    """获取待人工确认问题。"""

    state = _load_or_404(project_id)
    _migrate_question_decisions_if_needed(state)
    _cleanup_duplicate_questions_if_needed(state)
    return [q for q in state.questions if q.status == "pending"]


@app.get("/api/projects/{project_id}/questions/active")
def get_active_question(project_id: str):
    """获取当前阻塞流程的最高优先级问题。"""

    state = _load_or_404(project_id)
    _migrate_question_decisions_if_needed(state)
    _cleanup_duplicate_questions_if_needed(state)
    return active_question(state)


@app.post("/api/projects/{project_id}/questions/{question_id}/answer")
def answer_question(project_id: str, question_id: str, payload: AnswerRequest, background_tasks: BackgroundTasks) -> ProjectState:
    """提交人工确认答案，并更新项目状态。"""

    state = _load_or_404(project_id)
    _migrate_question_decisions_if_needed(state)
    question = next((q for q in state.questions if q.id == question_id), None)
    if not question:
        raise HTTPException(status_code=404, detail="问题不存在。")

    was_active_question = state.active_question_id == question_id
    try:
        record_question_answer(question, payload.answer, payload.action)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    state = _merge_answer_into_latest_state(project_id, state, question)
    question = next((q for q in state.questions if q.id == question_id), question)
    apply_question_effect(state, question)
    decision = record_decision_for_question(state, question)
    append_log(state, f"已处理人工确认问题：{question.title}。")
    _cleanup_duplicate_questions_if_needed(state)
    if was_active_question:
        save_project(state)
        background_tasks.add_task(
            resume_plan_stage,
            state.id,
            {
                "question_id": question.id,
                "answer": question.answer,
                "action": payload.action,
                "decision_id": decision.id,
            },
        )
        state.status = "running"
        state.current_step = "继续识别"
        state.active_question_id = None
        save_project(state)
    else:
        _refresh_status_after_question_answer(state)
        save_project(state)
    return state


@app.post("/api/projects/{project_id}/resume")
def resume_project(project_id: str, background_tasks: BackgroundTasks) -> dict[str, str]:
    """回答阻塞问题后继续 Agent 流程。"""

    state = _load_or_404(project_id)
    _migrate_question_decisions_if_needed(state)
    if state.status == "running":
        return {"status": "running"}
    if active_question(state):
        raise HTTPException(status_code=400, detail="仍有阻塞问题未处理，不能继续运行。")
    background_tasks.add_task(resume_plan_stage, state.id, {})
    return {"status": "started"}


@app.get("/api/projects/{project_id}/risks")
def get_risks(project_id: str):
    """获取风险物料预览。"""

    return _load_or_404(project_id).risks


@app.get("/api/projects/{project_id}/export/check", response_model=ExportCheckResponse)
def check_export(project_id: str) -> ExportCheckResponse:
    """检查是否允许下载正式 Excel。"""

    state = _load_or_404(project_id)
    return _build_export_check(state)


@app.get("/api/projects/{project_id}/export")
def download_export(project_id: str) -> FileResponse:
    """下载最终风险物料 Excel。"""

    state = _load_or_404(project_id)
    export_check = _build_export_check(state)
    if not export_check.allowed:
        raise HTTPException(status_code=409, detail=export_check.model_dump())
    if not state.export_path or not Path(state.export_path).exists():
        state.export_path = str(export_risks(state.id, state.config.project_name, state.risks))
        save_project(state)
    return FileResponse(
        state.export_path,
        filename=Path(state.export_path).name,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.get("/api/templates/rd-risk")
def download_rd_template() -> FileResponse:
    """下载研发自提风险 Excel 模板。"""

    path = create_rd_risk_template()
    return FileResponse(
        path,
        filename=path.name,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


async def _save_upload(file: UploadFile, target_dir: Path) -> Path:
    """保存上传文件。"""

    safe_name = Path(file.filename or "upload.bin").name
    path = target_dir / safe_name
    with path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return path


def _load_or_404(project_id: str) -> ProjectState:
    """读取项目，不存在时返回 404。"""

    try:
        return load_project(project_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="项目不存在。")


def _cleanup_duplicate_questions_if_needed(state: ProjectState) -> None:
    """按业务指纹清理历史重复 pending 问题。"""

    sync_all_question_keys(state)
    suppress_questions_by_decisions(state)
    removed_count = _dedupe_existing_questions(state)
    if not removed_count:
        return
    append_log(state, f"已合并重复确认问题 {removed_count} 个。")
    save_project(state)


def _migrate_question_decisions_if_needed(state: ProjectState) -> None:
    """懒迁移历史已处理问题到决策账本。"""

    sync_all_question_keys(state)
    created_count = ensure_decisions_from_completed_questions(state)
    suppressed_count = suppress_questions_by_decisions(state)
    if created_count or suppressed_count:
        if created_count:
            append_log(state, f"已同步历史人工决策 {created_count} 条。")
        if suppressed_count:
            append_log(state, f"已按人工决策隐藏重复确认问题 {suppressed_count} 个。")
        save_project(state)


def _refresh_status_after_question_answer(state: ProjectState) -> None:
    """按问题处理结果刷新项目状态和导出文件。

    阻塞问题处理完后，前端会调用 /resume 继续运行；非阻塞问题处理完后，
    如果已有风险清单，则立即重新导出 Excel。
    """

    if active_question(state):
        state.status = "waiting"
        state.current_step = "等待人工确认"
        return

    if state.risks:
        state.export_path = str(export_risks(state.id, state.config.project_name, state.risks))

    if state.status == "waiting":
        if state.risks:
            state.status = "done"
            state.current_step = "已完成"
        else:
            state.status = "created"
            state.current_step = "人工确认已完成，可继续运行"


def _merge_answer_into_latest_state(project_id: str, state: ProjectState, answered_question: Question) -> ProjectState:
    """把答案合并到磁盘上的最新项目状态，降低后台任务旧 state 覆盖答案的概率。"""

    latest = load_project(project_id)
    if latest.updated_at == state.updated_at:
        return state
    for index, question in enumerate(latest.questions):
        if question.id == answered_question.id:
            latest.questions[index] = answered_question
            break
    else:
        latest.questions.append(answered_question)
    return latest


def _should_auto_rerun_after_answer(state: ProjectState, question: Question, action: str) -> bool:
    """判断非阻塞必答问题回答后是否应自动重新识别。"""

    return False


def _schedule_rerun_after_answer(state: ProjectState, background_tasks: BackgroundTasks) -> None:
    """保存自动重跑状态并把识别任务放入后台。"""

    state.status = "running"
    state.current_step = "准备重新识别"
    state.export_path = None
    append_log(state, "所有必答确认已完成，自动重新识别以应用人工回答。")
    save_project(state)
    background_tasks.add_task(run_plan_stage, state)


def _build_export_check(state: ProjectState) -> ExportCheckResponse:
    """生成正式导出门禁检查结果。"""

    if state.status == "running":
        return ExportCheckResponse(
            allowed=False,
            message="项目正在识别或重新识别，完成后才能下载正式 Excel。",
        )

    pending_required = pending_required_questions(state)
    if not pending_required:
        return ExportCheckResponse(allowed=True, message="可以下载正式 Excel。")

    questions = [
        ExportCheckQuestion(
            id=question.id,
            title=question.title,
            question_kind=question.question_kind,
            message=question.message,
            reason=question.reason,
        )
        for question in pending_required[:10]
    ]
    return ExportCheckResponse(
        allowed=False,
        pending_required_count=len(pending_required),
        message=f"还有 {len(pending_required)} 个必答问题未处理，处理后才能下载正式 Excel。",
        questions=questions,
    )


if __name__ == "__main__":
    import uvicorn

    ensure_data_dirs()
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
