"""系统数据模型。

这些模型既用于 FastAPI 接口，也用于 Agent 内部状态传递。
字段命名尽量直接，避免复杂继承，方便 IT 查看。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


PLAN_STAGE = "计划阶段"

RISK_TYPES = [
    "新物料/新技术风险",
    "供应资源风险",
    "长周期风险",
    "定制工艺风险",
    "关键性能风险",
    "成本达成风险",
    "质量验证风险",
    "接口匹配风险",
]

FACT_TYPES = [
    "性能指标",
    "材料要求",
    "认证要求",
    "规格/版本",
    "成本目标",
    "使用环境",
    "结构/接口",
    "区域/销售约束",
    "开发/交付约束",
]


class DocumentSourceRef(BaseModel):
    """可追溯到原始文档位置的来源引用。"""

    source_name: str = ""
    file_name: str = ""
    sheet: str = ""
    row_number: int = 0
    page_number: int = 0
    parser: str = ""
    excerpt: str = ""


class DocumentTextUnit(BaseModel):
    """统一文档摄取后的一条文本单元。"""

    text: str
    source_name: str = ""
    file_name: str = ""
    sheet: str = ""
    row_number: int = 0
    page_number: int = 0
    parser: str = ""
    raw_text: str = ""

    def as_line(self) -> str:
        """转换成兼容现有大模型提示词的文本行。"""

        if self.sheet and self.row_number:
            prefix = f"{self.sheet} R{self.row_number}"
        elif self.page_number:
            prefix = f"{self.file_name} P{self.page_number}"
        else:
            prefix = self.file_name or self.source_name or "文档"
        return f"{prefix}: {self.text}"


class DocumentIngestionDiagnostics(BaseModel):
    """文档摄取质量诊断。"""

    file_name: str = ""
    parser: str = ""
    text_unit_count: int = 0
    non_empty_cell_count: int = 0
    suspicious_title_only: bool = False
    warnings: list[str] = Field(default_factory=list)


class DocumentIngestionResult(BaseModel):
    """统一文档摄取结果。"""

    units: list[DocumentTextUnit] = Field(default_factory=list)
    diagnostics: DocumentIngestionDiagnostics = Field(default_factory=DocumentIngestionDiagnostics)

    def lines(self, max_lines: int | None = None) -> list[str]:
        """返回兼容旧解析入口的文本行。"""

        lines = [unit.as_line() for unit in self.units]
        return lines[:max_lines] if max_lines else lines


class FactItem(BaseModel):
    """从 PRD / 规格书等文档抽取出的产品事实。"""

    id: str = Field(default_factory=lambda: uuid4().hex)
    fact_type: str
    subject: str
    value: str
    source_basis: str
    source_refs: list[DocumentSourceRef] = Field(default_factory=list)


class MaterialRecord(BaseModel):
    """从 BOM 中解析出的物料记录。"""

    name: str
    module: str = ""
    sheet: str = ""
    level: str = ""
    spec: str = ""
    material: str = ""
    quantity: str = ""
    row_number: int = 0


class LarkMessage(BaseModel):
    """飞书群聊消息的最小结构。"""

    message_id: str = ""
    chat_id: str = ""
    sender: str = ""
    create_time: str = ""
    content: str = ""
    raw: dict[str, Any] = Field(default_factory=dict)


class RiskItem(BaseModel):
    """最终输出表中的一条风险物料。"""

    id: str = Field(default_factory=lambda: uuid4().hex)
    stage: str = PLAN_STAGE
    material_name: str
    module: str = ""
    risk_type: str
    risk_reason: str
    source_basis: str
    evidence_items: list[DocumentSourceRef] = Field(default_factory=list)
    confidence: float | None = None
    unresolved_questions: list[str] = Field(default_factory=list)


QuestionInputType = Literal["single_select", "multi_select", "boolean", "text", "textarea"]
QuestionPermission = Literal["allow", "ask", "deny"]
QuestionStatus = Literal["pending", "answered", "skipped", "rejected"]
QuestionAction = Literal["submit", "skip", "reject"]


class Question(BaseModel):
    """Agent 执行中产生的结构化问题。

    这里借鉴 opencode question tool 的思想：问题是一种工具调用结果，
    前端按 input_type 渲染成控件，后端按 question_kind 应用答案影响。
    `type` 只用于兼容历史项目，后续业务判断都看 question_kind。
    """

    id: str = Field(default_factory=lambda: uuid4().hex)
    type: str | None = None
    question_kind: str = "general"
    input_type: QuestionInputType = "single_select"
    title: str
    message: str = ""
    reason: str = ""
    options: list[str] = Field(default_factory=list)
    default_value: Any | None = None
    blocking: bool = False
    required: bool = True
    permission: QuestionPermission = "ask"
    context: dict[str, Any] = Field(default_factory=dict)
    allow_custom: bool = True
    related_risk_ids: list[str] = Field(default_factory=list)
    status: QuestionStatus = "pending"
    answer: Any | None = None

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_question(cls, value: Any) -> Any:
        """把旧三类问题自动迁移成通用 Question。

        历史 state.json 中可能仍有 select_lark_chat / confirm_merge /
        confirm_keep，读取时在这里统一补齐新字段，避免历史记录打不开。
        """

        if not isinstance(value, dict):
            return value

        data = dict(value)
        legacy_type = data.get("type")
        mapping = {
            "select_lark_chat": {
                "question_kind": "lark_chat_selection",
                "input_type": "single_select",
                "blocking": True,
                "required": True,
                "permission": "ask",
            },
            "confirm_merge": {
                "question_kind": "risk_merge_review",
                "input_type": "boolean",
                "blocking": False,
                "required": False,
                "permission": "ask",
            },
            "confirm_keep": {
                "question_kind": "risk_keep_review",
                "input_type": "boolean",
                "blocking": False,
                "required": False,
                "permission": "ask",
            },
        }
        if legacy_type in mapping:
            for key, mapped_value in mapping[legacy_type].items():
                data.setdefault(key, mapped_value)
        data.setdefault("message", data.get("title", ""))
        data.setdefault("context", {})
        data.setdefault("allow_custom", True)
        return data


class ProjectConfig(BaseModel):
    """一次项目运行的输入配置。"""

    project_name: str
    bom_path: str
    prd_path: str | None = None
    spec_path: str | None = None
    drawing_dir: str | None = None
    rd_risk_path: str | None = None
    lark_chat: str | None = None


class ProjectState(BaseModel):
    """项目运行状态，持久化为 JSON 文件。"""

    id: str = Field(default_factory=lambda: uuid4().hex)
    stage: str = PLAN_STAGE
    status: Literal["created", "running", "waiting", "done", "error"] = "created"
    current_step: str = "已创建"
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    config: ProjectConfig
    logs: list[str] = Field(default_factory=list)
    risks: list[RiskItem] = Field(default_factory=list)
    facts: list[FactItem] = Field(default_factory=list)
    questions: list[Question] = Field(default_factory=list)
    export_path: str | None = None
    error: str | None = None


class ProjectSummary(BaseModel):
    """历史记录列表中展示的项目摘要。"""

    id: str
    project_name: str
    status: str
    current_step: str
    created_at: str
    updated_at: str
    risk_count: int
    question_count: int
    required_question_count: int = 0
    export_path: str | None = None


class ExportCheckQuestion(BaseModel):
    """导出门禁中需要前端展示的必答问题摘要。"""

    id: str
    title: str
    question_kind: str
    message: str = ""
    reason: str = ""


class ExportCheckResponse(BaseModel):
    """正式导出前的门禁检查结果。"""

    allowed: bool
    pending_required_count: int = 0
    message: str = ""
    questions: list[ExportCheckQuestion] = Field(default_factory=list)


class AnswerRequest(BaseModel):
    """前端提交人工确认答案。"""

    answer: Any
    action: QuestionAction = "submit"


class ModelSettingsRequest(BaseModel):
    """WebUI 提交的模型配置。"""

    openai_base_url: str | None = None
    openai_api_key: str | None = None
    text_model: str | None = None
    vision_model: str | None = None


class ModelSettingsResponse(BaseModel):
    """返回给前端的模型配置。

    出于安全考虑，不返回完整 API key，只返回是否已配置和掩码。
    """

    openai_base_url: str | None = None
    text_model: str
    vision_model: str
    has_api_key: bool
    api_key_masked: str = ""
