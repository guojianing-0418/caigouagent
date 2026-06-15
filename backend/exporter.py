"""风险物料 Excel 导出。"""

from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook
from openpyxl.worksheet.worksheet import Worksheet
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from .config import ensure_data_dirs, settings
from .models import DocumentSourceRef, RiskItem


EXPORT_HEADERS = ["风险物料名称", "所属模块", "风险类型", "风险原因", "来源与依据"]
EVIDENCE_HEADERS = [
    "风险物料名称",
    "风险类型",
    "置信度",
    "待确认点",
    "来源名称",
    "文件名",
    "Sheet",
    "行号",
    "页码",
    "解析器",
    "证据原文",
]
HEADER_FILL = "2F6F5E"


def export_risks(project_id: str, project_name: str, risks: list[RiskItem]) -> Path:
    """导出最终风险物料清单。"""

    ensure_data_dirs()
    export_dir = settings.data_dir / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in project_name)
    path = export_dir / f"{safe_name}_{project_id}_计划阶段风险物料.xlsx"

    wb = Workbook()
    ws = wb.active
    ws.title = "计划阶段风险物料"

    ws.append(EXPORT_HEADERS)
    _style_header(ws)

    for risk in risks:
        ws.append(
            [
                risk.material_name,
                risk.module,
                risk.risk_type,
                risk.risk_reason,
                risk.source_basis,
            ]
        )

    _style_body(ws, [24, 22, 20, 44, 80])
    _append_evidence_sheet(wb, risks)
    wb.save(path)
    return path


def create_rd_risk_template() -> Path:
    """生成研发自提风险 Excel 模板。"""

    ensure_data_dirs()
    path = settings.data_dir / "templates" / "研发自提风险模板.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.title = "研发自提风险"
    ws.append(["风险物料名称", "原因"])
    ws.column_dimensions["A"].width = 28
    ws.column_dimensions["B"].width = 60
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="2F6F5E")
        cell.alignment = Alignment(horizontal="center")
    wb.save(path)
    return path


def _append_evidence_sheet(wb: Workbook, risks: list[RiskItem]) -> None:
    """追加结构化证据详情页，主表字段保持不变。"""

    ws = wb.create_sheet("证据详情")
    ws.append(EVIDENCE_HEADERS)
    _style_header(ws)

    for risk in risks:
        for row in _evidence_rows_for_risk(risk):
            ws.append(row)

    _style_body(ws, [24, 20, 12, 40, 18, 28, 18, 10, 10, 18, 90])


def _evidence_rows_for_risk(risk: RiskItem) -> list[list[str | int]]:
    """把单条风险转换为证据明细行；没有结构化证据时用来源依据兜底。"""

    confidence = _format_confidence(risk.confidence)
    unresolved = "\n".join(risk.unresolved_questions)
    evidence_items = risk.evidence_items or [
        DocumentSourceRef(
            source_name="来源与依据",
            excerpt=risk.source_basis,
        )
    ]
    rows: list[list[str | int]] = []
    for evidence in evidence_items:
        rows.append(
            [
                risk.material_name,
                risk.risk_type,
                confidence,
                unresolved,
                evidence.source_name,
                evidence.file_name,
                evidence.sheet,
                evidence.row_number or "",
                evidence.page_number or "",
                evidence.parser,
                evidence.excerpt or risk.source_basis,
            ]
        )
    return rows


def _format_confidence(confidence: float | None) -> str:
    """把 0-1 置信度显示成百分比。"""

    if confidence is None:
        return ""
    return f"{max(0, min(100, round(confidence * 100)))}%"


def _style_header(ws: Worksheet) -> None:
    """统一表头样式。"""

    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor=HEADER_FILL)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)


def _style_body(ws: Worksheet, widths: list[int]) -> None:
    """统一列宽、换行和冻结首行。"""

    for index, width in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(index)].width = width

    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)

    ws.freeze_panes = "A2"

