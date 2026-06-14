"""风险物料 Excel 导出。"""

from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from .config import ensure_data_dirs, settings
from .models import RiskItem


EXPORT_HEADERS = ["风险物料名称", "所属模块", "风险类型", "风险原因", "来源与依据"]


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
    for cell in ws[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="2F6F5E")
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

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

    widths = [24, 22, 20, 44, 80]
    for index, width in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(index)].width = width

    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)

    ws.freeze_panes = "A2"
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

