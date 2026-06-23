"""草 BOM 解析。

当前 BOM 是多 sheet 结构，第一行一般是表头。
本模块只负责把 BOM 解析成物料清单，不在这里直接判断风险。
"""

from __future__ import annotations

from pathlib import Path
import re

from .excel_utils import find_header_index, get_by_header, header_map, read_sheet_rows
from ..models import MaterialRecord


def parse_bom(path: str | Path) -> list[MaterialRecord]:
    """读取草 BOM，抽取物料名称、模块、层级、规格和材质。"""

    materials: list[MaterialRecord] = []
    sheets = read_sheet_rows(path)

    for sheet_name, rows in sheets.items():
        if not rows:
            continue
        header_index = find_header_index(rows, "物料名称")
        headers = header_map(rows[header_index])

        sheet_materials: list[MaterialRecord] = []
        for offset, row in enumerate(rows[header_index + 1 :], start=header_index + 2):
            name = get_by_header(row, headers, ["物料名称"])
            if not name:
                continue
            level = get_by_header(row, headers, ["项目层级"])
            spec = get_by_header(row, headers, ["规格型号"])
            material = get_by_header(row, headers, ["材质"])
            quantity = get_by_header(row, headers, ["单位用量", "数量"])

            sheet_materials.append(
                MaterialRecord(
                    name=name,
                    module=sheet_name,
                    sheet=sheet_name,
                    level=level,
                    spec=spec,
                    material=material,
                    quantity=quantity,
                    row_number=offset,
                )
            )
        _enrich_bom_hierarchy(sheet_materials)
        materials.extend(sheet_materials)

    return materials


def _enrich_bom_hierarchy(materials: list[MaterialRecord]) -> None:
    """根据同一工作表内的项目层级补齐父级、路径和物料角色。"""

    stack_by_depth: dict[int, MaterialRecord] = {}
    for item in materials:
        level = _normalize_level(item.level)
        item.level = level or item.level
        item.level_depth = _level_depth(level)
        item.parent_level = _parent_level(level)
        parent = _nearest_parent(item, stack_by_depth)
        item.parent_name = parent.name if parent else ""
        item.bom_path = f"{parent.bom_path} > {item.name}" if parent and parent.bom_path else item.name
        item.item_role = _item_role(item)
        if parent:
            parent.has_children = True
            parent.item_role = _item_role(parent)
        if item.level_depth:
            _trim_stack(stack_by_depth, item.level_depth)
            stack_by_depth[item.level_depth] = item


def _normalize_level(level: str) -> str:
    """把 Excel 中可能出现的空格、中文点、尾随点归一成 1.1.2 形式。"""

    text = str(level or "").strip().replace("．", ".").replace("。", ".")
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"\.+", ".", text).strip(".")
    return text if re.fullmatch(r"\d+(?:\.\d+)*", text) else ""


def _parent_level(level: str) -> str:
    if not level or "." not in level:
        return ""
    return level.rsplit(".", 1)[0]


def _level_depth(level: str) -> int:
    return len(level.split(".")) if level else 0


def _item_role(item: MaterialRecord) -> str:
    if not item.level:
        return "待确认"
    if item.level_depth <= 1:
        return "总成"
    if item.has_children:
        return "组件"
    return "零件"


def _nearest_parent(item: MaterialRecord, stack_by_depth: dict[int, MaterialRecord]) -> MaterialRecord | None:
    if not item.level_depth or item.level_depth <= 1:
        return None
    parent = stack_by_depth.get(item.level_depth - 1)
    if parent and _normalize_level(parent.level) == item.parent_level:
        return parent
    return None


def _trim_stack(stack_by_depth: dict[int, MaterialRecord], depth: int) -> None:
    for existing_depth in list(stack_by_depth):
        if existing_depth >= depth:
            del stack_by_depth[existing_depth]

