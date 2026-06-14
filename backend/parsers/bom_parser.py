"""草 BOM 解析。

当前 BOM 是多 sheet 结构，第一行一般是表头。
本模块只负责把 BOM 解析成物料清单，不在这里直接判断风险。
"""

from __future__ import annotations

from pathlib import Path

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

        for offset, row in enumerate(rows[header_index + 1 :], start=header_index + 2):
            name = get_by_header(row, headers, ["物料名称"])
            if not name:
                continue
            level = get_by_header(row, headers, ["项目层级"])
            spec = get_by_header(row, headers, ["规格型号"])
            material = get_by_header(row, headers, ["材质"])
            quantity = get_by_header(row, headers, ["单位用量", "数量"])

            materials.append(
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

    return materials

