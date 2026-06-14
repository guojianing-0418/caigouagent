"""研发自提风险模板解析。

模板只有两列：风险物料名称、原因。
这部分是研发主动提报，优先级较高，不做复杂推理。
"""

from __future__ import annotations

from pathlib import Path

from .excel_utils import find_header_index, get_by_header, header_map, read_sheet_rows


def parse_rd_risk_excel(path: str | Path | None) -> list[dict[str, str]]:
    """读取研发自提风险 Excel。"""

    if not path:
        return []
    file_path = Path(path)
    if not file_path.exists():
        return []

    records: list[dict[str, str]] = []
    for sheet_name, rows in read_sheet_rows(file_path).items():
        if not rows:
            continue
        header_index = find_header_index(rows, "风险物料名称")
        headers = header_map(rows[header_index])
        for row in rows[header_index + 1 :]:
            name = get_by_header(row, headers, ["风险物料名称"])
            reason = get_by_header(row, headers, ["原因"])
            if name:
                records.append({"material_name": name, "reason": reason, "sheet": sheet_name})
    return records

