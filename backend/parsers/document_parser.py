"""PRD / 规格书解析兼容入口。

新实现委托给统一 DocumentIngestor，同时保留旧的 list[str] 返回形态。
"""

from __future__ import annotations

from pathlib import Path

from .document_ingestor import ingest_document


def parse_document_lines(path: str | Path | None, max_lines: int = 800) -> list[str]:
    """读取 PRD 或规格书文本，返回兼容旧流程的文本行。"""

    result = ingest_document(path, max_units=max_lines)
    if result.units:
        return result.lines(max_lines=max_lines)
    return result.diagnostics.warnings[:max_lines]
