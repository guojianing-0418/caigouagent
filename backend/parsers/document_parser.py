"""PRD / 规格书解析。

PRD 往往是“文档化 Excel”，不是标准二维数据库表。
第一版先把 Excel/PDF/TXT 转成文本行，再交给风险规则和可选大模型处理。
"""

from __future__ import annotations

from pathlib import Path

from .excel_utils import excel_as_text_lines


def parse_document_lines(path: str | Path | None, max_lines: int = 800) -> list[str]:
    """读取 PRD 或规格书文本。

    支持 xlsx/xlsm/txt/pdf。PDF 依赖 PyMuPDF，如果没有安装会返回提示行。
    """

    if not path:
        return []
    file_path = Path(path)
    if not file_path.exists():
        return [f"文件不存在: {file_path}"]

    suffix = file_path.suffix.lower()
    if suffix in [".xlsx", ".xlsm"]:
        return excel_as_text_lines(file_path, max_lines=max_lines)
    if suffix in [".txt", ".md", ".csv"]:
        return file_path.read_text(encoding="utf-8", errors="ignore").splitlines()[:max_lines]
    if suffix == ".pdf":
        return _read_pdf_text(file_path, max_lines=max_lines)
    return [f"暂不支持的规格书格式: {file_path.name}"]


def _read_pdf_text(path: Path, max_lines: int) -> list[str]:
    """抽取 PDF 文本，主要用于规格书，不替代图纸视觉识别。"""

    try:
        import fitz  # PyMuPDF
    except Exception:
        return ["未安装 PyMuPDF，无法抽取 PDF 文本。请安装 requirements.txt 后重试。"]

    lines: list[str] = []
    with fitz.open(path) as doc:
        for page_index, page in enumerate(doc, start=1):
            text = page.get_text("text")
            for line in text.splitlines():
                clean = line.strip()
                if clean:
                    lines.append(f"{path.name} P{page_index}: {clean}")
                if len(lines) >= max_lines:
                    return lines
    return lines

