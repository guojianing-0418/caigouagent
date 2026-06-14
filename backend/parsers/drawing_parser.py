"""PDF 图纸解析。

图纸需要“整张图都看”，所以本模块会：
1. 抽取 PDF 可复制文字；
2. 将页面渲染为图片；
3. 如果配置了视觉模型，则让模型识别标题栏、技术要求、公差和工艺难点。
"""

from __future__ import annotations

from pathlib import Path

from ..llm_client import call_vision


DRAWING_PROMPT = """你是采购早期风险识别助手。请阅读这页 PDF 图纸的整张图，重点识别：
1. 标题栏中的零件名称、材料、表面处理、版本；
2. 技术要求中的特殊加工、检验、装配、防水、可靠性要求；
3. 尺寸、公差、配合、薄壁、小孔、螺纹、复杂结构；
4. 可能影响采购供应、周期、成本、质量验证或工艺实现的风险点。
请用简短中文要点输出，不要编造看不到的信息。"""


def parse_drawing_folder(folder: str | Path | None, max_pages_per_pdf: int = 20) -> list[str]:
    """读取图纸文件夹中所有 PDF，并返回图纸风险线索文本。"""

    if not folder:
        return []
    drawing_dir = Path(folder)
    if not drawing_dir.exists():
        return [f"图纸文件夹不存在: {drawing_dir}"]

    pdfs = sorted(drawing_dir.glob("*.pdf"))
    if not pdfs:
        return []

    try:
        import fitz  # PyMuPDF
    except Exception:
        return ["未安装 PyMuPDF，无法解析 PDF 图纸。请安装 requirements.txt 后重试。"]

    lines: list[str] = []
    for pdf_path in pdfs:
        with fitz.open(pdf_path) as doc:
            for page_index, page in enumerate(doc, start=1):
                if page_index > max_pages_per_pdf:
                    lines.append(f"{pdf_path.name}: 超过 {max_pages_per_pdf} 页，后续页面本次跳过。")
                    break

                text = page.get_text("text").strip()
                if text:
                    compact = " ".join(text.split())
                    lines.append(f"{pdf_path.name} P{page_index} 可复制文字: {compact[:1200]}")

                # 渲染整页图片给视觉模型。模型配置会在 Agent 启动时强制检查。
                pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
                vision_text = call_vision(DRAWING_PROMPT, pix.tobytes("png"))
                if vision_text:
                    lines.append(f"{pdf_path.name} P{page_index} 视觉识别: {vision_text[:1800]}")
    return lines
