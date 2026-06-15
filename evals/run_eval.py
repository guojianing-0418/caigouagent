"""采购风险 Agent 评测脚本。

用途：
1. mock 模式不调用真实模型，用固定 JSON 验证解析、合并、问题门禁；
2. live 模式调用真实模型，用少量人工标注样本观察召回和误报。

示例：
    python evals/run_eval.py --case evals/cases/p725/case.yaml --mode mock
    python evals/run_eval.py --case evals/cases/p725/case.yaml --mode live
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.fact_rules import extract_document_facts, format_facts_for_prompt
from backend.exporter import EVIDENCE_HEADERS, EXPORT_HEADERS, export_risks
from backend.models import DocumentSourceRef, FactItem, MaterialRecord, ProjectConfig, ProjectState, Question, RiskItem
from backend.parsers.bom_parser import parse_bom
from backend.parsers.document_parser import parse_document_lines
from backend.parsers.document_ingestor import ingest_document
from backend.question_engine import build_human_answer_context, create_question, answer_question
from backend.risk_rules import (
    _chunk_hash,
    _items_and_questions_from_model,
    build_questions,
    extract_bom_risks,
    extract_document_risks,
    merge_risks,
    normalize_name,
)


def main() -> None:
    """读取评测配置并输出报告。"""

    parser = argparse.ArgumentParser(description="运行采购风险 Agent 评测")
    parser.add_argument("--case", required=True, help="评测 case.yaml 路径")
    parser.add_argument("--mode", choices=["mock", "live"], default="mock", help="mock 不调用模型，live 调用真实模型")
    args = parser.parse_args()

    case_path = Path(args.case)
    case = yaml.safe_load(case_path.read_text(encoding="utf-8"))
    expected = _load_expected(Path(case["expected_path"]))
    expected_facts = _load_expected_facts(case)
    materials = parse_bom(case["bom_path"])
    ingestion_report = _run_ingestion_checks(case)
    human_context_report = _run_human_context_check()

    if args.mode == "mock":
        risks, questions, facts = _run_mock(case, materials)
    else:
        risks, questions, facts = _run_live(case, materials)

    export_report = _run_export_check()
    report = _build_report(
        case["case_name"],
        args.mode,
        risks,
        questions,
        expected,
        facts,
        expected_facts,
        ingestion_report,
        human_context_report,
        export_report,
    )
    report_path = _write_report(case["case_name"], args.mode, report)

    print(f"case: {case['case_name']}")
    print(f"mode: {args.mode}")
    print(f"risks: {len(risks)}")
    print(f"questions: {len(questions)}")
    print(f"recall: {report['metrics']['recall']:.2f}")
    print(f"false_positives: {report['metrics']['false_positive_count']}")
    print(f"type_accuracy: {report['metrics']['risk_type_accuracy']:.2f}")
    print(f"evidence_hit_rate: {report['metrics']['evidence_hit_rate']:.2f}")
    print(f"fact_hit_rate: {report['metrics']['fact_hit_rate']:.2f}")
    print(f"prd_line_count: {report['ingestion']['prd_line_count']}")
    print(f"human_context_hash_differs: {report['human_context']['hash_differs']}")
    print(f"export_check: {report['export_check']['status']}")
    print(f"report: {report_path}")


def _run_mock(case: dict[str, Any], materials: list[MaterialRecord]) -> tuple[list[RiskItem], list[Question], list[FactItem]]:
    """用固定模型输出验证后处理链路。"""

    data = json.loads(Path(case["mock_output_path"]).read_text(encoding="utf-8"))
    risks, model_questions = _items_and_questions_from_model(data, "mock", materials)
    merged = merge_risks(risks)
    questions = model_questions + build_questions(merged)
    facts = _keyword_facts_from_ingestion(case)
    return merged, questions, facts


def _run_live(case: dict[str, Any], materials: list[MaterialRecord]) -> tuple[list[RiskItem], list[Question], list[FactItem]]:
    """调用真实模型运行一个轻量评测。"""

    project_id = f"eval-{case['case_name']}"
    risks: list[RiskItem] = []
    questions: list[Question] = []
    facts: list[FactItem] = []

    prd_path = case.get("prd_path")
    prd_lines: list[str] = []
    if prd_path:
        prd_result = ingest_document(prd_path, source_name="PRD")
        prd_lines = prd_result.lines()
        facts.extend(extract_document_facts(prd_result.units, source_name="PRD", project_id=project_id))

    fact_context = format_facts_for_prompt(facts)
    bom_risks, bom_questions = extract_bom_risks(materials, project_id, fact_context=fact_context)
    risks.extend(bom_risks)
    questions.extend(bom_questions)

    if prd_path:
        prd_risks, prd_questions = extract_document_risks(prd_lines, materials, "PRD", project_id, fact_context=fact_context)
        risks.extend(prd_risks)
        questions.extend(prd_questions)

    merged = merge_risks(risks)
    questions.extend(build_questions(merged))
    return merged, questions, facts


def _load_expected(path: Path) -> list[dict[str, Any]]:
    """读取人工期望风险，可用 JSON 或 Excel。"""

    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))

    if path.suffix.lower() in {".xlsx", ".xlsm"}:
        from openpyxl import load_workbook

        wb = load_workbook(path)
        ws = wb.active
        headers = [str(cell.value or "").strip() for cell in ws[1]]
        rows: list[dict[str, Any]] = []
        for row in ws.iter_rows(min_row=2, values_only=True):
            record = {headers[index]: value for index, value in enumerate(row) if index < len(headers)}
            keywords = str(record.get("证据关键词") or "").replace("，", ";").split(";")
            rows.append(
                {
                    "material_name": str(record.get("风险物料名称") or "").strip(),
                    "risk_type": str(record.get("风险类型") or "").strip(),
                    "evidence_keywords": [keyword.strip() for keyword in keywords if keyword.strip()],
                }
            )
        return rows

    raise ValueError(f"不支持的期望结果格式：{path}")


def _load_expected_facts(case: dict[str, Any]) -> list[dict[str, Any]]:
    """读取期望事实清单。"""

    path = case.get("expected_facts_path")
    if not path:
        return []
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _build_report(
    case_name: str,
    mode: str,
    risks: list[RiskItem],
    questions: list[Question],
    expected: list[dict[str, Any]],
    facts: list[FactItem],
    expected_facts: list[dict[str, Any]],
    ingestion_report: dict[str, Any],
    human_context_report: dict[str, Any],
    export_report: dict[str, Any],
) -> dict[str, Any]:
    """计算召回、误报、类型准确率和证据命中率。"""

    matches = []
    matched_risk_ids: set[str] = set()
    for item in expected:
        risk = _find_matching_risk(item, risks)
        if not risk:
            matches.append({"expected": item, "matched": None, "type_ok": False, "evidence_ok": False})
            continue
        matched_risk_ids.add(risk.id)
        type_ok = not item.get("risk_type") or risk.risk_type == item.get("risk_type")
        evidence_ok = _evidence_hit(item, risk)
        matches.append(
            {
                "expected": item,
                "matched": risk.model_dump(),
                "type_ok": type_ok,
                "evidence_ok": evidence_ok,
            }
        )

    matched = [match for match in matches if match["matched"]]
    false_positives = [risk.model_dump() for risk in risks if risk.id not in matched_risk_ids]
    expected_count = max(len(expected), 1)
    matched_count = len(matched)
    type_ok_count = sum(1 for match in matched if match["type_ok"])
    evidence_ok_count = sum(1 for match in matched if match["evidence_ok"])
    fact_matches = _match_expected_facts(expected_facts, facts, ingestion_report)
    fact_hit_count = sum(1 for match in fact_matches if match["matched"])

    return {
        "case_name": case_name,
        "mode": mode,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "metrics": {
            "expected_count": len(expected),
            "risk_count": len(risks),
            "question_count": len(questions),
            "pending_required_question_count": len([q for q in questions if q.status == "pending" and q.required]),
            "recall": matched_count / expected_count,
            "false_positive_count": len(false_positives),
            "risk_type_accuracy": type_ok_count / max(matched_count, 1),
            "evidence_hit_rate": evidence_ok_count / max(matched_count, 1),
            "fact_expected_count": len(expected_facts),
            "fact_count": len(facts),
            "fact_hit_rate": fact_hit_count / max(len(expected_facts), 1),
        },
        "ingestion": ingestion_report,
        "fact_matches": fact_matches,
        "facts": [fact.model_dump() for fact in facts],
        "human_context": human_context_report,
        "export_check": export_report,
        "matches": matches,
        "false_positives": false_positives,
        "questions": [question.model_dump() for question in questions],
    }


def _run_export_check() -> dict[str, Any]:
    """验证导出工作簿主表兼容，且证据详情页可用。"""

    from openpyxl import load_workbook

    risks = [
        RiskItem(
            material_name="结构化证据物料",
            module="功率模块",
            risk_type="关键性能风险",
            risk_reason="用于验证结构化证据明细导出。",
            source_basis="PRD：功率精度要求 ±1.0%。",
            evidence_items=[
                DocumentSourceRef(
                    source_name="PRD",
                    file_name="P725功率计PRD V1.xlsx",
                    sheet="需求规格",
                    row_number=12,
                    parser="openpyxl",
                    excerpt="功率精度要求 ±1.0%。",
                )
            ],
            confidence=0.8,
            unresolved_questions=["供应商是否已有量产校准能力？"],
        ),
        RiskItem(
            material_name="兜底证据物料",
            module="踏板模块",
            risk_type="定制工艺风险",
            risk_reason="用于验证没有 evidence_items 时仍可导出来源依据。",
            source_basis="BOM：轴心为定制加工件。",
        ),
    ]
    path = export_risks("eval-export-check", "eval-export-check", risks)
    wb = load_workbook(path)
    try:
        main_ws = wb["计划阶段风险物料"]
        evidence_ws = wb["证据详情"]
        main_headers = [cell.value for cell in main_ws[1]]
        evidence_headers = [cell.value for cell in evidence_ws[1]]
        evidence_rows = list(evidence_ws.iter_rows(min_row=2, values_only=True))
        checks = {
            "has_expected_sheets": "计划阶段风险物料" in wb.sheetnames and "证据详情" in wb.sheetnames,
            "main_headers_unchanged": main_headers == EXPORT_HEADERS,
            "evidence_headers_ok": evidence_headers == EVIDENCE_HEADERS,
            "structured_row_exists": any(row[0] == "结构化证据物料" and row[4] == "PRD" and row[7] == 12 for row in evidence_rows),
            "fallback_row_exists": any(row[0] == "兜底证据物料" and row[4] == "来源与依据" and "轴心为定制加工件" in str(row[10] or "") for row in evidence_rows),
            "confidence_formatted": any(row[0] == "结构化证据物料" and row[2] == "80%" for row in evidence_rows),
            "unresolved_questions_exported": any(row[0] == "结构化证据物料" and "量产校准能力" in str(row[3] or "") for row in evidence_rows),
        }
    finally:
        wb.close()

    return {
        "status": "ok" if all(checks.values()) else "failed",
        "path": str(path),
        **checks,
    }


def _run_ingestion_checks(case: dict[str, Any]) -> dict[str, Any]:
    """运行文档摄取回归检查。"""

    prd_path = case.get("prd_path")
    if not prd_path:
        return {"prd_line_count": 0, "suspicious_title_only": False, "warnings": []}
    result = ingest_document(prd_path, source_name="PRD")
    return {
        "prd_line_count": len(result.lines()),
        "non_empty_cell_count": result.diagnostics.non_empty_cell_count,
        "suspicious_title_only": result.diagnostics.suspicious_title_only,
        "warnings": result.diagnostics.warnings,
    }


def _keyword_facts_from_ingestion(case: dict[str, Any]) -> list[FactItem]:
    """mock 模式下用期望关键词从摄取文本中构造事实，用于验证事实评测链路。"""

    prd_path = case.get("prd_path")
    expected_facts = _load_expected_facts(case)
    if not prd_path or not expected_facts:
        return []
    lines = ingest_document(prd_path, source_name="PRD").lines()
    facts: list[FactItem] = []
    for expected in expected_facts:
        keywords = expected.get("keywords", [])
        matched_lines = [line for line in lines if all(str(keyword) in line for keyword in keywords)]
        if not matched_lines:
            matched_lines = [line for line in lines if any(str(keyword) in line for keyword in keywords)]
        if matched_lines:
            facts.append(
                FactItem(
                    fact_type=str(expected.get("fact_type") or ""),
                    subject=" / ".join(str(keyword) for keyword in keywords[:3]),
                    value="；".join(str(keyword) for keyword in keywords),
                    source_basis=matched_lines[0],
                )
            )
    return facts


def _match_expected_facts(expected_facts: list[dict[str, Any]], facts: list[FactItem], ingestion_report: dict[str, Any]) -> list[dict[str, Any]]:
    """按关键词匹配事实；若 live 未返回事实，也可从摄取文本诊断辅助定位。"""

    matches = []
    haystack = "\n".join(f"{fact.fact_type} {fact.subject} {fact.value} {fact.source_basis}" for fact in facts)
    for expected in expected_facts:
        keywords = [str(keyword) for keyword in expected.get("keywords", [])]
        type_filter = str(expected.get("fact_type") or "")
        matched = bool(keywords) and all(keyword in haystack for keyword in keywords)
        if type_filter and matched:
            matched = any(fact.fact_type == type_filter and all(keyword in f"{fact.subject} {fact.value} {fact.source_basis}" for keyword in keywords) for fact in facts)
        matches.append({"expected": expected, "matched": matched})
    return matches


def _run_human_context_check() -> dict[str, Any]:
    """验证人工回答会进入上下文并影响风险缓存 hash。"""

    state = ProjectState(config=ProjectConfig(project_name="eval-human-context", bom_path="mock.xlsx"))
    question = create_question(
        question_kind="procurement_confirmation",
        input_type="textarea",
        title="确认待定规格风险",
        message="请确认规格是否完整。",
        reason="规格不完整会影响风险判断。",
        context={"source_name": "PRD", "source_excerpt": "PRD R50: KEO / SPD-SL / SPD"},
        blocking=True,
        required=True,
    )
    answer_question(question, "无法确认，请保留风险。")
    state.questions.append(question)
    context = build_human_answer_context(state)
    base_hash = _chunk_hash("PRD", "materials", "chunk", human_context="")
    answer_hash = _chunk_hash("PRD", "materials", "chunk", human_context=context)
    return {
        "context_has_answer": "无法确认" in context and "保留风险" in context,
        "hash_differs": base_hash != answer_hash,
    }


def _find_matching_risk(expected: dict[str, Any], risks: list[RiskItem]) -> RiskItem | None:
    """按物料名做宽松匹配，适合早期少量评测样本。"""

    expected_name = normalize_name(str(expected.get("material_name") or ""))
    if not expected_name:
        return None
    for risk in risks:
        risk_name = normalize_name(risk.material_name)
        if risk_name == expected_name or expected_name in risk_name or risk_name in expected_name:
            return risk
    return None


def _evidence_hit(expected: dict[str, Any], risk: RiskItem) -> bool:
    """检查证据文本是否包含人工标注关键词。"""

    keywords = expected.get("evidence_keywords") or []
    if not keywords:
        return True
    evidence = f"{risk.risk_reason} {risk.source_basis}"
    return any(str(keyword) in evidence for keyword in keywords)


def _write_report(case_name: str, mode: str, report: dict[str, Any]) -> Path:
    """保存评测报告。"""

    reports_dir = ROOT / "evals" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = reports_dir / f"{case_name}_{mode}_{stamp}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    main()
