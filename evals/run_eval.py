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

from backend.models import MaterialRecord, Question, RiskItem
from backend.parsers.bom_parser import parse_bom
from backend.parsers.document_parser import parse_document_lines
from backend.risk_rules import (
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
    materials = parse_bom(case["bom_path"])

    if args.mode == "mock":
        risks, questions = _run_mock(case, materials)
    else:
        risks, questions = _run_live(case, materials)

    report = _build_report(case["case_name"], args.mode, risks, questions, expected)
    report_path = _write_report(case["case_name"], args.mode, report)

    print(f"case: {case['case_name']}")
    print(f"mode: {args.mode}")
    print(f"risks: {len(risks)}")
    print(f"questions: {len(questions)}")
    print(f"recall: {report['metrics']['recall']:.2f}")
    print(f"false_positives: {report['metrics']['false_positive_count']}")
    print(f"type_accuracy: {report['metrics']['risk_type_accuracy']:.2f}")
    print(f"evidence_hit_rate: {report['metrics']['evidence_hit_rate']:.2f}")
    print(f"report: {report_path}")


def _run_mock(case: dict[str, Any], materials: list[MaterialRecord]) -> tuple[list[RiskItem], list[Question]]:
    """用固定模型输出验证后处理链路。"""

    data = json.loads(Path(case["mock_output_path"]).read_text(encoding="utf-8"))
    risks, model_questions = _items_and_questions_from_model(data, "mock", materials)
    merged = merge_risks(risks)
    questions = model_questions + build_questions(merged)
    return merged, questions


def _run_live(case: dict[str, Any], materials: list[MaterialRecord]) -> tuple[list[RiskItem], list[Question]]:
    """调用真实模型运行一个轻量评测。"""

    project_id = f"eval-{case['case_name']}"
    risks: list[RiskItem] = []
    questions: list[Question] = []

    bom_risks, bom_questions = extract_bom_risks(materials, project_id)
    risks.extend(bom_risks)
    questions.extend(bom_questions)

    prd_path = case.get("prd_path")
    if prd_path:
        prd_lines = parse_document_lines(prd_path)
        prd_risks, prd_questions = extract_document_risks(prd_lines, materials, "PRD", project_id)
        risks.extend(prd_risks)
        questions.extend(prd_questions)

    merged = merge_risks(risks)
    questions.extend(build_questions(merged))
    return merged, questions


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


def _build_report(
    case_name: str,
    mode: str,
    risks: list[RiskItem],
    questions: list[Question],
    expected: list[dict[str, Any]],
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
        },
        "matches": matches,
        "false_positives": false_positives,
        "questions": [question.model_dump() for question in questions],
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
