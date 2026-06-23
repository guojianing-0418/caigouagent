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
from backend.agent import _dedupe_existing_questions, _question_signature, _remove_stale_risk_action_questions, _upsert_questions
from backend.main import _build_export_check, _should_auto_rerun_after_answer
from backend.decision_engine import parse_human_decision, record_decision_for_question, suppress_questions_by_decisions
from backend.models import DocumentSourceRef, FactItem, MaterialRecord, ProjectConfig, ProjectState, Question, RiskItem
from backend.parsers.bom_parser import parse_bom
from backend.parsers.document_parser import parse_document_lines
from backend.parsers.document_ingestor import ingest_document
from backend.question_engine import build_human_answer_context, create_question, answer_question
from backend.risk_classification import classify_risks
from backend.risk_rules import (
    _chunk_hash,
    _items_and_questions_from_model,
    _normalize_risk_type,
    build_questions,
    extract_bom_risks,
    extract_document_risks,
    merge_risks,
    normalize_name,
)


MOCK_THRESHOLDS = {
    "recall": 1.0,
    "risk_type_accuracy": 1.0,
    "evidence_hit_rate": 1.0,
    "structured_evidence_rate": 1.0,
    "fact_hit_rate": 1.0,
    "fact_source_ref_rate": 1.0,
}


def main() -> None:
    """读取评测配置并输出报告。"""

    parser = argparse.ArgumentParser(description="运行采购风险 Agent 评测")
    parser.add_argument("--case", help="评测 case.yaml 路径")
    parser.add_argument("--case-dir", help="批量运行目录下所有 case.yaml")
    parser.add_argument("--mode", choices=["mock", "live"], default="mock", help="mock 不调用模型，live 调用真实模型")
    args = parser.parse_args()

    case_paths = _resolve_case_paths(args.case, args.case_dir)
    reports = []
    failed = False
    for case_path in case_paths:
        report, report_path = _run_case(case_path, args.mode)
        reports.append((report, report_path))
        failed = _print_case_summary(report, report_path) or failed

    if len(reports) > 1:
        _print_batch_summary(reports)
    if failed:
        raise SystemExit(1)


def _resolve_case_paths(case: str | None, case_dir: str | None) -> list[Path]:
    """解析单 case 或批量 case 路径。"""

    if bool(case) == bool(case_dir):
        raise SystemExit("请传入 --case 或 --case-dir，且二者只能选择一个。")
    if case:
        return [Path(case)]
    paths = sorted(Path(case_dir or "").glob("*/case.yaml"))
    if not paths:
        raise SystemExit(f"未在目录中找到 case.yaml：{case_dir}")
    return paths


def _run_case(case_path: Path, mode: str) -> tuple[dict[str, Any], Path]:
    """运行单个评测 case。"""

    case = yaml.safe_load(case_path.read_text(encoding="utf-8"))
    expected = _load_expected(Path(case["expected_path"]))
    expected_facts = _load_expected_facts(case)
    materials = parse_bom(case["bom_path"])
    ingestion_report = _run_ingestion_checks(case)
    human_context_report = _run_human_context_check()
    auto_rerun_report = _run_auto_rerun_check()
    question_lifecycle_report = _run_question_lifecycle_check()
    checkpoint_report = _run_checkpoint_smoke_check()

    if mode == "mock":
        risks, questions, facts = _run_mock(case, materials)
    else:
        risks, questions, facts = _run_live(case, materials)

    export_report = _run_export_check()
    classification_report = _run_classification_check()
    bom_hierarchy_report = _run_bom_hierarchy_check(materials)
    report = _build_report(
        case["case_name"],
        mode,
        risks,
        questions,
        expected,
        facts,
        expected_facts,
        ingestion_report,
        human_context_report,
        export_report,
        classification_report,
        bom_hierarchy_report,
        auto_rerun_report,
        question_lifecycle_report,
        checkpoint_report,
    )
    report["thresholds"] = _threshold_check(report) if mode == "mock" else {"status": "skipped", "failures": []}
    report_path = _write_report(case["case_name"], mode, report)
    return report, report_path


def _print_case_summary(report: dict[str, Any], report_path: Path) -> bool:
    """打印单 case 摘要，返回是否失败。"""

    print(f"case: {report['case_name']}")
    print(f"mode: {report['mode']}")
    print(f"risks: {report['metrics']['risk_count']}")
    print(f"questions: {report['metrics']['question_count']}")
    print(f"recall: {report['metrics']['recall']:.2f}")
    print(f"false_positives: {report['metrics']['false_positive_count']}")
    print(f"type_accuracy: {report['metrics']['risk_type_accuracy']:.2f}")
    print(f"evidence_hit_rate: {report['metrics']['evidence_hit_rate']:.2f}")
    print(f"structured_evidence_rate: {report['metrics']['structured_evidence_rate']:.2f}")
    print(f"fact_hit_rate: {report['metrics']['fact_hit_rate']:.2f}")
    print(f"fact_source_ref_rate: {report['metrics']['fact_source_ref_rate']:.2f}")
    print(f"prd_line_count: {report['ingestion']['prd_line_count']}")
    print(f"human_context_hash_differs: {report['human_context']['hash_differs']}")
    print(f"auto_rerun_check: {report['auto_rerun']['status']}")
    print(f"question_lifecycle_check: {report['question_lifecycle']['status']}")
    print(f"checkpoint_check: {report['checkpoint']['status']}")
    print(f"export_check: {report['export_check']['status']}")
    print(f"classification_check: {report['classification']['status']}")
    print(f"bom_hierarchy_check: {report['bom_hierarchy']['status']}")
    print(f"thresholds: {report['thresholds']['status']}")
    for failure in report["thresholds"].get("failures", []):
        print(f"threshold_failure: {failure}")
    print(f"report: {report_path}")
    print("")
    return report["thresholds"]["status"] == "failed"


def _print_batch_summary(reports: list[tuple[dict[str, Any], Path]]) -> None:
    """打印批量评测简洁摘要。"""

    print("batch_summary:")
    for report, _ in reports:
        metrics = report["metrics"]
        print(
            "- "
            f"{report['case_name']} "
            f"recall={metrics['recall']:.2f} "
            f"fact={metrics['fact_hit_rate']:.2f} "
            f"evidence={metrics['structured_evidence_rate']:.2f} "
            f"export={report['export_check']['status']} "
            f"auto_rerun={report['auto_rerun']['status']} "
            f"questions={report['question_lifecycle']['status']} "
            f"checkpoint={report['checkpoint']['status']} "
            f"thresholds={report['thresholds']['status']}"
        )


def _threshold_check(report: dict[str, Any]) -> dict[str, Any]:
    """mock 模式下的硬阈值检查。"""

    failures = []
    metrics = report["metrics"]
    for key, minimum in MOCK_THRESHOLDS.items():
        value = float(metrics.get(key, 0))
        if value < minimum:
            failures.append(f"{key}={value:.2f} < {minimum:.2f}")
    if report["human_context"].get("hash_differs") is not True:
        failures.append("human_context.hash_differs is not true")
    for section in ["auto_rerun", "question_lifecycle", "checkpoint", "export_check", "classification", "bom_hierarchy"]:
        if report[section].get("status") != "ok":
            failures.append(f"{section}.status={report[section].get('status')}")
    if report["ingestion"].get("suspicious_title_only"):
        failures.append("ingestion.suspicious_title_only is true")
    if not report["ingestion"].get("dimension_repair_message_ok", True):
        failures.append("ingestion.dimension_repair_message_ok is not true")
    if report["question_lifecycle"].get("business_choice_boolean_compat") is not True:
        failures.append("question_lifecycle.business_choice_boolean_compat is not true")
    if report["question_lifecycle"].get("risk_type_alias_check") is not True:
        failures.append("question_lifecycle.risk_type_alias_check is not true")
    return {
        "status": "failed" if failures else "ok",
        "failures": failures,
        "required_metrics": MOCK_THRESHOLDS,
    }


def _run_mock(case: dict[str, Any], materials: list[MaterialRecord]) -> tuple[list[RiskItem], list[Question], list[FactItem]]:
    """用固定模型输出验证后处理链路。"""

    data = json.loads(Path(case["mock_output_path"]).read_text(encoding="utf-8"))
    risks, model_questions = _items_and_questions_from_model(data, "mock", materials)
    merged = classify_risks(merge_risks(risks), materials)
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

    merged = classify_risks(merge_risks(risks), materials)
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
    classification_report: dict[str, Any],
    bom_hierarchy_report: dict[str, Any],
    auto_rerun_report: dict[str, Any],
    question_lifecycle_report: dict[str, Any],
    checkpoint_report: dict[str, Any],
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
    structured_evidence_count = sum(1 for risk in risks if _has_structured_evidence(risk))
    fact_matches = _match_expected_facts(expected_facts, facts, ingestion_report)
    fact_hit_count = sum(1 for match in fact_matches if match["matched"])
    facts_with_source_ref_count = sum(1 for fact in facts if _has_source_ref(fact))

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
            "structured_evidence_rate": structured_evidence_count / max(len(risks), 1),
            "fact_expected_count": len(expected_facts),
            "fact_count": len(facts),
            "fact_hit_rate": fact_hit_count / max(len(expected_facts), 1),
            "fact_source_ref_rate": facts_with_source_ref_count / max(len(facts), 1),
        },
        "ingestion": ingestion_report,
        "fact_matches": fact_matches,
        "facts": [fact.model_dump() for fact in facts],
        "human_context": human_context_report,
        "auto_rerun": auto_rerun_report,
        "question_lifecycle": question_lifecycle_report,
        "checkpoint": checkpoint_report,
        "export_check": export_report,
        "classification": classification_report,
        "bom_hierarchy": bom_hierarchy_report,
        "matches": matches,
        "false_positives": false_positives,
        "questions": [question.model_dump() for question in questions],
    }


def _run_question_lifecycle_check() -> dict[str, Any]:
    """验证重跑时 pending 问题保留、刷新、去重和过期清理。"""

    state = ProjectState(config=ProjectConfig(project_name="eval-question-lifecycle", bom_path="mock.xlsx"))
    pending = create_question(
        question_kind="procurement_confirmation",
        input_type="single_select",
        title="确认供应商能力",
        message="是否具备量产能力？",
        reason="旧原因",
        options=["未知"],
        context={"source_excerpt": "同一依据"},
        related_risk_ids=["old-risk"],
        required=True,
        blocking=False,
    )
    answered = create_question(
        question_kind="procurement_confirmation",
        input_type="textarea",
        title="已回答问题",
        message="规格是否完整？",
        context={"source_excerpt": "已回答依据"},
        required=True,
        blocking=False,
    )
    answer_question(answered, "无法确认，请保留风险。")
    stale = create_question(
        question_kind="risk_keep_review",
        input_type="boolean",
        title="这条风险是否保留？",
        message="旧风险是否保留？",
        context={"source_excerpt": "旧风险依据"},
        related_risk_ids=["stale-risk"],
        required=False,
        blocking=False,
    )
    state.questions.extend([pending, answered, stale])
    pending_id = pending.id

    refreshed = create_question(
        question_kind="procurement_confirmation",
        input_type="single_select",
        title="确认供应商能力",
        message="是否具备量产能力？",
        reason="新原因",
        options=["具备", "不具备"],
        context={"source_excerpt": "同一依据", "risk_id": "new-risk"},
        related_risk_ids=["new-risk"],
        required=True,
        blocking=False,
    )
    ignored_answered = create_question(
        question_kind="procurement_confirmation",
        input_type="textarea",
        title="已回答问题",
        message="规格是否完整？",
        context={"source_excerpt": "已回答依据"},
        required=True,
        blocking=False,
    )
    new_question = create_question(
        question_kind="risk_material_mapping",
        input_type="single_select",
        title="确认接口件对应物料",
        message="请选择 BOM 物料。",
        context={"source_excerpt": "新依据"},
        required=True,
        blocking=False,
    )
    upsert_result = _upsert_questions(state, [refreshed, ignored_answered, new_question])
    answered_rephrased_base = create_question(
        question_kind="risk_material_mapping",
        input_type="textarea",
        title="产品有KEO、SPD-SL、SPD三种制式脚踏，当前BOM仅列出M山地踏板组件，采购需确认",
        message="请确认KEO制式和SPD-SL制式是否需要独立BOM。",
        context={"source_excerpt": "PRD R50: KEO制式脚踏、SPD-SL制式脚踏、SPD制式脚踏；BOM中仅见P725PRO M山地踏板组件"},
        required=True,
        blocking=False,
    )
    answer_question(answered_rephrased_base, "这个需要跟研发确认")
    state.questions.append(answered_rephrased_base)
    rephrased_same_question = create_question(
        question_kind="risk_material_mapping",
        input_type="textarea",
        title="三种脚踏制式的BOM结构确认",
        message="PRD要求 KEO / SPD-SL / SPD 三种规格，但草BOM只看到M山地踏板组件，请确认其他制式是否单独建BOM。",
        context={"source_excerpt": "[规格/版本] 规格: KEO制式脚踏、SPD-SL制式脚踏、SPD制式脚踏 | 依据:PRD R50；BOM中仅见P725PRO M左侧踏板组件"},
        required=True,
        blocking=False,
    )
    rephrased_upsert = _upsert_questions(state, [rephrased_same_question])
    merge_question = create_question(
        question_kind="risk_merge_review",
        input_type="boolean",
        title="以下两个风险物料是否合并？",
        message="“P725PRO轴心-左侧”和“P725PRO轴心-右侧”名称相近，风险类型相同，是否合并？",
        options=["合并", "不合并"],
        context={
            "left_material": "P725PRO轴心-左侧",
            "right_material": "P725PRO轴心-右侧",
            "source_excerpt": "PRD R48明确轴心材质为沉淀硬化型不锈钢。",
        },
        required=False,
        blocking=False,
    )
    state.questions.append(merge_question)
    merge_duplicate = create_question(
        question_kind="risk_merge_review",
        input_type="boolean",
        title="以下两个风险物料是否合并？",
        message="“P725PRO轴心-右侧”和“P725PRO轴心-左侧”名称相近，风险类型相同，是否合并？",
        options=["合并", "不合并"],
        context={
            "left_material": "P725PRO轴心-右侧",
            "right_material": "P725PRO轴心-左侧",
            "source_excerpt": "草 BOM：人工确认供应商正在开发中。",
        },
        required=False,
        blocking=False,
    )
    merge_duplicate_upsert = _upsert_questions(state, [merge_duplicate])
    answer_question(merge_question, "不合并")
    answered_merge_duplicate = create_question(
        question_kind="risk_merge_review",
        input_type="boolean",
        title="以下两个风险物料是否合并？",
        message="“P725PRO轴心-左侧”和“P725PRO轴心-右侧”是否合并？",
        options=["合并", "不合并"],
        context={
            "left_material": "P725PRO轴心-左侧",
            "right_material": "P725PRO轴心-右侧",
            "source_excerpt": "另一轮识别生成的不同依据。",
        },
        required=False,
        blocking=False,
    )
    answered_merge_duplicate_upsert = _upsert_questions(state, [answered_merge_duplicate])
    same_subject_state = ProjectState(config=ProjectConfig(project_name="eval-question-same-subject", bom_path="mock.xlsx"))
    carbon_supplier_mapping = create_question(
        question_kind="risk_material_mapping",
        input_type="textarea",
        title="P725PRO M山地踏板本体长碳纤维注塑供应商是否已有储备",
        message="PRD明确脚踏本体采用长碳纤维注塑工艺，请确认是否有已合作或推荐的长碳纤维注塑供应商？",
        context={"source_excerpt": "[材料要求] 材质: 脚踏本体 长碳纤维注塑 | 依据:功率锁踏- R49"},
        required=True,
        blocking=False,
    )
    carbon_supplier_confirmation = create_question(
        question_kind="procurement_confirmation",
        input_type="textarea",
        title="长碳纤维注塑踏板本体供应商确认",
        message="PRD要求脚踏本体材质为长碳纤维注塑，请确认是否已有该工艺的成熟供应商？模具开发周期预估多久？",
        context={"source_excerpt": "功率锁踏- R49: 脚踏本体 长碳纤维注塑"},
        required=True,
        blocking=False,
    )
    same_subject_state.questions.append(carbon_supplier_mapping)
    same_subject_upsert = _upsert_questions(same_subject_state, [carbon_supplier_confirmation])
    pcba_state = ProjectState(config=ProjectConfig(project_name="eval-question-pcba", bom_path="mock.xlsx"))
    pcba_supplier = create_question(
        question_kind="procurement_confirmation",
        input_type="textarea",
        title="PCBA供应商资源确认",
        message="主板PCBA是否已有成熟供应商？是否沿用现有平台方案？",
        context={"source_excerpt": "BOM: name=主板PCBA"},
        required=True,
        blocking=False,
    )
    pcba_cert_chip = create_question(
        question_kind="procurement_confirmation",
        input_type="textarea",
        title="PCBA认证芯片确认",
        message="主板PCBA是否必须包含特定蓝牙/ANT+认证芯片型号？",
        context={"source_excerpt": "PRD R25: Ant+、Bluetooth、CE、FCC、RoHS"},
        required=True,
        blocking=False,
    )
    _upsert_questions(pcba_state, [pcba_supplier])
    pcba_distinct_upsert = _upsert_questions(pcba_state, [pcba_cert_chip])
    cleanup_state = ProjectState(config=ProjectConfig(project_name="eval-question-cleanup", bom_path="mock.xlsx"))
    answered_cleanup_merge = create_question(
        question_kind="risk_merge_review",
        input_type="boolean",
        title="以下两个风险物料是否合并？",
        message="“M7尾端螺钉-左轴”和“M7尾端螺钉-右轴”名称相近，风险类型相同，是否合并？",
        options=["合并", "不合并"],
        context={
            "left_material": "M7尾端螺钉-左轴",
            "right_material": "M7尾端螺钉-右轴",
            "source_excerpt": "PRD R30盐雾要求72小时。",
        },
        required=False,
        blocking=False,
    )
    answer_question(answered_cleanup_merge, "不合并")
    pending_cleanup_duplicate = create_question(
        question_kind="risk_merge_review",
        input_type="boolean",
        title="以下两个风险物料是否合并？",
        message="“M7尾端螺钉-右轴”和“M7尾端螺钉-左轴”名称相近，风险类型相同，是否合并？",
        options=["合并", "不合并"],
        context={
            "left_material": "M7尾端螺钉-右轴",
            "right_material": "M7尾端螺钉-左轴",
            "source_excerpt": "另一轮识别的证据摘要。",
        },
        required=False,
        blocking=False,
    )
    cleanup_state.questions.extend([answered_cleanup_merge, pending_cleanup_duplicate])
    cleanup_removed_count = _dedupe_existing_questions(cleanup_state)

    def fake_decision_classifier(question: Question, answer_text: str) -> dict[str, Any] | None:
        if answer_text == "这个供应商需要跟研发确认":
            return {
                "disposition": "pending_external_confirmation",
                "owner": "研发",
                "risk_effect": "keep_risk",
                "ask_again": False,
                "export_blocking": False,
            }
        if answer_text == "不是不属于采购范围，只是供应商还需要研发确认":
            return {
                "disposition": "pending_external_confirmation",
                "owner": "研发",
                "risk_effect": "keep_risk",
                "ask_again": "false",
                "export_blocking": "false",
            }
        if answer_text == "目前有候选供应商，但价格和认证还没确认，先保留风险":
            return {
                "disposition": "unresolved",
                "owner": "",
                "risk_effect": "keep_risk",
                "ask_again": False,
                "export_blocking": False,
            }
        if answer_text == "这个物料本身已有供应商，不过连接方案需要客户确认":
            return {
                "disposition": "pending_external_confirmation",
                "owner": "客户",
                "risk_effect": "keep_risk",
                "ask_again": False,
                "export_blocking": False,
            }
        if answer_text == "模型返回非法字段":
            return {
                "disposition": "unknown",
                "owner": None,
                "risk_effect": "invalid",
                "ask_again": "false",
                "export_blocking": "false",
            }
        return None

    decision_state = ProjectState(config=ProjectConfig(project_name="eval-decision", bom_path="mock.xlsx"))
    supplier_rd_question = create_question(
        question_kind="procurement_confirmation",
        input_type="textarea",
        title="踏板供应商定厂状态确认",
        message="请确认踏板供应商是否已定厂。",
        required=True,
        blocking=False,
    )
    answer_question(supplier_rd_question, "这个供应商需要跟研发确认")
    decision_state.questions.append(supplier_rd_question)
    rd_decision = record_decision_for_question(decision_state, supplier_rd_question, llm_decision_fn=fake_decision_classifier)
    supplier_duplicate = create_question(
        question_kind="procurement_confirmation",
        input_type="textarea",
        title="踏板供应商定厂结论",
        message="当前踏板供应商是否已最终锁定？",
        required=True,
        blocking=False,
    )
    decision_state.questions.append(supplier_duplicate)
    suppressed_supplier_count = suppress_questions_by_decisions(decision_state)
    unresolved_state = ProjectState(config=ProjectConfig(project_name="eval-unresolved", bom_path="mock.xlsx"))
    mud_question = create_question(
        question_kind="risk_material_mapping",
        input_type="textarea",
        title="泥浆设备关联物料确认",
        message="泥浆设备对应 BOM 哪个物料？",
        required=True,
        blocking=False,
    )
    answer_question(mud_question, "这个先待定，保留为风险吧")
    unresolved_state.questions.append(mud_question)
    unresolved_decision = record_decision_for_question(unresolved_state, mud_question, llm_decision_fn=fake_decision_classifier)
    out_of_scope_question = create_question(
        question_kind="procurement_confirmation",
        input_type="textarea",
        title="泥浆设备采购归属",
        message="泥浆设备是否属于采购范围？",
        required=True,
        blocking=False,
    )
    answer_question(out_of_scope_question, "不属于采购范围")
    out_of_scope_decision = record_decision_for_question(unresolved_state, out_of_scope_question, llm_decision_fn=fake_decision_classifier)
    confirmed_question = create_question(
        question_kind="procurement_confirmation",
        input_type="textarea",
        title="锂电池供应商确认",
        message="是否已有供应商？",
        required=True,
        blocking=False,
    )
    answer_question(confirmed_question, "已有意向供应商")
    confirmed_decision = record_decision_for_question(unresolved_state, confirmed_question, llm_decision_fn=fake_decision_classifier)
    complex_scope_question = create_question(
        question_kind="procurement_confirmation",
        input_type="textarea",
        title="踏板供应商确认",
        message="踏板供应商是否不属于采购范围？",
        required=True,
        blocking=False,
    )
    complex_scope_decision = parse_human_decision(
        complex_scope_question,
        "不是不属于采购范围，只是供应商还需要研发确认",
        llm_decision_fn=fake_decision_classifier,
    )
    complex_supplier_question = create_question(
        question_kind="procurement_confirmation",
        input_type="textarea",
        title="供应商和认证确认",
        message="请确认供应商、价格和认证状态。",
        required=True,
        blocking=False,
    )
    complex_supplier_decision = parse_human_decision(
        complex_supplier_question,
        "目前有候选供应商，但价格和认证还没确认，先保留风险",
        llm_decision_fn=fake_decision_classifier,
    )
    complex_connector_question = create_question(
        question_kind="procurement_confirmation",
        input_type="textarea",
        title="电池连接方案确认",
        message="请确认电池连接方案和供应商状态。",
        required=True,
        blocking=False,
    )
    complex_connector_decision = parse_human_decision(
        complex_connector_question,
        "这个物料本身已有供应商，不过连接方案需要客户确认",
        llm_decision_fn=fake_decision_classifier,
    )
    invalid_llm_decision = parse_human_decision(
        complex_connector_question,
        "模型返回非法字段",
        llm_decision_fn=fake_decision_classifier,
    )
    fallback_out_of_scope_decision = parse_human_decision(
        out_of_scope_question,
        "不属于采购范围",
        llm_decision_fn=lambda _question, _answer: None,
    )
    fallback_custom_note_decision = parse_human_decision(
        out_of_scope_question,
        "这个结论记录一下，后续例会同步",
        llm_decision_fn=lambda _question, _answer: None,
    )
    business_choice = create_question(
        question_kind="procurement_confirmation",
        input_type="boolean",
        title="应变片供应商是否已定点？",
        options=["已定点", "未定点，有候选", "未定点，需寻找"],
        required=True,
        blocking=False,
    )
    answer_question(business_choice, "未定点，有候选")
    keep_question = create_question(
        question_kind="risk_keep_review",
        input_type="boolean",
        title="是否保留风险？",
        options=["保留", "删除"],
        required=False,
        blocking=False,
    )
    answer_question(keep_question, "删除")
    state.risks = [
        RiskItem(
            id="new-risk",
            material_name="结构化证据物料",
            risk_type="关键性能风险",
            risk_reason="验证问题关联风险仍有效。",
            source_basis="mock",
        )
    ]
    removed_count = _remove_stale_risk_action_questions(state)
    refreshed_pending = next(question for question in state.questions if question.id == pending_id)

    checks = {
        "pending_id_preserved": refreshed_pending.id == pending_id,
        "pending_refreshed": refreshed_pending.reason == "新原因" and refreshed_pending.related_risk_ids == ["new-risk"] and refreshed_pending.options == ["具备", "不具备"],
        "answered_not_duplicated": len([question for question in state.questions if question.title == "已回答问题"]) == 1,
        "new_question_added": any(question.title == "确认接口件对应物料" for question in state.questions),
        "upsert_counts_ok": len(upsert_result["added"]) == 1 and len(upsert_result["refreshed"]) == 1 and len(upsert_result["ignored"]) == 1,
        "stale_risk_question_removed": removed_count == 1 and all("stale-risk" not in question.related_risk_ids for question in state.questions),
        "business_choice_boolean_compat": business_choice.input_type == "single_select" and business_choice.answer == "未定点，有候选",
        "real_boolean_still_boolean": keep_question.input_type == "boolean" and keep_question.answer is False,
        "risk_type_alias_check": _normalize_risk_type("新技术风险") == "新物料/新技术风险",
        "rephrased_answered_question_ignored": len(rephrased_upsert["ignored"]) == 1
        and _question_signature(answered_rephrased_base) == _question_signature(rephrased_same_question),
        "merge_duplicate_refreshed": len(merge_duplicate_upsert["refreshed"]) == 1
        and merge_duplicate_upsert["refreshed"][0].id == merge_question.id,
        "answered_merge_duplicate_ignored": len(answered_merge_duplicate_upsert["ignored"]) == 1,
        "same_subject_cross_kind_refreshed": len(same_subject_upsert["refreshed"]) == 1
        and same_subject_upsert["refreshed"][0].id == carbon_supplier_mapping.id,
        "pcba_distinct_intents_not_merged": len(pcba_distinct_upsert["added"]) == 1
        and len(pcba_state.questions) == 2,
        "historical_pending_duplicate_removed": cleanup_removed_count == 1
        and len(cleanup_state.questions) == 1
        and cleanup_state.questions[0].status == "answered",
        "rd_confirmation_decision": rd_decision.disposition == "pending_external_confirmation"
        and rd_decision.owner == "研发"
        and rd_decision.risk_effect == "keep_risk"
        and rd_decision.ask_again is False
        and rd_decision.export_blocking is False,
        "decision_suppresses_duplicate": suppressed_supplier_count == 1
        and len([q for q in decision_state.questions if q.status == "pending"]) == 0,
        "unresolved_answer_decision": unresolved_decision.disposition == "unresolved"
        and unresolved_decision.ask_again is False
        and unresolved_decision.export_blocking is False,
        "out_of_scope_decision": out_of_scope_decision.disposition == "out_of_scope"
        and out_of_scope_decision.risk_effect == "remove_risk",
        "confirmed_decision": confirmed_decision.disposition == "confirmed"
        and confirmed_decision.risk_effect == "reduce_risk",
        "llm_first_complex_negation": complex_scope_decision.disposition == "pending_external_confirmation"
        and complex_scope_decision.owner == "研发"
        and complex_scope_decision.ask_again is False
        and complex_scope_decision.export_blocking is False,
        "llm_first_complex_unresolved": complex_supplier_decision.disposition == "unresolved"
        and complex_supplier_decision.risk_effect == "keep_risk",
        "llm_first_external_beats_confirmed": complex_connector_decision.disposition == "pending_external_confirmation"
        and complex_connector_decision.owner == "客户"
        and complex_connector_decision.risk_effect == "keep_risk",
        "invalid_llm_payload_normalized": invalid_llm_decision.disposition == "custom_note"
        and invalid_llm_decision.risk_effect == "keep_risk"
        and invalid_llm_decision.ask_again is False
        and invalid_llm_decision.export_blocking is False,
        "llm_failure_rule_fallback": fallback_out_of_scope_decision.disposition == "out_of_scope"
        and fallback_out_of_scope_decision.risk_effect == "remove_risk",
        "llm_failure_custom_note_fallback": fallback_custom_note_decision.disposition == "custom_note"
        and fallback_custom_note_decision.risk_effect == "keep_risk"
        and fallback_custom_note_decision.ask_again is False,
    }
    return {
        "status": "ok" if all(checks.values()) else "failed",
        **checks,
    }


def _run_auto_rerun_check() -> dict[str, Any]:
    """验证自动重跑只在所有非阻塞必答都答完后触发。"""

    state = ProjectState(config=ProjectConfig(project_name="eval-auto-rerun", bom_path="mock.xlsx"))
    required_question = create_question(
        question_kind="procurement_confirmation",
        input_type="textarea",
        title="确认供应商能力",
        required=True,
        blocking=False,
    )
    answer_question(required_question, "需要保留风险。")
    all_required_done_state = ProjectState(config=ProjectConfig(project_name="eval-auto-rerun-all-done", bom_path="mock.xlsx"))
    all_required_done_state.questions.append(required_question)
    partial_required_state = ProjectState(config=ProjectConfig(project_name="eval-auto-rerun-partial", bom_path="mock.xlsx"))
    partial_required_state.questions.extend([
        required_question,
        create_question(
            question_kind="procurement_confirmation",
            input_type="textarea",
            title="另一个必答问题",
            required=True,
            blocking=False,
        ),
    ])
    optional_question = create_question(
        question_kind="risk_keep_review",
        input_type="boolean",
        title="可选风险保留",
        required=False,
        blocking=False,
    )
    answer_question(optional_question, "保留")
    blocking_question = create_question(
        question_kind="risk_material_mapping",
        input_type="single_select",
        title="阻塞物料映射",
        required=True,
        blocking=True,
    )
    answer_question(blocking_question, "P725PRO左轴功率模组")
    running_state = ProjectState(config=ProjectConfig(project_name="eval-auto-rerun-running", bom_path="mock.xlsx"), status="running")
    waiting_state = ProjectState(config=ProjectConfig(project_name="eval-auto-rerun-waiting", bom_path="mock.xlsx"), status="waiting")
    waiting_state.questions.extend([required_question, create_question(
        question_kind="risk_material_mapping",
        input_type="single_select",
        title="仍待确认的阻塞问题",
        required=True,
        blocking=True,
    )])

    checks = {
        "all_required_answered_submit": not _should_auto_rerun_after_answer(all_required_done_state, required_question, "submit"),
        "partial_required_answered_ignored": not _should_auto_rerun_after_answer(partial_required_state, required_question, "submit"),
        "optional_submit_ignored": not _should_auto_rerun_after_answer(state, optional_question, "submit"),
        "blocking_submit_ignored": not _should_auto_rerun_after_answer(state, blocking_question, "submit"),
        "skip_ignored": not _should_auto_rerun_after_answer(state, required_question, "skip"),
        "running_state_ignored": not _should_auto_rerun_after_answer(running_state, required_question, "submit"),
        "other_blocking_pending_ignored": not _should_auto_rerun_after_answer(waiting_state, required_question, "submit"),
    }
    return {
        "status": "ok" if all(checks.values()) else "failed",
        **checks,
    }


def _run_checkpoint_smoke_check() -> dict[str, Any]:
    """验证 SQLite checkpoint interrupt/resume 不重跑已完成节点。"""

    import sqlite3
    import tempfile
    from typing import TypedDict

    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.graph import END, StateGraph
    from langgraph.types import Command, interrupt

    class SmokeState(TypedDict, total=False):
        parsed_count: int
        gate_count: int
        answer: str

    def parse_node(state: SmokeState) -> SmokeState:
        return {**state, "parsed_count": state.get("parsed_count", 0) + 1}

    def gate_node(state: SmokeState) -> SmokeState:
        if not state.get("answer"):
            answer = interrupt({"question": "mock"})
            return {**state, "gate_count": state.get("gate_count", 0) + 1, "answer": str(answer)}
        return state

    path = tempfile.NamedTemporaryFile(delete=False).name
    conn = sqlite3.connect(path, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    graph = StateGraph(SmokeState)
    graph.add_node("parse", parse_node)
    graph.add_node("gate", gate_node)
    graph.set_entry_point("parse")
    graph.add_edge("parse", "gate")
    graph.add_edge("gate", END)
    app = graph.compile(checkpointer=saver)
    config = {"configurable": {"thread_id": "eval-checkpoint"}}
    first = app.invoke({"parsed_count": 0}, config=config)
    second = app.invoke(Command(resume="ok"), config=config)
    checks = {
        "interrupted": "__interrupt__" in first,
        "resume_answer_applied": second.get("answer") == "ok",
        "completed_node_not_rerun": second.get("parsed_count") == 1,
        "gate_completed_once": second.get("gate_count") == 1,
    }
    return {"status": "ok" if all(checks.values()) else "failed", **checks}


def _run_export_check() -> dict[str, Any]:
    """验证导出工作簿主表、问题分发清单和证据详情页可用。"""

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
        RiskItem(
            material_name="结构化证据物料",
            module="功率模块",
            risk_type="认证风险",
            risk_reason="用于验证相同物料的不同风险在主表相邻展示。",
            source_basis="PRD：认证要求待补充。",
        ),
    ]
    path = export_risks("eval-export-check", "eval-export-check", risks)
    running_gate = _build_export_check(
        ProjectState(
            config=ProjectConfig(project_name="eval-running-export-gate", bom_path="mock.xlsx"),
            status="running",
        )
    )
    wb = load_workbook(path)
    try:
        main_ws = wb["计划阶段风险物料"]
        question_ws = wb["问题分发清单"]
        evidence_ws = wb["证据详情"]
        main_headers = [cell.value for cell in main_ws[1]]
        main_materials = [main_ws.cell(row=row, column=1).value for row in range(2, main_ws.max_row + 1)]
        merged_ranges = {str(cell_range) for cell_range in main_ws.merged_cells.ranges}
        question_headers = [cell.value for cell in question_ws[1]]
        question_rows = list(question_ws.iter_rows(min_row=2, values_only=True))
        evidence_headers = [cell.value for cell in evidence_ws[1]]
        evidence_rows = list(evidence_ws.iter_rows(min_row=2, values_only=True))
        checks = {
            "has_expected_sheets": "计划阶段风险物料" in wb.sheetnames and "问题分发清单" in wb.sheetnames and "证据详情" in wb.sheetnames,
            "main_headers_ok": main_headers == EXPORT_HEADERS,
            "same_material_rows_adjacent": main_materials[:3] == ["兜底证据物料", "结构化证据物料", None],
            "same_material_cells_merged": "A3:A4" in merged_ranges,
            "question_sheet_headers_ok": question_headers == [
                "风险物料名称",
                "风险类型",
                "主归口",
                "建议提问对象",
                "可发群问题",
                "需要补齐的信息",
                "信息成熟度",
                "来源与依据",
            ],
            "question_rows_exist": len(question_rows) >= 3,
            "classification_fields_exported": any(row[0] == "兜底证据物料" and row[5] for row in main_ws.iter_rows(min_row=2, values_only=True)),
            "evidence_headers_ok": evidence_headers == EVIDENCE_HEADERS,
            "structured_row_exists": any(row[0] == "结构化证据物料" and row[4] == "PRD" and row[7] == 12 for row in evidence_rows),
            "fallback_row_exists": any(row[0] == "兜底证据物料" and row[4] == "来源与依据" and "轴心为定制加工件" in str(row[10] or "") for row in evidence_rows),
            "confidence_formatted": any(row[0] == "结构化证据物料" and row[2] == "80%" for row in evidence_rows),
            "unresolved_questions_exported": any(row[0] == "结构化证据物料" and "量产校准能力" in str(row[3] or "") for row in evidence_rows),
            "running_export_blocked": running_gate.allowed is False and "正在识别" in running_gate.message,
        }
    finally:
        wb.close()

    return {
        "status": "ok" if all(checks.values()) else "failed",
        "path": str(path),
        **checks,
    }


def _run_classification_check() -> dict[str, Any]:
    """验证风险分类规则不改变风险数量，并能覆盖典型场景。"""

    materials = [
        MaterialRecord(name="主板 PCBA", module="电子", spec="PCBA组件", level="1.6.3", parent_name="透光罩组件", bom_path="P725PRO左轴功率模组 > 透光罩组件 > 主板 PCBA", item_role="零件"),
        MaterialRecord(name="轴心", module="结构", spec="按图加工", material="7075-T6", level="1.1.1", parent_name="贴片组件", bom_path="P725PRO左轴功率模组 > 贴片组件 > 轴心", item_role="零件"),
        MaterialRecord(name="未知风险物料", module=""),
    ]
    risks = [
        RiskItem(
            material_name="进口芯片",
            module="电子",
            risk_type="长周期风险",
            risk_reason="供应商反馈交期长，可能影响齐套。",
            source_basis="群聊：该芯片交期 12 周。",
        ),
        RiskItem(
            material_name="轴心",
            module="结构",
            risk_type="定制工艺风险",
            risk_reason="轴心需要按图加工，公差和热处理未冻结。",
            source_basis="BOM：轴心为定制加工件。",
        ),
        RiskItem(
            material_name="主板 PCBA",
            module="电子",
            risk_type="关键性能风险",
            risk_reason="功率精度和校准一致性需要验证。",
            source_basis="PRD：功率精度要求 ±1.0%。",
        ),
        RiskItem(
            material_name="待确认物料",
            module="",
            risk_type="接口匹配风险",
            risk_reason="资料中提到接口匹配风险，但未对应到 BOM 物料。",
            source_basis="群聊：接口件待确认。",
        ),
    ]
    classified = classify_risks(risks, materials)
    long_lead = classified[0]
    process = classified[1]
    performance = classified[2]
    unknown = classified[3]
    checks = {
        "risk_count_unchanged": len(classified) == len(risks),
        "long_lead_procurement": long_lead.primary_owner == "采购" and long_lead.risk_confirmation_method == "采购确认",
        "long_lead_tags": "长周期" in long_lead.risk_tags and "齐套交付" in long_lead.risk_tags,
        "custom_process_joint": process.risk_confirmation_method == "协同确认" and process.primary_owner in {"结构", "工艺"},
        "custom_process_questions": bool(process.followup_questions) and any("图纸" in question or "DFM" in question for question in process.followup_questions),
        "performance_rd_or_joint": performance.risk_confirmation_method in {"研发确认", "协同确认"} and performance.primary_owner in {"电子", "结构"},
        "classification_basis_has_bom_path": "路径" in performance.classification_basis and "透光罩组件" in performance.classification_basis,
        "unknown_not_fabricated": unknown.primary_owner == "待确认" or unknown.material_attribute == "待确认",
        "unknown_missing_info": "对应BOM物料确认" in unknown.missing_information,
    }
    return {"status": "ok" if all(checks.values()) else "failed", **checks}


def _run_bom_hierarchy_check(materials: list[MaterialRecord]) -> dict[str, Any]:
    """验证 BOM 项目层级已解析成父子路径。"""

    def find_material(name: str, level: str = "") -> MaterialRecord | None:
        return next((item for item in materials if item.name == name and (not level or item.level == level)), None)

    strain = find_material("电阻应变片", "1.1.2.1")
    right_strain = next((item for item in materials if item.name == "电阻应变片" and item.level == "1.1.2.1" and "右轴" in item.bom_path), None)
    pcba = find_material("主板PCBA", "1.6.3") or find_material("主板 PCBA", "1.6.3")
    root = find_material("P725PRO左轴功率模组", "1")
    malformed = MaterialRecord(name="异常层级物料", level="abc")
    checks = {
        "strain_parent_ok": bool(strain and strain.parent_name == "应变片贴片组件"),
        "strain_path_ok": bool(strain and "P725PRO左轴功率模组" in strain.bom_path and "应变片贴片组件" in strain.bom_path and strain.bom_path.endswith("电阻应变片")),
        "duplicate_level_right_path_ok": bool(right_strain and "P725PRO右轴功率模组" in right_strain.bom_path),
        "pcba_parent_ok": bool(pcba and pcba.parent_name == "透光罩组件"),
        "root_role_ok": bool(root and root.item_role == "总成" and root.has_children is True),
        "malformed_default_ok": malformed.level_depth == 0 and malformed.item_role == "",
    }
    return {"status": "ok" if all(checks.values()) else "failed", **checks}


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
        "dimension_repair_message_ok": any("Excel 内部维度元数据异常" in warning and "已自动重置读取范围" in warning for warning in result.diagnostics.warnings),
        "warnings": result.diagnostics.warnings,
    }


def _keyword_facts_from_ingestion(case: dict[str, Any]) -> list[FactItem]:
    """mock 模式下用期望关键词从摄取文本中构造事实，用于验证事实评测链路。"""

    prd_path = case.get("prd_path")
    expected_facts = _load_expected_facts(case)
    if not prd_path or not expected_facts:
        return []
    units = ingest_document(prd_path, source_name="PRD").units
    facts: list[FactItem] = []
    for expected in expected_facts:
        keywords = expected.get("keywords", [])
        matched_units = [unit for unit in units if all(str(keyword) in unit.as_line() for keyword in keywords)]
        if not matched_units:
            matched_units = [unit for unit in units if any(str(keyword) in unit.as_line() for keyword in keywords)]
        if matched_units:
            unit = matched_units[0]
            facts.append(
                FactItem(
                    fact_type=str(expected.get("fact_type") or ""),
                    subject=" / ".join(str(keyword) for keyword in keywords[:3]),
                    value="；".join(str(keyword) for keyword in keywords),
                    source_basis=unit.as_line(),
                    source_refs=[
                        DocumentSourceRef(
                            source_name=unit.source_name,
                            file_name=unit.file_name,
                            sheet=unit.sheet,
                            row_number=unit.row_number,
                            page_number=unit.page_number,
                            parser=unit.parser,
                            excerpt=unit.text,
                        )
                    ],
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
    evidence = " ".join(
        [
            risk.risk_reason,
            risk.source_basis,
            *[item.excerpt for item in risk.evidence_items],
        ]
    )
    return any(str(keyword) in evidence for keyword in keywords)


def _has_structured_evidence(risk: RiskItem) -> bool:
    """判断风险是否带有可追溯结构化证据。"""

    return any(item.excerpt.strip() for item in risk.evidence_items)


def _has_source_ref(fact: FactItem) -> bool:
    """判断事实是否带有有效来源引用。"""

    return any(ref.excerpt.strip() and (ref.file_name or ref.source_name) for ref in fact.source_refs)


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
