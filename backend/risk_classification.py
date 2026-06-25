"""计划阶段风险物料分类与采购前置问题生成。

本模块在风险识别和合并之后运行，只把已有风险转成采购代表可执行的信息。
它不新增风险、不删除风险，也不替代研发做技术结论。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .models import MaterialRecord, RiskItem


CONFIRM_RD = "研发确认"
CONFIRM_PROCUREMENT = "采购确认"
CONFIRM_JOINT = "协同确认"
UNKNOWN = "待确认"

OWNER_STRUCTURE = "结构"
OWNER_ELECTRONICS = "电子"
OWNER_PROCESS = "工艺"
OWNER_PROCUREMENT = "采购"

MAT_STANDARD = "标准件"
MAT_SPECIAL_STANDARD = "特殊标准件"
MAT_CUSTOM = "定制件"
MAT_OUTSOURCED = "外购件"
MAT_PROCESSING = "委外加工件"

MATURITY_CLEAR = "已有明确依据"
MATURITY_WEAK = "计划阶段待补充"
MATURITY_POTENTIAL = "计划阶段待补充"


@dataclass(frozen=True)
class KeywordRule:
    """一组关键词命中后追加同类标签。"""

    keywords: tuple[str, ...]
    tags: tuple[str, ...]


TAG_RULES = [
    KeywordRule(("长周期", "交期", "周期", "齐套", "延期", "lead time"), ("长周期", "齐套交付")),
    KeywordRule(("稀缺", "停产", "缺货", "单一", "独家", "垄断"), ("稀缺", "单一供应")),
    KeywordRule(("新供应商", "未买过", "首次采购", "新供方", "新资源"), ("新供应商", "供方能力")),
    KeywordRule(("二供", "备选", "替代供应商"), ("二供验证",)),
    KeywordRule(("审厂", "资质", "供应商能力", "做不了", "能力不足"), ("审厂", "供方能力")),
    KeywordRule(("成本", "报价", "降本", "目标价", "价格"), ("成本达成",)),
    KeywordRule(("moq", "最小起订量", "起订量"), ("MOQ",)),
    KeywordRule(("开模", "模具", "模费"), ("开模", "开模费")),
    KeywordRule(("dfm", "可制造", "加工性"), ("DFM",)),
    KeywordRule(("pfmea",), ("PFMEA",)),
    KeywordRule(("治具", "夹具", "工装"), ("工装",)),
    KeywordRule(("检具", "量具"), ("检具",)),
    KeywordRule(("良率", "一致性", "稳定性"), ("良率", "一致性")),
    KeywordRule(("低压注塑", "注塑"), ("低压注塑",)),
    KeywordRule(("委外", "外协", "加工"), ("委外加工",)),
    KeywordRule(("图纸", "冻结", "未定", "未明确"), ("图纸未冻结",)),
    KeywordRule(("规格", "型号", "参数"), ("规格未冻结",)),
    KeywordRule(("样件", "样品", "到样"), ("样件数量未明确", "到样时间未明确")),
    KeywordRule(("验收", "检验", "来料"), ("验收要求未明确", "来料合格率")),
    KeywordRule(("封样",), ("封样",)),
    KeywordRule(("防水", "ipx", "ip等级"), ("防水",)),
    KeywordRule(("盐雾",), ("盐雾",)),
    KeywordRule(("疲劳", "寿命"), ("疲劳", "可靠性")),
    KeywordRule(("精度", "校准", "误差"), ("精度", "可靠性")),
    KeywordRule(("认证", "法规", "ce", "fcc", "rohs", "reach"), ("认证", "认证资料")),
    KeywordRule(("电池", "锂电"), ("电池法规",)),
    KeywordRule(("通信", "蓝牙", "无线", "射频"), ("通信法规",)),
    KeywordRule(("pcba", "pcb", "贴片", "smt"), ("PCBA",)),
    KeywordRule(("芯片", "ic", "mcu", "传感器"), ("芯片", "替代料", "AVL")),
    KeywordRule(("替代", "替换", "替代料"), ("替代料",)),
    KeywordRule(("关键尺寸", "公差", "尺寸"), ("关键尺寸",)),
    KeywordRule(("材料", "材质", "7075", "6061", "铝合金", "不锈钢", "塑料"), ("材料",)),
    KeywordRule(("表面处理", "阳极", "喷涂", "电镀", "氧化"), ("表面处理",)),
    KeywordRule(("装配", "干涉", "配合", "接口"), ("装配匹配",)),
]

STRUCTURE_HINTS = (
    "结构",
    "壳",
    "外壳",
    "支架",
    "轴",
    "轴心",
    "螺钉",
    "螺丝",
    "螺母",
    "垫片",
    "密封",
    "o型圈",
    "胶",
    "塑",
    "铝",
    "钢",
    "cnc",
    "压铸",
    "注塑",
    "防水",
    "盐雾",
    "尺寸",
    "公差",
    "表面处理",
)
ELECTRONICS_HINTS = (
    "电子",
    "pcba",
    "pcb",
    "芯片",
    "ic",
    "mcu",
    "传感",
    "电池",
    "电容",
    "电阻",
    "连接器",
    "天线",
    "蓝牙",
    "通信",
    "功率",
    "电路",
    "模组",
)
PROCESS_HINTS = (
    "工艺",
    "dfm",
    "pfmea",
    "治具",
    "检具",
    "工装",
    "良率",
    "一致性",
    "低压注塑",
    "委外",
    "外协",
    "加工",
    "装配",
)
PROCUREMENT_HINTS = (
    "供应商",
    "供方",
    "交期",
    "周期",
    "长周期",
    "成本",
    "报价",
    "moq",
    "采购",
    "齐套",
    "二供",
    "新供应商",
    "未买过",
)

STANDARD_HINTS = ("gb", "iso", "din", "m2", "m3", "m4", "m5", "m6", "螺钉", "螺丝", "螺母", "垫圈", "垫片")
SPECIAL_STANDARD_HINTS = ("正反牙", "非标", "特殊", "7075", "盐雾", "表面处理", "定制长度", "特殊等级")
CUSTOM_HINTS = ("定制", "开模", "图纸", "专用", "自研", "结构件", "外壳", "支架", "轴心")
OUTSOURCED_HINTS = ("模组", "组件", "总成", "电池", "传感器", "芯片", "连接器", "品牌", "型号")
PROCESSING_HINTS = ("委外", "外协", "加工", "cnc", "注塑", "压铸", "热处理", "表面处理", "低压注塑")


def classify_risks(risks: list[RiskItem], materials: list[MaterialRecord] | None = None) -> list[RiskItem]:
    """补充每条风险的分类、标签和采购前置问题。"""

    material_index = _material_index(materials or [])
    classified: list[RiskItem] = []
    for risk in risks:
        try:
            material = material_index.get(normalize_name(risk.material_name))
            classified.append(classify_risk(risk, material))
        except Exception as exc:
            classified.append(_fallback_classification(risk, f"分类规则执行失败：{exc}"))
    return classified


def ensure_risk_classification(risks: list[RiskItem], materials: list[MaterialRecord] | None = None) -> list[RiskItem]:
    """只为缺少分类字段的风险补齐分类，用于导出前兜底。"""

    if not risks:
        return []
    material_index = _material_index(materials or [])
    output: list[RiskItem] = []
    for risk in risks:
        if _has_classification(risk):
            output.append(risk)
            continue
        try:
            output.append(classify_risk(risk, material_index.get(normalize_name(risk.material_name))))
        except Exception as exc:
            output.append(_fallback_classification(risk, f"导出前分类兜底失败：{exc}"))
    return output


def classify_risk(risk: RiskItem, material: MaterialRecord | None = None) -> RiskItem:
    """按规则分类单条风险。"""

    text = _risk_text(risk, material)
    tags = _dedupe([*risk.risk_tags, *_tags_from_risk_type(risk.risk_type), *_tags_from_text(text)])
    owner = _primary_owner(risk.risk_type, text, tags)
    confirmation_method = _confirmation_method(risk.risk_type, text, tags)
    material_attribute = _material_attribute(text, tags, material)
    maturity = _information_maturity(risk, material)
    suggested_owner = _suggested_question_owner(owner, confirmation_method)
    missing_info = _missing_information(risk, material, tags, material_attribute)
    followup_questions = _followup_questions(risk, material, owner, confirmation_method, material_attribute, tags, missing_info)
    basis = _classification_basis(risk, owner, confirmation_method, material_attribute, tags, material, maturity)

    risk.risk_confirmation_method = confirmation_method
    risk.primary_owner = owner
    risk.material_attribute = material_attribute
    risk.risk_tags = tags
    risk.information_maturity = maturity
    risk.suggested_question_owner = suggested_owner
    risk.followup_questions = followup_questions
    risk.missing_information = missing_info
    risk.classification_basis = basis
    return risk


def merge_classification_fields(left: RiskItem, right: RiskItem) -> RiskItem:
    """人工合并风险时同步合并分类字段。"""

    left.risk_tags = _dedupe([*left.risk_tags, *right.risk_tags])
    left.followup_questions = _dedupe([*left.followup_questions, *right.followup_questions])
    left.missing_information = _dedupe([*left.missing_information, *right.missing_information])
    if not left.risk_confirmation_method:
        left.risk_confirmation_method = right.risk_confirmation_method
    elif right.risk_confirmation_method and left.risk_confirmation_method != right.risk_confirmation_method:
        left.risk_confirmation_method = CONFIRM_JOINT
    if not left.primary_owner:
        left.primary_owner = right.primary_owner
    if not left.material_attribute:
        left.material_attribute = right.material_attribute
    if not left.information_maturity:
        left.information_maturity = right.information_maturity
    if not left.suggested_question_owner:
        left.suggested_question_owner = right.suggested_question_owner
    left.classification_basis = "；".join(_dedupe([left.classification_basis, right.classification_basis]))
    return left


def _material_index(materials: list[MaterialRecord]) -> dict[str, MaterialRecord]:
    return {normalize_name(material.name): material for material in materials if material.name}


def _has_classification(risk: RiskItem) -> bool:
    return bool(
        risk.risk_confirmation_method
        or risk.primary_owner
        or risk.material_attribute
        or risk.risk_tags
        or risk.followup_questions
        or risk.missing_information
    )


def _fallback_classification(risk: RiskItem, reason: str) -> RiskItem:
    risk.risk_confirmation_method = risk.risk_confirmation_method or UNKNOWN
    risk.primary_owner = risk.primary_owner or UNKNOWN
    risk.material_attribute = risk.material_attribute or UNKNOWN
    risk.information_maturity = risk.information_maturity or MATURITY_POTENTIAL
    risk.suggested_question_owner = risk.suggested_question_owner or "项目采购代表先确认主设计/采购归口"
    risk.followup_questions = risk.followup_questions or [f"请主设计确认“{risk.material_name or '该物料'}”在计划阶段是否需要提前关注。"]
    risk.missing_information = risk.missing_information or ["对应BOM物料确认", "风险物料识别清单", "初步物料清单", "关键件选型/风险说明"]
    risk.classification_basis = risk.classification_basis or reason
    return risk


def _risk_text(risk: RiskItem, material: MaterialRecord | None) -> str:
    parts = [
        risk.material_name,
        risk.module,
        risk.risk_type,
        risk.risk_reason,
        risk.source_basis,
        "；".join(risk.unresolved_questions),
    ]
    if material:
        parts.extend(
            [
                material.name,
                material.module,
                material.spec,
                material.material,
                material.level,
                material.parent_name,
                material.bom_path,
                material.item_role,
                material.quantity,
            ]
        )
    return " ".join(part for part in parts if part).lower()


def _tags_from_risk_type(risk_type: str) -> list[str]:
    mapping = {
        "新物料/新技术风险": ["新供应商", "供方能力"],
        "供应资源风险": ["供方能力", "二供验证"],
        "长周期风险": ["长周期", "齐套交付"],
        "定制工艺风险": ["DFM", "委外加工"],
        "关键性能风险": ["可靠性", "测试不过"],
        "成本达成风险": ["成本达成", "降本压力"],
        "质量验证风险": ["来料合格率", "检验规范"],
        "接口匹配风险": ["装配匹配"],
        "认证风险": ["认证", "认证资料"],
    }
    return mapping.get(risk_type, [])


def _tags_from_text(text: str) -> list[str]:
    tags: list[str] = []
    for rule in TAG_RULES:
        if any(keyword in text for keyword in rule.keywords):
            tags.extend(rule.tags)
    return _dedupe(tags)


def _primary_owner(risk_type: str, text: str, tags: list[str]) -> str:
    scores = {
        OWNER_STRUCTURE: _score(text, STRUCTURE_HINTS),
        OWNER_ELECTRONICS: _score(text, ELECTRONICS_HINTS),
        OWNER_PROCESS: _score(text, PROCESS_HINTS),
        OWNER_PROCUREMENT: _score(text, PROCUREMENT_HINTS),
    }
    if risk_type in {"供应资源风险", "长周期风险", "成本达成风险"}:
        scores[OWNER_PROCUREMENT] += 3
    if risk_type in {"定制工艺风险", "质量验证风险"}:
        scores[OWNER_PROCESS] += 2
    if risk_type in {"关键性能风险", "接口匹配风险", "认证风险"}:
        if scores[OWNER_ELECTRONICS] >= scores[OWNER_STRUCTURE]:
            scores[OWNER_ELECTRONICS] += 2
        else:
            scores[OWNER_STRUCTURE] += 2
    if {"开模", "关键尺寸", "材料", "表面处理", "防水", "盐雾", "强度", "装配匹配"} & set(tags):
        scores[OWNER_STRUCTURE] += 2
    if {"PCBA", "芯片", "替代料", "AVL", "认证资料", "通信法规", "电池法规"} & set(tags):
        scores[OWNER_ELECTRONICS] += 2
    if {"DFM", "PFMEA", "工装", "检具", "良率", "一致性", "低压注塑", "委外加工"} & set(tags):
        scores[OWNER_PROCESS] += 2
    if {"长周期", "单一供应", "新供应商", "二供验证", "供方能力", "齐套交付", "MOQ", "成本达成"} & set(tags):
        scores[OWNER_PROCUREMENT] += 2

    owner, score = max(scores.items(), key=lambda item: item[1])
    return owner if score > 0 else UNKNOWN


def _confirmation_method(risk_type: str, text: str, tags: list[str]) -> str:
    procurement_score = _score(text, PROCUREMENT_HINTS)
    rd_score = _score(text, STRUCTURE_HINTS + ELECTRONICS_HINTS) + len({"关键尺寸", "材料", "认证", "认证资料", "精度", "装配匹配"} & set(tags))
    process_score = _score(text, PROCESS_HINTS) + len({"DFM", "工装", "检具", "良率", "一致性", "委外加工", "开模"} & set(tags))

    rd_action_tags = ("规格未冻结", "图纸未冻结", "认证资料", "关键尺寸", "材料", "表面处理")
    rd_action_text = ("替代", "规格未定", "图纸未定", "认证资料", "关键尺寸", "材料未定")
    if risk_type in {"供应资源风险", "长周期风险", "成本达成风险"} and not any(tag in tags for tag in rd_action_tags) and not any(term in text for term in rd_action_text):
        return CONFIRM_PROCUREMENT
    if risk_type in {"关键性能风险", "接口匹配风险", "认证风险"} and procurement_score == 0 and process_score == 0:
        return CONFIRM_RD
    if procurement_score > 0 and (rd_score > 0 or process_score > 0):
        return CONFIRM_JOINT
    if risk_type in {"定制工艺风险", "质量验证风险"}:
        return CONFIRM_JOINT
    if procurement_score > rd_score + process_score:
        return CONFIRM_PROCUREMENT
    if rd_score or risk_type in {"关键性能风险", "接口匹配风险"}:
        return CONFIRM_RD
    return UNKNOWN


def _material_attribute(text: str, tags: list[str], material: MaterialRecord | None) -> str:
    if _score(text, PROCESSING_HINTS) or {"委外加工", "低压注塑", "表面处理"} & set(tags):
        return MAT_PROCESSING
    if _score(text, CUSTOM_HINTS) or {"开模", "DFM", "关键尺寸"} & set(tags):
        return MAT_CUSTOM
    if _score(text, SPECIAL_STANDARD_HINTS):
        return MAT_SPECIAL_STANDARD
    if _score(text, STANDARD_HINTS):
        return MAT_STANDARD
    if _score(text, OUTSOURCED_HINTS):
        return MAT_OUTSOURCED
    if material and (material.spec or material.material):
        return MAT_OUTSOURCED
    return UNKNOWN


def _information_maturity(risk: RiskItem, material: MaterialRecord | None) -> str:
    has_evidence = bool(risk.evidence_items or risk.source_basis)
    has_material = bool(material or (risk.material_name and risk.material_name not in {"待确认物料", "未知物料"}))
    has_open_questions = bool(risk.unresolved_questions)
    if has_evidence and has_material and not has_open_questions:
        return MATURITY_CLEAR
    return MATURITY_POTENTIAL


def _suggested_question_owner(owner: str, confirmation_method: str) -> str:
    if owner == OWNER_STRUCTURE:
        return "结构负责人/主设计"
    if owner == OWNER_ELECTRONICS:
        return "电子负责人/主设计"
    if owner == OWNER_PROCESS:
        return "工艺负责人"
    if owner == OWNER_PROCUREMENT:
        return "寻源采购/采购代表"
    if confirmation_method == CONFIRM_JOINT:
        return "主设计+工艺+寻源采购"
    return "项目采购代表先确认主设计/采购归口"


def _missing_information(risk: RiskItem, material: MaterialRecord | None, tags: list[str], material_attribute: str) -> list[str]:
    missing: list[str] = []
    text = _risk_text(risk, material)
    if not material or risk.material_name in {"待确认物料", "未知物料", ""}:
        missing.append("对应BOM物料确认")
    if "风险物料识别清单" not in text:
        missing.append("风险物料识别清单")
    if "长周期" in text or {"长周期", "齐套交付"} & set(tags):
        missing.append("长周期风险物料确认")
    if {"DFM", "PFMEA", "可靠性"} & set(tags) or "dfmea" in text:
        missing.append("概要设计DFMEA")
    if owner_hint := _plan_material_list_hint(risk, material, tags):
        missing.append(owner_hint)
    if material_attribute in {MAT_CUSTOM, MAT_PROCESSING, MAT_SPECIAL_STANDARD} or {"关键尺寸", "开模", "材料", "表面处理"} & set(tags):
        missing.append("初步零部件图纸/关键尺寸")
    if {"PCBA", "芯片", "AVL", "替代料"} & set(tags):
        missing.append("电子关键件选型/风险说明")
    if {"PCBA"} & set(tags):
        missing.append("PCBA测试工装初始需求")
    if {"DFM", "委外加工", "低压注塑"} & set(tags):
        missing.append("机加工可行性/整机DFM")
    if {"工装", "检具", "来料合格率"} & set(tags):
        missing.append("检具初始需求")
    if {"认证", "认证资料", "通信法规", "电池法规"} & set(tags):
        missing.append("产品认证清单/法规合规需求")
    if {"样件数量未明确", "到样时间未明确"} & set(tags) or "样" in text or "齐套" in text:
        missing.append("原理样机物料采购进度/齐套信息")
    return _dedupe(missing)


def _plan_material_list_hint(risk: RiskItem, material: MaterialRecord | None, tags: list[str]) -> str:
    text = _risk_text(risk, material)
    if {"PCBA", "芯片", "AVL", "替代料", "认证资料", "通信法规", "电池法规"} & set(tags) or _score(text, ELECTRONICS_HINTS):
        return "初步电子物料清单"
    if {"关键尺寸", "开模", "材料", "表面处理", "装配匹配", "防水", "盐雾"} & set(tags) or _score(text, STRUCTURE_HINTS):
        return "初步结构物料清单"
    if material:
        return "初步物料清单"
    return ""


def _followup_questions(
    risk: RiskItem,
    material: MaterialRecord | None,
    owner: str,
    confirmation_method: str,
    material_attribute: str,
    tags: list[str],
    missing_info: list[str],
) -> list[str]:
    name = risk.material_name or "该物料"
    missing = _join_cn(missing_info[:3])
    suffix = f"，并补充{missing}" if missing else ""
    if material and material.has_children:
        suffix += "，说明影响组件整体还是下级物料"

    if owner == OWNER_STRUCTURE:
        core = f"请结构确认“{name}”是否是计划阶段需要提前关注的结构风险{suffix}。"
    elif owner == OWNER_ELECTRONICS:
        core = f"请电子确认“{name}”是否是计划阶段需要提前关注的电子风险{suffix}。"
    elif owner == OWNER_PROCESS:
        core = f"请工艺/质量确认“{name}”是否需要在计划阶段提前做可行性、DFM或检具判断{suffix}。"
    elif owner == OWNER_PROCUREMENT:
        core = f"请采购确认“{name}”在计划阶段是否存在长周期、齐套或供方能力风险{suffix}。"
    else:
        core = f"请主设计确认“{name}”是否应纳入计划阶段风险物料清单{suffix}。"

    return [core]


def _classification_basis(
    risk: RiskItem,
    owner: str,
    confirmation_method: str,
    material_attribute: str,
    tags: list[str],
    material: MaterialRecord | None,
    maturity: str,
) -> str:
    basis_parts = [
        f"风险类型={risk.risk_type}",
        f"主归口={owner}",
        f"确认方式={confirmation_method}",
        f"物料属性={material_attribute}",
        f"信息成熟度={maturity}",
    ]
    if tags:
        basis_parts.append(f"命中标签={_join_cn(tags[:8])}")
    if material:
        material_desc = _join_cn(
            [
                part
                for part in [
                    f"层级{material.level}" if material.level else "",
                    f"路径{material.bom_path}" if material.bom_path else "",
                    f"上级{material.parent_name}" if material.parent_name else "",
                    f"角色{material.item_role}" if material.item_role else "",
                    material.spec,
                    material.material,
                    f"BOM行{material.row_number}" if material.row_number else "",
                ]
                if part
            ]
        )
        if material_desc:
            basis_parts.append(f"BOM信息={material_desc}")
    if risk.unresolved_questions:
        basis_parts.append("原待确认点已转入问题清单")
    return "；".join(basis_parts)


def _score(text: str, hints: tuple[str, ...]) -> int:
    return sum(1 for hint in hints if hint.lower() in text)


def normalize_name(name: str) -> str:
    """轻量物料名归一化，避免分类模块反向依赖风险识别模块。"""

    return re.sub(r"\s+", "", str(name or "")).lower()


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _join_cn(values: list[str]) -> str:
    clean = _dedupe(values)
    return "、".join(clean) if clean else "待确认信息"


def _material_scope(material: MaterialRecord | None) -> str:
    if not material:
        return ""
    if material.parent_name:
        return f"（上级：{material.parent_name}）"
    if material.bom_path and material.bom_path != material.name:
        return f"（路径：{material.bom_path}）"
    return ""


def split_multi_text(value: str | list[str]) -> list[str]:
    """供导出层复用的多值文本拆分。"""

    if isinstance(value, list):
        return _dedupe([str(item) for item in value])
    return _dedupe([item.strip() for item in re.split(r"[,，;；\n]+", str(value or "")) if item.strip()])
