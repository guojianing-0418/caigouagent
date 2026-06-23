# 计划阶段采购风险物料识别 Agent

这是一个本机/内网单机版 Web 应用，用于在 IPD 计划阶段识别采购需要提前关注的风险物料。第一版只做计划阶段，EVT / DVT / PVT 后续扩展。

## 功能范围

- 读取草 BOM Excel，解析模块、物料、规格、材质和层级，并交给大模型识别风险。
- 通过统一 `DocumentIngestor` 读取 PRD / 规格书 / PDF / TXT / MD / CSV，抽取带文件、sheet、行号、页码和解析器信息的文本线索。
- 从 PRD / 规格书中抽取 `FactItem` 产品事实，再交给风险识别引用。
- 读取 PDF 图纸文件，支持整页视觉识别。
- 读取研发自提风险 Excel，模板只有两列：`风险物料名称`、`原因`。
- 通过 `lark-cli` 读取飞书项目群历史消息。
- WebUI 支持配置 OpenAI 兼容 `Base URL`、`API Key`、文本模型和视觉模型。
- WebUI 支持历史记录入口，可查看之前创建过的项目、风险结果和导出文件。
- 人工确认采用通用 Question Engine：阻塞问题会弹窗暂停 Agent，非阻塞问题进入“人工确认”面板；非阻塞必答问题回答后会自动重新识别以应用人工回答。
- 风险识别优先使用 OpenAI 兼容 Structured Outputs；不支持时自动回退到 JSON prompt。
- 同一项目的模型分块输出会缓存到 SQLite，重跑或继续运行时尽量复用。
- 下载正式 Excel 前会检查未处理的必答问题，避免跳过关键采购确认。
- 未配置或未连通大模型时，系统不会执行风险物料识别，也不会输出规则兜底结果。
- 输出 Excel 主表保留：`风险物料名称`、`所属模块`、`风险类型`、`风险原因`、`来源与依据`，并追加风险确认方式、主归口、物料属性、风险标签、信息成熟度、可发群问题和待补齐信息。
- 输出 Excel 增加 `问题分发清单` sheet，方便采购代表拿表去问主设计、结构、电子、工艺或寻源采购；`证据详情` sheet 继续导出结构化证据、置信度和待确认点。
- 风险分类和问题清单用于计划阶段资料尚不完整时的前置识别，后续可随图纸、样机、测试、工艺和供应商反馈滚动更新。

## 目录结构

```text
backend/        FastAPI 后端、Agent 流程、解析器、飞书读取、Excel 导出
frontend/       React + Vite 前端
scripts/        命令行调试脚本
examples/       示例 YAML 配置
data/           本地上传、缓存、导出和运行状态
```

## 后端启动

建议先创建虚拟环境，然后安装依赖：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

启动 FastAPI：

```bash
python -m backend.main
```

默认后端地址：

```text
http://127.0.0.1:8000
```

## 前端启动

```bash
cd frontend
npm install
npm run dev
```

默认前端地址：

```text
http://127.0.0.1:5173
```

## 模型配置

可以在 WebUI 的“模型配置”区域填写：

- Base URL
- API Key
- 文本模型
- 视觉模型

也可以用环境变量配置：

```bash
set OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
set OPENAI_API_KEY=your-api-key
set TEXT_MODEL=your-text-model
set VISION_MODEL=your-vision-model
```

WebUI 保存的配置优先级高于环境变量。API key 保存在本机 `data/config/model_settings.json`，接口只返回掩码，不回显明文。

点击“开始识别”时，后端会先做一次模型连通性检查。检查失败时，本次识别会停止，不会生成风险物料结果。

## 飞书授权

第一版使用 `lark-cli` 本地授权读取当前用户可见群聊。

常用步骤：

```bash
lark-cli config init --new
lark-cli auth login --scope "im:chat:read im:message:readonly"
```

如果公司飞书应用权限不足，需要在飞书开发者后台开通对应 scope。WebUI 中“飞书项目群”可以填写群名或 `oc_xxx` chat_id。

## 人工确认问题模块

问题模块借鉴 opencode 的 question tool 思路：Agent 不把问题写进普通日志，而是创建结构化 `Question`。前端根据 `input_type` 渲染控件，后端根据 `question_kind` 应用答案影响。

当前支持的输入类型：

- `single_select`：单选，支持自定义答案。
- `multi_select`：多选，支持补充自定义答案。
- `boolean`：是/否类确认，例如保留、删除、合并、不合并。
- `text`：短文本。
- `textarea`：长文本。

当前支持的关键字段：

- `question_kind`：业务分类，例如 `lark_chat_selection`、`risk_material_mapping`、`procurement_confirmation`、`risk_merge_review`。
- `blocking`：为 `true` 时前端弹窗，Agent 状态进入 `waiting`，回答后通过 `/api/projects/{id}/resume` 继续。
- `blocking=true` 仅用于“不回答就无法继续正确执行”的问题；风险保留、风险合并、风险类型修正等结果确认默认应为 `blocking=false`。
- 大模型如果没有明确返回 `blocking=true`，后端会按非阻塞问题处理，避免运行过程中频繁打断采购。
- `required`：表示正式输出前必须处理；未处理时可以预览候选结果，但不能下载正式 Excel。
- `permission`：保留 `allow / ask / deny` 语义；首版主要使用 `ask`。
- `context`：保存来源摘要、相关物料、风险 ID、轻量后续动作等上下文。

问题生命周期：

- 已回答、已跳过、已拒绝的问题会作为人工上下文保留，后续模型调用必须参考。
- 重跑时未回答的 pending 问题不会被直接清掉；新生成的同类问题会刷新旧问题，避免重复卡片。
- 风险保留 / 风险合并类问题如果关联的风险已经不存在，会自动清理。

问题接口：

```text
GET  /api/projects/{id}/questions
GET  /api/projects/{id}/questions/active
POST /api/projects/{id}/questions/{question_id}/answer
POST /api/projects/{id}/resume
GET  /api/projects/{id}/export/check
```

回答 payload：

```json
{
  "answer": "用户答案",
  "action": "submit"
}
```

## 可靠性机制

- `backend/risk_rules.py` 中的 `PROMPT_VERSION` 用于提示词和缓存版本管理；修改风险识别提示词时应同步更新它。
- 模型输出缓存保存在 `data/cache/risk_agent.sqlite3` 的 `llm_cache` 表中，缓存 key 包含项目、来源、分块 hash、模型名和 prompt 版本。
- Structured Outputs 不可用时，系统自动回退到普通 JSON 调用，并继续执行后端字段校验。
- 正式导出接口会阻止未处理 `required=true` 问题的项目下载 Excel；前端会先调用 `/export/check` 给出提示。
- 项目运行中或自动重新识别中，正式导出会被暂时阻止，避免下载旧结果。
- `backend/parsers/document_ingestor.py` 保留实验解析器注册入口，可在实验分支接入 MarkItDown / Docling / Unstructured；默认部署不依赖这些库。

## 评测

首版评测框架放在 `evals/` 下，默认样本是 P725。

不调用真实模型的 mock 评测：

```bash
python evals/run_eval.py --case evals/cases/p725/case.yaml --mode mock
```

批量运行所有 case：

```bash
python evals/run_eval.py --case-dir evals/cases --mode mock
```

调用真实模型的 live 评测：

```bash
python evals/run_eval.py --case evals/cases/p725/case.yaml --mode live
```

评测报告会输出到 `evals/reports/`，包含召回率、误报数、风险类型准确率、证据命中率、结构化证据覆盖率、事实命中率、事实来源覆盖率、导出检查、自动重跑检查和问题生命周期检查。`expected_risks.json` 是人工期望风险清单，`expected_facts.json` 是人工期望事实清单，后续每个项目都可以按同样结构增加样本。mock 模式会做阈值检查，关键指标不达标时以非 0 退出。

本地生成物不纳入提交：

- `backend-fastapi*.log`
- `frontend-vite*.log`
- `data/templates/研发自提风险模板.xlsx`
- `evals/reports/`

## 研发自提风险模板

WebUI 启动后会自动生成模板，也可下载：

```text
GET /api/templates/rd-risk
```

模板字段：

| 风险物料名称 | 原因 |
|---|---|

## 命令行调试

不经过 WebUI，直接运行示例项目：

```bash
python scripts/run_plan_stage.py --config examples/p725.yaml
```

示例配置字段：

```yaml
project_name: P725计划阶段风险识别
bom_path: P725脚踏功率计 结构EBOM.xlsx
prd_path: P725功率计PRD V1.xlsx
spec_path:
drawing_dir:
rd_risk_path:
lark_chat:
```

## 风险类型

第一版固定 8 类：

- 新物料/新技术风险
- 供应资源风险
- 长周期风险
- 定制工艺风险
- 关键性能风险
- 成本达成风险
- 质量验证风险
- 接口匹配风险

风险识别由大模型完成。`backend/risk_rules.py` 只负责准备提示词、校验模型 JSON 输出、合并去重和生成需要人工确认的问题，不再保留关键词兜底规则。
