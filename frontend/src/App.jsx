import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  AlertCircle,
  Bot,
  Check,
  Download,
  FileSpreadsheet,
  History,
  KeyRound,
  Loader2,
  MessageSquareText,
  Play,
  RefreshCw,
  Save,
  Settings,
} from "lucide-react";
import "./styles.css";

const API_BASE = "";

// 统一封装 fetch，方便集中处理错误。
async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, options);
  if (!response.ok) {
    let message = `请求失败：${response.status}`;
    try {
      const data = await response.json();
      if (typeof data.detail === "string") {
        message = data.detail;
      } else if (data.detail?.message) {
        message = data.detail.message;
      } else if (data.message) {
        message = data.message;
      }
    } catch {
      // 忽略非 JSON 错误体。
    }
    throw new Error(message);
  }
  return response.json();
}

function App() {
  const [modelForm, setModelForm] = useState({
    openai_base_url: "",
    openai_api_key: "",
    text_model: "",
    vision_model: "",
  });
  const [modelStatus, setModelStatus] = useState(null);
  const [projectForm, setProjectForm] = useState({
    project_name: "P725计划阶段风险识别",
    lark_chat: "",
  });
  const [files, setFiles] = useState({});
  const [project, setProject] = useState(null);
  const [history, setHistory] = useState([]);
  const [questions, setQuestions] = useState([]);
  const [activeQuestion, setActiveQuestion] = useState(null);
  const [risks, setRisks] = useState([]);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");

  const projectId = project?.id;
  const modelReady = Boolean(modelStatus?.openai_base_url && modelStatus?.has_api_key && modelStatus?.text_model);

  useEffect(() => {
    loadModelSettings();
    loadHistory();
  }, []);

  useEffect(() => {
    if (!projectId) return;
    const timer = setInterval(() => refreshProject(projectId), 2500);
    return () => clearInterval(timer);
  }, [projectId]);

  const canRun = useMemo(
    () => Boolean(project?.id && project.status !== "running" && modelReady && !activeQuestion),
    [project, modelReady, activeQuestion]
  );
  const panelQuestions = useMemo(
    () => questions.filter((question) => !activeQuestion || question.id !== activeQuestion.id),
    [questions, activeQuestion]
  );
  const requiredQuestionCount = useMemo(
    () => questions.filter((question) => question.required).length,
    [questions]
  );

  async function loadModelSettings() {
    try {
      const data = await request("/api/settings/model");
      setModelStatus(data);
      setModelForm({
        openai_base_url: data.openai_base_url || "",
        openai_api_key: "",
        text_model: data.text_model || "",
        vision_model: data.vision_model || "",
      });
    } catch (err) {
      setError(err.message);
    }
  }

  async function loadHistory() {
    try {
      const rows = await request("/api/projects");
      setHistory(rows);
    } catch (err) {
      setError(err.message);
    }
  }

  async function saveModelSettings(event) {
    event.preventDefault();
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const data = await request("/api/settings/model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(modelForm),
      });
      setModelStatus(data);
      setModelForm((prev) => ({ ...prev, openai_api_key: "" }));
      setNotice("模型配置已保存。");
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function createProject(event) {
    event.preventDefault();
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const form = new FormData();
      form.append("project_name", projectForm.project_name);
      form.append("lark_chat", projectForm.lark_chat || "");
      appendFile(form, "bom_file", files.bom_file);
      appendFile(form, "prd_file", files.prd_file);
      appendFile(form, "spec_file", files.spec_file);
      appendFile(form, "rd_risk_file", files.rd_risk_file);
      if (files.drawing_files) {
        Array.from(files.drawing_files).forEach((file) => form.append("drawing_files", file));
      }

      const data = await request("/api/projects", { method: "POST", body: form });
      setProject(data);
      setRisks(data.risks || []);
      setQuestions([]);
      setActiveQuestion(null);
      await loadHistory();
      setNotice("项目已创建，可以开始识别。");
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function runProject() {
    if (!project?.id) return;
    setBusy(true);
    setError("");
    setNotice("");
    try {
      await request(`/api/projects/${project.id}/run`, { method: "POST" });
      setNotice("已启动识别任务。");
      await refreshProject(project.id);
      await loadHistory();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function refreshProject(id = projectId) {
    if (!id) return;
    try {
      const [state, pendingQuestions, active, riskRows] = await Promise.all([
        request(`/api/projects/${id}`),
        request(`/api/projects/${id}/questions`),
        request(`/api/projects/${id}/questions/active`),
        request(`/api/projects/${id}/risks`),
      ]);
      setProject(state);
      setQuestions(pendingQuestions);
      setActiveQuestion(active);
      setRisks(riskRows);
      await loadHistory();
    } catch (err) {
      setError(err.message);
    }
  }

  async function answerQuestion(question, answer, action = "submit") {
    setBusy(true);
    setError("");
    try {
      const data = await request(`/api/projects/${project.id}/questions/${question.id}/answer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ answer, action }),
      });
      setProject(data);
      if (question.blocking) {
        await request(`/api/projects/${project.id}/resume`, { method: "POST" });
      }
      await refreshProject(project.id);
      await loadHistory();
      setNotice(question.blocking ? "确认结果已保存，Agent 已继续运行。" : "确认结果已保存。");
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function downloadExport() {
    if (!project?.id) return;
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const check = await request(`/api/projects/${project.id}/export/check`);
      if (!check.allowed) {
        setError(check.message || "仍有必答问题未处理，不能下载正式 Excel。");
        await refreshProject(project.id);
        return;
      }
      window.location.href = `/api/projects/${project.id}/export`;
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function openHistoryProject(id) {
    setBusy(true);
    setError("");
    try {
      await refreshProject(id);
      setNotice("已打开历史项目。");
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">IPD 计划阶段</p>
          <h1>采购风险物料识别 Agent</h1>
        </div>
        <button className="icon-button" type="button" onClick={() => refreshProject()} title="刷新状态">
          <RefreshCw size={18} />
        </button>
      </header>

      {(notice || error) && (
        <section className={error ? "banner banner-error" : "banner"}>
          {error ? <AlertCircle size={18} /> : <Check size={18} />}
          <span>{error || notice}</span>
        </section>
      )}

      {activeQuestion && (
        <BlockingQuestionModal question={activeQuestion} busy={busy} onAnswer={answerQuestion} />
      )}

      <div className="workspace">
        <section className="left-pane">
          <Panel
            icon={<KeyRound size={18} />}
            title="模型配置"
            description="OpenAI 兼容接口，保存后用于文本抽取和 PDF 图纸视觉识别。"
          >
            <form className="stack" onSubmit={saveModelSettings}>
              <TextInput
                label="Base URL"
                value={modelForm.openai_base_url}
                placeholder="https://your-endpoint/v1"
                onChange={(value) => setModelForm({ ...modelForm, openai_base_url: value })}
              />
              <TextInput
                label="API Key"
                type="password"
                value={modelForm.openai_api_key}
                placeholder={modelStatus?.has_api_key ? `已保存：${modelStatus.api_key_masked}` : "请输入 API key"}
                onChange={(value) => setModelForm({ ...modelForm, openai_api_key: value })}
              />
              <div className="two-col">
                <TextInput
                  label="文本模型"
                  value={modelForm.text_model}
                  placeholder="例如 gpt-4.1-mini"
                  onChange={(value) => setModelForm({ ...modelForm, text_model: value })}
                />
                <TextInput
                  label="视觉模型"
                  value={modelForm.vision_model}
                  placeholder="例如 gpt-4.1"
                  onChange={(value) => setModelForm({ ...modelForm, vision_model: value })}
                />
              </div>
              <button className="primary-button" type="submit" disabled={busy}>
                <Save size={17} />
                保存模型配置
              </button>
            </form>
          </Panel>

          <Panel
            icon={<FileSpreadsheet size={18} />}
            title="项目输入"
            description="草 BOM 必填，其余文件可为空。飞书群可填群名或 chat_id。"
          >
            <form className="stack" onSubmit={createProject}>
              <TextInput
                label="项目名称"
                value={projectForm.project_name}
                onChange={(value) => setProjectForm({ ...projectForm, project_name: value })}
              />
              <TextInput
                label="飞书项目群"
                value={projectForm.lark_chat}
                placeholder="群名或 oc_xxx，可为空"
                onChange={(value) => setProjectForm({ ...projectForm, lark_chat: value })}
              />
              <FileInput label="草 BOM Excel（必填）" accept=".xlsx,.xlsm" onChange={(file) => setFiles({ ...files, bom_file: file })} />
              <FileInput label="PRD Excel（建议）" accept=".xlsx,.xlsm" onChange={(file) => setFiles({ ...files, prd_file: file })} />
              <FileInput label="规格书（可为空）" accept=".xlsx,.xlsm,.pdf,.txt,.md" onChange={(file) => setFiles({ ...files, spec_file: file })} />
              <FileInput label="研发自提风险 Excel（可为空）" accept=".xlsx,.xlsm" onChange={(file) => setFiles({ ...files, rd_risk_file: file })} />
              <FileInput
                label="PDF 图纸文件（可多选）"
                accept=".pdf"
                multiple
                onChange={(fileList) => setFiles({ ...files, drawing_files: fileList })}
              />
              <div className="button-row">
                <button className="secondary-button" type="submit" disabled={busy}>
                  <Settings size={17} />
                  创建项目
                </button>
                <button className="primary-button" type="button" disabled={!canRun || busy} onClick={runProject}>
                  {project?.status === "running" ? <Loader2 className="spin" size={17} /> : <Play size={17} />}
                  开始识别
                </button>
              </div>
              {!modelReady && <p className="hint-text">请先保存可用的大模型配置，否则不能开始识别。</p>}
            </form>
          </Panel>

          <Panel
            icon={<History size={18} />}
            title="历史记录"
            description="查看本机之前创建过的识别项目。"
          >
            <HistoryList rows={history} currentId={project?.id} onOpen={openHistoryProject} />
          </Panel>
        </section>

        <section className="right-pane">
          <Panel icon={<Bot size={18} />} title="运行状态" description={project ? project.current_step : "请先创建项目。"}>
            <StatusStrip project={project} />
            <LogList logs={project?.logs || []} />
          </Panel>

          <Panel
            icon={<MessageSquareText size={18} />}
            title="人工确认"
            description={`待处理 ${questions.length} 个，必答 ${requiredQuestionCount} 个。`}
          >
            {panelQuestions.length === 0 ? (
              <EmptyText text="当前没有待确认问题。" />
            ) : (
              <div className="question-list">
                {panelQuestions.map((question) => (
                  <QuestionRenderer key={question.id} question={question} busy={busy} onAnswer={answerQuestion} />
                ))}
              </div>
            )}
          </Panel>

          <Panel icon={<FileSpreadsheet size={18} />} title="风险物料预览" description={`当前 ${risks.length} 条。`}>
            <RiskTable risks={risks} />
            {requiredQuestionCount > 0 && (
              <p className="hint-text export-gate-text">
                还有 {requiredQuestionCount} 个必答问题未处理，处理后可下载正式 Excel。
              </p>
            )}
            <div className="button-row end">
              <button className="primary-button" type="button" disabled={!project?.id || risks.length === 0 || busy} onClick={downloadExport}>
                <Download size={17} />
                下载 Excel
              </button>
            </div>
          </Panel>
        </section>
      </div>
    </main>
  );
}

function BlockingQuestionModal({ question, busy, onAnswer }) {
  return (
    <div className="modal-backdrop">
      <section className="question-modal">
        <div className="modal-header">
          <h2>需要采购确认</h2>
          <p>Agent 已暂停，处理后会继续执行。</p>
        </div>
        <QuestionRenderer question={question} busy={busy} onAnswer={onAnswer} blocking />
      </section>
    </div>
  );
}

function QuestionRenderer({ question, busy, onAnswer, blocking = false }) {
  const inputType = question.input_type || "single_select";
  const options = Array.isArray(question.options) ? question.options : [];
  const allowCustom = question.allow_custom !== false;
  const [textAnswer, setTextAnswer] = useState(stringDefault(question.default_value));
  const [multiAnswer, setMultiAnswer] = useState(arrayDefault(question.default_value));

  useEffect(() => {
    setTextAnswer(stringDefault(question.default_value));
    setMultiAnswer(arrayDefault(question.default_value));
  }, [question.id, question.default_value]);

  const contextExcerpt =
    question.context?.source_excerpt ||
    question.context?.source ||
    question.context?.source_name ||
    question.context?.query ||
    "";

  function submitTextAnswer() {
    onAnswer(question, textAnswer);
  }

  function submitMultiAnswer() {
    const customValues = parseCustomValues(textAnswer);
    onAnswer(question, [...multiAnswer, ...customValues]);
  }

  function toggleMulti(option) {
    setMultiAnswer((prev) =>
      prev.includes(option) ? prev.filter((item) => item !== option) : [...prev, option]
    );
  }

  return (
    <article className={`question question-renderer ${blocking ? "blocking" : ""}`}>
      <div className="question-meta">
        <span>{questionKindLabel(question.question_kind)}</span>
        {question.blocking && <span>阻塞</span>}
        {question.required && <span>必答</span>}
        <span>{permissionLabel(question.permission)}</span>
      </div>
      <div className="question-copy">
        <h3>{question.title}</h3>
        {question.message && <p>{question.message}</p>}
        {question.reason && <p className="question-reason">{question.reason}</p>}
      </div>
      {contextExcerpt && <div className="question-context">{contextExcerpt}</div>}

      {inputType === "single_select" && (
        <div className="question-control">
          <div className="answer-row">
            {options.map((option) => (
              <button className="secondary-button compact" type="button" key={option} onClick={() => onAnswer(question, option)} disabled={busy}>
                {option}
              </button>
            ))}
          </div>
          {allowCustom && (
            <div className="custom-answer">
              <input value={textAnswer} placeholder="自定义答案" onChange={(event) => setTextAnswer(event.target.value)} />
              <button className="primary-button compact" type="button" onClick={submitTextAnswer} disabled={busy || !textAnswer.trim()}>
                <Check size={15} />
                提交
              </button>
            </div>
          )}
        </div>
      )}

      {inputType === "multi_select" && (
        <div className="question-control">
          <div className="checkbox-list">
            {options.map((option) => (
              <label className="checkbox-option" key={option}>
                <input type="checkbox" checked={multiAnswer.includes(option)} onChange={() => toggleMulti(option)} />
                <span>{option}</span>
              </label>
            ))}
          </div>
          {allowCustom && (
            <input value={textAnswer} placeholder="补充自定义答案，可用分号分隔" onChange={(event) => setTextAnswer(event.target.value)} />
          )}
          <div className="answer-row">
            <button className="primary-button compact" type="button" onClick={submitMultiAnswer} disabled={busy || (multiAnswer.length === 0 && !textAnswer.trim())}>
              <Check size={15} />
              提交
            </button>
            {!question.required && (
              <button className="secondary-button compact" type="button" onClick={() => onAnswer(question, [], "skip")} disabled={busy}>
                跳过
              </button>
            )}
          </div>
        </div>
      )}

      {inputType === "boolean" && (
        <div className="answer-row">
          {(options.length ? options : ["是", "否"]).map((option) => (
            <button className="secondary-button compact" type="button" key={option} onClick={() => onAnswer(question, option)} disabled={busy}>
              {option}
            </button>
          ))}
          {!question.required && !question.blocking && (
            <button className="secondary-button compact" type="button" onClick={() => onAnswer(question, null, "skip")} disabled={busy}>
              跳过
            </button>
          )}
        </div>
      )}

      {inputType === "text" && (
        <div className="custom-answer">
          <input value={textAnswer} placeholder="请输入答案" onChange={(event) => setTextAnswer(event.target.value)} />
          <button className="primary-button compact" type="button" onClick={submitTextAnswer} disabled={busy || (question.required && !textAnswer.trim())}>
            <Check size={15} />
            提交
          </button>
        </div>
      )}

      {inputType === "textarea" && (
        <div className="question-control">
          <textarea value={textAnswer} placeholder="请输入详细说明" onChange={(event) => setTextAnswer(event.target.value)} />
          <div className="answer-row">
            <button className="primary-button compact" type="button" onClick={submitTextAnswer} disabled={busy || (question.required && !textAnswer.trim())}>
              <Check size={15} />
              提交
            </button>
          </div>
        </div>
      )}
    </article>
  );
}

function questionKindLabel(kind) {
  const labels = {
    lark_chat_selection: "飞书群选择",
    risk_material_mapping: "物料映射",
    procurement_confirmation: "采购确认",
    risk_merge_review: "风险合并",
    risk_keep_review: "风险保留",
  };
  return labels[kind] || "人工确认";
}

function permissionLabel(permission) {
  const labels = {
    allow: "自动允许",
    ask: "需要确认",
    deny: "拒绝",
  };
  return labels[permission] || "需要确认";
}

function stringDefault(value) {
  if (value === null || value === undefined || Array.isArray(value)) return "";
  return String(value);
}

function arrayDefault(value) {
  if (!Array.isArray(value)) return [];
  return value.map((item) => String(item));
}

function parseCustomValues(value) {
  return String(value || "")
    .split(/[，,；;\n]/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function appendFile(form, name, file) {
  if (file) form.append(name, file);
}

function Panel({ icon, title, description, children }) {
  return (
    <section className="panel">
      <div className="panel-header">
        <span className="panel-icon">{icon}</span>
        <div>
          <h2>{title}</h2>
          <p>{description}</p>
        </div>
      </div>
      {children}
    </section>
  );
}

function TextInput({ label, value, onChange, placeholder = "", type = "text" }) {
  return (
    <label className="field">
      <span>{label}</span>
      <input type={type} value={value} placeholder={placeholder} onChange={(event) => onChange(event.target.value)} />
    </label>
  );
}

function FileInput({ label, accept, multiple = false, onChange }) {
  return (
    <label className="field">
      <span>{label}</span>
      <input type="file" accept={accept} multiple={multiple} onChange={(event) => onChange(multiple ? event.target.files : event.target.files?.[0])} />
    </label>
  );
}

function StatusStrip({ project }) {
  const status = project?.status || "created";
  return (
    <div className="status-strip">
      <span className={`status-dot ${status}`} />
      <strong>{statusLabel(status)}</strong>
      {project?.error && <span className="error-text">{project.error}</span>}
    </div>
  );
}

function statusLabel(status) {
  const labels = {
    created: "已创建",
    running: "运行中",
    waiting: "等待人工确认",
    done: "已完成",
    error: "异常",
  };
  return labels[status] || status;
}

function LogList({ logs }) {
  if (!logs.length) return <EmptyText text="暂无运行日志。" />;
  return (
    <div className="log-list">
      {logs.slice(-10).map((log, index) => (
        <p key={`${log}-${index}`}>{log}</p>
      ))}
    </div>
  );
}

function RiskTable({ risks }) {
  if (!risks.length) return <EmptyText text="暂无风险物料结果。" />;
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>风险物料名称</th>
            <th>所属模块</th>
            <th>风险类型</th>
            <th>风险原因</th>
            <th>来源与依据</th>
          </tr>
        </thead>
        <tbody>
          {risks.map((risk) => (
            <tr key={risk.id}>
              <td>{risk.material_name}</td>
              <td>{risk.module}</td>
              <td>{risk.risk_type}</td>
              <td>{risk.risk_reason}</td>
              <td>{risk.source_basis}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function HistoryList({ rows, currentId, onOpen }) {
  if (!rows.length) return <EmptyText text="暂无历史项目。" />;
  return (
    <div className="history-list">
      {rows.map((row) => (
        <button
          className={`history-item ${row.id === currentId ? "active" : ""}`}
          type="button"
          key={row.id}
          onClick={() => onOpen(row.id)}
        >
          <span className="history-title">{row.project_name}</span>
          <span className="history-meta">
            {statusLabel(row.status)} · {row.risk_count} 条风险 · {row.question_count} 个待确认 · {row.required_question_count || 0} 个必答
          </span>
          <span className="history-time">{row.updated_at}</span>
        </button>
      ))}
    </div>
  );
}

function EmptyText({ text }) {
  return <p className="empty-text">{text}</p>;
}

createRoot(document.getElementById("root")).render(<App />);
