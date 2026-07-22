const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

const appState = {
  bootstrap: null,
  system: null,
  datasets: [],
  runs: [],
  selectedRunId: null,
  selectedRun: null,
  detailTab: "summary",
  workflowRunId: null,
  workflow: null,
  workflowFlow: "ingest",
  selectedNode: null,
  swe: null,
  sweHistory: [],
  sweArmDraft: null,
  environment: null,
  polling: false,
};

const activeRunStates = new Set(["running", "preparing", "paused", "stopping", "retrying"]);
const activeSweStates = new Set(["preparing", "building_memory", "running_agent", "running_harness", "finalizing", "running_all"]);
const pageMeta = {
  overview: ["CONTROL ROOM", "实验总览"],
  swe: ["SWE-BENCH", "SWE 双模式"],
  runs: ["RUN CENTER", "运行中心"],
  workflow: ["INTERNAL WORKFLOW", "内部工作流"],
  environment: ["ENVIRONMENT", "环境与连接"],
  changes: ["CHANGE LOG", "更新记录"],
};

const workflowLabels = {
  source: ["数据集长对话", "原始会话与消息"],
  normalize: ["消息规范化", "角色、内容、时间与元数据"],
  buffer: ["消息缓冲", "PostgreSQL message buffer"],
  segment: ["主题边界分段", "LLM 自适应分段"],
  narrative: ["生成情景叙事", "对话转为可检索故事"],
  episode_embed: ["情景向量化", "标题与叙事 Embedding"],
  episode_store: ["保存情景记忆", "PostgreSQL + Qdrant"],
  merge: ["相似情景合并", "向量候选与 LLM 决策"],
  semantic_context: ["召回已有知识", "为预测纠错提供先验"],
  semantic_extract: ["语义事实提炼", "预测—纠错抽取"],
  semantic_embed: ["语义向量化", "逐条事实 Embedding"],
  semantic_consolidate: ["语义知识合并", "新增、合并或冲突替换"],
  semantic_store: ["保存语义记忆", "最终持久化输出"],
  question: ["评测问题", "Query 与问题日期"],
  query_embed: ["问题向量化", "使用相同 Embedding 模型"],
  episode_recall: ["情景记忆召回", "向量或全文检索"],
  semantic_recall: ["语义记忆召回", "细粒度事实检索"],
  rrf: ["Hybrid RRF 融合", "合并向量与文本排序"],
  topk: ["Top-K 上下文", "情景与语义双通道"],
  answer_prompt: ["答案提示词组装", "时间推理与证据约束"],
  answer: ["生成回答", "低温度确定性输出"],
  checkpoint: ["结果检查点", "增量写入与断点恢复"],
  image: ["实例镜像", "官方 SWE-bench 环境"],
  checkout: ["检出精确提交", "定位 base_commit"],
  dependencies: ["准备依赖", "构建可复现测试环境"],
  prediction: ["候选补丁", "Agent 生成的 model_patch"],
  apply: ["应用补丁", "容器内 git apply"],
  tests: ["运行回归测试", "FAIL_TO_PASS + PASS_TO_PASS"],
  grade: ["判定是否解决", "确定性测试评分"],
  report: ["生成评测报告", "解决率与逐实例详情"],
};

async function api(path, options = {}) {
  const init = { ...options, headers: { ...(options.headers || {}) } };
  if (options.body && !init.headers["Content-Type"]) init.headers["Content-Type"] = "application/json";
  const response = await fetch(path, init);
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("json") ? await response.json().catch(() => ({})) : await response.text();
  if (!response.ok) {
    const detail = typeof payload === "object" ? payload.detail : payload;
    throw new Error(detail || `HTTP ${response.status}`);
  }
  return payload;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[char]);
}

function fmt(value, maximumFractionDigits = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toLocaleString("zh-CN", { maximumFractionDigits }) : "--";
}

function fmtDate(value) {
  if (!value) return "--";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" });
}

function titleCase(value) {
  return String(value || "").replace(/[_-]/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function statusLabel(value) {
  return ({
    not_started: "未开始", ready: "待启动", awaiting: "等待继续", preparing: "准备中", running: "运行中",
    building_memory: "构建记忆", running_agent: "Agent 运行中", running_harness: "Harness 运行中",
    finalizing: "汇总中", partial: "阶段完成", completed: "已完成", failed: "失败", stopped: "已停止",
    interrupted: "已中断", running_all: "全流程运行", pending: "待执行", paused: "已暂停", skipped: "已跳过",
    stopping: "停止中", retrying: "重试中",
  })[value] || value || "未知";
}

function statusClass(value) {
  if (["completed", "running", "running_all", "building_memory", "running_agent", "running_harness", "preparing", "finalizing"].includes(value)) return value === "completed" ? "completed" : "running";
  if (["failed", "stopped", "interrupted"].includes(value)) return value;
  if (["ready", "awaiting", "partial", "paused"].includes(value)) return "ready";
  return "";
}

function toast(message, isError = false) {
  const node = $("#toast");
  node.textContent = message;
  node.className = `toast show${isError ? " error" : ""}`;
  clearTimeout(node.timer);
  node.timer = setTimeout(() => { node.className = "toast"; }, isError ? 5500 : 2800);
}

function setLoading(enabled, copy = "处理中…") {
  $("#loading-copy").textContent = copy;
  $("#loading-overlay").classList.toggle("hidden", !enabled);
}

async function withLoading(copy, task) {
  setLoading(true, copy);
  try { return await task(); }
  finally { setLoading(false); }
}

function navigate(viewName) {
  const name = pageMeta[viewName] ? viewName : "overview";
  $$(".nav-item").forEach((item) => item.classList.toggle("active", item.dataset.view === name));
  $$(".view").forEach((view) => view.classList.toggle("active", view.id === `view-${name}`));
  $("#page-eyebrow").textContent = pageMeta[name][0];
  $("#page-title").textContent = pageMeta[name][1];
  $("#sidebar").classList.remove("open");
  history.replaceState(null, "", `#${name}`);
  if (name === "runs") refreshRuns().catch(reportError);
  if (name === "environment") loadEnvironment().catch(reportError);
  if (name === "workflow") syncWorkflowSelector();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function reportError(error) {
  console.error(error);
  toast(error.message || String(error), true);
}

async function loadBootstrap({ quiet = false } = {}) {
  try {
    const data = await api("/api/bootstrap");
    appState.bootstrap = data;
    appState.system = data.system || {};
    appState.datasets = data.datasets || [];
    appState.runs = data.runs || [];
    appState.swe = data.nemori_swe || {};
    appState.sweHistory = data.nemori_swe_history || [];
    appState.environment = data.environment || {};
    renderSystem(appState.system);
    renderDatasets(appState.datasets);
    renderRuns();
    renderRecentRuns();
    renderSwe(appState.swe);
    renderSweHistory(appState.sweHistory);
    populateEnvironment(appState.environment);
    syncWorkflowSelector();
    $("#nav-run-count").textContent = appState.runs.length;
    setServiceState(true);
    return data;
  } catch (error) {
    setServiceState(false);
    if (!quiet) reportError(error);
    throw error;
  }
}

function setServiceState(ok) {
  const service = $("#service-state");
  service.textContent = ok ? "服务在线" : "连接失败";
  service.classList.toggle("ok", ok);
  const sidebar = $("#sidebar-health");
  sidebar.className = `sidebar-health ${ok ? "ok" : "error"}`;
  sidebar.lastElementChild.textContent = ok ? "控制面在线" : "控制面离线";
}

function renderSystem(system) {
  appState.system = system || {};
  const definitions = [
    ["docker", "Docker", "◫"], ["postgres", "PostgreSQL", "▤"], ["qdrant", "Qdrant", "◈"],
    ["ollama", "Ollama", "◎"], ["swebench", "SWE Runner", "⌘"],
  ];
  $("#system-cards").innerHTML = definitions.map(([key, label, icon]) => {
    const item = system?.[key] || { ok: false, error: "未检测" };
    const version = item.version || item.container || item.model || item.error || "未运行";
    const latency = Number.isFinite(Number(item.latency_ms)) ? `${item.latency_ms} ms` : (item.ok ? "可用" : "离线");
    return `<article class="system-card ${item.ok ? "" : "offline"}">
      <div class="system-card-head"><span class="system-card-icon">${icon}</span><span class="status-pill ${item.ok ? "ok" : "failed"}">${item.ok ? "在线" : "离线"}</span></div>
      <h3>${label}</h3><p title="${escapeHtml(version)}">${escapeHtml(version)}</p><small class="muted">${escapeHtml(latency)}</small>
    </article>`;
  }).join("");

  const disk = system?.disk || {};
  const percent = Number(disk.percent || 0);
  $("#disk-percent").textContent = Number.isFinite(percent) ? `${percent.toFixed(1)}%` : "--";
  $("#disk-free").textContent = disk.free_gb != null ? `${fmt(disk.free_gb, 1)} GB` : "--";
  $("#disk-copy").textContent = disk.total_gb != null
    ? `总计 ${fmt(disk.total_gb, 1)} GB · 建议为 SWE 预留至少 120 GB`
    : (disk.free_gb != null ? "建议为 SWE 预留至少 120 GB 可用空间" : "无法读取磁盘信息");
  const bar = $("#disk-bar");
  bar.style.width = `${Math.max(0, Math.min(100, percent))}%`;
  bar.className = percent >= 95 ? "danger" : (percent >= 85 ? "warning" : "");
  const warning = $("#disk-warning");
  warning.classList.toggle("hidden", percent < 90);
  warning.textContent = percent >= 90 ? `磁盘仅余 ${fmt(disk.free_gb, 1)} GB` : "";
  $("#environment-ready").textContent = system?.environment_ready ? "配置就绪" : "需要配置";
  $("#environment-ready").className = `status-pill ${system?.environment_ready ? "ok" : "ready"}`;
  $("#active-runs").textContent = appState.runs.filter((run) => activeRunStates.has(run.status)).length;
  $("#swe-overview-status").textContent = statusLabel(appState.swe?.status);
}

async function refreshSystem() {
  const system = await api("/api/system");
  renderSystem(system);
  return system;
}

async function startEnvironment() {
  const button = $("#start-environment");
  button.disabled = true;
  try {
    const result = await withLoading("正在启动 Docker、PostgreSQL、Qdrant 与 Ollama…", () => api("/api/system/start", {
      method: "POST", body: JSON.stringify({ services: ["docker", "postgres", "qdrant", "ollama"] }),
    }));
    renderSystem(result.status || {});
    const steps = $("#start-steps");
    steps.classList.remove("hidden");
    steps.innerHTML = (result.steps || []).map((step) => `<div class="step-line ${step.ok ? "ok" : "failed"}"><strong>${escapeHtml(step.service)}</strong><span>${escapeHtml(step.action || step.error || "")}${step.error ? ` · ${escapeHtml(step.error)}` : ""}</span></div>`).join("") || '<p class="muted">没有需要执行的启动步骤。</p>';
    toast("启动流程已完成，状态已重新检测");
  } catch (error) { reportError(error); }
  finally { button.disabled = false; }
}

function renderDatasets(datasets) {
  const names = { locomo: "LoCoMo", longmemeval: "LongMemEval", swebench: "SWE-bench Lite" };
  $("#dataset-cards").innerHTML = datasets.length ? datasets.map((item) => {
    const detail = item.id === "locomo" ? `${fmt(item.questions)} 个问题` : item.id === "longmemeval" ? `${fmt(item.sessions)} 个会话` : `${fmt(item.repository_count)} 个仓库`;
    return `<article class="dataset-card"><span class="status-pill ${item.available ? "ok" : "failed"}">${item.available ? "可用" : "缺失"}</span><strong>${escapeHtml(names[item.id] || item.id)}</strong><small>${fmt(item.samples)} 个样本 · ${detail}</small><div class="dataset-meta"><span>${item.bytes ? `${fmt(item.bytes / 1024 / 1024, 1)} MB` : "本地数据"}</span><span>${escapeHtml(item.split || "evaluation")}</span></div></article>`;
  }).join("") : '<p class="muted">未发现数据集。</p>';
}

function filteredRuns() {
  const filter = $("#run-filter")?.value || "all";
  if (filter === "all") return appState.runs;
  if (filter === "active") return appState.runs.filter((run) => activeRunStates.has(run.status));
  return appState.runs.filter((run) => run.benchmark === filter);
}

function runListMarkup(runs, limit = Infinity) {
  return runs.slice(0, limit).map((run) => `<button class="run-item selectable-item ${run.id === appState.selectedRunId ? "active" : ""}" data-run-id="${escapeHtml(run.id)}">
    <div><strong>${escapeHtml(run.name || run.id)}</strong><small>${escapeHtml(run.benchmark)} · ${fmt(run.selection?.sample_count)} 样本 · ${fmtDate(run.updated_at)}</small></div>
    <div><span class="status-pill ${statusClass(run.status)}">${escapeHtml(statusLabel(run.status))}</span><div class="run-progress"><i style="width:${Math.max(0, Math.min(100, Number(run.progress || 0)))}%"></i></div></div>
  </button>`).join("");
}

function renderRuns() {
  const target = $("#run-list");
  if (!target) return;
  const runs = filteredRuns();
  target.innerHTML = runs.length ? runListMarkup(runs) : '<p class="muted">当前筛选下暂无运行。</p>';
  $$('[data-run-id]', target).forEach((button) => button.addEventListener("click", () => selectRun(button.dataset.runId)));
  $("#nav-run-count").textContent = appState.runs.length;
  $("#active-runs").textContent = appState.runs.filter((run) => activeRunStates.has(run.status)).length;
}

function renderRecentRuns() {
  const target = $("#recent-runs");
  target.innerHTML = appState.runs.length ? runListMarkup(appState.runs, 5) : '<p class="muted">暂无运行。到运行中心创建第一次评测。</p>';
  $$('[data-run-id]', target).forEach((button) => button.addEventListener("click", () => { navigate("runs"); selectRun(button.dataset.runId); }));
}

async function refreshRuns() {
  appState.runs = await api("/api/runs");
  renderRuns();
  renderRecentRuns();
  syncWorkflowSelector();
  return appState.runs;
}

function formNumber(formData, key, fallback, nullable = false) {
  const raw = String(formData.get(key) || "").trim();
  if (!raw && nullable) return null;
  const value = Number(raw);
  return Number.isFinite(value) ? value : fallback;
}

async function createRun(event) {
  event.preventDefault();
  const submitter = event.submitter;
  const shouldStart = submitter?.dataset.createMode === "start";
  const form = event.currentTarget;
  const data = new FormData(form);
  const payload = {
    benchmark: data.get("benchmark"), name: String(data.get("name") || "").trim() || null, notes: String(data.get("notes") || ""),
    start_index: formNumber(data, "start_index", 0), limit: formNumber(data, "limit", 1), run_mode: data.get("run_mode"),
    llm_model: String(data.get("llm_model") || "").trim() || null, embedding_model: String(data.get("embedding_model") || "").trim() || null,
    judge_model: String(data.get("judge_model") || "").trim() || null, search_method: data.get("search_method"),
    top_k_episodes: formNumber(data, "top_k_episodes", 10), top_k_semantic: formNumber(data, "top_k_semantic", 20),
    include_original_messages_top_k: formNumber(data, "include_original_messages_top_k", 2), observation_window: formNumber(data, "observation_window", 20),
    batch_size: formNumber(data, "batch_size", 1), ingest_concurrency: formNumber(data, "ingest_concurrency", 1),
    search_concurrency: formNumber(data, "search_concurrency", 10), judge_concurrency: formNumber(data, "judge_concurrency", 10),
    checkpoint_every: formNumber(data, "checkpoint_every", 5), llm_timeout: formNumber(data, "llm_timeout", 120),
    max_process_vms_mb: formNumber(data, "max_process_vms_mb", null, true), auto_advance: data.has("auto_advance"),
    paper_profile: data.has("paper_profile"), strict_quality: data.has("strict_quality"), create_direct_pair: data.has("create_direct_pair"),
  };
  try {
    const run = await withLoading(shouldStart ? "正在创建并执行启动前检查…" : "正在创建运行…", async () => {
      const created = await api("/api/runs", { method: "POST", body: JSON.stringify(payload) });
      if (shouldStart) await api(`/api/runs/${created.id}/start`, { method: "POST" });
      return created;
    });
    await refreshRuns();
    $("#create-run-panel").classList.remove("open");
    await selectRun(run.id);
    toast(shouldStart ? "运行已创建并启动" : "运行已创建");
  } catch (error) { reportError(error); }
}

async function selectRun(runId, { quiet = false } = {}) {
  appState.selectedRunId = runId;
  try {
    const run = await api(`/api/runs/${encodeURIComponent(runId)}`);
    appState.selectedRun = run;
    renderRunDetail(run);
    renderRuns();
    if (appState.detailTab === "logs") await loadRunLogs();
    if (appState.detailTab === "results") await loadRunResults();
    return run;
  } catch (error) {
    if (!quiet) reportError(error);
    throw error;
  }
}

function renderRunDetail(run) {
  $("#run-empty").classList.add("hidden");
  $("#run-detail-content").classList.remove("hidden");
  $("#detail-benchmark").textContent = `${String(run.benchmark || "run").toUpperCase()} / ${String(run.settings?.run_mode || "nemori").toUpperCase()}`;
  $("#detail-name").textContent = run.name || run.id;
  $("#detail-id").textContent = `${run.id} · 更新于 ${fmtDate(run.updated_at)}`;
  const status = $("#detail-status");
  status.textContent = statusLabel(run.status);
  status.className = `status-pill ${statusClass(run.status)}`;
  $("#detail-progress").textContent = `${fmt(run.progress, 1)}%`;
  $("#detail-samples").textContent = `${fmt(run.selection?.sample_count)} / ${fmt(run.selection?.dataset_total)}`;
  $("#detail-cpu").textContent = run.process?.cpu != null ? `${fmt(run.process.cpu, 1)}%` : "--";
  $("#detail-memory").textContent = run.process?.memory_mb != null ? `${fmt(run.process.memory_mb, 1)} MB` : "--";
  $("#detail-tokens").textContent = fmt(run.token_usage?.total_tokens);
  $("#run-notes").value = run.notes || "";
  renderPhases(run.phases || []);
  renderRunSummary(run);
  renderArtifacts(run);
  renderCompareOptions(run);
  updateRunActionButtons(run.status);
}

function renderPhases(phases) {
  $("#phase-list").innerHTML = phases.map((phase) => `<div class="phase"><div class="phase-name"><strong>${escapeHtml(phase.label || titleCase(phase.id))}</strong><small>${escapeHtml(statusLabel(phase.status))}</small></div><div class="phase-meter"><i style="width:${Math.max(0, Math.min(100, Number(phase.progress || 0)))}%"></i></div><span>${fmt(phase.progress)}%</span></div>`).join("") || '<p class="muted">没有阶段信息。</p>';
}

function flattenSummary(object, prefix = "") {
  const rows = [];
  if (!object || typeof object !== "object") return rows;
  Object.entries(object).forEach(([key, value]) => {
    const name = prefix ? `${prefix}.${key}` : key;
    if (value && typeof value === "object" && !Array.isArray(value)) rows.push(...flattenSummary(value, name));
    else if (["string", "number", "boolean"].includes(typeof value)) rows.push([name, value]);
  });
  return rows;
}

function renderRunSummary(run) {
  const summaryRows = flattenSummary(run.summary).slice(0, 18);
  const baseRows = [
    ["当前活动", run.current_activity || "等待"], ["当前阶段", run.current_phase || "--"],
    ["LLM", run.settings?.llm_model || "--"], ["Embedding", run.settings?.embedding_model || "--"],
    ["Judge", run.settings?.judge_model || "--"], ["可比论文分数", run.paper_alignment?.score_comparable ? "是" : "否"],
  ];
  const rows = summaryRows.length ? summaryRows : baseRows;
  $("#run-summary").innerHTML = rows.map(([key, value]) => `<div class="summary-card"><span>${escapeHtml(key)}</span><strong>${escapeHtml(typeof value === "number" ? fmt(value, 4) : value)}</strong></div>`).join("");
}

function updateRunActionButtons(status) {
  const states = {
    start: ["ready", "awaiting"], retry: ["failed", "stopped", "interrupted"], pause: ["running"], resume: ["paused"],
    skip: ["running", "paused"], stop: ["running", "paused", "preparing", "retrying"], clone: ["ready", "awaiting", "completed", "failed", "stopped", "interrupted"],
    delete: ["ready", "awaiting", "completed", "failed", "stopped", "interrupted"],
  };
  $$('[data-run-action]').forEach((button) => { button.disabled = !(states[button.dataset.runAction] || []).includes(status); });
}

async function runAction(action) {
  const runId = appState.selectedRunId;
  if (!runId) return;
  if (action === "delete" && !window.confirm(`确认删除运行 ${runId}？运行目录和产物会一并删除。`)) return;
  const method = action === "delete" ? "DELETE" : "POST";
  const copy = { start: "正在执行启动前检查…", retry: "正在准备重试…", delete: "正在删除运行…", clone: "正在克隆运行…" }[action] || `正在${$("[data-run-action='" + action + "']")?.textContent || "处理"}…`;
  try {
    const result = await withLoading(copy, () => api(`/api/runs/${encodeURIComponent(runId)}${action === "delete" ? "" : `/${action}`}`, { method }));
    await refreshRuns();
    if (action === "delete") {
      appState.selectedRunId = null;
      appState.selectedRun = null;
      $("#run-detail-content").classList.add("hidden");
      $("#run-empty").classList.remove("hidden");
      toast("运行已删除");
    } else {
      const nextId = action === "clone" ? result.id : runId;
      await selectRun(nextId);
      toast(action === "clone" ? "已克隆为新运行" : "操作已提交");
    }
  } catch (error) { reportError(error); }
}

async function saveRunNotes() {
  if (!appState.selectedRunId) return;
  try {
    const run = await api(`/api/runs/${encodeURIComponent(appState.selectedRunId)}/notes`, { method: "PATCH", body: JSON.stringify({ notes: $("#run-notes").value }) });
    appState.selectedRun = run;
    toast("运行备注已保存");
  } catch (error) { reportError(error); }
}

async function loadRunLogs() {
  if (!appState.selectedRunId) return;
  const data = await api(`/api/runs/${encodeURIComponent(appState.selectedRunId)}/logs?tail=500`);
  const node = $("#generic-log");
  node.textContent = (data.items || []).join("\n") || "尚无日志。";
  node.scrollTop = node.scrollHeight;
}

async function loadRunResults() {
  if (!appState.selectedRunId) return;
  const data = await api(`/api/runs/${encodeURIComponent(appState.selectedRunId)}/results?limit=100&offset=0`);
  const items = data.items || [];
  const target = $("#generic-results");
  if (!items.length) {
    target.innerHTML = `<div class="empty-state small"><p>当前没有结果记录。来源：${escapeHtml(data.source || "尚未生成")}</p></div>`;
    return;
  }
  target.innerHTML = `<div class="section-head compact"><span class="muted">${fmt(data.total)} 条 · ${escapeHtml(data.source || "live")}</span></div><div class="table-wrap"><table><thead><tr><th>ID</th><th>问题</th><th>响应</th><th>类别</th><th>LLM</th><th>F1</th><th>记忆</th></tr></thead><tbody>${items.map((item) => `<tr><td>${escapeHtml(item.question_id || "--")}</td><td title="${escapeHtml(item.question)}">${escapeHtml(String(item.question || "--").slice(0,80))}</td><td title="${escapeHtml(item.response)}">${escapeHtml(String(item.response || "--").slice(0,90))}</td><td>${escapeHtml(item.category || "--")}</td><td>${item.llm_score == null ? "--" : fmt(item.llm_score,3)}</td><td>${item.f1_score == null ? "--" : fmt(item.f1_score,3)}</td><td>${fmt(item.memory_count)}</td></tr>`).join("")}</tbody></table></div>`;
}

function renderArtifacts(run) {
  const labels = { results: "原始结果", metrics: "评测指标", report: "报告", summary: "汇总", log: "完整日志", config: "运行配置", input: "输入切片", agent_input: "Agent 输入", predictions: "预测补丁" };
  const names = Object.keys(run.paths || {}).filter((name) => labels[name] && !(run.benchmark === "swebench" && name === "input"));
  $("#artifact-list").innerHTML = names.map((name) => `<div class="artifact"><div><strong>${labels[name]}</strong><small class="muted">${escapeHtml(name)}</small></div><a href="/api/runs/${encodeURIComponent(run.id)}/artifacts/${encodeURIComponent(name)}" target="_blank" rel="noopener">下载</a></div>`).join("") || '<p class="muted">暂无可列出的产物。</p>';
}

function renderCompareOptions(run) {
  const others = appState.runs.filter((item) => item.id !== run.id && item.benchmark === run.benchmark);
  $("#compare-run").innerHTML = others.length ? others.map((item) => `<option value="${escapeHtml(item.id)}">${escapeHtml(item.name)} · ${statusLabel(item.status)}</option>`).join("") : '<option value="">没有同数据集运行</option>';
}

async function compareRuns() {
  const against = $("#compare-run").value;
  if (!appState.selectedRunId || !against) return toast("请选择可对比的运行", true);
  try {
    const data = await api(`/api/runs/${encodeURIComponent(appState.selectedRunId)}/compare/${encodeURIComponent(against)}`);
    const delta = data.overall?.delta || {};
    $("#compare-result").innerHTML = `<p class="muted">${escapeHtml(data.run?.name)} 相对 ${escapeHtml(data.against?.name)}</p><div class="comparison-grid">${Object.entries(delta).map(([key, value]) => `<div class="comparison-card"><span class="muted">${escapeHtml(key)}</span><strong class="${Number(value) > 0 ? "positive" : Number(value) < 0 ? "negative" : ""}">${Number(value) > 0 ? "+" : ""}${fmt(value,4)}</strong></div>`).join("")}</div>`;
  } catch (error) { reportError(error); }
}

function switchDetailTab(name) {
  appState.detailTab = name;
  $$(".subtab[data-detail-tab]").forEach((button) => button.classList.toggle("active", button.dataset.detailTab === name));
  $$(".detail-tab").forEach((tab) => tab.classList.toggle("active", tab.id === `detail-tab-${name}`));
  if (name === "logs") loadRunLogs().catch(reportError);
  if (name === "results") loadRunResults().catch(reportError);
}

function syncWorkflowSelector() {
  const select = $("#workflow-run");
  if (!select) return;
  const current = appState.workflowRunId || select.value;
  select.innerHTML = '<option value="">选择运行</option>' + appState.runs.map((run) => `<option value="${escapeHtml(run.id)}">${escapeHtml(run.name)} · ${statusLabel(run.status)}</option>`).join("");
  if (current && appState.runs.some((run) => run.id === current)) select.value = current;
}

async function loadWorkflow(runId) {
  appState.workflowRunId = runId;
  if (!runId) {
    appState.workflow = null;
    renderWorkflow();
    return;
  }
  try {
    appState.workflow = await api(`/api/runs/${encodeURIComponent(runId)}/workflow`);
    appState.selectedNode = null;
    renderWorkflow();
  } catch (error) { reportError(error); }
}

function cleanWorkflowText(value, fallback = "") {
  const text = String(value || "");
  return /[鏁鐨鍐浣璇缁銆鈥鍚]/.test(text) ? fallback : text;
}

function renderWorkflow() {
  const target = $("#workflow-canvas");
  const inspector = $("#node-inspector");
  if (!appState.workflow) {
    $("#workflow-title").textContent = "尚未选择运行";
    $("#workflow-subtitle").textContent = "";
    $("#workflow-params").innerHTML = "";
    target.innerHTML = '<div class="empty-state"><span>◇</span><p>从上方选择一次运行。</p></div>';
    return;
  }
  const definition = appState.workflow.definitions?.[appState.workflowFlow];
  const runtime = appState.workflow.runtime || {};
  if (!definition) {
    target.innerHTML = '<div class="empty-state"><p>此运行没有该工作流定义。</p></div>';
    return;
  }
  const fallbackTitle = appState.workflowFlow === "ingest" ? "构建记忆内部工作流" : "检索与回答内部工作流";
  $("#workflow-title").textContent = cleanWorkflowText(definition.title, fallbackTitle);
  $("#workflow-subtitle").textContent = cleanWorkflowText(definition.subtitle, appState.workflowFlow === "ingest" ? "从原始长对话到情景记忆与语义记忆" : "从问题到双路召回、融合、上下文与最终答案");
  $("#workflow-params").innerHTML = Object.entries(definition.parameters || {}).filter(([, value]) => value != null).map(([key, value]) => `<span class="parameter">${escapeHtml(key)} = ${escapeHtml(value)}</span>`).join("");
  target.innerHTML = `<div class="workflow-track">${(definition.nodes || []).map((node) => {
    const label = workflowLabels[node.id] || [cleanWorkflowText(node.title, titleCase(node.id)), cleanWorkflowText(node.subtitle, "流程节点")];
    const nodeRuntime = runtime.nodes?.[node.id] || {};
    const active = runtime.active_node === node.id || nodeRuntime.status === "running";
    return `<button class="workflow-node ${active ? "runtime-active" : ""}" data-node-id="${escapeHtml(node.id)}"><small>${escapeHtml(node.kind || "node")} · ${escapeHtml(statusLabel(nodeRuntime.status || "pending"))}</small><strong>${escapeHtml(label[0])}</strong><span>${escapeHtml(label[1])}</span></button>`;
  }).join("")}</div>`;
  $$('[data-node-id]', target).forEach((button) => button.addEventListener("click", () => inspectNode(button.dataset.nodeId)));
  renderWorkflowEvents(runtime.events || []);
  if (!appState.selectedNode) inspector.innerHTML = '<div class="empty-state small"><p>点击工作流节点查看决策逻辑和运行状态。</p></div>';
}

function inspectNode(nodeId) {
  const definition = appState.workflow?.definitions?.[appState.workflowFlow];
  const node = definition?.nodes?.find((item) => item.id === nodeId);
  if (!node) return;
  appState.selectedNode = nodeId;
  $$(".workflow-node").forEach((button) => button.classList.toggle("active", button.dataset.nodeId === nodeId));
  const label = workflowLabels[node.id] || [cleanWorkflowText(node.title, titleCase(node.id)), cleanWorkflowText(node.subtitle, "流程节点")];
  const runtime = appState.workflow.runtime?.nodes?.[nodeId] || {};
  const decision = cleanWorkflowText(node.decision, `执行“${label[0]}”节点的核心转换或决策；具体参数以当前运行配置为准。`);
  const inspect = cleanWorkflowText(node.inspect, "检查输入、输出、耗时、错误和运行状态。");
  $("#node-inspector").innerHTML = `<div class="node-details"><h3>${escapeHtml(label[0])}</h3><p class="muted">${escapeHtml(label[1])}</p><dl><dt>节点 ID</dt><dd class="mono">${escapeHtml(node.id)}</dd><dt>类型</dt><dd>${escapeHtml(node.kind || "node")}</dd><dt>代码</dt><dd class="mono">${escapeHtml(node.code || "--")}</dd><dt>运行状态</dt><dd>${escapeHtml(statusLabel(runtime.status || "pending"))}</dd><dt>决策逻辑</dt><dd>${escapeHtml(decision)}</dd><dt>检查重点</dt><dd>${escapeHtml(inspect)}</dd></dl></div>`;
}

function renderWorkflowEvents(events) {
  $("#workflow-events").innerHTML = events.length ? events.slice(-30).reverse().map((event) => `<div class="event-item"><strong>${escapeHtml(event.event || event.type || "event")}</strong> · ${escapeHtml(event.node || event.question_id || "")}${event.time ? `<br>${escapeHtml(fmtDate(event.time))}` : ""}</div>`).join("") : '<p class="muted">暂无事件。</p>';
}

function selectedSweArm() {
  return appState.sweArmDraft || $("input[name='swe-arm']:checked")?.value || "nemori";
}

function renderSweMode(arm, busy = false) {
  const isApi = arm === "api";
  $("#swe-mode-title").innerHTML = isApi ? "API + SWE<br><span>纯 Coding Agent 基线。</span>" : "Nemori + SWE<br><span>原生代码记忆测评。</span>";
  $("#swe-mode-copy").innerHTML = isApi
    ? "使用同一批答案隔离的 <code>SWE-bench Lite</code> 任务和同一个 Coding Agent API，不构建、不注入任何 Nemori 记忆，再由官方 Harness 验证。"
    : "在精确的 <code>base_commit</code> 工作区构建 Nemori 记忆，检索与问题相关的代码上下文，交给同一个 Coding Agent，再由官方 Harness 验证。";
  $("#memory-stage-title").textContent = isApi ? "基线输入" : "代码记忆";
  $("#memory-stage-copy").textContent = isApi ? "跳过记忆构建，冻结无答案 Agent 输入" : "扫描代码并构建原生 Nemori 记忆";
  $("#run-all").textContent = `运行完整 ${isApi ? "API + SWE" : "Nemori + SWE"}`;
  $$("input[name='swe-arm']").forEach((radio) => {
    radio.checked = radio.value === arm;
    radio.disabled = busy;
  });
}

function renderSweHistory(items) {
  appState.sweHistory = items || [];
  const target = $("#swe-history");
  if (!target) return;
  target.innerHTML = appState.sweHistory.length ? appState.sweHistory.map((item) => {
    const result = item.result || {};
    return `<tr><td>${escapeHtml(fmtDate(item.created_at))}</td><td><span class="status-pill ${item.arm === "nemori" ? "ok" : "ready"}">${item.arm === "nemori" ? "Nemori + SWE" : "API + SWE"}</span></td><td>${escapeHtml(statusLabel(item.status))}</td><td>${fmt(item.sample_count)}</td><td>${result.resolved == null ? "--" : fmt(result.resolved)}</td><td>${result.success_rate == null ? "--" : `${(Number(result.success_rate) * 100).toFixed(1)}%`}</td><td>${result.agent_tokens == null ? "--" : fmt(result.agent_tokens)}</td><td>${result.memory_hits == null ? "--" : fmt(result.memory_hits)}</td></tr>`;
  }).join("") : '<tr><td colspan="8" class="empty">暂无 SWE 运行记录。</td></tr>';
}

async function refreshSweHistory() {
  const data = await api("/api/nemori-swe/history?limit=20");
  renderSweHistory(data.items || []);
}

function renderSwe(current) {
  appState.swe = current || {};
  const status = current?.status || "not_started";
  const badge = $("#swe-status");
  badge.textContent = statusLabel(status);
  badge.className = `status-pill ${statusClass(status)}`;
  $("#swe-run-id").textContent = current?.id || "--";
  $("#swe-activity").textContent = current?.current_activity || "可运行完整流程，也可逐阶段执行。";
  $("#swe-tasks").textContent = current?.selection?.sample_count || current?.config?.task_count || 3;
  $("#swe-episodes").textContent = current?.memory ? fmt(current.memory.episodes) : "--";
  $("#swe-semantic").textContent = current?.memory ? fmt(current.memory.semantic_memories) : "--";
  const memoryHits = current?.result?.memory_hits ?? current?.memory?.tasks?.reduce((sum, item) => sum + Number(item.memory_hits || 0), 0);
  $("#swe-hits").textContent = memoryHits == null ? "--" : fmt(memoryHits);
  $("#swe-overview-status").textContent = statusLabel(status);
  const busy = activeSweStates.has(status);
  if (appState.sweArmDraft == null) {
    appState.sweArmDraft = current?.arm || current?.config?.arm || "nemori";
  }
  renderSweMode(selectedSweArm(), busy);
  $("#run-all").disabled = busy;
  $("#stop-swe").disabled = !busy;
  $("#task-count").disabled = busy;
  const ordered = ["prepare", "memory", "agent", "harness", "result"];
  ordered.forEach((name, index) => {
    const step = current?.steps?.[name] || { status: "pending" };
    const card = $(`.stage[data-stage="${name}"]`);
    if (!card) return;
    card.classList.remove("running", "completed", "failed");
    if (["running", "completed", "failed"].includes(step.status)) card.classList.add(step.status);
    const button = card.querySelector("button");
    const previousDone = index === 0 || current?.steps?.[ordered[index - 1]]?.status === "completed";
    button.disabled = busy || !previousDone || step.status === "completed" || name === "result";
    button.textContent = step.status === "running" ? "运行中" : step.status === "completed" ? "已完成" : step.status === "failed" ? "重试" : ({ prepare: "准备", memory: selectedSweArm() === "api" ? "冻结输入" : "构建", agent: "运行", harness: "评测", result: "自动汇总" })[name];
  });
  renderSweResult(current?.result);
  renderSweLog(current?.logs || []);
}

function renderSweResult(result) {
  $("#swe-rate").textContent = result ? `${(Number(result.success_rate || 0) * 100).toFixed(1)}%` : "--";
  $("#swe-resolved").textContent = result ? fmt(result.resolved) : "--";
  $("#swe-failed").textContent = result ? fmt(result.failed) : "--";
  $("#swe-tokens").textContent = result ? fmt(result.agent_tokens) : "--";
  $("#swe-time").textContent = result ? `${fmt(result.agent_time_seconds, 1)}s` : "--";
  $("#swe-files").textContent = result ? fmt(result.memory?.indexed_files) : "--";
  const body = $("#swe-result-tasks");
  body.innerHTML = result?.instances?.length ? result.instances.map((item) => `<tr><td>${escapeHtml(item.instance_id)}</td><td>${item.resolved ? '<span class="status-pill ok">已解决</span>' : '<span class="status-pill failed">失败</span>'}</td><td>${fmt(item.tokens)}</td><td>${fmt(item.time_seconds,1)}s</td><td>${fmt(item.memory_hits)}</td></tr>`).join("") : '<tr><td colspan="5" class="empty">完成后显示逐任务结果。</td></tr>';
}

function renderSweLog(logs) {
  const terminal = $("#swe-log");
  if (!logs.length) { terminal.innerHTML = '<p class="muted">等待实验启动。</p>'; return; }
  terminal.innerHTML = logs.slice(-160).map((item) => `<p class="log-line ${escapeHtml(item.level || "")}"><span class="time">${escapeHtml(item.timestamp ? new Date(item.timestamp).toLocaleTimeString("zh-CN") : "")}</span> [${escapeHtml(item.stage)}] ${escapeHtml(item.message)}</p>`).join("");
  terminal.scrollTop = terminal.scrollHeight;
}

async function refreshSwe() {
  const state = await api("/api/nemori-swe");
  renderSwe(state);
  return state;
}

async function sweAction(action) {
  try {
    const payload = action === "prepare" ? { task_count: Number($("#task-count").value || 3), arm: selectedSweArm() } : undefined;
    const result = await api(`/api/nemori-swe/${action}`, { method: "POST", ...(payload ? { body: JSON.stringify(payload) } : {}) });
    renderSwe(result);
    toast("阶段已提交");
  } catch (error) { reportError(error); }
}

async function runAllSwe() {
  try {
    const arm = selectedSweArm();
    const result = await api("/api/nemori-swe/run-all", { method: "POST", body: JSON.stringify({ task_count: Number($("#task-count").value || 3), arm }) });
    renderSwe(result);
    await refreshSweHistory();
    toast(`完整 ${arm === "api" ? "API + SWE" : "Nemori + SWE"} 实验已启动`);
  } catch (error) { reportError(error); }
}

async function stopSwe() {
  try { renderSwe(await api("/api/nemori-swe/stop", { method: "POST" })); toast("停止请求已提交"); }
  catch (error) { reportError(error); }
}

async function loadEnvironment() {
  const data = await api("/api/environment");
  appState.environment = data;
  populateEnvironment(data);
  return data;
}

function populateEnvironment(environment) {
  const form = $("#environment-form");
  if (!form || !environment?.fields) return;
  Object.entries(environment.fields).forEach(([key, field]) => {
    const input = form.elements.namedItem(key);
    if (!input) return;
    if (input.type === "checkbox") {
      const rendered = field.value || field.default || "";
      input.checked = ["1", "true", "yes", "on"].includes(String(rendered).toLowerCase());
      return;
    }
    if (field.secret) {
      input.value = "";
      input.placeholder = field.configured ? "已配置；留空保持不变" : "尚未配置";
      input.dataset.configured = field.configured ? "true" : "false";
    } else {
      input.value = field.value || field.default || "";
      input.placeholder = field.default || "";
    }
  });
}

function collectEnvironmentValues() {
  const form = $("#environment-form");
  const values = {};
  $$('[name]', form).forEach((input) => {
    if (input.type === "checkbox") {
      values[input.name] = input.checked ? "true" : "false";
      return;
    }
    const value = input.value.trim();
    if (value) values[input.name] = value;
  });
  return values;
}

async function saveEnvironment() {
  try {
    const data = await withLoading("正在保存项目环境配置…", () => api("/api/environment", { method: "POST", body: JSON.stringify({ values: collectEnvironmentValues(), clear: [] }) }));
    appState.environment = data;
    populateEnvironment(data);
    toast("环境配置已保存");
  } catch (error) { reportError(error); }
}

function applyLocalEmbeddingPreset() {
  const form = $("#environment-form");
  form.elements.EMBEDDING_BASE_URL.value = "http://127.0.0.1:21434/v1";
  form.elements.EMBEDDING_API_KEY.value = "ollama";
  form.elements.NEMORI_EMBEDDING_MODEL.value = "qwen3-embedding:4b";
  form.elements.NO_PROXY.value = "localhost,127.0.0.1,::1";
  toast("已填入本地 Ollama Embedding，尚未保存");
}

async function testConnections(targets) {
  const buttons = $$(".test-button, #test-all-connections");
  buttons.forEach((button) => { button.disabled = true; });
  const form = $("#environment-form");
  try {
    const result = await api("/api/environment/test", {
      method: "POST",
      body: JSON.stringify({
        targets, values: collectEnvironmentValues(), clear: [],
        llm_model: form.elements.NEMORI_LLM_MODEL.value || null,
        agent_model: form.elements.AGENT_MODEL.value || null,
        judge_model: form.elements.NEMORI_JUDGE_MODEL.value || form.elements.LLM_JUDGE_MODEL.value || null,
        embedding_model: form.elements.NEMORI_EMBEDDING_MODEL.value || null,
      }),
    });
    renderConnectionResults(result);
    const ok = Object.values(result).every((item) => item.ok);
    toast(ok ? "所选连接全部通过" : "部分连接未通过，请查看结果", !ok);
  } catch (error) { reportError(error); }
  finally { buttons.forEach((button) => { button.disabled = false; }); }
}

function renderConnectionResults(results) {
  const labels = { postgres: "PostgreSQL", qdrant: "Qdrant", ollama: "Ollama", docker: "Docker", swebench: "SWE Runner", llm: "生成模型", agent: "Coding Agent", judge: "Judge", embedding: "Embedding" };
  $("#connection-results").innerHTML = Object.entries(results).map(([key, item]) => {
    const detail = item.ok ? [item.model, item.version, item.dimension ? `${item.dimension} 维` : null, item.latency_ms != null ? `${item.latency_ms} ms` : null].filter(Boolean).join(" · ") || "连接正常" : item.error || "连接失败";
    return `<article class="connection-result ${item.ok ? "ok" : "failed"}"><strong>${escapeHtml(labels[key] || key)}</strong><span class="status-pill ${item.ok ? "ok" : "failed"}">${item.ok ? "通过" : "失败"}</span><span class="${item.ok ? "" : "error"}">${escapeHtml(detail)}</span>${item.base_url ? `<span>${escapeHtml(item.base_url)}</span>` : ""}</article>`;
  }).join("") || '<p class="muted">没有测试结果。</p>';
}

function bindEvents() {
  $$(".nav-item").forEach((item) => item.addEventListener("click", () => navigate(item.dataset.view)));
  $$('[data-jump]').forEach((button) => button.addEventListener("click", () => navigate(button.dataset.jump)));
  $("#menu-toggle").addEventListener("click", () => $("#sidebar").classList.toggle("open"));
  $("#global-refresh").addEventListener("click", () => loadBootstrap().catch(() => {}));
  $("#refresh-system").addEventListener("click", () => refreshSystem().catch(reportError));
  $("#start-environment").addEventListener("click", startEnvironment);
  $("#run-all").addEventListener("click", runAllSwe);
  $("#stop-swe").addEventListener("click", stopSwe);
  $("#refresh-swe").addEventListener("click", () => refreshSwe().catch(reportError));
  $("#refresh-swe-history").addEventListener("click", () => refreshSweHistory().catch(reportError));
  $$("input[name='swe-arm']").forEach((radio) => radio.addEventListener("change", () => {
    appState.sweArmDraft = radio.value;
    renderSweMode(radio.value, activeSweStates.has(appState.swe?.status));
  }));
  $$('[data-swe-action]').forEach((button) => button.addEventListener("click", () => sweAction(button.dataset.sweAction)));
  $("#toggle-create-run").addEventListener("click", () => $("#create-run-panel").classList.toggle("open"));
  $("#close-create-run").addEventListener("click", () => $("#create-run-panel").classList.remove("open"));
  $("#run-form").addEventListener("submit", createRun);
  $("#run-filter").addEventListener("change", renderRuns);
  $("#refresh-runs").addEventListener("click", () => refreshRuns().catch(reportError));
  $$('[data-run-action]').forEach((button) => button.addEventListener("click", () => runAction(button.dataset.runAction)));
  $$(".subtab[data-detail-tab]").forEach((button) => button.addEventListener("click", () => switchDetailTab(button.dataset.detailTab)));
  $("#save-run-notes").addEventListener("click", saveRunNotes);
  $("#refresh-run-logs").addEventListener("click", () => loadRunLogs().catch(reportError));
  $("#run-compare").addEventListener("click", compareRuns);
  $("#workflow-run").addEventListener("change", (event) => loadWorkflow(event.target.value));
  $$(".subtab[data-flow]").forEach((button) => button.addEventListener("click", () => {
    appState.workflowFlow = button.dataset.flow;
    $$(".subtab[data-flow]").forEach((item) => item.classList.toggle("active", item === button));
    appState.selectedNode = null;
    renderWorkflow();
  }));
  $("#save-environment").addEventListener("click", saveEnvironment);
  $("#local-embedding-preset").addEventListener("click", applyLocalEmbeddingPreset);
  $$("[data-test-target]").forEach((button) => button.addEventListener("click", () => testConnections([button.dataset.testTarget])));
  $("#test-all-connections").addEventListener("click", () => testConnections(["infrastructure", "llm", "agent", "judge", "embedding"]));
}

async function poll() {
  if (appState.polling) return;
  appState.polling = true;
  try {
    const activeView = $(".view.active")?.id.replace("view-", "");
    if (activeSweStates.has(appState.swe?.status) || activeView === "swe") {
      await refreshSwe();
      if (activeView === "swe") await refreshSweHistory();
    }
    if (appState.selectedRunId && (activeRunStates.has(appState.selectedRun?.status) || activeView === "runs")) {
      await selectRun(appState.selectedRunId, { quiet: true });
      if (appState.detailTab === "logs") await loadRunLogs();
    }
    if (activeView === "workflow" && appState.workflowRunId) await loadWorkflow(appState.workflowRunId);
    if (["overview", "environment"].includes(activeView)) await refreshSystem();
    if (activeView === "overview" || activeRunStates.has(appState.selectedRun?.status)) await refreshRuns();
    setServiceState(true);
  } catch (error) {
    console.warn("poll failed", error);
    setServiceState(false);
  } finally {
    appState.polling = false;
  }
}

async function init() {
  bindEvents();
  const initialView = location.hash.replace("#", "");
  navigate(pageMeta[initialView] ? initialView : "overview");
  try {
    await loadBootstrap();
    if (appState.runs.length && !appState.selectedRunId) {
      appState.selectedRunId = appState.runs[0].id;
      renderRuns();
    }
  } catch { /* loadBootstrap already reports */ }
  setInterval(poll, 5000);
}

document.addEventListener("DOMContentLoaded", init);
