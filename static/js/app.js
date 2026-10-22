const input = document.getElementById("sentenceInput");
const analyzeBtn = document.getElementById("analyzeBtn");
const errorMessage = document.getElementById("errorMessage");
const animationStatus = document.getElementById("animationStatus");
const progress = document.getElementById("pipelineProgress");
const stages = ["preprocess", "tokenize", "regex", "morphology", "default", "context", "results"];
let runId = 0;

function wait(ms) { return new Promise(resolve => setTimeout(resolve, ms)); }
function esc(value) {
  return String(value).replace(/[&<>"']/g, ch => ({ "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#039;" }[ch]));
}
function resetStages() {
  document.querySelectorAll(".stage-card").forEach(card => {
    card.classList.remove("active", "done");
    card.querySelector(".stage-state").textContent = "WAITING";
  });
  progress.style.width = "0%";
}
function setStage(id, state) {
  const card = document.querySelector(`[data-stage="${id}"]`);
  if (!card) return;
  card.classList.remove("active", "done");
  if (state === "active") {
    card.classList.add("active");
    card.querySelector(".stage-state").textContent = "PROCESSING";
  } else if (state === "done") {
    card.classList.add("done");
    card.querySelector(".stage-state").textContent = "COMPLETE";
  }
  const index = stages.indexOf(id);
  progress.style.width = `${Math.max(0, (index / (stages.length - 1)) * 100)}%`;
}
function setOutput(id, html) {
  const el = document.getElementById(`${id}Output`);
  if (el) el.innerHTML = html;
}
function pill(tag) { return `<span class="tag-pill ${esc(tag)}">${esc(tag)}</span>`; }

async function animateResult(data, thisRun) {
  resetStages();
  const stageText = {
    preprocess: "Cleaning whitespace and preparing the sentence…",
    tokenize: "Splitting the sentence into tokens…",
    regex: "Matching tokens against ordered regex rules…",
    morphology: "Checking word-form suffix clues…",
    default: "Checking whether any token needed a fallback…",
    context: "Applying nearby-word context corrections…",
    results: "Building the final token-by-token table…"
  };
  for (let i = 0; i < stages.length; i++) {
    if (thisRun !== runId) return;
    const id = stages[i];
    setStage(id, "active");
    animationStatus.textContent = stageText[id];
    if (id === "preprocess") {
      setOutput(id, `<span>${esc(data.preprocessed)}</span>`);
    } else if (id === "tokenize") {
      setOutput(id, data.tokens.map(t => `<span class="mini-token">${esc(t)}</span>`).join(""));
    } else if (id === "regex") {
      const count = data.trace.filter(t => t.source === "Regex: lexical rule").length;
      setOutput(id, `${count} token(s) matched explicit lexical regex rules.`);
    } else if (id === "morphology") {
      const morphs = data.trace.filter(t => t.source === "Regex: morphological rule");
      setOutput(id, morphs.length ? `${morphs.length} morphology match(es): ${morphs.map(t => esc(t.token)).join(", ")}` : "No morphological regex matched; lexical rules handled the tokens.");
    } else if (id === "default") {
      const defaults = data.trace.filter(t => t.source === "DefaultTagger fallback");
      setOutput(id, defaults.length ? `${defaults.length} fallback token(s): ${defaults.map(t => esc(t.token)).join(", ")}` : "Every token matched a regex rule; fallback was not needed.");
    } else if (id === "context") {
      const changes = data.trace.filter(t => t.context_applied);
      const decisions = data.trace.filter(t => t.context_detail !== "No contextual correction applied.");
      setOutput(id, changes.length ? `${changes.length} tag correction(s): ${changes.map(t => `${esc(t.token)} ${esc(t.initial_tag)}→${esc(t.final_tag)}`).join(", ")}` : (decisions.length ? `Context checked ${decisions.length} token(s); no tag changed.` : "No ambiguous-word context rule triggered."));
    } else if (id === "results") {
      setOutput(id, `${data.results.length} token(s) tagged.`);
    }
    await wait(520);
    if (thisRun !== runId) return;
    setStage(id, "done");
  }
  renderTable(data);
  animationStatus.textContent = "Pipeline complete. Review the final tags and rule explanations below.";
}

function renderTable(data) {
  const tbody = document.getElementById("resultsBody");
  tbody.innerHTML = data.trace.map((row, i) => `
    <tr class="row-enter" style="animation-delay:${Math.min(i * 35, 350)}ms">
      <td>${i + 1}</td>
      <td><span class="token-cell">${esc(row.token)}</span></td>
      <td>${pill(row.initial_tag)}</td>
      <td><span class="rule-label">${esc(row.source)}</span><br><span class="explanation">${esc(row.matched_pattern ? `Pattern: ${row.matched_pattern}` : "No matching regex")}</span></td>
      <td>${pill(row.final_tag)}</td>
      <td><span class="explanation">${esc(row.context_detail === "No contextual correction applied." ? row.rule_detail : row.context_detail)}</span></td>
    </tr>`).join("");
  const badge = document.getElementById("accuracyBadge");
  badge.textContent = `${data.results.length} TOKENS TAGGED`;
  badge.classList.add("ready");
  if (window.lucide) lucide.createIcons();
}

async function analyze() {
  const sentence = input.value.trim();
  errorMessage.style.display = "none";
  if (!sentence) {
    errorMessage.textContent = "Please enter a sentence before analyzing.";
    errorMessage.style.display = "block";
    input.focus();
    return;
  }
  runId++;
  const thisRun = runId;
  analyzeBtn.disabled = true;
  analyzeBtn.querySelector("span").textContent = "Analyzing…";
  document.getElementById("accuracyBadge").textContent = "PROCESSING";
  document.getElementById("accuracyBadge").classList.remove("ready");
  try {
    const response = await fetch("/api/tag", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({sentence})
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Unable to analyze this sentence.");
    await animateResult(data, thisRun);
  } catch (err) {
    errorMessage.textContent = err.message || "Something went wrong. Please try again.";
    errorMessage.style.display = "block";
    animationStatus.textContent = "Analysis could not be completed.";
  } finally {
    if (thisRun === runId) {
      analyzeBtn.disabled = false;
      analyzeBtn.querySelector("span").textContent = "Analyze sentence";
      if (window.lucide) lucide.createIcons();
    }
  }
}
analyzeBtn.addEventListener("click", analyze);
input.addEventListener("keydown", e => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") analyze();
});
document.querySelectorAll(".example-chip").forEach(btn => {
  btn.addEventListener("click", () => { input.value = btn.dataset.example; analyze(); });
});
if (window.lucide) lucide.createIcons();
