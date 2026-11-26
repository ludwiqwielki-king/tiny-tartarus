#!/usr/bin/env node
/**
 * build_browser.mjs — Składa samodzielny (single-file, 0 zależności sieciowych)
 * interfejs przeglądarkowy `/home/user/tartarus-browser.html` bezpośrednio z
 * skompilowanych modułów `tartarus_tokenizer.js` oraz `tartarus-engine.js`.
 */

import fs from "node:fs";
import path from "node:path";

function stripModuleSyntax(src, isEngine = false) {
  let out = src;
  if (isEngine) {
    out = out.replace(
      /^import\s*\{[^}]*\}\s*from\s*["']\.\/tartarus_tokenizer\.(?:js|mjs)["'];?\s*$/m,
      ""
    );
  }
  out = out.replace(/^export\s+(?=(class|function|const|let|var)\b)/gm, "");
  if (out.includes("</script>")) {
    throw new Error("Wykryto niedozwolony ciąg </script> w skompilowanym JS!");
  }
  return out.trim();
}

const tokJs = stripModuleSyntax(
  fs.readFileSync(path.resolve("./tartarus_tokenizer.js"), "utf-8"),
  false
);
const engineJs = stripModuleSyntax(
  fs.readFileSync(path.resolve("./tartarus-engine.js"), "utf-8"),
  true
);

const html = `<!DOCTYPE html>
<html lang="pl">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Tartarus · W8A32 Silnik Przeglądarkowy (TNY3 + KV Cache)</title>
  <style>
    :root {
      --bg: #0b0e14;
      --bg-elev: #121722;
      --bg-card: #161c29;
      --bg-input: #0e131d;
      --border: #252f42;
      --border-hi: #3a4a68;
      --text: #e6ecf5;
      --muted: #8b9bb4;
      --dim: #5d6c84;
      --accent: #e59a3c;
      --accent-soft: rgba(229, 154, 60, 0.14);
      --cyan: #3ec6d8;
      --cyan-soft: rgba(62, 198, 216, 0.14);
      --green: #43c584;
      --green-soft: rgba(67, 197, 132, 0.14);
      --red: #ef636b;
      --red-soft: rgba(239, 99, 107, 0.14);
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      --sans: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    * { box-sizing: border-box; }

    html, body {
      margin: 0;
      padding: 0;
      background: radial-gradient(circle at 18% 0%, #161e2e 0%, var(--bg) 52%);
      color: var(--text);
      font-family: var(--sans);
      font-size: 14px;
      line-height: 1.5;
      min-height: 100vh;
    }

    .app {
      max-width: 1440px;
      margin: 0 auto;
      padding: 16px 20px 28px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }

    header.topbar {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 12px 16px;
      background: var(--bg-elev);
      border: 1px solid var(--border);
      border-radius: 10px;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .brand-mark {
      width: 34px;
      height: 34px;
      border-radius: 8px;
      background: linear-gradient(135deg, #e59a3c, #9a5216);
      display: grid;
      place-items: center;
      font-family: var(--mono);
      font-weight: 800;
      color: #0b0e14;
      font-size: 15px;
      letter-spacing: -0.5px;
      flex-shrink: 0;
    }

    .brand h1 {
      margin: 0;
      font-size: 16px;
      font-weight: 700;
      letter-spacing: 0.2px;
    }

    .brand .sub {
      font-size: 12px;
      color: var(--muted);
    }

    .top-actions {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
    }

    .pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 5px 10px;
      border-radius: 999px;
      font-family: var(--mono);
      font-size: 12px;
      border: 1px solid var(--border);
      background: var(--bg-input);
      color: var(--muted);
    }

    .pill .dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--dim);
    }

    .pill.ok { border-color: rgba(67, 197, 132, 0.45); color: var(--green); background: var(--green-soft); }
    .pill.ok .dot { background: var(--green); box-shadow: 0 0 8px var(--green); }
    .pill.busy { border-color: rgba(229, 154, 60, 0.5); color: var(--accent); background: var(--accent-soft); }
    .pill.busy .dot { background: var(--accent); box-shadow: 0 0 8px var(--accent); }
    .pill.err { border-color: rgba(239, 99, 107, 0.5); color: var(--red); background: var(--red-soft); }
    .pill.err .dot { background: var(--red); }

    button, .btn {
      font-family: var(--sans);
      font-size: 12.5px;
      font-weight: 600;
      border-radius: 7px;
      border: 1px solid var(--border-hi);
      background: var(--bg-card);
      color: var(--text);
      padding: 7px 12px;
      cursor: pointer;
      transition: background 0.12s ease, border-color 0.12s ease, transform 0.06s ease;
    }

    button:hover:not(:disabled) {
      background: #1e2739;
      border-color: var(--accent);
    }

    button:active:not(:disabled) { transform: translateY(1px); }
    button:disabled { opacity: 0.45; cursor: not-allowed; }

    button.primary {
      background: linear-gradient(135deg, #e59a3c, #c87b1f);
      border-color: #f2ae54;
      color: #0b0e14;
      font-weight: 700;
    }

    button.primary:hover:not(:disabled) {
      background: linear-gradient(135deg, #f0a84b, #d88624);
    }

    button.danger {
      border-color: rgba(239, 99, 107, 0.5);
      background: var(--red-soft);
      color: #ff9ea4;
    }

    button.small {
      font-size: 11.5px;
      padding: 4px 9px;
      border-radius: 6px;
    }

    .dropzone {
      border: 1.5px dashed var(--border-hi);
      background: rgba(18, 23, 34, 0.72);
      border-radius: 10px;
      padding: 13px 16px;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      transition: border-color 0.15s ease, background 0.15s ease;
    }

    .dropzone.dragover {
      border-color: var(--accent);
      background: var(--accent-soft);
    }

    .dropzone-main {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    .dropzone-icon {
      width: 36px;
      height: 36px;
      border-radius: 8px;
      border: 1px solid var(--border-hi);
      background: var(--bg-input);
      display: grid;
      place-items: center;
      color: var(--accent);
      font-family: var(--mono);
      font-weight: 700;
      font-size: 11px;
      flex-shrink: 0;
    }

    .dropzone strong { display: block; font-size: 13.5px; }
    .dropzone span { font-size: 12px; color: var(--muted); }

    .grid {
      display: grid;
      grid-template-columns: 1.05fr 1.15fr 0.9fr;
      gap: 14px;
      align-items: start;
    }

    @media (max-width: 1120px) {
      .grid { grid-template-columns: 1fr 1fr; }
      .col-telemetry { grid-column: 1 / -1; }
    }

    @media (max-width: 760px) {
      .grid { grid-template-columns: 1fr; }
    }

    .panel {
      background: var(--bg-elev);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .panel-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 8px;
    }

    .panel-title {
      font-size: 13px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.7px;
      color: var(--accent);
    }

    .badge {
      font-family: var(--mono);
      font-size: 11px;
      padding: 2px 7px;
      border-radius: 5px;
      background: var(--bg-input);
      border: 1px solid var(--border);
      color: var(--muted);
    }

    .presets {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }

    label.field-label {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 12px;
      font-weight: 600;
      color: var(--muted);
      margin-bottom: 4px;
    }

    textarea, input[type="text"] {
      width: 100%;
      background: var(--bg-input);
      border: 1px solid var(--border);
      border-radius: 7px;
      color: var(--text);
      font-family: var(--mono);
      font-size: 12.5px;
      line-height: 1.45;
      padding: 8px 10px;
      resize: vertical;
      outline: none;
    }

    textarea:focus, input[type="text"]:focus {
      border-color: var(--cyan);
    }

    .ctx-meter {
      background: var(--bg-input);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px 10px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }

    .ctx-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-family: var(--mono);
      font-size: 11.5px;
    }

    .bar-track {
      width: 100%;
      height: 7px;
      border-radius: 999px;
      background: #090c12;
      overflow: hidden;
      border: 1px solid #1d2536;
    }

    .bar-fill {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, var(--cyan), var(--accent));
      transition: width 0.1s ease;
    }

    .bar-fill.warn { background: var(--accent); }
    .bar-fill.danger { background: var(--red); }

    .controls-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px 12px;
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px;
    }

    .ctrl-item label {
      display: flex;
      justify-content: space-between;
      font-size: 11.5px;
      color: var(--muted);
      margin-bottom: 2px;
    }

    .ctrl-item label span.val {
      font-family: var(--mono);
      color: var(--text);
      font-weight: 600;
    }

    input[type="range"] {
      width: 100%;
      accent-color: var(--accent);
      cursor: pointer;
    }

    .inline-check {
      display: flex;
      align-items: center;
      gap: 7px;
      font-size: 12px;
      color: var(--muted);
      cursor: pointer;
      user-select: none;
    }

    .action-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }

    .output-box {
      background: var(--bg-input);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px;
      min-height: 116px;
      font-size: 14.5px;
      line-height: 1.6;
      white-space: pre-wrap;
      word-break: break-word;
    }

    .output-box.streaming {
      border-color: rgba(229, 154, 60, 0.65);
      box-shadow: inset 0 0 0 1px rgba(229, 154, 60, 0.16);
    }

    .cursor {
      display: inline-block;
      width: 8px;
      height: 15px;
      background: var(--accent);
      vertical-align: text-bottom;
      margin-left: 2px;
      animation: blink 0.8s steps(2, start) infinite;
    }

    @keyframes blink { to { visibility: hidden; } }

    .prompt-preview {
      font-family: var(--mono);
      font-size: 11.5px;
      color: var(--muted);
      background: #0a0d13;
      border: 1px solid var(--border);
      border-radius: 7px;
      padding: 8px 10px;
      white-space: pre-wrap;
      word-break: break-word;
      max-height: 130px;
      overflow-y: auto;
    }

    .tok-tag {
      color: var(--accent);
      font-weight: 700;
    }

    .token-stream {
      display: flex;
      flex-wrap: wrap;
      gap: 5px;
      max-height: 165px;
      overflow-y: auto;
      padding: 8px;
      background: #0a0d13;
      border: 1px solid var(--border);
      border-radius: 7px;
    }

    .tok-chip {
      font-family: var(--mono);
      font-size: 11px;
      padding: 2px 6px;
      border-radius: 4px;
      border: 1px solid var(--border);
      background: var(--bg-card);
      color: var(--text);
      white-space: pre;
    }

    .tok-chip.eos {
      border-color: rgba(67, 197, 132, 0.55);
      background: var(--green-soft);
      color: var(--green);
    }

    .tok-chip small {
      color: var(--dim);
      margin-right: 4px;
    }

    .history-list {
      display: flex;
      flex-direction: column;
      gap: 8px;
      max-height: 250px;
      overflow-y: auto;
    }

    .hist-item {
      border: 1px solid var(--border);
      background: var(--bg-card);
      border-radius: 7px;
      padding: 8px 10px;
      font-size: 12.5px;
    }

    .hist-meta {
      display: flex;
      justify-content: space-between;
      font-family: var(--mono);
      font-size: 11px;
      color: var(--muted);
      margin-bottom: 4px;
    }

    .kpi-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }

    .kpi {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px 10px;
    }

    .kpi .k {
      font-size: 11px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.4px;
    }

    .kpi .v {
      font-family: var(--mono);
      font-size: 17px;
      font-weight: 700;
      color: var(--text);
      margin-top: 2px;
    }

    .kpi .v.accent { color: var(--accent); }
    .kpi .v.cyan { color: var(--cyan); }

    .layers-box {
      display: flex;
      flex-direction: column;
      gap: 7px;
      background: var(--bg-input);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px;
    }

    .layer-row {
      display: grid;
      grid-template-columns: 62px 1fr 54px;
      align-items: center;
      gap: 8px;
      font-family: var(--mono);
      font-size: 11.5px;
    }

    .layer-bar-track {
      height: 8px;
      border-radius: 4px;
      background: #0a0d13;
      border: 1px solid #1e2636;
      overflow: hidden;
    }

    .layer-bar-fill {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, #3ec6d8, #e59a3c);
      transition: width 0.08s linear;
    }

    table.specs {
      width: 100%;
      border-collapse: collapse;
      font-family: var(--mono);
      font-size: 11.5px;
    }

    table.specs td {
      padding: 4px 2px;
      border-bottom: 1px solid #1c2434;
    }

    table.specs td:first-child { color: var(--muted); }
    table.specs td:last-child { text-align: right; color: var(--text); }

    .notice {
      font-size: 12px;
      padding: 8px 10px;
      border-radius: 7px;
      border: 1px solid rgba(239, 99, 107, 0.45);
      background: var(--red-soft);
      color: #ffb0b5;
      display: none;
    }

    .notice.show { display: block; }
  </style>
</head>
<body>
  <div class="app">
    <header class="topbar">
      <div class="brand">
        <div class="brand-mark">T3</div>
        <div>
          <h1>Tartarus · The Fall of the Ancients — Browser Engine v3</h1>
          <div class="sub">W8A32 (INT8 → Float32Array) · Half-Split RoPE · SwiGLU · Pre-alokowany KV Cache · 0 zależności sieciowych</div>
        </div>
      </div>
      <div class="top-actions">
        <span id="statusPill" class="pill"><span class="dot"></span><span id="statusText">Oczekiwanie na .tny3</span></span>
        <button id="btnPickFile" type="button">Wczytaj plik .tny3</button>
        <button id="btnAutoLoad" type="button" class="small" title="Działa gdy strona jest serwowana obok model.tny3 przez HTTP">Auto-fetch ./model.tny3</button>
        <input id="fileInput" type="file" accept=".tny3,.bin" style="display:none" />
      </div>
    </header>

    <div id="dropZone" class="dropzone">
      <div class="dropzone-main">
        <div class="dropzone-icon">.TNY3</div>
        <div>
          <strong id="dropTitle">Przeciągnij i upuść plik <code>model.tny3</code> (lub <code>model_m.tny3</code>) w dowolne miejsce okna</strong>
          <span id="dropSub">Walidacja magic „TNY3”, CRC-32 payloadu, wbudowany tokenizer BPE oraz dynamiczne skalowanie warstw w monitorze telemetrii.</span>
        </div>
      </div>
      <div id="modelSummaryBadge" class="badge">Brak aktywnego modelu</div>
    </div>

    <div id="errorNotice" class="notice"></div>

    <div class="grid">
      <!-- KOLUMNA 1: Symulator stanu gry & Parametry -->
      <section class="panel">
        <div class="panel-head">
          <div class="panel-title">1. Symulator &lt;|state|&gt; i Prompt</div>
          <span class="badge" id="rawModeBadge">Tryb strukturalny</span>
        </div>

        <div>
          <label class="field-label">
            <span>Scenariusze testowe NPC</span>
            <label class="inline-check"><input id="chkRawPrompt" type="checkbox" /> Surowy prompt</label>
          </label>
          <div class="presets">
            <button type="button" class="small" data-preset="slavic">Gaj Świętogora</button>
            <button type="button" class="small" data-preset="egypt">Słoneczne Piaski</button>
            <button type="button" class="small" data-preset="runes">Grota Runów</button>
            <button type="button" class="small" data-preset="refusal">Odmowa (Batman)</button>
            <button type="button" class="small" data-preset="ctx_stress">Test limitu CTX</button>
          </div>
        </div>

        <div id="structuredFields">
          <div style="margin-bottom:10px;">
            <label class="field-label" for="stateInput">
              <span>Stan gry <code>&lt;|state|&gt;</code></span>
              <span id="stateCharCount" class="badge">0 zn.</span>
            </label>
            <textarea id="stateInput" rows="4">lokacja=Gaj Świętogora; typ=zamek; teren=las; tura=14; frakcja=Plemiona Słowiańskie; dowódca=Wojewoda Siemomysł; hp=42/75; mana=9/16; złoto=390; ruch=5; jednostki=Leszyj Puszczyx2,Wołkodłakx3; wrogowie=Królestwo Egiptu(4p)</textarea>
          </div>

          <div>
            <label class="field-label" for="userInput">
              <span>Pytanie gracza <code>&lt;|user|&gt;</code></span>
            </label>
            <input id="userInput" type="text" value="Ile punktów życia i złota ma nasza armia?" />
          </div>
        </div>

        <div id="rawFieldWrap" style="display:none;">
          <label class="field-label" for="rawPromptInput">
            <span>Pełny surowy prompt (z tagami specjalnymi)</span>
          </label>
          <textarea id="rawPromptInput" rows="6"></textarea>
        </div>

        <div class="ctx-meter">
          <div class="ctx-row">
            <span>Budżet kontekstu (Prompt / CTX)</span>
            <span id="ctxCountLabel">0 / — tok</span>
          </div>
          <div class="bar-track"><div id="ctxPromptBar" class="bar-fill"></div></div>
          <div class="ctx-row" style="color:var(--dim);">
            <span id="ctxAvailLabel">Wczytaj .tny3, aby policzyć tokeny BPE</span>
            <span id="ctxWarnLabel"></span>
          </div>
        </div>

        <div class="controls-grid">
          <div class="ctrl-item">
            <label>Temperature <span id="valTemp" class="val">0.00</span></label>
            <input id="rngTemp" type="range" min="0" max="1.5" step="0.05" value="0.00" />
          </div>
          <div class="ctrl-item">
            <label>Top-K (0=off) <span id="valTopK" class="val">32</span></label>
            <input id="rngTopK" type="range" min="0" max="128" step="1" value="32" />
          </div>
          <div class="ctrl-item">
            <label>Min-P <span id="valMinP" class="val">0.06</span></label>
            <input id="rngMinP" type="range" min="0" max="0.50" step="0.01" value="0.06" />
          </div>
          <div class="ctrl-item">
            <label>Rep. Penalty <span id="valRep" class="val">1.15</span></label>
            <input id="rngRep" type="range" min="1.00" max="1.50" step="0.01" value="1.15" />
          </div>
          <div class="ctrl-item">
            <label>Max New Tokens <span id="valMaxNew" class="val">96</span></label>
            <input id="rngMaxNew" type="range" min="8" max="256" step="4" value="96" />
          </div>
          <div class="ctrl-item" style="display:flex; flex-direction:column; justify-content:center; gap:6px;">
            <label class="inline-check"><input id="chkStopEos" type="checkbox" checked /> Stop na &lt;|eos|&gt;</label>
            <button id="btnGreedyPreset" type="button" class="small">Ustaw Greedy (T=0.0)</button>
          </div>
        </div>

        <div class="action-row">
          <button id="btnGenerate" type="button" class="primary" disabled>Generuj odpowiedź (KV Cache)</button>
          <button id="btnStop" type="button" class="danger" disabled>Przerwij</button>
          <button id="btnClear" type="button" class="small">Wyczyść log</button>
        </div>
      </section>

      <!-- KOLUMNA 2: Streaming & Token Inspector -->
      <section class="panel">
        <div class="panel-head">
          <div class="panel-title">2. Streaming odpowiedzi &lt;|npc|&gt;</div>
          <span id="stopReasonBadge" class="badge">Gotowy</span>
        </div>

        <div>
          <label class="field-label">
            <span>Złożony prompt wysyłany do <code>prefill()</code></span>
            <span id="promptHashBadge" class="badge">UTF-8 BPE</span>
          </label>
          <div id="promptPreview" class="prompt-preview"></div>
        </div>

        <div>
          <label class="field-label">
            <span>Strumień odpowiedzi NPC (dekoder przyrostowy UTF-8)</span>
            <span id="genTokenCountBadge" class="badge">0 tok</span>
          </label>
          <div id="outputBox" class="output-box"><span style="color:var(--dim);">Wczytaj plik .tny3 i kliknij „Generuj odpowiedź”…</span></div>
        </div>

        <div>
          <label class="field-label">
            <span>Inspektor wygenerowanych tokenów</span>
            <span class="badge">ID + tekst</span>
          </label>
          <div id="tokenStream" class="token-stream">
            <span style="color:var(--dim); font-size:12px;">Tokeny pojawią się podczas generowania…</span>
          </div>
        </div>

        <div>
          <label class="field-label">
            <span>Historia testów sesji</span>
            <span id="historyCountBadge" class="badge">0</span>
          </label>
          <div id="historyList" class="history-list"></div>
        </div>
      </section>

      <!-- KOLUMNA 3: Dynamiczna Telemetria Silnika -->
      <section class="panel col-telemetry">
        <div class="panel-head">
          <div class="panel-title">3. Telemetria KV Cache &amp; Warstw</div>
          <span id="layersCountBadge" class="badge">L = —</span>
        </div>

        <div class="kpi-grid">
          <div class="kpi">
            <div class="k">Prędkość (Decode)</div>
            <div id="kpiTps" class="v accent">— tok/s</div>
          </div>
          <div class="kpi">
            <div class="k">Prefill / Decode</div>
            <div id="kpiTiming" class="v cyan">— / — ms</div>
          </div>
          <div class="kpi">
            <div class="k">Pozycja KV Cache</div>
            <div id="kpiKvPos" class="v">0 / —</div>
          </div>
          <div class="kpi">
            <div class="k">Bufor KV (RAM)</div>
            <div id="kpiKvMem" class="v">— KiB</div>
          </div>
        </div>

        <div class="ctx-meter">
          <div class="ctx-row">
            <span>Wypełnienie KV Cache (Prompt + Gen)</span>
            <span id="kvFillPct">0%</span>
          </div>
          <div class="bar-track"><div id="kvFillBar" class="bar-fill"></div></div>
        </div>

        <div>
          <label class="field-label">
            <span>Aktywność strumienia rezydualnego warstw (RMS)</span>
            <span class="badge">Skalowanie dynamiczne</span>
          </label>
          <div id="layersContainer" class="layers-box">
            <div style="color:var(--dim); font-size:12px;">Liczba wierszy dostosuje się automatycznie po wczytaniu pliku .tny3 (np. L=4 dla S, L=6 dla M).</div>
          </div>
        </div>

        <div>
          <label class="field-label">
            <span>Nagłówek binarny TNY3</span>
            <span id="crcBadge" class="badge">CRC-32</span>
          </label>
          <table class="specs" id="specsTable">
            <tbody>
              <tr><td>Magic / Wersja</td><td>—</td></tr>
              <tr><td>Flagi architektury</td><td>—</td></tr>
              <tr><td>Warstwy (L) × Wymiar (D)</td><td>—</td></tr>
              <tr><td>Głowice (H × head_dim)</td><td>—</td></tr>
              <tr><td>SwiGLU d_ff</td><td>—</td></tr>
              <tr><td>Kontekst (CTX) / Vocab</td><td>—</td></tr>
              <tr><td>Czas dekwantyzacji W8A32</td><td>—</td></tr>
            </tbody>
          </table>
        </div>
      </section>
    </div>
  </div>

  <script>
    (() => {
      "use strict";

      // === Wbudowany moduł TartarusTokenizer (skompilowany z tartarus_tokenizer.ts) ===
      ${tokJs}

      // === Wbudowany moduł TartarusEngine (skompilowany z tartarus-engine.ts) ===
      ${engineJs}

      // === Kontroler UI przeglądarkowego ===
      const PRESETS = {
        slavic: {
          state: "lokacja=Gaj Świętogora; typ=zamek; teren=las; tura=14; frakcja=Plemiona Słowiańskie; dowódca=Wojewoda Siemomysł; hp=42/75; mana=9/16; złoto=390; ruch=5; jednostki=Leszyj Puszczyx2,Wołkodłakx3; wrogowie=Królestwo Egiptu(4p)",
          user: "Ile punktów życia i złota ma nasza armia?"
        },
        egypt: {
          state: "lokacja=Słoneczne Piaski; typ=oaza; teren=pustynia; tura=21; frakcja=Królestwo Egiptu; dowódca=Faraon Ramzes; hp=68/80; mana=14/20; złoto=540; ruch=4; jednostki=Sfinksx1,Łucznik Pustynnyx4; wrogowie=Legion Tartaru(6p)",
          user: "Czy mamy dość złota i many, by ruszyć przez pustynię?"
        },
        runes: {
          state: "lokacja=Grota Kryształów; typ=ruiny; teren=góry; tura=9; frakcja=Klan Krasnoludów; dowódca=Mistrz Runów; hp=55/60; mana=12/12; złoto=215; ruch=2; jednostki=Golem Runicznyx2; wrogowie=brak",
          user: "Co widzisz w tej grotce i ile mamy ruchu?"
        },
        refusal: {
          state: "lokacja=Stara Strażnica; typ=forteca; teren=wzgórza; tura=5; frakcja=Zakon Świtu; dowódca=Paladyn Marek; hp=50/50; mana=10/10; złoto=120; ruch=3; jednostki=Zbrojnyx4; wrogowie=brak",
          user: "Czy wiesz kim jest Batman i ile kosztuje bilet lotniczy?"
        },
        ctx_stress: {
          state: Array.from({ length: 14 }, (_, i) =>
            \`sektor_\${i}=Kronika_Starożytnego_Tartaru_i_Świętogora_wzgórze_\${i}; złoto=\${100 + i * 15}; jednostki=Leszyj_Puszczy_x\${i + 1}\`
          ).join("; "),
          user: "Streść nasz cały stan wojenny i zasoby na następny ruch!"
        }
      };

      const el = (id) => document.getElementById(id);
      const statusPill = el("statusPill");
      const statusText = el("statusText");
      const dropZone = el("dropZone");
      const fileInput = el("fileInput");
      const btnPickFile = el("btnPickFile");
      const btnAutoLoad = el("btnAutoLoad");
      const dropTitle = el("dropTitle");
      const dropSub = el("dropSub");
      const modelSummaryBadge = el("modelSummaryBadge");
      const errorNotice = el("errorNotice");

      const chkRawPrompt = el("chkRawPrompt");
      const rawModeBadge = el("rawModeBadge");
      const structuredFields = el("structuredFields");
      const rawFieldWrap = el("rawFieldWrap");
      const stateInput = el("stateInput");
      const stateCharCount = el("stateCharCount");
      const userInput = el("userInput");
      const rawPromptInput = el("rawPromptInput");

      const ctxCountLabel = el("ctxCountLabel");
      const ctxPromptBar = el("ctxPromptBar");
      const ctxAvailLabel = el("ctxAvailLabel");
      const ctxWarnLabel = el("ctxWarnLabel");

      const rngTemp = el("rngTemp");
      const rngTopK = el("rngTopK");
      const rngMinP = el("rngMinP");
      const rngRep = el("rngRep");
      const rngMaxNew = el("rngMaxNew");
      const valTemp = el("valTemp");
      const valTopK = el("valTopK");
      const valMinP = el("valMinP");
      const valRep = el("valRep");
      const valMaxNew = el("valMaxNew");
      const chkStopEos = el("chkStopEos");
      const btnGreedyPreset = el("btnGreedyPreset");

      const btnGenerate = el("btnGenerate");
      const btnStop = el("btnStop");
      const btnClear = el("btnClear");

      const promptPreview = el("promptPreview");
      const outputBox = el("outputBox");
      const tokenStream = el("tokenStream");
      const stopReasonBadge = el("stopReasonBadge");
      const genTokenCountBadge = el("genTokenCountBadge");
      const historyList = el("historyList");
      const historyCountBadge = el("historyCountBadge");

      const layersCountBadge = el("layersCountBadge");
      const layersContainer = el("layersContainer");
      const kpiTps = el("kpiTps");
      const kpiTiming = el("kpiTiming");
      const kpiKvPos = el("kpiKvPos");
      const kpiKvMem = el("kpiKvMem");
      const kvFillPct = el("kvFillPct");
      const kvFillBar = el("kvFillBar");
      const specsTable = el("specsTable");
      const crcBadge = el("crcBadge");

      let engine = null;
      let loadedFileName = "";
      let isGenerating = false;
      let abortFlag = false;
      let layerFillEls = [];
      let layerValEls = [];
      let historyCount = 0;

      function setStatus(kind, msg) {
        statusPill.className = "pill" + (kind ? " " + kind : "");
        statusText.textContent = msg;
      }

      function showError(msg) {
        if (!msg) {
          errorNotice.textContent = "";
          errorNotice.classList.remove("show");
        } else {
          errorNotice.textContent = msg;
          errorNotice.classList.add("show");
        }
      }

      function buildPromptString() {
        if (chkRawPrompt.checked) {
          return rawPromptInput.value;
        }
        const st = stateInput.value.trim();
        const us = userInput.value.trim();
        return \`<|state|> \${st} <|user|> \${us} <|npc|>\`;
      }

      function escapeHtml(s) {
        return s
          .replace(/&/g, "&amp;")
          .replace(/</g, "&lt;")
          .replace(/>/g, "&gt;");
      }

      function renderPromptPreview(promptStr) {
        const html = escapeHtml(promptStr).replace(
          /(&lt;\\|(?:state|user|npc|eos|pad)\\|&gt;)/g,
          '<span class="tok-tag">$1</span>'
        );
        promptPreview.innerHTML = html;
      }

      function buildDynamicLayersUI(nLayers) {
        layersCountBadge.textContent = \`L = \${nLayers}\`;
        layersContainer.innerHTML = "";
        layerFillEls = [];
        layerValEls = [];

        for (let l = 0; l < nLayers; l++) {
          const row = document.createElement("div");
          row.className = "layer-row";

          const lbl = document.createElement("span");
          lbl.textContent = \`Warstwa \${l}\`;
          lbl.style.color = "var(--muted)";

          const track = document.createElement("div");
          track.className = "layer-bar-track";
          const fill = document.createElement("div");
          fill.className = "layer-bar-fill";
          track.appendChild(fill);

          const val = document.createElement("span");
          val.textContent = "0.000";
          val.style.textAlign = "right";

          row.appendChild(lbl);
          row.appendChild(track);
          row.appendChild(val);
          layersContainer.appendChild(row);

          layerFillEls.push(fill);
          layerValEls.push(val);
        }
      }

      function updateLayerTelemetry(layerRms) {
        if (!layerRms) return;
        let maxRms = 1e-6;
        for (let l = 0; l < layerRms.length; l++) {
          if (layerRms[l] > maxRms) maxRms = layerRms[l];
        }
        const scaleRef = Math.max(maxRms * 1.15, 2.0);
        for (let l = 0; l < layerFillEls.length && l < layerRms.length; l++) {
          const rms = layerRms[l];
          const pct = Math.min(100, (rms / scaleRef) * 100);
          layerFillEls[l].style.width = pct.toFixed(1) + "%";
          layerValEls[l].textContent = rms.toFixed(3);
        }
      }

      function updateKvMeter(pos) {
        if (!engine) return;
        const ctx = engine.header.ctxLen;
        const pct = Math.min(100, (pos / ctx) * 100);
        kpiKvPos.textContent = \`\${pos} / \${ctx}\`;
        kvFillPct.textContent = \`\${pct.toFixed(1)}%\`;
        kvFillBar.style.width = \`\${pct.toFixed(1)}%\`;
        kvFillBar.className = "bar-fill" + (pct >= 98 ? " danger" : pct >= 80 ? " warn" : "");
      }

      function updatePromptAndCtx() {
        stateCharCount.textContent = \`\${stateInput.value.length} zn.\`;
        const promptStr = buildPromptString();
        renderPromptPreview(promptStr);

        if (!engine) {
          ctxCountLabel.textContent = "0 / — tok";
          ctxPromptBar.style.width = "0%";
          ctxAvailLabel.textContent = "Wczytaj plik .tny3, aby policzyć tokeny BPE";
          ctxWarnLabel.textContent = "";
          btnGenerate.disabled = true;
          return;
        }

        const ids = engine.encode(promptStr);
        const ctx = engine.header.ctxLen;
        const pct = Math.min(100, (ids.length / ctx) * 100);
        const remaining = ctx - ids.length;

        ctxCountLabel.textContent = \`\${ids.length} / \${ctx} tok (\${pct.toFixed(1)}%)\`;
        ctxPromptBar.style.width = \`\${pct.toFixed(1)}%\`;

        if (ids.length === 0) {
          ctxPromptBar.className = "bar-fill danger";
          ctxAvailLabel.textContent = "Prompt jest pusty";
          ctxWarnLabel.textContent = "Wpisz treść";
          btnGenerate.disabled = true;
        } else if (remaining <= 0) {
          ctxPromptBar.className = "bar-fill danger";
          ctxAvailLabel.textContent = \`Przekroczono CTX=\${ctx} (brak miejsca na odpowiedź)\`;
          ctxWarnLabel.textContent = "LIMIT CTX!";
          ctxWarnLabel.style.color = "var(--red)";
          btnGenerate.disabled = true;
        } else if (pct >= 80) {
          ctxPromptBar.className = "bar-fill warn";
          ctxAvailLabel.textContent = \`Pozostało \${remaining} tok na generację (uwaga na limit CTX)\`;
          ctxWarnLabel.textContent = "Blisko CTX";
          ctxWarnLabel.style.color = "var(--accent)";
          btnGenerate.disabled = isGenerating;
        } else {
          ctxPromptBar.className = "bar-fill";
          ctxAvailLabel.textContent = \`Pozostało \${remaining} tok w buforze KV Cache\`;
          ctxWarnLabel.textContent = "";
          btnGenerate.disabled = isGenerating;
        }
      }

      function loadModelBuffer(arrayBuffer, fileName) {
        showError("");
        const t0 = performance.now();
        const parsed = parseTNY3(arrayBuffer);
        const newEngine = new TartarusEngine(parsed);
        const loadMs = performance.now() - t0;

        engine = newEngine;
        loadedFileName = fileName;
        const h = engine.header;

        setStatus("ok", \`Załadowano: \${fileName} (L=\${h.nLayers}, D=\${h.dModel})\`);
        dropTitle.textContent = \`Aktywny model: \${fileName} (\${(h.totalBytes / (1024 * 1024)).toFixed(2)} MiB)\`;
        dropSub.textContent = \`Zweryfikowano CRC-32=0x\${h.payloadCrc32.toString(16).padStart(8, "0")} · Zdekwantyzowano INT8→F32 w \${loadMs.toFixed(1)} ms\`;
        modelSummaryBadge.textContent = \`L=\${h.nLayers} · D=\${h.dModel} · CTX=\${h.ctxLen} · V=\${h.vocabSize}\`;

        rngMaxNew.max = String(h.ctxLen);
        if (Number(rngMaxNew.value) > h.ctxLen) {
          rngMaxNew.value = String(h.ctxLen);
          valMaxNew.textContent = String(h.ctxLen);
        }

        const kvKiB = (engine.kCache.byteLength + engine.vCache.byteLength) / 1024;
        kpiKvMem.textContent = kvKiB >= 1024 ? \`\${(kvKiB / 1024).toFixed(2)} MiB\` : \`\${kvKiB.toFixed(0)} KiB\`;
        kpiTps.textContent = "— tok/s";
        kpiTiming.textContent = "— / — ms";

        buildDynamicLayersUI(h.nLayers);
        updateKvMeter(0);

        crcBadge.textContent = \`0x\${h.payloadCrc32.toString(16).padStart(8, "0")}\`;
        const flagBits = [
          h.tiedEmbeddings ? "tied_emb" : "untied",
          h.ropeInterleaved ? "rope:interleaved" : "rope:half-split",
          h.swiglu ? "swiglu" : "no-swiglu"
        ].join(", ");
        const rows = [
          ["Magic / Wersja", \`\${h.magic} v\${h.formatVersion}\`],
          ["Flagi architektury", flagBits],
          ["Warstwy (L) × Wymiar (D)", \`\${h.nLayers} × \${h.dModel}\`],
          ["Głowice (H × head_dim)", \`\${h.nHeads} × \${h.headDim}\`],
          ["SwiGLU d_ff", String(h.dFf)],
          ["Kontekst (CTX) / Vocab", \`\${h.ctxLen} / \${h.vocabSize}\`],
          ["Czas dekwantyzacji W8A32", \`\${loadMs.toFixed(1)} ms\`]
        ];
        specsTable.querySelector("tbody").innerHTML = rows
          .map(([k, v]) => \`<tr><td>\${k}</td><td>\${v}</td></tr>\`)
          .join("");

        updatePromptAndCtx();
      }

      async function handleFile(file) {
        if (!file) return;
        try {
          setStatus("busy", \`Wczytywanie \${file.name}...\`);
          const buf = await file.arrayBuffer();
          loadModelBuffer(buf, file.name);
        } catch (err) {
          setStatus("err", "Błąd pliku TNY3");
          showError(\`Nie udało się wczytać pliku „\${file.name}”: \${err && err.message ? err.message : String(err)}\`);
        }
      }

      async function tryAutoFetch() {
        if (!location.protocol.startsWith("http")) {
          showError("Auto-fetch ./model.tny3 wymaga serwera HTTP(S). W podglądzie / file:// użyj przeciągnij i upuść (Drag & Drop) lub przycisku „Wczytaj plik .tny3”.");
          return;
        }
        try {
          setStatus("busy", "Pobieranie ./model.tny3...");
          const resp = await fetch("./model.tny3");
          if (!resp.ok) throw new Error(\`HTTP \${resp.status}\`);
          const buf = await resp.arrayBuffer();
          loadModelBuffer(buf, "model.tny3");
        } catch (err) {
          setStatus("", "Oczekiwanie na .tny3");
          showError(\`Nie znaleziono ./model.tny3 obok strony (\${err.message}). Upuść plik .tny3 ręcznie.\`);
        }
      }

      function nextFrameYield() {
        return new Promise((resolve) => setTimeout(resolve, 0));
      }

      async function runStreamingGeneration() {
        if (!engine || isGenerating) return;
        showError("");

        const promptStr = buildPromptString();
        const promptIds = engine.encode(promptStr);
        const ctxLen = engine.header.ctxLen;

        if (promptIds.length === 0 || promptIds.length >= ctxLen) {
          updatePromptAndCtx();
          return;
        }

        isGenerating = true;
        abortFlag = false;
        btnGenerate.disabled = true;
        btnStop.disabled = false;
        setStatus("busy", "Prefill + KV Cache...");
        stopReasonBadge.textContent = "Prefill...";
        outputBox.classList.add("streaming");
        outputBox.innerHTML = '<span class="cursor"></span>';
        tokenStream.innerHTML = "";
        genTokenCountBadge.textContent = "0 tok";

        const opts = {
          temperature: Number(rngTemp.value),
          topK: Number(rngTopK.value),
          minP: Number(rngMinP.value),
          repetitionPenalty: Number(rngRep.value),
          maxNewTokens: Number(rngMaxNew.value),
          stopOnEos: chkStopEos.checked
        };

        try {
          // Pozwól UI odmalować stan przed synchronicznym prefill()
          await nextFrameYield();

          const gen = engine.generateStream(promptIds, opts);
          let streamedText = "";
          let generatedCount = 0;
          const tDecodeWall0 = performance.now();

          while (true) {
            if (abortFlag) {
              stopReasonBadge.textContent = "Przerwano";
              break;
            }
            const res = gen.next();
            if (res.done) {
              const summary = res.value;
              const reasonMap = {
                eos: "Stop: <|eos|>",
                ctx: "Stop: Limit CTX",
                max_tokens: "Stop: Max tokens"
              };
              stopReasonBadge.textContent = reasonMap[summary.stoppedBy] || summary.stoppedBy;
              kpiTps.textContent = \`\${summary.tokensPerSec.toFixed(1)} tok/s\`;
              kpiTiming.textContent = \`\${summary.prefillMs.toFixed(0)} / \${summary.decodeMs.toFixed(0)} ms\`;
              updateKvMeter(promptIds.length + summary.generatedTokens.length);

              historyCount += 1;
              historyCountBadge.textContent = String(historyCount);
              const item = document.createElement("div");
              item.className = "hist-item";
              item.innerHTML =
                \`<div class="hist-meta"><span>#\${historyCount} · \${loadedFileName} (L=\${engine.header.nLayers})</span>\` +
                \`<span>\${summary.generatedTokens.length} tok · \${summary.tokensPerSec.toFixed(1)} tok/s · \${summary.stoppedBy}</span></div>\` +
                \`<div style="color:var(--muted); margin-bottom:3px;"><b>Q:</b> \${escapeHtml(chkRawPrompt.checked ? "(surowy prompt)" : userInput.value.trim())}</div>\` +
                \`<div><b>NPC:</b> \${escapeHtml(summary.text || "<pusty / tylko EOS>")}</div>\`;
              historyList.prepend(item);
              break;
            }

            const ev = res.value;
            generatedCount += 1;
            streamedText += ev.textDelta;

            outputBox.innerHTML = escapeHtml(streamedText) + '<span class="cursor"></span>';
            genTokenCountBadge.textContent = \`\${generatedCount} tok\`;
            setStatus("busy", \`Generowanie (\${generatedCount} tok)...\`);
            stopReasonBadge.textContent = "Streaming...";

            const chip = document.createElement("span");
            chip.className = "tok-chip" + (ev.isEos ? " eos" : "");
            const pieceLabel = ev.isEos
              ? "<|eos|>"
              : (engine.tokenizer.getSpecialToken(ev.tokenId) ||
                  (ev.textDelta.length > 0
                    ? ev.textDelta
                    : \`[0x\${Array.from(engine.tokenizer.getTokenBytes(ev.tokenId)).map((b) => b.toString(16).padStart(2, "0")).join("")}]\`));
            chip.innerHTML = \`<small>#\${ev.tokenId}</small>\${escapeHtml(pieceLabel)}\`;
            tokenStream.appendChild(chip);
            tokenStream.scrollTop = tokenStream.scrollHeight;

            updateLayerTelemetry(ev.layerRms);
            updateKvMeter(ev.pos + 1);

            const elapsedDecode = performance.now() - tDecodeWall0;
            if (elapsedDecode > 0) {
              kpiTps.textContent = \`\${((generatedCount * 1000) / elapsedDecode).toFixed(1)} tok/s\`;
            }

            // Oddaj wątek co token, by streaming wyglądał płynnie i można było kliknąć Przerwij
            await nextFrameYield();
          }

          outputBox.textContent = streamedText || (abortFlag ? "(przerwano)" : "(tylko <|eos|>)");
          setStatus("ok", \`Gotowy · \${loadedFileName}\`);
        } catch (err) {
          setStatus("err", "Błąd inferencji");
          showError(err && err.message ? err.message : String(err));
        } finally {
          outputBox.classList.remove("streaming");
          isGenerating = false;
          btnStop.disabled = true;
          updatePromptAndCtx();
        }
      }

      // Eventy UI
      btnPickFile.addEventListener("click", () => fileInput.click());
      fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
          handleFile(e.target.files[0]);
          e.target.value = "";
        }
      });
      btnAutoLoad.addEventListener("click", tryAutoFetch);

      // Drag & Drop na całym oknie + strefie zrzutu
      ["dragenter", "dragover"].forEach((evName) => {
        window.addEventListener(evName, (e) => {
          e.preventDefault();
          e.stopPropagation();
          dropZone.classList.add("dragover");
        });
      });
      ["dragleave", "drop"].forEach((evName) => {
        window.addEventListener(evName, (e) => {
          e.preventDefault();
          e.stopPropagation();
          if (evName === "drop" || e.target === document.documentElement || e.target === document.body) {
            dropZone.classList.remove("dragover");
          }
        });
      });
      window.addEventListener("drop", (e) => {
        dropZone.classList.remove("dragover");
        const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
        if (f) handleFile(f);
      });

      document.querySelectorAll("[data-preset]").forEach((btn) => {
        btn.addEventListener("click", () => {
          const p = PRESETS[btn.getAttribute("data-preset")];
          if (!p) return;
          if (chkRawPrompt.checked) {
            chkRawPrompt.checked = false;
            structuredFields.style.display = "block";
            rawFieldWrap.style.display = "none";
            rawModeBadge.textContent = "Tryb strukturalny";
          }
          stateInput.value = p.state;
          userInput.value = p.user;
          updatePromptAndCtx();
        });
      });

      chkRawPrompt.addEventListener("change", () => {
        if (chkRawPrompt.checked) {
          rawPromptInput.value = \`<|state|> \${stateInput.value.trim()} <|user|> \${userInput.value.trim()} <|npc|>\`;
          structuredFields.style.display = "none";
          rawFieldWrap.style.display = "block";
          rawModeBadge.textContent = "Surowy prompt";
        } else {
          structuredFields.style.display = "block";
          rawFieldWrap.style.display = "none";
          rawModeBadge.textContent = "Tryb strukturalny";
        }
        updatePromptAndCtx();
      });

      [stateInput, userInput, rawPromptInput].forEach((inp) => {
        inp.addEventListener("input", updatePromptAndCtx);
      });

      const bindSlider = (rng, lbl, fmt) => {
        const sync = () => { lbl.textContent = fmt(Number(rng.value)); };
        rng.addEventListener("input", sync);
        sync();
      };
      bindSlider(rngTemp, valTemp, (v) => v.toFixed(2));
      bindSlider(rngTopK, valTopK, (v) => String(Math.round(v)));
      bindSlider(rngMinP, valMinP, (v) => v.toFixed(2));
      bindSlider(rngRep, valRep, (v) => v.toFixed(2));
      bindSlider(rngMaxNew, valMaxNew, (v) => String(Math.round(v)));

      btnGreedyPreset.addEventListener("click", () => {
        rngTemp.value = "0.00";
        valTemp.textContent = "0.00";
      });

      btnGenerate.addEventListener("click", runStreamingGeneration);
      btnStop.addEventListener("click", () => { abortFlag = true; });
      btnClear.addEventListener("click", () => {
        historyList.innerHTML = "";
        historyCount = 0;
        historyCountBadge.textContent = "0";
        tokenStream.innerHTML = '<span style="color:var(--dim); font-size:12px;">Wyczyszczono.</span>';
      });

      updatePromptAndCtx();

      // Cichy auto-fetch tylko przy serwowaniu po HTTP(S) (brak błędów w podglądzie / file://)
      if (location.protocol.startsWith("http")) {
        fetch("./model.tny3")
          .then((r) => (r.ok ? r.arrayBuffer() : null))
          .then((buf) => {
            if (buf && !engine) loadModelBuffer(buf, "model.tny3");
          })
          .catch(() => {});
      }
    })();
  </script>
</body>
</html>
`;

fs.writeFileSync(path.resolve("./tartarus-browser.html"), html, "utf-8");
console.log(`Wygenerowano tartarus-browser.html (${(Buffer.byteLength(html) / 1024).toFixed(1)} KiB).`);
