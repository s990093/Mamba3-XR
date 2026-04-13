/**
 * Mamba3 pre-train dashboard — ECharts + plain CSV parse.
 * Expects columns: step, loss, ce_loss, lb_contrib, z_contrib, router_temp,
 * lr, grad_norm, loss_scale, tokens_seen, elapsed_s, step_time_s
 * PPL (perplexity) = exp(ce_loss); chart uses log Y-axis.
 */
(function () {
  "use strict";

  const REQUIRED = [
    "step",
    "loss",
    "ce_loss",
    "lb_contrib",
    "z_contrib",
    "router_temp",
    "lr",
    "grad_norm",
    "loss_scale",
    "tokens_seen",
    "elapsed_s",
    "step_time_s",
  ];

  let chartInstances = [];
  let themeRegistered = false;

  function $(id) {
    return document.getElementById(id);
  }

  function registerChartTheme() {
    if (themeRegistered || typeof echarts === "undefined" || !echarts.registerTheme) return;
    themeRegistered = true;
    echarts.registerTheme("mambaDark", {
      color: ["#38bdf8", "#4ade80", "#a78bfa", "#fb923c", "#f472b6", "#94a3b8", "#22d3ee", "#c084fc"],
      backgroundColor: "transparent",
    });
  }

  function setStatus(msg, isErr) {
    const el = $("status");
    if (!el) return;
    el.textContent = msg;
    el.classList.toggle("err", !!isErr);
  }

  function parseLogCsv(text) {
    const lines = text
      .trim()
      .split(/\r?\n/)
      .map((l) => l.trim())
      .filter(Boolean);
    if (lines.length < 2) return null;
    const headers = lines[0].split(",").map((h) => h.trim());
    const idx = {};
    headers.forEach((h, i) => {
      idx[h] = i;
    });
    for (const k of REQUIRED) {
      if (idx[k] === undefined) return null;
    }
    const rows = [];
    for (let li = 1; li < lines.length; li++) {
      const parts = lines[li].split(",");
      if (parts.length < headers.length) continue;
      const nums = parts.map((p) => parseFloat(String(p).trim()));
      if (nums.some((n) => Number.isNaN(n))) continue;
      rows.push(nums);
    }
    if (!rows.length) return null;
    const col = {};
    REQUIRED.forEach((k) => {
      const j = idx[k];
      col[k] = rows.map((r) => r[j]);
    });
    return col;
  }

  function ema(arr, span) {
    if (!arr.length) return [];
    const alpha = 2 / (span + 1);
    const out = new Array(arr.length);
    let e = arr[0];
    out[0] = e;
    for (let i = 1; i < arr.length; i++) {
      e = alpha * arr[i] + (1 - alpha) * e;
      out[i] = e;
    }
    return out;
  }

  function humanInt(n) {
    const x = Number(n);
    if (x >= 1e9) return (x / 1e9).toFixed(3) + "B";
    if (x >= 1e6) return (x / 1e6).toFixed(3) + "M";
    if (x >= 1e3) return (x / 1e3).toFixed(3) + "K";
    return String(Math.round(x));
  }

  function humanDuration(seconds) {
    const s = Number(seconds);
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = s % 60;
    if (h > 0) return `${h}h ${m}m ${Math.round(sec)}s`;
    if (m > 0) return `${m}m ${sec.toFixed(1)}s`;
    return `${sec.toFixed(1)}s`;
  }

  /** Perplexity from cross-entropy loss: PPL = exp(ce_loss). */
  function pplFromCe(ce) {
    return Math.exp(ce);
  }

  function formatPpl(x) {
    const v = Number(x);
    if (!isFinite(v) || v <= 0) return "—";
    if (v >= 1e9) return (v / 1e9).toFixed(3) + "B";
    if (v >= 1e6) return (v / 1e6).toFixed(3) + "M";
    if (v >= 1e3) return (v / 1e3).toFixed(2) + "K";
    return v.toFixed(2);
  }

  function zipXY(x, y) {
    const n = Math.min(x.length, y.length);
    const out = new Array(n);
    for (let i = 0; i < n; i++) out[i] = [x[i], y[i]];
    return out;
  }

  function disposeCharts() {
    chartInstances.forEach((c) => {
      try {
        c.dispose();
      } catch (_) {}
    });
    chartInstances = [];
  }

  function baseOption() {
    const axisMuted = "#94a3b8";
    const gridLine = "rgba(148, 163, 184, 0.08)";
    const axisLine = "rgba(148, 163, 184, 0.2)";
    return {
      backgroundColor: "transparent",
      textStyle: {
        color: "#e2e8f0",
        fontFamily: '"Plus Jakarta Sans", "Noto Sans TC", system-ui, sans-serif',
        fontSize: 12,
      },
      animation: true,
      animationDuration: 480,
      animationDurationUpdate: 320,
      animationEasing: "cubicOut",
      animationEasingUpdate: "cubicOut",
      tooltip: {
        trigger: "axis",
        axisPointer: {
          type: "cross",
          crossStyle: { color: "rgba(148, 163, 184, 0.5)" },
          lineStyle: { color: "rgba(56, 189, 248, 0.35)", width: 1 },
        },
        backgroundColor: "rgba(15, 23, 42, 0.94)",
        borderColor: "rgba(56, 189, 248, 0.35)",
        borderWidth: 1,
        padding: [12, 16],
        textStyle: { color: "#f1f5f9", fontSize: 12 },
        extraCssText: "border-radius:12px;box-shadow:0 16px 48px rgba(0,0,0,0.45);backdrop-filter:blur(8px);",
      },
      grid: { left: 58, right: 44, top: 58, bottom: 72, containLabel: false },
      xAxis: {
        type: "value",
        name: "step",
        nameLocation: "middle",
        nameTextStyle: { color: axisMuted, fontSize: 11 },
        nameGap: 28,
        scale: true,
        splitNumber: 6,
        axisLine: { lineStyle: { color: axisLine, width: 1 } },
        axisTick: { show: false },
        splitLine: { lineStyle: { color: gridLine, type: "dashed" } },
        axisLabel: {
          color: axisMuted,
          fontSize: 11,
          hideOverlap: true,
          formatter: function (v) {
            const n = Number(v);
            if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(2) + "M";
            if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(1) + "k";
            return String(Math.round(n));
          },
        },
      },
      dataZoom: [
        { type: "inside", xAxisIndex: 0, filterMode: "none" },
        {
          type: "slider",
          xAxisIndex: 0,
          filterMode: "none",
          height: 22,
          bottom: 8,
          borderColor: "rgba(148, 163, 184, 0.2)",
          backgroundColor: "rgba(15, 23, 42, 0.5)",
          fillerColor: "rgba(56, 189, 248, 0.18)",
          handleStyle: { color: "#38bdf8", borderColor: "#0ea5e9" },
          handleSize: "100%",
          textStyle: { color: axisMuted, fontSize: 11 },
          dataBackground: {
            lineStyle: { color: "rgba(148, 163, 184, 0.25)" },
            areaStyle: { color: "rgba(148, 163, 184, 0.06)" },
          },
        },
      ],
    };
  }

  function renderKpis(col) {
    const step = col.step;
    const loss = col.loss;
    const tokens = col.tokens_seen;
    const elapsed = col.elapsed_s;
    const stime = col.step_time_s;
    const scale = col.loss_scale;
    const n = step.length;

    const ce = col.ce_loss;
    const finalLoss = loss[n - 1];
    let minI = 0;
    let minV = loss[0];
    for (let i = 1; i < n; i++) {
      if (loss[i] < minV) {
        minV = loss[i];
        minI = i;
      }
    }
    const minStep = step[minI];

    let minCeI = 0;
    let minCeV = ce[0];
    for (let i = 1; i < n; i++) {
      if (ce[i] < minCeV) {
        minCeV = ce[i];
        minCeI = i;
      }
    }
    const finalPpl = pplFromCe(ce[n - 1]);
    const minPpl = pplFromCe(minCeV);
    const minPplStep = step[minCeI];

    const tokPerStep = new Array(n);
    tokPerStep[0] = tokens[0];
    for (let i = 1; i < n; i++) tokPerStep[i] = tokens[i] - tokens[i - 1];
    const tps = tokPerStep.map((t, i) => t / Math.max(stime[i], 1e-9));
    const tpsMean = tps.reduce((a, b) => a + b, 0) / n;

    let avgStep = 0;
    for (let i = 0; i < n; i++) avgStep += stime[i];
    avgStep /= n;

    const sorted = stime.slice().sort((a, b) => a - b);
    const medStep = sorted[Math.floor(n / 2)];

    const tail = Math.min(1000, n);
    let rMean = 0;
    for (let i = n - tail; i < n; i++) rMean += loss[i];
    rMean /= tail;
    let rVar = 0;
    for (let i = n - tail; i < n; i++) {
      const d = loss[i] - rMean;
      rVar += d * d;
    }
    const rStd = Math.sqrt(rVar / tail);

    let ceTailMean = 0;
    for (let i = n - tail; i < n; i++) ceTailMean += ce[i];
    ceTailMean /= tail;
    const recentPpl = Math.exp(ceTailMean);

    const scaleFixed = scale.every((v) => Math.abs(v - scale[0]) < 1e-6);
    const scaleNote = scaleFixed ? String(scale[0]) : "變動";

    const items = [
      ["總步數", String(Math.round(step[n - 1])).replace(/\B(?=(\d{3})+(?!\d))/g, ",")],
      ["總 tokens（累計）", humanInt(tokens[n - 1])],
      ["牆鐘時間", humanDuration(elapsed[n - 1])],
      ["最終 loss", finalLoss.toFixed(4)],
      ["最終 PPL (CE)", formatPpl(finalPpl)],
      ["最低 loss", `${minV.toFixed(4)} <small>@ step ${Math.round(minStep).toLocaleString()}</small>`],
      ["最低 PPL (CE)", `${formatPpl(minPpl)} <small>@ step ${Math.round(minPplStep).toLocaleString()}</small>`],
      ["平均 step 時間", `${avgStep.toFixed(2)}s <small>（中位 ${medStep.toFixed(2)}s）</small>`],
      ["平均吞吐", `${Math.round(tpsMean).toLocaleString()} tok/s`],
      [`近期 loss（最後 ${tail} 步）`, `${rMean.toFixed(4)} ± ${rStd.toFixed(4)}`],
      [`近期 PPL（最後 ${tail} 步）`, `${formatPpl(recentPpl)} <small>exp(mean(ce))</small>`],
      ["loss_scale", scaleNote],
    ];

    const grid = $("kpiGrid");
    if (grid) {
      grid.innerHTML = items
        .map(
          ([label, val]) =>
            `<div class="kpi"><span class="label">${label}</span><span class="val">${val}</span></div>`
        )
        .join("");
    }
    const kpiWrap = $("kpiSection");
    if (kpiWrap) kpiWrap.classList.remove("hidden");
  }

  function bindChart(domId, option) {
    const dom = $(domId);
    if (!dom) return;
    registerChartTheme();
    const chart = echarts.init(dom, "mambaDark", { renderer: "canvas" });
    chart.setOption(option, true);
    chartInstances.push(chart);
  }

  /** ECharts measures 0×0 when the container is display:none — must run after section is visible. */
  function resizeAllCharts() {
    chartInstances.forEach((c) => {
      try {
        c.resize();
      } catch (_) {}
    });
  }

  function renderCharts(col) {
    disposeCharts();

    const chartsEl = $("charts");
    if (chartsEl) chartsEl.classList.remove("hidden");

    const step = col.step;
    const n = step.length;
    const span = Math.max(20, Math.min(200, Math.floor(n / 30)));
    const lossE = ema(col.loss, span);
    const ceE = ema(col.ce_loss, span);

    const tokPerStep = new Array(n);
    tokPerStep[0] = col.tokens_seen[0];
    for (let i = 1; i < n; i++) tokPerStep[i] = col.tokens_seen[i] - col.tokens_seen[i - 1];
    const tps = tokPerStep.map((t, i) => t / Math.max(col.step_time_s[i], 1e-9));

    const lineLarge = {
      large: true,
      largeThreshold: 1500,
      smooth: false,
      emphasis: { focus: "series", blurScope: "coordinateSystem" },
    };

    // Loss
    bindChart("cLoss", {
      ...baseOption(),
      legend: {
        type: "scroll",
        data: ["loss", "ce_loss", `loss EMA(${span})`, `ce EMA(${span})`],
        textStyle: { color: "#8b949e", fontSize: 11 },
        top: 4,
        left: "center",
        width: "92%",
        itemGap: 14,
      },
      yAxis: {
        type: "value",
        name: "loss",
        nameTextStyle: { color: "#8b949e" },
        splitLine: { lineStyle: { color: "#21262d" } },
        axisLabel: { color: "#8b949e" },
      },
      series: [
        {
          name: "loss",
          type: "line",
          ...lineLarge,
          showSymbol: false,
          lineStyle: { width: 1, color: "#58a6ff", opacity: 0.35 },
          data: zipXY(step, col.loss),
        },
        {
          name: "ce_loss",
          type: "line",
          ...lineLarge,
          showSymbol: false,
          lineStyle: { width: 1, color: "#3fb950", opacity: 0.35 },
          data: zipXY(step, col.ce_loss),
        },
        {
          name: `loss EMA(${span})`,
          type: "line",
          showSymbol: false,
          lineStyle: { width: 2.2, color: "#a371f7" },
          data: zipXY(step, lossE),
        },
        {
          name: `ce EMA(${span})`,
          type: "line",
          showSymbol: false,
          lineStyle: { width: 2, color: "#79c0ff" },
          data: zipXY(step, ceE),
        },
      ],
    });

    const pplRaw = new Array(n);
    const pplEma = new Array(n);
    for (let i = 0; i < n; i++) {
      pplRaw[i] = pplFromCe(col.ce_loss[i]);
      pplEma[i] = pplFromCe(ceE[i]);
    }

    bindChart("cPpl", {
      ...baseOption(),
      legend: {
        type: "scroll",
        data: ["PPL (每步)", `PPL (CE EMA ${span})`],
        textStyle: { color: "#8b949e", fontSize: 11 },
        top: 4,
        left: "center",
        width: "92%",
        itemGap: 14,
      },
      yAxis: {
        type: "log",
        name: "PPL",
        scale: true,
        nameTextStyle: { color: "#8b949e" },
        splitLine: { lineStyle: { color: "#21262d" } },
        axisLabel: { color: "#8b949e" },
      },
      series: [
        {
          name: "PPL (每步)",
          type: "line",
          ...lineLarge,
          showSymbol: false,
          lineStyle: { width: 1, color: "#58a6ff", opacity: 0.35 },
          data: zipXY(step, pplRaw),
        },
        {
          name: `PPL (CE EMA ${span})`,
          type: "line",
          showSymbol: false,
          lineStyle: { width: 2.2, color: "#f0883e" },
          data: zipXY(step, pplEma),
        },
      ],
    });

    // Token / PPL（依你指定的估算：token ~= step * 512(seq_len) * 32(batch)）
    const SEQ_LEN = 512;
    const BATCH_SIZE = 32;
    const tokPerPplRaw = new Array(n);
    const tokPerPplEma = new Array(n);
    for (let i = 0; i < n; i++) {
      const tokensEst = Math.max(step[i] * SEQ_LEN * BATCH_SIZE, 1e-12);
      tokPerPplRaw[i] = tokensEst / Math.max(pplRaw[i], 1e-12);
      tokPerPplEma[i] = tokensEst / Math.max(pplEma[i], 1e-12);
    }

    bindChart("cTokPerPpl", {
      ...baseOption(),
      legend: {
        type: "scroll",
        data: ["token/PPL (每步)", `token/PPL (CE EMA ${span})`],
        textStyle: { color: "#8b949e", fontSize: 11 },
        top: 4,
        left: "center",
        width: "92%",
        itemGap: 14,
      },
      yAxis: {
        type: "log",
        name: "token / PPL",
        scale: true,
        nameTextStyle: { color: "#8b949e" },
        splitLine: { lineStyle: { color: "#21262d" } },
        axisLabel: { color: "#8b949e" },
      },
      series: [
        {
          name: "token/PPL (每步)",
          type: "line",
          ...lineLarge,
          showSymbol: false,
          lineStyle: { width: 1, color: "#22d3ee", opacity: 0.35 },
          data: zipXY(step, tokPerPplRaw),
        },
        {
          name: `token/PPL (CE EMA ${span})`,
          type: "line",
          showSymbol: false,
          lineStyle: { width: 2.2, color: "#34d399" },
          data: zipXY(step, tokPerPplEma),
        },
      ],
    });

    // LR (log)
    bindChart("cLr", {
      ...baseOption(),
      legend: { data: ["lr"], textStyle: { color: "#8b949e" }, top: 0 },
      yAxis: {
        type: "log",
        name: "lr",
        nameTextStyle: { color: "#8b949e" },
        splitLine: { lineStyle: { color: "#21262d" } },
        axisLabel: { color: "#8b949e" },
      },
      series: [
        {
          name: "lr",
          type: "line",
          ...lineLarge,
          showSymbol: false,
          lineStyle: { width: 1.5, color: "#f0883e" },
          data: zipXY(step, col.lr),
        },
      ],
    });

    bindChart("cGrad", {
      ...baseOption(),
      legend: { data: ["grad_norm"], textStyle: { color: "#8b949e" }, top: 0 },
      yAxis: {
        type: "value",
        name: "norm",
        nameTextStyle: { color: "#8b949e" },
        splitLine: { lineStyle: { color: "#21262d" } },
        axisLabel: { color: "#8b949e" },
      },
      series: [
        {
          name: "grad_norm",
          type: "line",
          ...lineLarge,
          showSymbol: false,
          lineStyle: { width: 1.2, color: "#ffa657" },
          data: zipXY(step, col.grad_norm),
        },
      ],
    });

    bindChart("cContrib", {
      ...baseOption(),
      legend: { data: ["lb_contrib", "z_contrib"], textStyle: { color: "#8b949e" }, top: 0 },
      yAxis: {
        type: "value",
        name: "contrib",
        nameTextStyle: { color: "#8b949e" },
        splitLine: { lineStyle: { color: "#21262d" } },
        axisLabel: { color: "#8b949e" },
      },
      series: [
        {
          name: "lb_contrib",
          type: "line",
          ...lineLarge,
          showSymbol: false,
          lineStyle: { width: 1.2, color: "#ff7b72" },
          data: zipXY(step, col.lb_contrib),
        },
        {
          name: "z_contrib",
          type: "line",
          ...lineLarge,
          showSymbol: false,
          lineStyle: { width: 1.2, color: "#56d364" },
          data: zipXY(step, col.z_contrib),
        },
      ],
    });

    bindChart("cRouter", {
      ...baseOption(),
      legend: { data: ["router_temp"], textStyle: { color: "#8b949e" }, top: 0 },
      yAxis: {
        type: "value",
        name: "temp",
        nameTextStyle: { color: "#8b949e" },
        splitLine: { lineStyle: { color: "#21262d" } },
        axisLabel: { color: "#8b949e" },
      },
      series: [
        {
          name: "router_temp",
          type: "line",
          ...lineLarge,
          showSymbol: false,
          lineStyle: { width: 1.2, color: "#d2a8ff" },
          data: zipXY(step, col.router_temp),
        },
      ],
    });

    const boPerf = baseOption();
    bindChart("cPerf", {
      ...boPerf,
      grid: { ...boPerf.grid, right: 58 },
      legend: {
        data: ["step_time_s", "tokens/s"],
        textStyle: { color: "#8b949e" },
        top: 0,
      },
      yAxis: [
        {
          type: "value",
          name: "秒",
          nameTextStyle: { color: "#8b949e" },
          position: "left",
          splitLine: { lineStyle: { color: "#21262d" } },
          axisLabel: { color: "#8b949e" },
        },
        {
          type: "value",
          name: "tokens/s",
          nameTextStyle: { color: "#8b949e" },
          position: "right",
          splitLine: { show: false },
          axisLabel: { color: "#8b949e" },
        },
      ],
      series: [
        {
          name: "step_time_s",
          type: "line",
          yAxisIndex: 0,
          ...lineLarge,
          showSymbol: false,
          lineStyle: { width: 1.1, color: "#8b949e" },
          data: zipXY(step, col.step_time_s),
        },
        {
          name: "tokens/s",
          type: "line",
          yAxisIndex: 1,
          ...lineLarge,
          showSymbol: false,
          lineStyle: { width: 1.1, color: "#39d353" },
          data: zipXY(step, tps),
        },
      ],
    });

    requestAnimationFrame(function () {
      requestAnimationFrame(function () {
        resizeAllCharts();
      });
    });
  }

  function parseTargets() {
    const qs = new URLSearchParams(typeof location !== "undefined" ? location.search : "");
    let ts = Number(qs.get("target_steps"));
    if (!Number.isFinite(ts) || ts <= 0) ts = 60000;
    let tb = Number(qs.get("target_tokens_b"));
    if (!Number.isFinite(tb) || tb <= 0) tb = null;
    let idealB = Number(qs.get("ideal_tokens_b"));
    if (!Number.isFinite(idealB) || idealB <= 0) idealB = 4.6;
    return { targetSteps: ts, targetTokensB: tb, idealTokensB: idealB };
  }

  function renderProgress(col) {
    const wrap = $("progressSection");
    if (!wrap) return;
    const step = col.step;
    const tokens = col.tokens_seen;
    const elapsed = col.elapsed_s;
    const n = step.length;
    const curStep = step[n - 1];
    const curTok = tokens[n - 1];
    const wall = elapsed[n - 1];
    const { targetSteps, targetTokensB, idealTokensB } = parseTargets();

    const pct = targetSteps > 0 ? Math.min(100, (curStep / targetSteps) * 100) : 0;
    let etaH = 0;
    if (curStep > 0 && targetSteps > curStep) {
      etaH = (wall * (targetSteps - curStep)) / curStep / 3600;
    } else if (curStep >= targetSteps) {
      etaH = 0;
    }

    const statusEl = $("progressStatusLine");
    const fillEl = $("progressBarFill");
    const footEl = $("progressFoot");
    const tokEl = $("tokenCompare");
    if (statusEl) {
      if (curStep >= targetSteps) {
        statusEl.textContent = `🚀 Status: 100% | 已達或超過目標 step（${Math.round(curStep).toLocaleString()} / ${Math.round(targetSteps).toLocaleString()}）`;
      } else {
        statusEl.textContent = `🚀 Status: ${pct.toFixed(1)}% | ETA: ${etaH.toFixed(1)}h (${(etaH / 24).toFixed(1)} days)`;
      }
    }
    if (fillEl) {
      fillEl.style.width = `${pct}%`;
    }
    const barOuter = $("progressBarOuter");
    if (barOuter) barOuter.setAttribute("aria-valuenow", String(Math.round(pct)));
    if (footEl) {
      footEl.innerHTML = `Step <strong>${Math.round(curStep).toLocaleString()}</strong> / 目標 <strong>${Math.round(targetSteps).toLocaleString()}</strong> · 已跑 <strong>${(wall / 3600).toFixed(2)}h</strong>`;
    }
    if (tokEl) {
      const curB = curTok / 1e9;
      const tgtB = targetTokensB != null ? targetTokensB : null;
      const maxB = Math.max(idealTokensB, tgtB || 0, curB) * 1.08;
      const pctCur = maxB > 0 ? (curB / maxB) * 100 : 0;
      const pctTgt = tgtB != null && maxB > 0 ? (tgtB / maxB) * 100 : null;
      const pctIdeal = maxB > 0 ? (idealTokensB / maxB) * 100 : 100;
      let tokHtml =
        '<div class="tok-head">Tokens（10⁹）</div>' +
        '<div class="tok-scale">刻度上限 ≈ ' +
        maxB.toFixed(2) +
        "B（含基準與目標）</div>";
      tokHtml += '<div class="tok-row"><span class="tok-name ideal">Ideal（FineWeb-Edu）</span><div class="tok-track"><div class="tok-fill ideal" style="width:' + pctIdeal + '%"></div></div><span class="tok-num">' + idealTokensB.toFixed(3) + " B</span></div>";
      if (pctTgt != null) {
        tokHtml +=
          '<div class="tok-row"><span class="tok-name target">本 run 目標</span><div class="tok-track"><div class="tok-fill target" style="width:' +
          pctTgt +
          '%"></div></div><span class="tok-num">' +
          tgtB.toFixed(3) +
          " B</span></div>";
      }
      tokHtml +=
        '<div class="tok-row"><span class="tok-name current">目前累計</span><div class="tok-track"><div class="tok-fill current" style="width:' +
        pctCur +
        '%"></div></div><span class="tok-num">' +
        curB.toFixed(3) +
        " B</span></div>";
      if (tgtB == null) {
        tokHtml +=
          '<p class="tok-hint">在網址加上 <code>target_tokens_b=0.983</code> 可對照本 run token 目標。</p>';
      }
      tokEl.innerHTML = tokHtml;
    }
    wrap.classList.remove("hidden");
  }

  function processText(text) {
    const col = parseLogCsv(text);
    if (!col) {
      setStatus("無法解析 CSV：請確認表頭與欄位與訓練 log 一致。", true);
      $("kpiSection").classList.add("hidden");
      const ps = $("progressSection");
      if (ps) ps.classList.add("hidden");
      $("charts").classList.add("hidden");
      disposeCharts();
      return;
    }
    renderKpis(col);
    renderProgress(col);
    renderCharts(col);
    setStatus(`已載入 ${col.step.length.toLocaleString()} 筆 step。`, false);
  }

  function tryFetchDefault() {
    fetch("log.csv", { cache: "no-store" })
      .then((r) => {
        if (!r.ok) throw new Error(String(r.status));
        return r.text();
      })
      .then((text) => {
        processText(text);
        setStatus("已自動載入同目錄 log.csv。", false);
      })
      .catch(() => {
        setStatus("請點「選擇 log.csv」載入檔案（直接開檔案時無法自動讀取）。");
      });
  }

  function onResize() {
    chartInstances.forEach((c) => c.resize());
  }

  document.addEventListener("DOMContentLoaded", () => {
    const input = $("csvFile");
    if (input) {
      input.addEventListener("change", (e) => {
        const f = e.target.files && e.target.files[0];
        if (!f) return;
        const reader = new FileReader();
        reader.onload = () => processText(String(reader.result || ""));
        reader.readAsText(f);
      });
    }
    tryFetchDefault();
    window.addEventListener("resize", onResize);
  });
})();
