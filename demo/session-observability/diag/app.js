(function () {
  const COLORS = dash.colors || [];
  const col = (i) => COLORS[i % Math.max(1, COLORS.length)] || "currentColor";
  const CRIT = "var(--cds-chart-status-critical)";
  const $ = (id) => document.getElementById(id);

  const fmtTok = (n) => (n == null ? "-" : n >= 1e9 ? (n / 1e9).toFixed(2) + "B" : n >= 1e6 ? (n / 1e6).toFixed(n >= 1e7 ? 0 : 1) + "M" : n >= 1e3 ? Math.round(n / 1e3) + "k" : String(Math.round(n)));
  const fmtInt = d3.format(",");
  const fmtS = (v) => (v == null ? "-" : v < 1 ? Math.round(v * 1000) + " ms" : (v >= 10 ? v.toFixed(0) : v.toFixed(1)) + " s");
  const fmtMs = (v) => (v == null ? "-" : v.toFixed(v < 10 ? 1 : 0) + " ms");
  const pct = (x) => (x == null ? "-" : Math.round(x) + "%");
  const num1 = (x) => (x == null ? "-" : x.toFixed(1));
  const quantile = (xs, q) => { const a = xs.filter((v) => v != null).sort((x, y) => x - y); return a.length ? a[Math.min(a.length - 1, Math.floor(q * a.length))] : null; };
  const median = (xs) => quantile(xs, 0.5);
  const sum = (xs) => xs.reduce((a, v) => a + (v || 0), 0);
  // A nested array in a dataset row can arrive as JSON text.
  const parseList = (v) => { if (typeof v === "string") { try { return JSON.parse(v); } catch (e) { return null; } } return v; };
  const ok = (id) => { const d = dash.data(id); return d.status === "ok" ? d.data : null; };

  function legend(id, items) {
    const el = $(id);
    el.replaceChildren(...items.map((it) => {
      const s = document.createElement("span");
      const i = document.createElement("i");
      if (it.dash) i.className = "dash"; else i.style.background = it.color;
      s.append(i, it.label);
      return s;
    }));
  }

  function tile(k, v, d) {
    const t = document.createElement("div"); t.className = "tile";
    for (const [cls, text] of [["k", k], ["v", v], ["d", d || ""]]) { const e = document.createElement("div"); e.className = cls; e.textContent = text; t.append(e); }
    return t;
  }
  const tiles = (id, list) => $(id).replaceChildren(...list.filter(Boolean).map((t) => tile(...t)));
  const muted = (el, text) => { el.replaceChildren(); const p = document.createElement("p"); p.className = "note"; p.textContent = text; el.append(p); };

  // Centered moving average over n samples, skipping gaps, so sparse per-minute counts do not whip the lines.
  function smooth(rows, keys, n) {
    const h = Math.floor(n / 2);
    return rows.map((r, i) => {
      const o = { ...r };
      for (const k of keys) {
        const v = rows.slice(Math.max(0, i - h), i + h + 1).map((x) => x[k]).filter((x) => x != null);
        o[k] = r[k] == null || !v.length ? r[k] : v.reduce((p, q) => p + q, 0) / v.length;
      }
      return o;
    });
  }

  // One chart: lines, stacked areas and an optional secondary axis, drawn from timeline rows.
  function lineChart(el, spec) {
    el.replaceChildren();
    const rows = spec.rows;
    // A series with nothing in it (for example reloads when there is no host tier) draws a flat line over the others.
    spec.series = spec.series.filter((s) => spec.keepEmpty || rows.some((r) => r[s.key] != null && r[s.key] !== 0));
    if (spec.legendId) legend(spec.legendId, [...spec.series.map((s) => ({ label: s.label, color: s.color })), ...(spec.refs && spec.refs.length ? [{ label: spec.refLabel || "Capacity", dash: true }] : [])]);
    const has = spec.series.length > 0;
    if (!has) { muted(el, spec.empty || "This run did not report this series."); return; }
    const W = Math.max(280, el.clientWidth || 600), H = 220;
    const right = spec.series.some((s) => s.axis === "right");
    const m = { l: 54, r: right ? 54 : 14, t: 10, b: 24 };
    const iw = W - m.l - m.r, ih = H - m.t - m.b;
    const xs = rows.map((r) => r.t), xMax = Math.max(...xs) || 1;
    const x = d3.scaleLinear([0, xMax], [m.l, m.l + iw]);
    const stacked = spec.series.map((s) => !!s.stack);
    const base = rows.map(() => 0);
    const tops = spec.series.map((s, k) => {
      if (!stacked[k]) return rows.map((r) => r[s.key]);
      s.lower = base.slice();
      rows.forEach((r, i) => (base[i] += r[s.key] || 0));
      return base.slice();
    });
    let topL = 0, topR = 0;
    spec.series.forEach((s, k) => { const mx = d3.max(tops[k]) || 0; if (s.axis === "right") topR = Math.max(topR, mx); else topL = Math.max(topL, mx); });
    for (const r of spec.refs || []) topL = Math.max(topL, r.y);
    let lowL = 0;
    spec.series.forEach((s, k) => { if (s.axis !== "right" && !stacked[k]) lowL = Math.min(lowL, d3.min(tops[k]) || 0); });
    const yL = d3.scaleLinear([lowL, spec.max || topL * 1.05 || 1], [m.t + ih, m.t]).nice(4);
    const yR = d3.scaleLinear([0, topR * 1.05 || 1], [m.t + ih, m.t]).nice(4);
    const Y = (s) => (s.axis === "right" ? yR : yL);
    const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${W} ${H}`).attr("width", "100%").attr("role", "img").attr("aria-label", spec.label);
    svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(yL).ticks(4).tickSize(-iw).tickFormat(spec.fmt)).call((g) => { g.select(".domain").remove(); g.selectAll("line").attr("stroke", "var(--cds-chart-grid)"); g.selectAll("text").attr("font-size", null).attr("font-family", null); });
    svg.append("g").attr("transform", `translate(0,${m.t + ih})`).call(d3.axisBottom(x).ticks(Math.min(8, Math.round(iw / 80))).tickFormat((d) => d + "m")).call((g) => { g.select(".domain").attr("stroke", "var(--cds-chart-axis)"); g.selectAll("line").attr("stroke", "var(--cds-chart-axis)"); g.selectAll("text").attr("font-size", null).attr("font-family", null); });
    if (right) svg.append("g").attr("transform", `translate(${m.l + iw},0)`).call(d3.axisRight(yR).ticks(4).tickFormat(spec.fmt2 || spec.fmt)).call((g) => { g.select(".domain").remove(); g.selectAll("line").remove(); g.selectAll("text").attr("font-size", null).attr("font-family", null); });
    if (lowL < 0) svg.append("line").attr("x1", m.l).attr("x2", m.l + iw).attr("y1", yL(0)).attr("y2", yL(0)).attr("stroke", "var(--cds-chart-axis)").attr("stroke-width", 1);
    for (const r of spec.refs || []) {
      svg.append("line").attr("x1", m.l).attr("x2", m.l + iw).attr("y1", yL(r.y)).attr("y2", yL(r.y)).attr("stroke", "var(--cds-chart-reference)").attr("stroke-width", 1.5).attr("stroke-dasharray", "6 4");
      svg.append("text").attr("x", m.l + iw - 4).attr("y", yL(r.y) - 5).attr("text-anchor", "end").text(r.label);
    }
    spec.series.forEach((s, k) => {
      const y = Y(s);
      if (stacked[k]) {
        const area = d3.area().x((_, i) => x(xs[i])).y0((_, i) => y(s.lower[i])).y1((_, i) => y(tops[k][i]));
        svg.append("path").datum(rows).attr("d", area).style("fill", s.color).attr("fill-opacity", 0.5).style("stroke", s.color).attr("stroke-width", 1.5).attr("stroke-linejoin", "round");
      } else {
        const line = d3.line().defined((_, i) => tops[k][i] != null).x((_, i) => x(xs[i])).y((_, i) => y(tops[k][i]));
        const p = svg.append("path").datum(rows).attr("d", line).attr("fill", "none").style("stroke", s.color).attr("stroke-width", 2).attr("stroke-linejoin", "round").attr("stroke-linecap", "round");
        if (s.dashed) p.attr("stroke-dasharray", "5 4");
      }
    });
    const cx = svg.append("line").attr("y1", m.t).attr("y2", m.t + ih).attr("stroke", "var(--cds-chart-axis)").style("display", "none");
    const tip = document.createElement("div"); tip.className = "tip"; el.append(tip);
    const clock = (t) => { const s = Math.round(t * 60); return Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0"); };
    svg.append("rect").attr("x", m.l).attr("y", m.t).attr("width", iw).attr("height", ih).attr("fill", "transparent")
      .on("pointermove", (ev) => {
        const box = svg.node().getBoundingClientRect();
        const t = x.invert(((ev.clientX - box.left) / box.width) * W);
        const i = Math.max(0, Math.min(rows.length - 1, d3.bisector((v) => v).center(xs, t)));
        cx.attr("x1", x(xs[i])).attr("x2", x(xs[i])).style("display", null);
        tip.replaceChildren();
        const head = document.createElement("b"); head.textContent = clock(xs[i]) + " into the run"; tip.append(head);
        spec.series.forEach((s) => {
          const v = rows[i][s.key];
          const d = document.createElement("div"); const dot = document.createElement("i"); dot.style.background = s.color;
          d.append(dot, `${s.label}: ${v == null ? "-" : (s.axis === "right" && spec.fmt2 ? spec.fmt2 : spec.fmt)(v)}`); tip.append(d);
        });
        tip.style.display = "block";
        const px = (x(xs[i]) / W) * el.clientWidth;
        tip.style.left = Math.max(0, Math.min(el.clientWidth - tip.offsetWidth, px + 12)) + "px";
        tip.style.top = "6px";
      })
      .on("pointerleave", () => { cx.style("display", "none"); tip.style.display = "none"; });
  }

  function bars(el, items, edges, fmtEdge, color) {
    const counts = edges.map(() => 0).concat([0]);
    items.forEach((v) => { const k = edges.findIndex((e) => v <= e); counts[k < 0 ? edges.length : k]++; });
    const top = Math.max(...counts, 1);
    return counts.map((c, k) => {
      const row = document.createElement("div"); row.className = "row";
      const lab = document.createElement("div"); lab.className = "lab"; lab.textContent = k < edges.length ? "up to " + fmtEdge(edges[k]) : "over " + fmtEdge(edges[edges.length - 1]);
      const bar = document.createElement("div"); bar.className = "bar"; const sp = document.createElement("span"); sp.style.width = (c / top) * 100 + "%"; sp.style.background = color; bar.append(sp);
      const n = document.createElement("div"); n.className = "n"; n.textContent = c;
      row.append(lab, bar, n); return row;
    });
  }

  let SC = null, LEN = null;
  function draw() {
    const allTl = ok("timeline"), allMetas = ok("run_meta");
    if (!allTl || !allMetas || !allMetas.length) return;
    // Runs of different lengths share the page: a selector picks the group, the tabs list its scenarios.
    const groups = [...new Set(allMetas.map((m) => m.group || "30 minutes"))];
    if (!LEN || !groups.includes(LEN)) LEN = groups[0];
    const metas = allMetas.filter((m) => (m.group || "30 minutes") === LEN);
    const lenSel = $("runlen");
    lenSel.parentElement.hidden = groups.length < 2;
    lenSel.replaceChildren(...groups.map((g) => { const o = document.createElement("option"); o.value = g; o.textContent = g; o.selected = g === LEN; return o; }));
    lenSel.onchange = () => { LEN = lenSel.value; SC = null; draw(); };
    if (!SC || !metas.some((m) => m.scenario === SC)) SC = (metas.find((m) => m.scenario === "mirror") || metas[0]).scenario;
    const meta = metas.find((m) => m.scenario === SC);
    const mine = (rows) => (rows || []).filter((r) => r.scenario === SC);
    const tl = mine(allTl), sessions = mine(ok("sessions")), returns = mine(ok("returns")), routing = mine(ok("routing"));
    $("tabs").replaceChildren(...metas.map((m) => {
      const b = document.createElement("button"); b.className = "tab"; b.type = "button"; b.setAttribute("aria-pressed", String(m.scenario === SC)); b.textContent = m.label;
      b.addEventListener("click", () => { SC = m.scenario; draw(); });
      return b;
    }));
    $("setup").textContent = meta.setup || "";
    $("blurb").textContent = meta.blurb;
    const act = tl.filter((r) => (r.in_tps || 0) > 0);
    // The idle wait after the last request ends every session at once; leave that tail out of the charts it would distort.
    const busyFloor = 0.2 * (median(act.map((r) => r.in_tps)) || 0);
    const lastBusy = tl.reduce((n, r, i) => ((r.in_tps || 0) > busyFloor ? i : n), 0);
    const tlBusy = tl.slice(0, lastBusy + 1);
    const tEnd = tlBusy.length ? tlBusy[tlBusy.length - 1].t : Infinity;
    const minutes = Math.round(Math.max(...tl.map((r) => r.t)));
    const wl = parseList(meta.window_labels) || ["1m", "5m", "15m"];
    const words0 = (w) => ({ "30s": "30 seconds", "1m": "1 minute", "5m": "5 minutes", "15m": "15 minutes", "30m": "30 minutes" }[w] || w);

    // 1. workload and response
    const ttft50 = median(act.map((r) => r.ttft_p50)), ttft90 = median(act.map((r) => r.ttft_p90)), ttft90w = d3.max(act, (r) => r.ttft_p90);
    const itl50 = median(act.map((r) => r.itl_p50_ms)), itl90 = median(act.map((r) => r.itl_p90_ms));
    tiles("t-workload", [
      ["Input tokens per second", fmtTok(median(act.map((r) => r.in_tps))) + " median", "peak " + fmtTok(d3.max(act, (r) => r.in_tps))],
      ["Output tokens per second", fmtTok(median(act.map((r) => r.out_tps))) + " median", "peak " + fmtTok(d3.max(act, (r) => r.out_tps))],
      ["Time to first token", fmtS(ttft50) + " median", "p90 " + fmtS(ttft90) + " typical, " + fmtS(ttft90w) + " worst minute"],
      ["Inter-token latency", fmtMs(itl50) + " median", "p90 " + fmtMs(itl90) + " typical"],
      ["Requests", fmtInt(meta.requests + (meta.chat_requests || 0)), fmtInt(meta.failed) + " failed"],
    ]);
    legend("l-thru", [{ label: "Input tokens per second (left)", color: col(0) }, { label: "Output tokens per second (right)", color: col(2) }]);
    lineChart($("c-thru"), { rows: tl, series: [{ key: "in_tps", label: "Input tokens per second", color: col(0) }, { key: "out_tps", label: "Output tokens per second", color: col(2), axis: "right" }], fmt: fmtTok, fmt2: (v) => (v >= 1000 ? (v / 1000).toFixed(1) + "k" : String(Math.round(v))), label: "Throughput" });
    legend("l-ttft", [{ label: "p50", color: col(0) }, { label: "p90", color: col(1) }]);
    lineChart($("c-ttft"), { rows: tl, series: [{ key: "ttft_p50", label: "p50", color: col(0) }, { key: "ttft_p90", label: "p90", color: col(1) }], fmt: fmtS, label: "Time to first token" });
    legend("l-itl", [{ label: "p50", color: col(0) }, { label: "p90", color: col(1) }]);
    lineChart($("c-itl"), { rows: tl, series: [{ key: "itl_p50_ms", label: "p50", color: col(0) }, { key: "itl_p90_ms", label: "p90", color: col(1) }], fmt: fmtMs, label: "Inter-token latency" });
    lineChart($("c-queue"), { legendId: "l-queue", rows: tl, series: [{ key: "prefilling", label: "Prefilling", color: col(1) }, { key: "decoding", label: "Decoding", color: col(0) }, { key: "queued", label: "Queued before dispatch", color: col(3) }], fmt: d3.format("d"), label: "Queues" });

    // 2. mix
    const reqAgent = sum(tl.map((r) => r.agent_req_ps)), reqChat = sum(tl.map((r) => r.chat_req_ps));
    const inAgent = sum(tl.map((r) => r.agent_in_tps)), inChat = sum(tl.map((r) => r.chat_in_tps));
    const hasOut = tl.some((r) => r.chat_out_tps != null);
    const outAgent = sum(tl.map((r) => r.agent_out_tps)), outChat = sum(tl.map((r) => r.chat_out_tps));
    tiles("t-mix", [
      ["Requests from agents", pct((100 * reqAgent) / (reqAgent + reqChat || 1)), fmtInt(meta.chat_requests || 0) + " chat requests, no session ID"],
      ["Input tokens from agents", pct((100 * inAgent) / (inAgent + inChat || 1)), "chat is " + pct((100 * inChat) / (inAgent + inChat || 1)) + " of input"],
      hasOut ? ["Output tokens from agents", pct((100 * outAgent) / (outAgent + outChat || 1)), "chat is " + pct((100 * outChat) / (outAgent + outChat || 1)) + " of output"] : null,
      meta.has_chat ? ["Chat prompt size", fmtTok(meta.chat_prompt_p50) + " median", "single turn, shared system prompts"] : null,
    ]);
    legend("l-mixin", [{ label: "Agents (with a session ID)", color: col(0) }, { label: "Chat (no session ID)", color: col(1) }]);
    lineChart($("c-mixin"), { rows: tl, series: [{ key: "agent_in_tps", label: "Agents (with a session ID)", color: col(0), stack: true }, { key: "chat_in_tps", label: "Chat (no session ID)", color: col(1), stack: true }], fmt: fmtTok, label: "Input tokens per second by traffic type" });
    legend("l-mixout", [{ label: "Agents (with a session ID)", color: col(0) }, { label: "Chat (no session ID)", color: col(1) }]);
    lineChart($("c-mixout"), { rows: tl, series: [{ key: "agent_out_tps", label: "Agents (with a session ID)", color: col(0), stack: true }, { key: "chat_out_tps", label: "Chat (no session ID)", color: col(1), stack: true }], fmt: fmtTok, label: "Output tokens per second by traffic type", empty: "This run did not split output tokens by traffic type." });

    // 3. cache funnel
    const pot = median(tl.map((r) => r.pot_pct)), f2 = median(tl.map((r) => r.f2_pct)), f3 = median(tl.map((r) => r.f3_pct)), f4 = median(tl.map((r) => r.f4_pct));
    tiles("t-cache", [
      pot != null ? ["Session history could reuse", pct(pot), "of all input tokens (chat is about 1%)"] : null,
      ["Highest-cache worker holds", pct(f2), "the most of the prefix any one worker has"],
      ["Chosen worker holds", pct(f3), "the router weighs load as well as cache"],
      f4 != null ? ["Worker actually reused", pct(f4), f3 - f4 > 5 ? "the router's view was stale by " + Math.round(f3 - f4) + " points by the time the worker admitted the request" : "matches what the router expected"] : null,
      pot != null ? ["Not held by any worker", Math.max(0, pot - f2).toFixed(0) + " points", "history says reusable; evicted or never stored"] : null,
      ["Passed over by routing", Math.max(0, f2 - f3).toFixed(0) + " points", "cache on a worker the request did not use"],
    ]);
    const fun = [{ key: "pot_pct", label: "Session history could reuse", color: col(0) }, { key: "f2_pct", label: "Highest-cache worker", color: col(1) }, { key: "f3_pct", label: "Chosen worker", color: col(2) }, { key: "f4_pct", label: "Worker-reported reuse", color: col(3) }];
    legend("l-fun", fun.filter((s) => tl.some((r) => r[s.key] != null)).map((s) => ({ label: s.label, color: s.color })));
    lineChart($("c-fun"), { rows: tlBusy, series: fun, fmt: (v) => Math.round(v) + "%", max: 100, label: "Cache reuse funnel" });

    // 4. fit
    const dm = tl.map((r) => r.demand);
    const peakW = d3.max(tl, (r) => r.w2) || d3.max(dm);
    tiles("t-fit", [
      ["Peak demand, last " + words0(wl[1]), fmtTok(peakW), pct((100 * peakW) / meta.cap_gpu) + " of GPU memory"],
      ["Against GPU plus host", pct((100 * peakW) / meta.cap_all), fmtTok(meta.cap_all) + " in all"],
      ["Peak demand, active window", fmtTok(d3.max(dm)), "distinct blocks, " + fmtTok(d3.max(tl, (r) => r.demand_sum)) + " summed per session"],
      ["Fleet", meta.workers + " workers", fmtTok(meta.cap_gpu) + " GPU" + (meta.cap_host ? ", " + fmtTok(meta.cap_host) + " host" : ", no host tier")],
    ]);
    // Windows that hold the same sessions draw the same line; show one and say so.
    const sameRows = (p, q) => tl.every((r) => r[p] === r[q]);
    const w23 = sameRows("w2", "w3"), w12 = sameRows("w1", "w2");
    const fitSeries = [{ key: "w1", label: "Active in the last " + words0(wl[0]) + (w12 ? " (same as " + words0(wl[1]) + (w23 ? " and " + words0(wl[2]) : "") + ")" : ""), color: col(2) }]
      .concat(w12 ? [] : [{ key: "w2", label: "Active in the last " + words0(wl[1]) + (w23 ? " and " + words0(wl[2]) + " (identical here)" : ""), color: col(0) }])
      .concat(w23 ? [] : [{ key: "w3", label: "Active in the last " + words0(wl[2]), color: col(1) }]);
    const fitRefs = [{ y: meta.cap_gpu, label: "GPU " + fmtTok(meta.cap_gpu) }].concat(meta.cap_host ? [{ y: meta.cap_all, label: "GPU + host " + fmtTok(meta.cap_all) }] : []);
        lineChart($("c-fit"), { legendId: "l-fit", rows: tl, series: fitSeries, refs: fitRefs, fmt: fmtTok, label: "Demand over several timescales" });
    const held = [{ key: "gpu_resident", label: "On GPU (each block once)", color: col(1), stack: true }, { key: "gpu_extra", label: "On GPU, extra copies on other workers", color: col(4), stack: true }].concat(meta.cap_host ? [{ key: "overlap", label: "In host memory, also on GPU (a copy)", color: col(3), stack: true }, { key: "host_only", label: "In host memory only", color: col(2), stack: true }] : []).concat([{ key: "demand", label: "Distinct demand", color: col(0) }]);
    if (!meta.cap_host) $("n-held").textContent = "What is resident in GPU memory, stacked, counting every worker's copy because each takes room, against the distinct demand and the GPU capacity. Memory that is full is normal for a cache; what matters is how much of it the active sessions need (the next section splits that) and how much is extra copies.";
    lineChart($("c-held"), { legendId: "l-held", rows: tl, series: held, refs: fitRefs, fmt: fmtTok, label: "What the fleet holds" });
    const thrRows = smooth(tlBusy.map((r) => {
      const inp = (r.agent_in_tps || 0) * 60;
      return { ...r, rebuilt_recomputed_pct: inp > 0 ? (100 * (r.thrash_recomputed_pm || 0)) / inp : null, rebuilt_reloaded_pct: inp > 0 && r.thrash_reloaded_pm != null ? (100 * r.thrash_reloaded_pm) / inp : null };
    }), ["rebuilt_recomputed_pct", "rebuilt_reloaded_pct"], 5);
    lineChart($("c-thr"), { legendId: "l-thr", rows: thrRows, series: [{ key: "rebuilt_recomputed_pct", label: "Recomputed", color: col(1), stack: true }, { key: "rebuilt_reloaded_pct", label: "Reloaded from host", color: col(2), stack: true }], fmt: (v) => v.toFixed(v < 10 ? 1 : 0) + "%", label: "Prefix rebuilt by returning sessions, share of agent input" });

    // 5. storage
    const dupG = median(tl.map((r) => r.dup_gpu)), dupH = median(tl.map((r) => r.dup_host));
    const ovlShare = meta.has_overlap ? median(tl.filter((r) => r.host_resident > 0).map((r) => (100 * (r.overlap || 0)) / r.host_resident)) : null;
    const imb = median(tl.filter((r) => r.wd_mean > 0).map((r) => r.wd_max / r.wd_mean));
    const strPeak = d3.max(tl, (r) => r.stranded_gpu);
    const shrShare = median(tl.filter((r) => r.demand > 0 && r.shared != null).map((r) => (100 * r.shared) / r.demand));
    tiles("t-storage", [
      ["Copies of the cache", dupG.toFixed(2) + "x on GPU", meta.cap_host ? dupH.toFixed(2) + "x in host memory" : "no host tier"],
      ovlShare != null ? ["Host memory that is a GPU copy", pct(ovlShare), "median over the run"] : ["Host and GPU overlap", meta.cap_host ? "not reported" : "no host tier", meta.cap_host ? "this run predates the overlap metric" : "GPU memory only in this scenario"],
      imb != null ? ["Busiest worker against average", imb.toFixed(2) + "x", "median distinct demand"] : null,
      ["Held for finished sessions", fmtTok(strPeak), pct((100 * strPeak) / meta.cap_gpu) + " of GPU at the peak"],
      shrShare != null ? ["Shared between sessions", pct(shrShare), "of distinct demand"] : null,
    ]);
    lineChart($("c-dup"), { legendId: "l-dup", rows: tl, series: [{ key: "dup_gpu", label: "GPU copies", color: col(1) }].concat(meta.cap_host ? [{ key: "dup_host", label: "Host copies", color: col(2) }] : []), fmt: (v) => v.toFixed(1) + "x", refs: [{ y: 1, label: "1x: one copy" }], label: "Redundant copies" });
    const wref = [{ y: meta.per_worker_gpu, label: "one worker's GPU " + fmtTok(meta.per_worker_gpu) }];
    const wser = [{ key: "wr_gpu_max", label: "Busiest worker", color: col(1) }, { key: "wr_gpu_mean", label: "Average worker", color: col(0) }, { key: "wr_gpu_min", label: "Least busy worker", color: col(2) }];
    legend("l-wrk", [...wser.map((x) => ({ label: x.label, color: x.color })), { label: "One worker's GPU capacity", dash: true }]);
    lineChart($("c-wrk"), { rows: tl, series: wser, refs: wref, fmt: fmtTok, label: "GPU-resident session tokens per worker", empty: "This run did not log per-worker cache." });
    const imbS = [{ key: "imb_res_gpu", label: "GPU-resident tokens, busiest over average", color: col(1) }, { key: "imb_demand", label: "Homed demand, busiest over average", color: col(0) }].concat(meta.cap_host ? [{ key: "imb_res_host", label: "Host-resident tokens, busiest over average", color: col(2) }] : []);
    legend("l-imb", imbS.map((x) => ({ label: x.label, color: x.color })).concat([{ label: "1.0 is even", dash: true }]));
    lineChart($("c-imb"), { rows: tlBusy, series: imbS, refs: [{ y: 1, label: "even" }], fmt: (v) => v.toFixed(1) + "x", label: "Imbalance ratio", min1: true, empty: "This run did not report the imbalance ratios." });
    const comp = [
      { key: "needed_gpu", label: "Needed by active sessions", color: col(0), stack: true },
      { key: "left_gpu_total", label: "Left behind by sessions that moved", color: col(1), stack: true },
      { key: "stranded_gpu", label: "Held for finished sessions", color: col(3), stack: true },
    ];
    legend("l-left", comp.map((x) => ({ label: x.label, color: x.color })).concat([{ label: "GPU capacity", dash: true }]));
    lineChart($("c-left"), { rows: tlBusy, series: comp, refs: [{ y: meta.cap_gpu, label: "GPU " + fmtTok(meta.cap_gpu) }], fmt: fmtTok, label: "What is in GPU memory", empty: "This run did not report cache left behind by moved sessions." });
    legend("l-str", [{ label: "GPU", color: col(1) }, ...(meta.cap_host ? [{ label: "Host", color: col(2) }] : [])]);
    lineChart($("c-str"), { rows: tlBusy, series: [{ key: "stranded_gpu", label: "GPU", color: col(1) }].concat(meta.cap_host ? [{ key: "stranded_host", label: "Host", color: col(2) }] : []), refs: [{ y: meta.cap_gpu, label: "GPU " + fmtTok(meta.cap_gpu) }], fmt: fmtTok, label: "State kept for finished sessions" });
    legend("l-shr", [{ label: "Shared by more than one session", color: col(2) }, { label: "Distinct demand", color: col(0) }]);
    lineChart($("c-shr"), { rows: tl, series: [{ key: "shared", label: "Shared by more than one session", color: col(2) }, { key: "demand", label: "Distinct demand", color: col(0) }], fmt: fmtTok, label: "Prefix shared between sessions" });

    // 6. returns
    const tot = sum(returns.map((g) => g.turns));
    legend("l-ret", [{ label: "Prefix on GPU", color: col(0) }, ...(meta.cap_host ? [{ label: "Prefix in host memory", color: col(2) }] : []), { label: "Recomputed", color: CRIT }]);
    $("b-ret").replaceChildren(...returns.map((g) => {
      const row = document.createElement("div"); row.className = "row";
      const lab = document.createElement("div"); lab.className = "lab"; lab.textContent = g.gap;
      const bar = document.createElement("div"); bar.className = "bar";
      for (const [v, c] of [[g.gpu, col(0)], [g.host, col(2)], [g.miss, CRIT]]) if (v) { const s = document.createElement("span"); s.style.width = (v / g.turns) * 100 + "%"; s.style.background = c; bar.append(s); }
      const n = document.createElement("div"); n.className = "n"; n.textContent = fmtInt(g.turns) + " turns";
      row.append(lab, bar, g.turns ? n : document.createElement("div")); if (!g.turns) lab.style.opacity = 0.5; return row;
    }));
    const lateMiss = returns.length ? returns[returns.length - 1] : null;
    $("n-ret").textContent = `These are counts over the whole run, not a rolling window. A turn is counted when its request arrives, by the time since that session's previous turn finished, and counts as recomputed when more than a tenth of its prior prefix was in no tier. ${fmtInt(tot)} returning turns in this run. This run replays recorded think times divided by ${meta.time_scale}, so gaps over a few minutes are rare${lateMiss && lateMiss.turns ? "" : "; none fell in the longest bucket"}. A real session that runs over days will land mostly in the longest bucket.`;

    // 7. sessions
    const prim = sessions.filter((s) => s.role === "primary"), subs = sessions.filter((s) => s.role === "subagent");
    tiles("t-sess", [
      ["Sessions", String(sessions.length), prim.length + " primary, " + subs.length + " subagents"],
      ["Turns per session", String(Math.round(median(sessions.map((s) => s.turns)))) + " median", "p90 " + Math.round(quantile(sessions.map((s) => s.turns), 0.9))],
      ["Input per turn", fmtTok(median(sessions.map((s) => s.isl_per_turn))) + " median", "p90 " + fmtTok(quantile(sessions.map((s) => s.isl_per_turn), 0.9))],
      ["Output per turn", Math.round(median(sessions.map((s) => s.osl_per_turn))) + " median", "p90 " + Math.round(quantile(sessions.map((s) => s.osl_per_turn), 0.9))],
      ["Duration", num1(median(sessions.map((s) => s.duration_min))) + " min median", "p90 " + num1(quantile(sessions.map((s) => s.duration_min), 0.9)) + " min"],
      ["Time between turns", num1(median(prim.map((s) => s.mean_gap_s))) + " s primary", num1(median(subs.map((s) => s.mean_gap_s))) + " s for subagents"],
    ]);
    const hist = (title, items, edges, f) => { const h = document.createElement("div"); h.className = "bars hist"; const t = document.createElement("div"); t.className = "note"; t.textContent = title; h.append(t, ...bars(h, items, edges, f, col(0))); return h; };
    $("h-sess").replaceChildren(
      hist("Turns per session", sessions.map((s) => s.turns), [1, 2, 4, 8, 16, 32, 64, 128], String),
      hist("Input tokens per turn", sessions.map((s) => s.isl_per_turn), [1000, 4000, 16000, 32000, 64000, 128000, 256000], fmtTok),
      hist("Output tokens per turn", sessions.map((s) => s.osl_per_turn), [32, 64, 128, 192, 256, 512], String),
      hist("Time between turns, primary sessions (s)", prim.map((s) => s.mean_gap_s), [0.5, 1, 2, 5, 10, 20, 40, 80], String),
    );
    const distRows = [["Turns", "turns", (v) => String(Math.round(v))], ["Input tokens per turn", "isl_per_turn", fmtTok], ["Output tokens per turn", "osl_per_turn", (v) => String(Math.round(v))], ["Context growth per turn", "growth_per_turn", fmtTok], ["Peak context", "peak_context", fmtTok], ["Cache hit rate", "hit_rate", (v) => pct(v * 100)], ["Duration (min)", "duration_min", num1], ["Mean gap between turns (s)", "mean_gap_s", num1], ["Subagents spawned", "subagents", (v) => String(Math.round(v))]];
    const tbl = document.createElement("table");
    tbl.innerHTML = "<thead><tr><th>Per session</th><th>p50</th><th>p90</th><th>max</th></tr></thead>";
    const tb = document.createElement("tbody");
    for (const [name, key, f] of distRows) { const xs = sessions.map((s) => s[key]); const tr = document.createElement("tr"); for (const v of [name, f(median(xs)), f(quantile(xs, 0.9)), f(d3.max(xs))]) { const td = document.createElement("td"); td.textContent = v; tr.append(td); } tb.append(tr); }
    tbl.append(tb); $("tb-sess").replaceChildren(tbl);

    // 3b. routing decisions, from the router's decision log
    const rr = routing.filter((x) => x.t >= 0 && x.t <= tEnd && x.differ >= 5).map((x) => ({ ...x, pct_best: x.differ ? (100 * x.best_cache) / x.differ : null, pct_load: x.differ ? (100 * (x.left_cache - x.no_load_advantage)) / x.differ : null, pct_other: x.differ ? (100 * x.no_load_advantage) / x.differ : null, decode_tokens: x.decode_saved_blocks * meta.block_size }));
    $("g-route").hidden = !rr.length;
    if (!rr.length) $("n-route").textContent = "This run did not record the router decision log.";
    if (rr.length) {
      const rs = [
        { key: "pct_best", label: "Picked the highest-cache worker", color: col(0), stack: true },
        { key: "pct_load", label: "Picked another worker that had less load", color: col(2), stack: true },
        { key: "pct_other", label: "Picked another worker, no load reason in the inputs", color: col(1), stack: true },
      ];
      legend("l-rsplit", rs.map((x) => ({ label: x.label, color: x.color })));
      lineChart($("c-rsplit"), { rows: smooth(rr, ["pct_best", "pct_load", "pct_other"], 5), series: rs, fmt: (v) => Math.round(v) + "%", max: 100, label: "How decisions related to cache" });
      const pctOf = (x, n) => (x.left_cache > 0 ? (100 * n) / x.left_cache : null);
      const rr2 = rr.map((x) => ({ ...x, p_pre: pctOf(x, x.lighter_prefill), p_dec: pctOf(x, x.lighter_decode), p_req: pctOf(x, x.lighter_requests) }));
      const rt = [
        { key: "p_pre", label: "Chosen worker had less prefill backlog", color: col(1) },
        { key: "p_dec", label: "Chosen worker had less decode load", color: col(2) },
        { key: "p_req", label: "Chosen worker had fewer active requests", color: col(3) },
      ];
      legend("l-rtrade", rt.map((x) => ({ label: x.label, color: x.color })));
      lineChart($("c-rtrade"), { rows: smooth(rr2, ["p_pre", "p_dec", "p_req"], 5), series: rt, fmt: (v) => Math.round(v) + "%", max: 100, label: "Why the router left the highest-cache worker" });
      const nDec = d3.sum(rr, (x) => x.decisions), nDiff = d3.sum(rr, (x) => x.differ), left = d3.sum(rr, (x) => x.left_cache), gave = d3.sum(rr, (x) => x.gave_up_tokens * x.left_cache);
      $("n-route").textContent = left ? `Of ${fmtInt(nDec)} decisions, ${pct((100 * (nDec - nDiff)) / nDec)} had the same cache on every candidate, so there was no cache choice to make. Of the ${fmtInt(nDiff)} where it differed, the router left the highest-cache worker on ${fmtInt(left)} (${pct((100 * left) / nDiff)}), giving up ${fmtTok(gave / left)} cached tokens on average. A decision can count in more than one line.` : "";
    }

    // 8. metric reference
    const NEW_GROUPS = new Set(["Working set", "Load across workers", "Thrash and returns", "Coverage", "Per-session shape (observed when a session ends)"]);
    const NEW_ROWS = new Set(["router_selection trace record", "session_worker_load log", "session_summary log"]);
    const REF = [
      ["Working set", [
        ["sessions_active_by_role{role}", "gauge", "Active sessions, primary and subagent."],
        ["session_demand_by_role_tokens{role}", "gauge", "Distinct prompt tokens the active sessions will present next, each block counted once. A subagent counts only the blocks its primaries do not cover. The sum is the working set."],
        ["session_demand_by_idle_tokens{idle}", "gauge", "Per-session demand of sessions that have not ended, by time since last activity (up to 1 m, 5 m, 15 m, over). Disjoint buckets, the same as the return gaps. Cumulative sums give demand over several timescales."],
        ["session_shared_tokens", "gauge", "Prompt tokens referenced by more than one active session."],
        ["session_resident_tokens{tier}", "gauge", "Distinct tokens the session index holds in each tier (device, host, disk)."],
        ["session_stranded_tokens{tier}", "gauge", "Tokens held only for inactive sessions (device, host): cache held but not needed."],
        ["session_replicated_tokens{tier}", "gauge", "Copies of session prefix beyond the first on other workers (device, host). The duplication factor is 1 + replicated / resident."],
        ["session_tier_overlap_tokens", "gauge", "Tokens held in both the GPU and host tier of a worker, summed: how much host memory is a copy of GPU memory."],
      ]],
      ["Load across workers", [
        ["session_left_behind_tokens{tier}", "gauge", "Tokens workers hold for active sessions whose last turn ran elsewhere: cache left behind when a session moves."],
        ["session_worker_demand_imbalance_ratio", "gauge", "Distinct demand homed on the busiest worker over the mean worker's. 1.0 is even."],
        ["session_worker_resident_imbalance_ratio{tier}", "gauge", "Distinct session tokens on the busiest worker over the mean worker's, per tier. 1.0 is even."],
        ["session_worker_load log", "log", "One event per worker each minute: demand, resident and left-behind tokens and capacity. The busiest, least busy and mean worker on this page come from it. A worker identity is never a metric label."],
      ]],
      ["Thrash and returns", [
        ["session_thrash_tokens_total{outcome}", "counter", "Prior-prefix tokens a returning session had to recompute, or reload from host or disk. Recomputed is the cost of running out of cache."],
        ["session_return_turns_total{gap,outcome}", "counter", "Returning turns by time since the previous turn (up to 1 m, 5 m, 15 m, or over) and where the prior prefix was found (gpu, host, disk, miss). A count over the run."],
        ["session_prior_prefix_tokens_total", "counter", "Prefix tokens returning sessions could reuse if nothing had been evicted: the prior turn's input, not its output. The session-history line in the funnel."],
      ]],
      ["Coverage", [
        ["session_coverage_requests_total{coverage}", "counter", "Routed requests with (covered) or without (uncovered) a session ID."],
        ["session_coverage_input_tokens_total{coverage}", "counter", "The same split in input tokens: how much of the traffic the session metrics describe."],
        ["session_covered_output_tokens_total", "counter", "Output tokens of requests that carried a session ID. With the frontend output total it gives the output from requests without one."],
      ]],
      ["Per-session shape (observed when a session ends)", [
        ["session_turns", "histogram", "Turns per session. Its _count is the number of sessions."],
        ["session_peak_context_tokens", "histogram", "Largest context a session presented."],
        ["session_cache_hit_ratio", "histogram", "Share of a session's input served from cache, 0 to 1."],
        ["session_duration_seconds", "histogram", "First request to end."],
        ["session_turn_gap_seconds", "histogram", "Mean time between a session's turns."],
        ["session_subagents", "histogram", "Subagent sessions a session spawned."],
        ["session_summary log", "log", "One event per ended session with its ID, parent, turns, input and output tokens, cache hit, duration, gaps, peak context, fan-out and thrash. The per-session figures on this page come from it."],
      ]],
      ["Cache reuse funnel (already in Dynamo)", [
        ["input_sequence_tokens", "histogram", "F0: every prompt token. Its sum and count give mean input size per request."],
        ["kv_best_eligible_cached_prefix_tokens_total", "counter", "F2: the prefix cached on the highest-cache eligible worker."],
        ["kv_selected_cached_prefix_tokens_total", "counter", "F3: the same on the worker the router chose."],
        ["kv_worker_reused_tokens_total", "counter", "F4: tokens the worker reports reusing."],
        ["output_sequence_tokens", "histogram", "Output tokens per request."],
      ]],
      ["Routing decisions (already in Dynamo, #16395)", [
        ["router_selection trace record", "trace", "One record per routed request with every candidate worker's cost, and (added in this work) its cached tokens, prefill backlog, decode load and active requests. A request-trace record, not a metric: DYN_REQUEST_TRACE_RECORDS=router_selection."],
      ]],
      ["Compute health (already in Dynamo)", [
        ["frontend time_to_first_token_seconds", "histogram", "Request arrival to first token."],
        ["frontend inter_token_latency_seconds", "histogram", "Time between streamed tokens."],
        ["frontend queued_requests", "gauge", "Requests waiting in the HTTP queue before a first response."],
        ["frontend inflight_requests", "gauge", "Requests being served."],
        ["frontend router_queue_pending_requests", "gauge", "Requests held in the router's scheduler queue."],
        ["frontend worker_active_requests{request_phase}", "gauge", "Requests booked per worker, split prefill (not yet prefill-complete) and decode."],
        ["frontend worker_active_prefill_tokens", "gauge", "Prompt tokens queued or being prefilled per worker."],
        ["frontend worker_active_decode_blocks", "gauge", "KV blocks booked by active requests per worker."],
      ]],
    ];
    const rt2 = document.createElement("table"); rt2.className = "reftable";
    rt2.innerHTML = "<thead><tr><th>Metric</th><th>Type</th><th>What it tells you</th></tr></thead>";
    const rb = document.createElement("tbody");
    for (const [group, ms] of REF) {
      const gr = document.createElement("tr"); const gc = document.createElement("td"); gc.colSpan = 3; const gb = document.createElement("b"); gb.textContent = group; gc.append(gb);
      if (NEW_GROUPS.has(group)) { const pill = document.createElement("span"); pill.className = "newpill"; pill.textContent = "new"; gc.append(pill); }
      gr.append(gc); rb.append(gr);
      for (const [n, ty, d] of ms) {
        const tr = document.createElement("tr");
        const a1 = document.createElement("td"); a1.textContent = n;
        if (NEW_GROUPS.has(group) || NEW_ROWS.has(n)) { const pill = document.createElement("span"); pill.className = "newpill"; pill.textContent = "new"; a1.append(pill); }
        const b1 = document.createElement("td"); b1.textContent = ty; const c1 = document.createElement("td"); c1.textContent = d;
        tr.append(a1, b1, c1); rb.append(tr);
      }
    }
    rt2.append(rb); $("ref").replaceChildren(rt2);

    $("foot").textContent = `The workers are mock workers: the router, the session index and the workload are real, but latency, queues and KV behaviour come from the mocker's model, so read the response times as a demonstration. Demand counts each distinct prompt block once across active sessions. Thrash and the session-history line are estimates from the session index.`;
  }

  dash.onData(draw);
})();
