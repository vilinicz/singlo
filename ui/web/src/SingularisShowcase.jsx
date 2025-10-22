import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import axios from "axios";
import cytoscape from "cytoscape";

// One‑pager showcase for the Singularis pipeline
// Color palette inspired by the third screenshot: neon‑green on dark "matrix" background.
// Tailwind required.

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";
const STAGES_ORDER = ["S0", "S1", "S2", "finalize"]; // expected order

export default function SingularisShowcase() {
  const [file, setFile] = useState(null);
  const [docId, setDocId] = useState("");
  const [status, setStatus] = useState(null); // { state: queued|running|done|error, stage, duration_ms, stages: [] }
  const [polling, setPolling] = useState(false);
  const cyRef = useRef(null);
  const cyContainerRef = useRef(null);

  // --- Derived values
  const pipelineProgress = useMemo(() => {
    if (!status?.stage) return 0;
    const idx = STAGES_ORDER.indexOf(status.stage);
    const doneIdx = status.state === "done" ? STAGES_ORDER.length : Math.max(0, idx + 1);
    return Math.min(100, Math.round((doneIdx / STAGES_ORDER.length) * 100));
  }, [status]);

  // --- Drag & Drop
  const onDrop = useCallback((e) => {
    e.preventDefault();
    const f = e.dataTransfer?.files?.[0];
    if (f) setFile(f);
  }, []);

  const onBrowse = (e) => {
    const f = e.target.files?.[0];
    if (f) setFile(f);
  };

  // --- Upload & Kickoff (S0) then Extract (S1+S2)
  const startPipeline = async () => {
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    // Auto theme selection; change to &theme=biomed if you want to pin
    const { data } = await axios.post(`${API}/parse?theme=auto`, form, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    const id = data?.doc_id;
    if (!id) {
      alert("Server didn't return doc_id");
      return;
    }
    setDocId(id);
    setPolling(true);
  };

  // --- Poll status
  useEffect(() => {
    if (!polling || !docId) return;
    const t = setInterval(async () => {
      try {
        const { data } = await axios.get(`${API}/status/${encodeURIComponent(docId)}`);
        setStatus(data);
        if (data?.state === "done" || data?.state === "error") {
          clearInterval(t);
          setPolling(false);
          if (data?.state === "done") {
            // Scroll to graph and render it
            setTimeout(() => {
              cyContainerRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
              renderGraph();
            }, 350);
          }
        }
      } catch (e) {
        console.error(e);
      }
    }, 900);
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [polling, docId]);

  // --- Cytoscape graph
  async function renderGraph() {
    // fetch full graph
    const { data } = await axios.get(`${API}/graph/${encodeURIComponent(docId)}`);
    const cyHost = cyContainerRef.current;
    if (!cyHost) return;

    if (!cyRef.current) {
      cyRef.current = cytoscape({
        container: cyHost,
        style: [
          {
            selector: "node",
            style: {
              label: "data(label)",
              "background-color": "#0f172a",
              color: "#e2e8f0",
              "text-wrap": "wrap",
              "text-max-width": "280px",
              "font-size": "12px",
              shape: "round-rectangle",
              padding: 16,
              width: "label",
              height: "label",
              "border-color": "#22c55e",
              "border-width": 1,
              "shadow-blur": 8,
              "shadow-color": "#00000055",
              "shadow-opacity": 0.6,
              "shadow-offset-x": 0,
              "shadow-offset-y": 2,
            },
          },
          { selector: "edge", style: { width: 2, "line-color": "#22c55e", "target-arrow-color": "#22c55e", "target-arrow-shape": "triangle" } },
          { selector: ".hl", style: { width: 3, opacity: 1 } },
        ],
        layout: { name: "cose", animate: false },
      });
    }

    const cy = cyRef.current;
    const elements = [];
    const trunc = (s, n = 140) => (s && s.length > n ? s.slice(0, n).trim() + "…" : s || "");
    for (const n of data.nodes || []) {
      elements.push({ data: { id: n.id, label: trunc(n.text || ""), type: n.type } });
    }
    for (const e of data.edges || []) {
      elements.push({ data: { id: `${e.from}->${e.to}:${e.type}`, source: e.from, target: e.to, type: e.type } });
    }

    cy.batch(() => {
      cy.elements().remove();
      cy.add(elements);
    });

    cy.layout({ name: "cose", fit: true, padding: 32 }).run();

    cy.off("tap");
    cy.on("tap", "node", (ev) => {
      cy.elements("edge").removeClass("hl");
      ev.target.connectedEdges().addClass("hl");
    });
    cy.on("tap", (ev) => {
      if (ev.target === cy) cy.elements("edge").removeClass("hl");
    });
  }

  // --- UI helpers
  const StageCard = ({ stage, active, duration, notes }) => (
    <div
      className={`transition-all rounded-2xl border border-emerald-500/40 bg-slate-900/60 px-5 py-4 shadow-xl ${
        active ? "scale-[1.05] ring-2 ring-emerald-400/70" : "opacity-80"
      }`}
    >
      <div className="flex items-center justify-between">
        <div className={`font-semibold tracking-wide ${active ? "text-emerald-300" : "text-emerald-200"}`}>
          {stage}
        </div>
        <div className="text-xs text-emerald-200/70">{duration ? `${(duration / 1000).toFixed(2)} s` : "—"}</div>
      </div>
      <p className="mt-2 text-xs leading-relaxed text-emerald-100/80 min-h-10">
        {notes || descriptionFor(stage)}
      </p>
    </div>
  );

  const descriptionFor = (stage) =>
    ({
      S0: "Parsing PDF → text/blocks with coordinates (TEI/JSON).",
      S1: "Heuristic extraction of atoms (Hypothesis/Method/Result/…).",
      S2: "Building typed edges, layout hints, dedup & cleaning.",
      finalize: "Packaging previews and final graph payload.",
    }[stage] || "Processing…");

  return (
    <div className="min-h-screen w-full bg-slate-950 text-emerald-100">
      {/* Matrix-esque animated backdrop */}
      <div className="pointer-events-none fixed inset-0 -z-10 bg-[radial-gradient(ellipse_at_top,rgba(16,185,129,0.08),transparent_60%),radial-gradient(ellipse_at_bottom,rgba(16,185,129,0.06),transparent_60%)]" />
      <div className="pointer-events-none fixed inset-0 -z-10 opacity-[0.12] [mask-image:linear-gradient(to_bottom,black,transparent)]">
        <SvgMatrixRain />
      </div>

      {/* HERO (state: pick & upload) */}
      <section className="mx-auto max-w-6xl px-5 py-12 md:py-16">
        <header className="mb-8 text-center">
          <h1 className="text-4xl font-bold tracking-widest text-emerald-300 drop-shadow md:text-5xl">
            SINGULARIS
          </h1>
          <p className="mt-3 text-sm text-emerald-200/80">Reforming scientific publishing with structured graphs</p>
          <div className="mx-auto mt-3 h-1 w-28 rounded-full bg-emerald-400/70" />
        </header>

        {!docId && (
          <div className="mx-auto grid max-w-4xl gap-6 md:grid-cols-[1.2fr_.8fr]">
            {/* Drop zone */}
            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={onDrop}
              className="flex min-h-[260px] items-center justify-center rounded-3xl border-2 border-dashed border-emerald-500/50 bg-slate-900/50 p-6 text-center shadow-2xl"
            >
              <label className="w-full cursor-pointer">
                <input type="file" accept="application/pdf" className="hidden" onChange={onBrowse} />
                <div className="mx-auto max-w-sm">
                  <div className="text-lg font-medium text-emerald-200/90">Drop PDF here</div>
                  <div className="mt-1 text-xs text-emerald-200/70">or click to choose a file</div>
                  {file && <div className="mt-3 truncate text-xs text-emerald-300">{file.name}</div>}
                </div>
              </label>
            </div>

            {/* Action card */}
            <div className="flex flex-col justify-between rounded-3xl border border-emerald-600/40 bg-slate-900/60 p-6 shadow-xl">
              <div>
                <h3 className="text-lg font-semibold text-emerald-200">Ready?</h3>
                <p className="mt-2 text-sm text-emerald-100/70">
                  Upload a paper and we will parse it, extract atoms, build edges, and draw a knowledge graph.
                </p>
              </div>
              <button
                onClick={startPipeline}
                disabled={!file}
                className="mt-5 inline-flex items-center justify-center rounded-2xl bg-emerald-500 px-5 py-3 text-sm font-semibold text-slate-900 shadow-lg shadow-emerald-500/30 transition-all hover:bg-emerald-400 disabled:cursor-not-allowed disabled:opacity-50"
              >
                Start parsing
              </button>
            </div>
          </div>
        )}

        {/* PROGRESS (state: running) */}
        {docId && status?.state !== "done" && (
          <div className="mx-auto mt-6 max-w-5xl rounded-3xl border border-emerald-600/40 bg-slate-900/60 p-6 shadow-2xl">
            <div className="flex flex-col gap-6 md:flex-row md:items-start">
              {/* Steps */}
              <div className="grid flex-1 grid-cols-1 gap-4 lg:grid-cols-4">
                {STAGES_ORDER.map((s) => {
                  const rec = status?.stages?.find((x) => x.name === s);
                  const active = status?.stage === s && status?.state === "running";
                  return (
                    <StageCard key={s} stage={s} active={active} duration={rec?.duration_ms} notes={rec?.notes} />
                  );
                })}
              </div>

              {/* Gauge */}
              <div className="flex w-full max-w-[220px] flex-col items-center justify-center gap-3">
                <SpinnerCircle progress={pipelineProgress} />
                <div className="text-xs text-emerald-200/80">{pipelineProgress}%</div>
                <div className="text-[11px] text-emerald-200/60">
                  {status?.state === "running" ? `Processing: ${status?.stage}` : status?.state || "—"}
                </div>
              </div>
            </div>
          </div>
        )}
      </section>

      {/* GRAPH (state: done) */}
      {status?.state === "done" && (
        <section className="relative h-[100vh] w-full" ref={cyContainerRef}>
          <h2 className="sr-only">Knowledge Graph</h2>
          <div className="absolute inset-0" id="cy-host" ref={cyContainerRef}></div>
          {/* Real container for Cytoscape */}
          <div ref={cyContainerRef} id="cy" className="absolute inset-0" />
        </section>
      )}
    </div>
  );
}

// --- Visual widgets ---
function SpinnerCircle({ progress = 0 }) {
  return (
    <div className="relative h-24 w-24">
      <div className="absolute inset-0 animate-spin rounded-full border-2 border-emerald-400/30 border-t-emerald-400" />
      <div className="absolute inset-1 rounded-full border border-emerald-400/40" />
      <div className="absolute inset-0 grid place-items-center text-sm font-semibold text-emerald-200">
        {Math.round(progress)}%
      </div>
    </div>
  );
}

function SvgMatrixRain() {
  // a simple svg grid of characters to emulate matrix rain background (static)
  const rows = 20;
  const cols = 36;
  const chars = "ACGT01";
  const cells = [];
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      const ch = chars[(x * 17 + y * 13) % chars.length];
      cells.push(
        <text key={`${x}-${y}`} x={x * 32 + 12} y={y * 28 + 22} className="fill-emerald-400/90 text-[12px]">
          {ch}
        </text>
      );
    }
  }
  return (
    <svg width="100%" height="100%" viewBox="0 0 1152 768" xmlns="http://www.w3.org/2000/svg">
      {cells}
    </svg>
  );
}
