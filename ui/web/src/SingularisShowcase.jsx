import React, {useCallback, useEffect, useMemo, useRef, useState} from "react";
import axios from "axios";
import cytoscape from "cytoscape";

// ─── constants ─────────────────────────────────────────────
const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

const STAGE_ORDER = ["S0", "S1", "S2", "finalize"];
const STAGE_DISPLAY = {S0: "Step 1", S1: "Step 2", S2: "Step 3", finalize: "Total"};
const STAGE_DESC = {
    S0: "Reading a document (GROBID)",
    S1: "Heuristic extraction of atoms (Hypothesis/Method/Result/...).",
    S2: "Building typed edges, layout hints, dedup & cleaning.",
    finalize: "Packaging previews and final graph payload."
};
const stageDisplayName = (n) => STAGE_DISPLAY[n] || n;

// ── grid & legend helpers (ported from App.jsx) ─────────────────
const EDGE_STYLES = {
    supports: {color: "#16a34a"},
    refutes: {color: "#ef4444"},
    produces: {color: "#7c3aed"},
    uses: {color: "#0f766e"},
    feeds: {color: "#2563eb"},
    informs: {color: "#0284c7"},
    summarizes: {color: "#a855f7"},
    used_by: {color: "#94a3b8"},
};

// рядом с CARD_W
const CARD_W = 340;          // фиксированная ширина всех блоков
const COLUMN_GAP = 120;      // ← расстояние между столбцами (меняешь как нужно)
const COLS_COUNT = 8;        // сколько колонок у тебя в легенде
// edges — порог, фан-аут, цвет + «волнистые» контрольные точки
const EDGE_CONF_MIN = 0.55, MAX_FAN_OUT = 3;
const CARD_PAD = 18;       // padding узлов
const TEXT_MAX_W = CARD_W - CARD_PAD * 2; // чтобы текст красиво переносился

function makeGrid(cy) {
    const host = cy.container();
    const W = host?.clientWidth || 1200;
    const H = host?.clientHeight || 700;

    // поля слева/справа — тоже можно крутить
    const COL_LEFT = 120;
    const COL_RIGHT = 120;

    // шаг между центрами колонок:
    const COL_STEP = CARD_W + COLUMN_GAP;

    // если нужно «впихнуть» все столбцы в текущую ширину — раскомментируй строку ниже
    // const COL_STEP = Math.max(CARD_W + COLUMN_GAP, (W - COL_LEFT - COL_RIGHT) / (COLS_COUNT - 1));

    const ROW_TOP = 250;
    const ROW_STEP = 100;

    const colX = (idx) => COL_LEFT + idx * COL_STEP;
    const rowY = (idx) => ROW_TOP + idx * ROW_STEP;
    return {colX, rowY};
}

function addColumnHeaderNodes(cy) {
    const COLS = ["Input Fact", "Hypothesis", "Experiment", "Technique", "Result", "Dataset", "Analysis", "Conclusion"];
    const grid = makeGrid(cy);
    const ys = cy.nodes().map(n => n.position("y"));
    const topY = (ys.length ? Math.min(...ys) : 120) - 90;
    const toAdd = [];
    for (let idx = 0; idx < COLS.length; idx++) {
        const id = `hdr:${idx}`;
        const x = grid.colX(idx);
        const existing = cy.getElementById(id);
        const data = {id, type: "Header", label: COLS[idx]};
        const pos = {x, y: topY};
        if (existing.nonempty()) {
            existing.data(data);
            existing.position(pos);
        } else toAdd.push({data, position: pos, selectable: false, grabbable: false});
    }
    if (toAdd.length) cy.add(toAdd);
    cy.style().selector('node[type = "Header"]').style({
        "background-color": "#ffffff",
        "border-color": "#e5e7eb",
        "border-width": 1,
        "label": "data(label)",
        "font-size": "14px",
        "font-weight": "600",
        "text-halign": "center",
        "text-valign": "center",
        "text-wrap": "wrap",
        "text-max-width": `${TEXT_MAX_W}px`,
        "padding": `${CARD_PAD}px`,
        "border-radius": "8px",
        "width": CARD_W,       // ← тот же размер
        "height": "label",
        "color": "#374151",
        "text-outline-width": 3,
        "text-outline-color": "#ffffff",
        "shadow-blur": 8,
        "shadow-color": "#0000001a",
        "shadow-opacity": 1,
        "shadow-offset-x": 0,
        "shadow-offset-y": 2
    }).update();
}

function renderLegend(cy) {
    const host = cy.container();
    host.style.position = host.style.position || "relative";
    let box = host.querySelector(".legend-box");
    if (!box) {
        box = document.createElement("div");
        box.className = "legend-box";
        host.appendChild(box);
    }
    const items = [
        ["supports", "Result → Hypothesis"], ["refutes", "Result ↛ Hypothesis"], ["produces", "Experiment → Result"],
        ["uses", "Technique → * / * → Analysis"], ["feeds", "Dataset → Experiment"], ["informs", "Result/Dataset → Analysis"],
        ["summarizes", "Analysis → Conclusion"], ["used_by", "reverse use"]
    ];
    box.innerHTML = `
    <div class="legend-title">Legend</div>
    ${items.map(([k, text]) => `<div class="legend-row"><span class="swatch" style="background:${(EDGE_STYLES[k] || {}).color || "#64748b"}"></span><span class="legend-text">${text}</span></div>`).join("")}
  `;
}

function bindInteractions(cy) {
    let clickTimer = null, lastId = null, lastTs = 0;
    const DOUBLE_MS = 280;
    cy.off('tap');
    cy.off('tap', 'node');
    cy.off('tap', 'core');

    function highlightNodeEdges(node) {
        cy.elements('edge.hl').removeClass('hl');
        cy.nodes('.active').removeClass('active').unselect();
        if (!node || node.empty()) return;
        node.addClass('active').select();
        const inc = node.connectedEdges();
        if (inc && inc.length) inc.addClass('hl');
    }

    function clearHighlight() {
        cy.elements('edge.hl').removeClass('hl');
        cy.nodes('.active').removeClass('active').unselect();
    }

    cy.on('tap', 'node', (ev) => {
        const n = ev.target, now = Date.now();
        if (lastId === n.id() && (now - lastTs) < DOUBLE_MS) {
            clearTimeout(clickTimer);
            clickTimer = null;
            lastId = null;
            lastTs = 0;
            const d = n.data();
            alert(`${d.type}\n\n${d.text || "(no text)"}\n\nconf=${(d.conf ?? 0).toFixed(2)}\n${d.id}`);
            return;
        }
        lastId = n.id();
        lastTs = now;
        clearTimeout(clickTimer);
        clickTimer = setTimeout(() => {
            highlightNodeEdges(n);
            clickTimer = null;
            lastId = null;
            lastTs = 0;
        }, DOUBLE_MS);
    });
    cy.on('tap', (ev) => {
        if (ev.target === cy) {
            clearTimeout(clickTimer);
            clearHighlight();
            lastId = null;
            lastTs = 0;
            clickTimer = null;
        }
    });
}

export default function SingularisShowcase() {
    const [file, setFile] = useState(null);
    const [docId, setDocId] = useState("");
    const [status, setStatus] = useState(null); // { state, stage, stages: [{name,duration_ms,notes}] }
    const [polling, setPolling] = useState(false);
    const [graphVisible, setGraphVisible] = useState(false);
    const cyRef = useRef(null);
    const graphHostRef = useRef(null);
    const fileInputRef = useRef(null);

    function progressValue() {
        if (!status || !status.stage) return 0;
        const idx = STAGE_ORDER.indexOf(status.stage);
        if (idx < 0) return 0;
        const doneIdx = status.state === "done" ? STAGE_ORDER.length : idx + 1;
        return Math.min(100, Math.round((doneIdx / STAGE_ORDER.length) * 100));
    }

    // Drag & drop
    const onDrop = useCallback((e) => {
        e.preventDefault();
        const f = e.dataTransfer?.files?.[0];
        if (f) setFile(f);
    }, []);
    const onBrowse = (e) => {
        const f = e.target.files?.[0];
        if (f) setFile(f);
    };

    // Start
    const startPipeline = async () => {
        if (!file) return;
        const form = new FormData();
        form.append("file", file);
        const {data} = await axios.post(`${API}/parse?theme=auto`, form, {headers: {"Content-Type": "multipart/form-data"}});
        const id = data?.doc_id;
        if (!id) return alert("Server didn't return doc_id");
        setDocId(id);
        setPolling(true);
    };

    // Poll status
    useEffect(() => {
        if (!polling || !docId) return;
        const t = setInterval(async () => {
            try {
                const {data} = await axios.get(`${API}/status/${encodeURIComponent(docId)}`);
                setStatus(data);
                if (data?.state === "done" || data?.state === "error") {
                    clearInterval(t);
                    setPolling(false);
                    if (data?.state === "done") {
                        // 1) сразу строим граф
                        await renderGraph();
                        // 2) ждём 2 секунды и плавно показываем секцию графа
                        setTimeout(() => {
                            setGraphVisible(true);
                            graphHostRef.current?.scrollIntoView({behavior: "smooth", block: "center"});
                        }, 2000);
                    }
                }
            } catch (e) {
                console.error(e);
            }
        }, 900);
        return () => clearInterval(t);
    }, [polling, docId]);

    // Render graph
    async function renderGraph() {
        try {
            const {data} = await axios.get(`${API}/graph/${encodeURIComponent(docId)}`);
            const host = graphHostRef.current;
            if (!host) return;

            let cy = cyRef.current;
            if (!cy) {
                cy = cyRef.current = cytoscape({
                    container: host,
                    pixelRatio: 1,
                    wheelSensitivity: 0.2,
                    minZoom: 0.2,
                    maxZoom: 2,
                    style: [
                        {selector: "edge.hl", style: {width: 3, opacity: 1}}
                    ],
                    layout: {name: "preset", animate: false},
                });
            }

            const elements = [];
            const have = new Set();
            const grid = makeGrid(cy);
            const trunc = (s, n = 140) => (s && s.length > n ? s.slice(0, n).trim() + "…" : (s || ""));

            // nodes — раскладываем по col/row (из S2), иначе — уважаем входные позиции
            for (const n of data.nodes || []) {
                const col = Number.isFinite(n.data?.col) ? n.data.col : null;
                const row = Number.isFinite(n.data?.row) ? n.data.row : null;
                elements.push({
                    data: {
                        id: n.id, type: n.type, text: n.text || "", conf: n.conf ?? 0,
                        col, row, label: trunc(n.text || "")
                    },
                    position: (col != null && row != null) ? {
                        x: grid.colX(col),
                        y: grid.rowY(row)
                    } : (n.position || undefined),
                    selectable: true, grabbable: false,
                });
                have.add(n.id);
            }

            const outCnt = {};
            for (const e of data.edges || []) {
                const conf = +e.conf || 0;
                if (conf < EDGE_CONF_MIN) continue;
                if (!have.has(e.from) || !have.has(e.to)) continue;
                const k = e.from;
                outCnt[k] = (outCnt[k] || 0);
                if (outCnt[k] >= MAX_FAN_OUT) continue;
                outCnt[k]++;
                const cpd = [-40, 40, -40], cpw = [0.25, 0.5, 0.75];
                const style = EDGE_STYLES[e.type] || {color: "#64748b"};
                elements.push({
                    data: {
                        id: `${e.from}->${e.to}:${e.type}`, source: e.from, target: e.to, type: e.type,
                        conf, cpd, cpw, color: style.color
                    }
                });
            }

            // paint
            cy.batch(() => {
                cy.elements().remove();
                cy.add(elements);
            });

            // стили (узлы-карточки + цвета типов, рёбра с underlay и подсветкой)
            cy.style().fromJson([
                {
                    selector: "node",
                    style: {
                        "background-color": "#e5e7eb",
                        "label": "data(label)",
                        "text-wrap": "wrap",
                        "text-max-width": `${TEXT_MAX_W}px`,
                        "font-size": "12px",
                        "text-halign": "center",
                        "text-valign": "center",
                        "shape": "round-rectangle",
                        "padding": `${CARD_PAD}px`,
                        "border-width": 1,
                        "border-color": "#1f2937",
                        "width": CARD_W,          // ← фиксированная ширина
                        "height": "label",         // авто-высота под текст
                        "shadow-blur": 8,
                        "shadow-color": "#00000022",
                        "shadow-opacity": 0.5,
                        "shadow-offset-x": 0,
                        "shadow-offset-y": 1
                    }
                },
                {selector: "node[type = 'Input Fact']", style: {"background-color": "#b8c1ec"}},
                {selector: "node[type = 'Hypothesis']", style: {"background-color": "#ffd166"}},
                {selector: "node[type = 'Experiment']", style: {"background-color": "#ef476f"}},
                {selector: "node[type = 'Technique']", style: {"background-color": "#06d6a0"}},
                {selector: "node[type = 'Result']", style: {"background-color": "#118ab2"}},
                {selector: "node[type = 'Dataset']", style: {"background-color": "#c77dff"}},
                {selector: "node[type = 'Analysis']", style: {"background-color": "#73d2de"}},
                {selector: "node[type = 'Conclusion']", style: {"background-color": "#83c5be"}},
                {
                    selector: "edge",
                    style: {
                        "curve-style": "unbundled-bezier",
                        "control-point-distances": "data(cpd)",
                        "control-point-weights": "data(cpw)",
                        "line-color": "data(color)",
                        "target-arrow-color": "data(color)",
                        "target-arrow-shape": "triangle",
                        "width": 2,
                        "opacity": "mapData(conf, 0, 1, 0.4, 0.85)",
                        "font-size": "10px",
                        "text-rotation": "autorotate",
                        "text-background-opacity": 0,
                        "text-margin-y": -2,
                        "underlay-color": "#212121",
                        "underlay-opacity": 0,
                        "underlay-padding": 1
                    }
                },
                {
                    selector: "edge.hl",
                    style: {
                        "opacity": 1, "width": 3,
                        "line-color": "data(color)", "target-arrow-color": "data(color)",
                        "z-index-compare": "manual", "z-index": 9999,
                        "underlay-opacity": 1, "underlay-padding": 2,
                        "target-arrow-shape": "triangle", "arrow-scale": 1.2,
                        "transition-property": "opacity, width, underlay-padding", "transition-duration": "120ms"
                    }
                },
                {selector: "node:selected", style: {"border-width": 2, "border-color": "#000"}},
                {selector: "edge:selected", style: {"line-color": "#000", "target-arrow-color": "#000", "width": 2}},
            ]).update();

            // layout — preset (S2 уже дал col/row)
            cy.layout({name: "preset", fit: true, padding: 32}).run();

            // заголовки колонок + легенда
            addColumnHeaderNodes(cy);
            renderLegend(cy);

            // fit on resize
            const container = cy.container();
            const fitOnce = () => {
                try {
                    cy.fit(undefined, 48);
                } catch {
                }
            };
            fitOnce();
            const ro = new ResizeObserver(() => {
                cy.resize();
                fitOnce();
            });
            if (container) ro.observe(container);

            // interactions
            bindInteractions(cy);
        } catch
            (e) {
            console.error(e);
        }
    }

    function Donut({value = 0, size = 120, stroke = 10}) {
        const r = (size - stroke) / 2;
        const c = 2 * Math.PI * r;
        const off = c - (c * Math.min(100, Math.max(0, value))) / 100;
        return (
            <svg width={size} height={size} className="block">
                <g transform={`translate(${size / 2},${size / 2})`}>
                    <circle r={r} className="fill-none stroke-emerald-500/20" strokeWidth={stroke}/>
                    <circle
                        r={r}
                        className="fill-none stroke-emerald-400 transition-[stroke-dashoffset] duration-200"
                        strokeWidth={stroke}
                        strokeLinecap="round"
                        strokeDasharray={c}
                        strokeDashoffset={off}
                        transform="rotate(-90)"
                    />
                </g>
                <foreignObject x="0" y="0" width={size} height={size}>
                    <div className="w-full h-full flex items-center justify-center">
                        <div className="text-emerald-200 font-semibold text-lg">{Math.round(value)}%</div>
                    </div>
                </foreignObject>
            </svg>
        );
    }

    return (
        <div className="min-h-screen w-full bg-slate-950 text-emerald-100">
            {/* Matrix rain background */}
            <div
                className="pointer-events-none fixed inset-0 z-0 bg-[radial-gradient(ellipse_at_top,rgba(16,185,129,0.08),transparent_60%),radial-gradient(ellipse_at_bottom,rgba(16,185,129,0.06),transparent_60%)]"/>
            <div
                className="pointer-events-none fixed inset-0 z-0 opacity-[0.21] [mask-image:linear-gradient(to_bottom,black,transparent)]">
                <MatrixRain/></div>

            <section className="mx-auto max-w-6xl px-5 py-12 md:py-16">
                <header className="mb-8 text-center">
                    <h1 className="text-4xl font-bold tracking-widest text-emerald-300 drop-shadow md:text-5xl">SINGULARIS</h1>
                    <p className="mt-3 text-sm text-emerald-200/80">Reforming scientific publishing with structured
                        graphs</p>
                    <div className="mx-auto mt-3 h-1 w-28 rounded-full bg-emerald-400/70"/>
                </header>

                {/* HERO stacked */}
                {!docId && (
                    <div className="mx-auto max-w-3xl">
                        <div onDragOver={(e) => e.preventDefault()} onDrop={onDrop}
                             onClick={() => fileInputRef.current?.click()}
                             className="flex min-h-[260px] items-center justify-center rounded-3xl border-2 border-dashed border-emerald-500/50 bg-slate-900/50 p-6 text-center shadow-2xl cursor-pointer">
                            <label className="w-full cursor-pointer">
                                <input ref={fileInputRef} type="file" accept="application/pdf" className="hidden"
                                       onChange={onBrowse}/>
                                <div className="mx-auto max-w-sm">
                                    <div className="text-lg font-medium text-emerald-200/90">Drop PDF here</div>
                                    <div className="mt-1 text-xs text-emerald-200/70">or click to choose a file
                                    </div>
                                    {file &&
                                        <div className="mt-3 truncate text-xs text-emerald-300">{file.name}</div>}
                                </div>
                            </label>
                        </div>

                        <div
                            className="mt-6 rounded-3xl border border-emerald-600/40 bg-slate-900/60 p-8 text-center shadow-xl">
                            <h3 className="text-xl font-semibold text-emerald-200">Ready?</h3>
                            <p className="mx-auto mt-2 max-w-md text-sm text-emerald-100/70">Upload a paper and we
                                will
                                parse it, extract atoms, build edges, and draw a knowledge graph.</p>
                            <button onClick={startPipeline} disabled={!file}
                                    className="mx-auto mt-6 inline-flex items-center justify-center rounded-2xl bg-emerald-500 px-7 py-4 text-base font-semibold text-slate-900 shadow-lg shadow-emerald-500/30 transition-all hover:bg-emerald-400 disabled:cursor-not-allowed disabled:opacity-50">Start
                                parsing
                            </button>
                        </div>
                    </div>
                )}

                {/* PROGRESS (always visible after start; vertical stages) */}
                {docId && (
                    <>
                        {/* ── Fancy status/stages card (как в исходном дизайне) ── */}
                        <div
                            className="rounded-[28px] border border-emerald-500/30 p-6 md:p-8 bg-white/0 backdrop-blur-sm">
                            {/* шапка слева + индикатор справа */}
                            <div className="grid grid-cols-1 md:grid-cols-[1fr_auto] gap-6 md:gap-10 items-start">
                                {/* левая колонка */}
                                <div className="space-y-2 text-emerald-200/90">
                                    <div><b className="text-emerald-200">State:</b> <span
                                        className="text-emerald-200/80">{status?.state || "—"}</span></div>
                                    <div><b className="text-emerald-200">Stage:</b> <span
                                        className="text-emerald-200/80">
                                        {status?.stage ? stageDisplayName(status.stage) : "—"}
                                      </span></div>
                                    <div><b className="text-emerald-200">Total:</b> <span
                                        className="text-emerald-200/80">
                                        {status?.duration_ms ? `${(status.duration_ms / 1000).toFixed(2)} s` : "—"}
                                      </span></div>
                                </div>

                                {/* индикатор справа */}
                                <div className="flex flex-col items-center justify-center">
                                    <Donut value={progressValue()}/>
                                    <div className="mt-1 text-emerald-200/70 text-sm">{status?.state || "—"}</div>
                                </div>
                            </div>

                            {/* список стадий как карточки */}
                            {(() => {
                                const map = Object.fromEntries((status?.stages || []).map(s => [s.name, s]));
                                const current = status?.stage || null;
                                const items = STAGE_ORDER.map(name => {
                                    const meta = map[name];
                                    const isDone = !!meta || status?.state === "done";
                                    const isRunning = status?.state === "running" && current === name;
                                    const dur = meta?.duration_ms;
                                    const desc = name === "S0"
                                        ? (meta?.notes || STAGE_DESC.S0)
                                        : STAGE_DESC[name];

                                    const base =
                                        "flex items-start justify-between gap-4 rounded-2xl border px-5 py-4 md:px-6 md:py-5";
                                    const palette = isRunning
                                        ? "border-emerald-400/60 bg-emerald-400/5 ring-2 ring-emerald-400/40"
                                        : isDone
                                            ? "border-emerald-500/30 bg-white/0"
                                            : "border-emerald-500/20 bg-white/0";

                                    return (
                                        <div key={name} className={`${base} ${palette} mt-4`}>
                                            <div className="min-w-0">
                                                <div
                                                    className="text-emerald-200 font-semibold text-base md:text-lg">
                                                    {stageDisplayName(name)}
                                                </div>
                                                {desc && (
                                                    <div
                                                        className="text-emerald-200/70 text-sm md:text-[15px] mt-2">
                                                        {desc}
                                                    </div>
                                                )}
                                            </div>
                                            <div className="shrink-0 text-emerald-200/80 text-sm md:text-base">
                                                {isRunning ? "running…" : (dur != null ? `${(dur / 1000).toFixed(2)} s` : (isDone ? "done" : ""))}
                                            </div>
                                        </div>
                                    );
                                });

                                return <div className="mt-6 md:mt-8">{items}</div>;
                            })()}
                        </div>

                    </>
                )}
            </section>
            <div className="mx-auto mt-3 h-1 w-28 rounded-full bg-emerald-400/70 mb-8"/>
            {/* GRAPH: fades in and slides up when visible; progress panel stays above */}
            {status?.state && (
                <section
                    className={`relative h-[100vh] w-full px-2 pb-10 transition-all duration-700 ${graphVisible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"}`}>
                    <h2 className="sr-only">Knowledge Graph</h2>
                    <div ref={graphHostRef}
                         className="h-full w-full rounded-3xl border border-emerald-600/40 bg-slate-900/60 shadow-2xl"/>
                </section>
            )}
        </div>
    );
}

function SpinnerCircle({progress = 0, done = false}) {
    return (
        <div className="relative h-24 w-24">
            <div
                className={`absolute inset-0 rounded-full border-2 ${done ? "border-emerald-400" : "animate-spin border-emerald-400/30 border-t-emerald-400"}`}/>
            <div className="absolute inset-1 rounded-full border border-emerald-400/40"/>
            <div
                className="absolute inset-0 grid place-items-center text-sm font-semibold text-emerald-200">{done ? 100 : Math.round(progress)}%
            </div>
        </div>
    );
}

// Canvas-based animated Matrix rain
function MatrixRain() {
    const canvasRef = useRef(null);
    useEffect(() => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");
        let raf;
        const resize = () => {
            canvas.width = canvas.offsetWidth;
            canvas.height = canvas.offsetHeight;
        };
        const onResize = () => {
            resize();
            init();
        };
        resize();
        window.addEventListener("resize", onResize);


        const chars = "ACGT01".split("");
        const fontSize = 16;
        let columns = 0;
        let drops = [];
        let frame = 0;
        const SLOW = 7; // update every 5rd frame to slow down


        function init() {
            columns = Math.floor(canvas.width / fontSize);
            drops = new Array(columns).fill(1);
        }


        init();


        const draw = () => {
            // throttle frames to slow the rain down
            frame++;
            ctx.fillStyle = "rgba(2,8,23,0.18)";
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            if (frame % SLOW !== 0) {
                raf = requestAnimationFrame(draw);
                return;
            }
            ctx.fillStyle = "rgba(16,185,129,0.85)";
            ctx.font = `${fontSize}px monospace`;
            for (let i = 0; i < drops.length; i++) {
                const text = chars[Math.floor(Math.random() * chars.length)];
                ctx.fillText(text, i * fontSize, drops[i] * fontSize);
                if (drops[i] * fontSize > canvas.height && Math.random() > 0.99) drops[i] = 0;
                drops[i]++;
            }
            raf = requestAnimationFrame(draw);
        };
        draw();


        return () => {
            cancelAnimationFrame(raf);
            window.removeEventListener("resize", onResize);
        };
    }, []);
    return <canvas ref={canvasRef} className="h-full w-full"/>;
}
