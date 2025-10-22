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
            if (!cyRef.current) {
                cyRef.current = cytoscape({
                    container: host,
                    pixelRatio: 1,
                    wheelSensitivity: 0.2,
                    minZoom: 0.2,
                    maxZoom: 2,
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
                                "shadow-blur": 6,
                                "shadow-color": "#00000055",
                                "shadow-opacity": 0.6
                            }
                        },
                        {
                            selector: "edge",
                            style: {
                                width: 2,
                                opacity: 0.9,
                                "curve-style": "bezier",
                                "line-color": "#22c55e",
                                "target-arrow-color": "#22c55e",
                                "target-arrow-shape": "triangle",
                                "arrow-scale": 0.8,
                                label: "data(type)",
                                "font-size": 9,
                                "text-background-color": "#0b1220",
                                "text-background-opacity": 0.7,
                                "text-background-padding": 2
                            }
                        },
                        {selector: ".hl", style: {width: 3, opacity: 1}},
                    ],
                });
            }
            const cy = cyRef.current;
            const elements = [];
            const trunc = (s, n = 140) => (s && s.length > n ? s.slice(0, n).trim() + "…" : s || "");
            for (const n of data.nodes || []) elements.push({
                data: {
                    id: n.id,
                    label: trunc(n.text || ""),
                    type: n.type
                }
            });
            for (const e of data.edges || []) elements.push({
                data: {
                    id: `${e.from}->${e.to}:${e.type}`,
                    source: e.from,
                    target: e.to,
                    type: e.type
                }
            });
            cy.batch(() => {
                cy.elements().remove();
                cy.add(elements);
            });
            cy.layout({
                name: "cose",
                fit: true,
                padding: 40,
                animate: false,
                nodeRepulsion: 8000,
                gravity: 1,
                componentSpacing: 80,
                idealEdgeLength: 160
            }).run();
            cy.center();
            cy.fit(undefined, 50);
            cy.off("tap");
            cy.on("tap", "node", (ev) => {
                cy.elements("edge").removeClass("hl");
                ev.target.connectedEdges().addClass("hl");
            });
            cy.on("tap", (ev) => {
                if (ev.target === cy) cy.elements("edge").removeClass("hl");
            });
        } catch (e) {
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
                                    <div className="mt-1 text-xs text-emerald-200/70">or click to choose a file</div>
                                    {file && <div className="mt-3 truncate text-xs text-emerald-300">{file.name}</div>}
                                </div>
                            </label>
                        </div>

                        <div
                            className="mt-6 rounded-3xl border border-emerald-600/40 bg-slate-900/60 p-8 text-center shadow-xl">
                            <h3 className="text-xl font-semibold text-emerald-200">Ready?</h3>
                            <p className="mx-auto mt-2 max-w-md text-sm text-emerald-100/70">Upload a paper and we will
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
                                                <div className="text-emerald-200 font-semibold text-base md:text-lg">
                                                    {stageDisplayName(name)}
                                                </div>
                                                {desc && (
                                                    <div className="text-emerald-200/70 text-sm md:text-[15px] mt-2">
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
