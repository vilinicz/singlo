import React, {useCallback, useEffect, useMemo, useRef, useState} from "react";
import axios from "axios";
import cytoscape from "cytoscape";

// One‑pager showcase for the Singularis pipeline (Tailwind UI)
// Updates in this version:
// 1) Matrix rain animation (canvas)
// 2) "Ready" centered and larger CTA
// 3) Progress section stable; indicator stops spinning when done
// 4) Stages laid out VERTICALLY, active stage highlighted
// 5) Smooth fade/translate transition to graph section; progress stays above

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";
const STAGES_ORDER = ["S0", "S1", "S2", "finalize"]; // expected order

export default function SingularisShowcase() {
    const [file, setFile] = useState(null);
    const [docId, setDocId] = useState("");
    const [status, setStatus] = useState(null); // { state, stage, stages: [{name,duration_ms,notes}] }
    const [polling, setPolling] = useState(false);
    const [graphVisible, setGraphVisible] = useState(false);
    const cyRef = useRef(null);
    const graphHostRef = useRef(null);

    // total duration for finalize display
    const totalDurationMs = useMemo(() => (status?.stages || []).reduce((s, x) => s + (x?.duration_ms || 0), 0), [status]);

    // progress in %
    const pipelineProgress = useMemo(() => {
        if (!status) return 0;
        if (status.state === "done") return 100;
        const idx = STAGES_ORDER.indexOf(status.stage);
        const doneIdx = Math.max(0, idx + (status.state === "running" ? 1 : 0));
        return Math.min(100, Math.round((doneIdx / STAGES_ORDER.length) * 100));
    }, [status]);

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
                        setTimeout(() => {
                            renderGraph();
                            setGraphVisible(true);
                            graphHostRef.current?.scrollIntoView({behavior: "smooth", block: "start"});
                        }, 250);
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

    const StageCard = ({stage, active, duration, notes}) => (
        <div
            className={`transition-all rounded-2xl border border-emerald-500/40 bg-slate-900/60 px-5 py-4 shadow-xl ${active ? "scale-[1.03] ring-2 ring-emerald-400/70" : "opacity-85"}`}>
            <div className="flex items-center justify-between gap-2">
                <div
                    className={`font-semibold tracking-wide ${active ? "text-emerald-300" : "text-emerald-200"}`}>{stage}</div>
                <div
                    className="text-xs text-emerald-200/70">{duration ? `${(duration / 1000).toFixed(2)} s` : "—"}</div>
            </div>
            <p className="mt-2 line-clamp-4 text-xs leading-relaxed text-emerald-100/80 min-h-10">{notes || descriptionFor(stage)}</p>
        </div>
    );

    const descriptionFor = (stage) => ({
        S0: "Parsing PDF → text/blocks with coordinates (TEI/JSON).",
        S1: "Heuristic extraction of atoms (Hypothesis/Method/Result/…).",
        S2: "Building typed edges, layout hints, dedup & cleaning.",
        finalize: "Packaging previews and final graph payload.",
    }[stage] || "Processing…");

    return (
        <div className="min-h-screen w-full bg-slate-950 text-emerald-100">
            {/* Matrix rain background */}
            <div
                className="pointer-events-none fixed inset-0 -z-20 bg-[radial-gradient(ellipse_at_top,rgba(16,185,129,0.08),transparent_60%),radial-gradient(ellipse_at_bottom,rgba(16,185,129,0.06),transparent_60%)]"/>
            <div
                className="pointer-events-none fixed inset-0 -z-10 opacity-[0.15] [mask-image:linear-gradient(to_bottom,black,transparent)]">
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
                             className="flex min-h-[260px] items-center justify-center rounded-3xl border-2 border-dashed border-emerald-500/50 bg-slate-900/50 p-6 text-center shadow-2xl">
                            <label className="w-full cursor-pointer">
                                <input type="file" accept="application/pdf" className="hidden" onChange={onBrowse}/>
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
                    <div
                        className="mx-auto mt-6 max-w-6xl rounded-3xl border border-emerald-600/40 bg-slate-900/60 p-6 shadow-2xl">
                        <div className="flex flex-col gap-6 xl:flex-row xl:items-start">
                            {/* Vertical stages */}
                            <div className="flex flex-1 flex-col gap-4">
                                {STAGES_ORDER.map((s) => {
                                    const rec = status?.stages?.find((x) => x.name === s);
                                    const active = status?.stage === s && status?.state === "running";
                                    const dur = s === "finalize" ? (totalDurationMs || rec?.duration_ms) : rec?.duration_ms;
                                    return <StageCard key={s} stage={s} active={active} duration={dur}
                                                      notes={rec?.notes}/>;
                                })}
                            </div>
                            {/* Gauge */}
                            <div className="flex w-full max-w-[240px] flex-col items-center justify-center gap-3">
                                <SpinnerCircle progress={pipelineProgress} done={status?.state === "done"}/>
                                <div className="text-xs text-emerald-200/80">{pipelineProgress}%</div>
                                <div
                                    className="text-[11px] text-emerald-200/60">{status?.state === "running" ? `Processing: ${status?.stage}` : status?.state || "—"}</div>
                            </div>
                        </div>
                    </div>
                )}
            </section>
            <div className="mx-auto mt-3 h-1 w-28 rounded-full bg-emerald-400/70"/>
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
        const fontSize = 14;
        let columns = 0;
        let drops = [];

        function init() {
            columns = Math.floor(canvas.width / fontSize);
            drops = new Array(columns).fill(1);
        }

        init();

        const draw = () => {
            ctx.fillStyle = "rgba(2,8,23,0.25)"; // fade trail
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = "rgba(16,185,129,0.85)";
            ctx.font = `${fontSize}px monospace`;
            for (let i = 0; i < drops.length; i++) {
                const text = chars[Math.floor(Math.random() * chars.length)];
                ctx.fillText(text, i * fontSize, drops[i] * fontSize);
                if (drops[i] * fontSize > canvas.height && Math.random() > 0.975) drops[i] = 0;
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
