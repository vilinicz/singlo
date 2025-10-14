import React, {useEffect, useRef, useState} from "react";
import axios from "axios";
import cytoscape from "cytoscape";
import "./App.css";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

// ─── constants ─────────────────────────────────────────────
const COLS = [
    "Input Fact", "Hypothesis", "Experiment", "Technique",
    "Result", "Dataset", "Analysis", "Conclusion"
];

const EDGE_STYLES = {
    supports: {color: "#16a34a", label: "Result → Hypothesis"},
    refutes: {color: "#ef4444", label: "Result ↛ Hypothesis"},
    produces: {color: "#7c3aed", label: "Experiment → Result"},
    uses: {color: "#0f766e", label: "Technique → * / * → Analysis"},
    feeds: {color: "#2563eb", label: "Dataset → Experiment"},
    informs: {color: "#0284c7", label: "Result/Dataset → Analysis"},
    summarizes: {color: "#a855f7", label: "Analysis → Conclusion"},
    used_by: {color: "#94a3b8", label: "reverse use"}
};

const STAGES = ["S0", "S1", "S2", "finalize"];

// ─── helpers: headers & legend overlays ────────────────────
function addColumnHeaderNodes(cy) {
    const grid = makeGrid(cy);

    // найдём самый верхний Y реальных нод, чтобы заголовки были чуть выше
    const ys = cy.nodes().map(n => n.position("y"));
    const topY = (ys.length ? Math.min(...ys) : 120) - 70;

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
        } else {
            toAdd.push({data, position: pos, selectable: false, grabbable: false});
        }
    }
    if (toAdd.length) cy.add(toAdd);

    cy.style().selector('node[type = "Header"]').style({
        "shape": "round-rectangle",
        "background-color": "#ffffff",
        "border-color": "#e5e7eb",
        "border-width": 1,
        "label": "data(label)",
        "font-size": "12px",
        "font-weight": "600",
        "text-halign": "center",
        "text-valign": "center",
        "padding": "6px 10px",
        "border-radius": "8px",
        "events": "no",
        "shadow-blur": 8,
        "shadow-color": "#0000001a",
        "shadow-opacity": 1,
        "shadow-offset-x": 0,
        "shadow-offset-y": 2,
        "color": "#374151",
        "text-outline-width": 3,
        "text-outline-color": "#ffffff"
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
        ["supports", "Результат подтверждает гипотезу (Result → Hypothesis)"],
        ["refutes", "Результат опровергает гипотезу (Result ↛ Hypothesis)"],
        ["produces", "Эксперимент приводит к результату (Experiment → Result)"],
        ["uses", "Техника используется (Technique → *) или применена в анализе (* → Analysis)"],
        ["feeds", "Датасет подаётся в эксперимент (Dataset → Experiment)"],
        ["informs", "Результат/датасет информирует анализ (Result/Dataset → Analysis)"],
        ["summarizes", "Анализ суммируется в вывод (Analysis → Conclusion)"],
        ["used_by", "Обратная зависимость (reverse use)"],
    ];

    box.innerHTML = `
    <div class="legend-title">Легенда связей</div>
    ${items.map(([k, text]) => {
        const color = (EDGE_STYLES[k] || {}).color || "#64748b";
        return `
        <div class="legend-row">
          <span class="swatch" style="background:${color}"></span>
          <span class="legend-text">${text}</span>
        </div>
      `;
    }).join("")}
  `;
}

// фиксированная сетка: колонки/ряды сразу в нужных местах
function makeGrid(cy) {
    const host = cy.container();
    const W = host?.clientWidth || 1200;
    const H = host?.clientHeight || 700;

    const COL_LEFT = 120;                 // левый отступ
    const COL_RIGHT = 120;                // правый отступ
    const COLS_COUNT = 8;
    const COL_STEP = Math.max(340, (W - COL_LEFT - COL_RIGHT) / (COLS_COUNT - 1)); // шире = больше первое число

    const ROW_TOP = 250;                  // отступ сверху для первой строки
    const ROW_STEP = 100;                 // межстрочный шаг

    const colX = (idx) => COL_LEFT + idx * COL_STEP;
    const rowY = (idx) => ROW_TOP + idx * ROW_STEP;

    return {colX, rowY};
}

// хелпер для docId по имени файла
function suggestIdFromFile(file) {
    if (!file?.name) return "";
    const stem = file.name.replace(/\.[^.]+$/, "");
    const slug = stem.toLowerCase().replace(/[^a-z0-9_-]+/g, "-").replace(/^-+|-+$/g, "");
    const rnd = Math.random().toString(36).slice(2, 8);
    return `${slug || "doc"}-${rnd}`;
}

export default function App() {
    const [s1Preview, setS1Preview] = useState(null);
    const cyRef = useRef(null);
    const [docId, setDocId] = useState("");
    const [file, setFile] = useState(null);
    const [status, setStatus] = useState(null);
    const [polling, setPolling] = useState(false);
    const [s0Preview, setS0Preview] = useState(null);
    const [graphPreview, setGraphPreview] = useState(null);
    const [s2Preview, setS2Preview] = useState(null);
    const [themes, setThemes] = useState([]);
    const [themeSel, setThemeSel] = useState("auto"); // "auto" | "biomed" | ...

    // init cytoscape
    useEffect(() => {
        if (!cyRef.current) {
            cyRef.current = cytoscape({
                container: document.getElementById("cy"),
                style: [
                    {selector: "node[type='Hypothesis']", style: {"background-color": "#7c3aed"}},
                    {selector: "node[type='Experiment']", style: {"background-color": "#2563eb"}},
                    {selector: "node[type='Result']", style: {"background-color": "#16a34a"}},
                    {selector: "node[type='Method']", style: {"background-color": "#374151"}},
                    {selector: "node[type='Conclusion']", style: {"background-color": "#f59e0b"}},
                    {
                        selector: "edge[type='supports']",
                        style: {
                            "line-color": "#16a34a",
                            "target-arrow-color": "#16a34a",
                            "target-arrow-shape": "triangle"
                        }
                    },
                    {
                        selector: "edge[type='refutes']",
                        style: {
                            "line-color": "#ef4444",
                            "target-arrow-color": "#ef4444",
                            "target-arrow-shape": "triangle"
                        }
                    },
                    {
                        selector: "edge[type='produces']",
                        style: {
                            "line-color": "#2563eb",
                            "target-arrow-color": "#2563eb",
                            "target-arrow-shape": "triangle"
                        }
                    },
                    {
                        selector: "edge[type='uses']",
                        style: {
                            "line-color": "#6b7280",
                            "target-arrow-color": "#6b7280",
                            "target-arrow-shape": "triangle"
                        }
                    },
                    {selector: "edge", style: {width: 2, "curve-style": "bezier"}},
                ],
                layout: {name: "cose", animate: false},
            });
        }
    }, []);

    async function loadS1Preview(idOverride) {
        const id = idOverride || docId;
        if (!id) return;
        try {
            const {data} = await axios.get(`${API}/preview/${encodeURIComponent(id)}/s1`);
            setS1Preview(data.preview);
        } catch (e) {
            console.error("loadS1Preview failed", e);
        }
    }

    async function loadS2Preview(idOverride) {
        const id = idOverride || docId;
        if (!id) return;
        try {
            const {data} = await axios.get(`${API}/preview/${encodeURIComponent(id)}/s2`);
            setS2Preview(data.preview);
        } catch (e) {
            console.error("loadS2Preview failed", e);
        }
    }

    function onPickFile(f) {
        setFile(f);
        if (!docId) setDocId(suggestIdFromFile(f));      // автоподстановка
    }

    async function uploadPdf() {
        if (!file) {
            alert("Choose a PDF");
            return;
        }
        const form = new FormData();
        form.append("file", file);

        const {data} = await axios.post(
            `${API}/parse?theme=${encodeURIComponent(themeSel)}`,
            form,
            {headers: {"Content-Type": "multipart/form-data"}}
        );

        const id = data?.doc_id;
        if (!id) {
            alert("No doc_id from server");
            return;
        }

        setDocId(id);
        setPolling(true);           // ← сразу начинаем следить за статусом
        await loadStatus(id);       // первый снимок
        // s0.json появится, когда закончится S0 — фронт сам подтянет в следующем poll
    }


    function highlightNodeEdges(cy, node) {
        // снять прошлое
        cy.elements('edge.hl').removeClass('hl');
        cy.nodes('.active').removeClass('active').unselect();

        if (!node || node.empty()) return;

        // активируем узел
        node.addClass('active').select();

        // подсветить ВСЕ рёбра, инцидентные узлу
        const inc = node.connectedEdges();
        if (inc && inc.length) inc.addClass('hl');
    }

    function clearHighlight(cy) {
        cy.elements('edge.hl').removeClass('hl');
        cy.nodes('.active').removeClass('active').unselect();
    }

    function bindInteractions(cy) {
        let clickTimer = null;
        let lastId = null;
        let lastTs = 0;
        const DOUBLE_MS = 280;

        cy.off('tap');            // очистим старые бинды
        cy.off('tap', 'node');
        cy.off('tap', 'core');

        // одиночный / двойной клик по ноде
        cy.on('tap', 'node', (ev) => {
            const n = ev.target;
            const now = Date.now();

            if (lastId === n.id() && (now - lastTs) < DOUBLE_MS) {
                // двойной клик
                clearTimeout(clickTimer);
                clickTimer = null;
                lastId = null;
                lastTs = 0;

                const d = n.data();
                alert(`${d.type}\n\n${d.text || "(no text)"}\n\nconf=${(d.conf ?? 0).toFixed(2)}\n${d.id}`);
                return;
            }

            // готовим одиночный клик
            lastId = n.id();
            lastTs = now;
            clearTimeout(clickTimer);
            clickTimer = setTimeout(() => {
                highlightNodeEdges(cy, n);
                clickTimer = null;
                lastId = null;
                lastTs = 0;
            }, DOUBLE_MS);
        });

        // клик по пустому месту — снятие подсветки
        cy.on('tap', (ev) => {
            if (ev.target === cy) {
                clearTimeout(clickTimer);
                clearHighlight(cy);
                lastId = null;
                lastTs = 0;
                clickTimer = null;
            }
        });
    }

    // --- дайте функциям опциональный id ---
    async function loadStatus(idOverride) {
        const id = idOverride || docId;
        if (!id) return;
        const {data} = await axios.get(`${API}/status/${encodeURIComponent(id)}`).catch(() => ({data: null}));
        if (data) setStatus(data);
    }

    async function loadS0Preview(idOverride) {
        const id = idOverride || docId;
        if (!id) return;
        const {data} = await axios.get(`${API}/preview/${encodeURIComponent(id)}/s0`);
        setS0Preview(data.preview);
    }

    async function loadGraphPreview(idOverride) {
        const id = idOverride || docId;
        if (!id) return;
        const {data} = await axios.get(`${API}/preview/${encodeURIComponent(id)}/graph`);
        setGraphPreview(data.preview);
    }

    async function loadGraph(idOverride) {
        const id = idOverride || docId;
        if (!id) return;

        try {
            const {data} = await axios.get(`${API}/graph/${encodeURIComponent(id)}`);

            // helpers
            const trunc = (s, n = 140) => (s && s.length > n ? s.slice(0, n).trim() + "…" : (s || ""));
            const elements = [];
            const have = new Set();

            const cy = cyRef.current;
            if (!cy) return;
            const grid = makeGrid(cy);

            // nodes: label = only text (без типа) + сразу правильные позиции
            for (const n of data.nodes || []) {
                const col = Number.isFinite(n.data?.col) ? n.data.col : null;
                const row = Number.isFinite(n.data?.row) ? n.data.row : null;

                elements.push({
                    data: {
                        id: n.id,
                        type: n.type,
                        text: n.text || "",
                        conf: n.conf ?? 0,
                        col, row,
                        label: trunc(n.text || "")
                    },
                    position: (col != null && row != null)
                        ? {x: grid.colX(col), y: grid.rowY(row)}
                        : (n.position || undefined),            // если нет col/row — уважаем то, что пришло
                    selectable: true,
                    grabbable: false,
                });
                have.add(n.id);
            }

            // edges: threshold + fan-out + wavy control points
            const EDGE_CONF_MIN = 0.55;
            const MAX_FAN_OUT = 3;
            const outCnt = {};
            for (const e of (data.edges || [])) {
                const conf = +e.conf || 0;
                if (conf < EDGE_CONF_MIN) continue;
                if (!have.has(e.from) || !have.has(e.to)) continue;

                const k = e.from;
                outCnt[k] = (outCnt[k] || 0);
                if (outCnt[k] >= MAX_FAN_OUT) continue;
                outCnt[k]++;

                // лёгкая «волнистость»: три контрольные точки
                const cpd = [-40, 40, -40];           // distances
                const cpw = [0.25, 0.5, 0.75];        // weights (позиции)
                const style = EDGE_STYLES[e.type] || {color: "#64748b"};

                elements.push({
                    data: {
                        id: `${e.from}->${e.to}:${e.type}`,
                        source: e.from,
                        target: e.to,
                        type: e.type,
                        conf,
                        cpd, cpw, color: style.color
                    },
                    selectable: true,
                });
            }

            // paint
            cy.batch(() => {
                cy.elements().remove();
                cy.add(elements);
            });

            addColumnHeaderNodes(cy);
            renderLegend(cy);

            // styles
            cy.style().fromJson([
                // base nodes
                {
                    selector: "node",
                    style: {
                        "background-color": "#e5e7eb",
                        "label": "data(label)",
                        "text-wrap": "wrap",
                        "text-max-width": "280px",
                        "font-size": "12px",
                        "text-halign": "center",
                        "text-valign": "center",
                        "shape": "round-rectangle",
                        "padding": "18px",                 // ↑ padding
                        "border-width": 1,
                        "border-color": "#1f2937",
                        "width": "label",
                        "height": "label",
                        "shadow-blur": 8,
                        "shadow-color": "#00000022",
                        "shadow-opacity": 0.5,
                        "shadow-offset-x": 0,
                        "shadow-offset-y": 1
                    }
                },
                // node colors by type (как раньше)
                {selector: "node[type = 'Input Fact']", style: {"background-color": "#b8c1ec"}},
                {selector: "node[type = 'Hypothesis']", style: {"background-color": "#ffd166"}},
                {selector: "node[type = 'Experiment']", style: {"background-color": "#ef476f"}},
                {selector: "node[type = 'Technique']", style: {"background-color": "#06d6a0"}},
                {selector: "node[type = 'Result']", style: {"background-color": "#118ab2"}},
                {selector: "node[type = 'Dataset']", style: {"background-color": "#c77dff"}},
                {selector: "node[type = 'Analysis']", style: {"background-color": "#73d2de"}},
                {selector: "node[type = 'Conclusion']", style: {"background-color": "#83c5be"}},

                // edges: color per type + wavy
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
                        // ↓↓↓ чёрная обводка (underlay)
                        "underlay-color": "#212121",
                        "underlay-opacity": 0,
                        "underlay-padding": 1,
                    }
                },
                {
                    selector: "edge.hl", style: {
                        "opacity": 1,
                        "width": 3,
                        "line-color": "data(color)",
                        "target-arrow-color": "data(color)",
                        "z-index-compare": "manual",
                        "z-index": 9999,
                        // ↓↓↓ чёрная обводка (underlay)
                        "underlay-opacity": 1,
                        "underlay-padding": 2,    // толщина «канта»
                        // чуть крупнее стрелка — так контрастнее
                        "target-arrow-shape": "triangle",
                        "arrow-scale": 1.2,
                        "transition-property": "opacity, width, underlay-padding",
                        "transition-duration": "120ms"
                    }
                },
                {selector: "node:selected", style: {"border-width": 2, "border-color": "#000"}},
                {selector: "edge:selected", style: {"line-color": "#000", "target-arrow-color": "#000", "width": 2}},
            ]).update();

            // layout: preset (S2 даёт позиции по колонкам)
            cy.layout({name: "preset", fit: true, padding: 32}).run();

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

            cy.off("tap", "node");
            bindInteractions(cy);
        } catch (e) {
            console.error("loadGraph failed", e);
        }
    }

    useEffect(() => {
        (async () => {
            try {
                const {data} = await axios.get(`${API}/themes`);
                setThemes(data.themes || []);
            } catch (e) {
                console.error("load themes failed", e);
                setThemes([]);
            }
        })();
    }, []);

    useEffect(() => {
        if (!polling) return;
        const id = setInterval(async () => {
            await loadStatus();
            const st = await (async () => {
                try {
                    const {data} = await axios.get(`${API}/status/${encodeURIComponent(docId)}`);
                    return data;
                } catch {
                    return null;
                }
            })();
            if (st?.state === "done" || st?.state === "error") {
                setPolling(false);
                await loadS0Preview();
                await loadS1Preview();
                await loadS2Preview();
                await loadGraphPreview();
                await loadGraph();
            }
        }, 1000);
        return () => clearInterval(id);
    }, [polling, docId]);

    // прогресс по стадиям
    function progressValue() {
        if (!status || !status.stage) return 0;
        const idx = STAGES.indexOf(status.stage);
        if (idx < 0) return 0;
        const doneIdx = status.state === "done" ? STAGES.length : idx + 1;
        return Math.min(100, Math.round((doneIdx / STAGES.length) * 100));
    }

    return (
        <div style={{
            fontFamily: "system-ui",
            padding: 12,
            display: "grid",
            gridTemplateColumns: "360px 1fr",
            gap: 16,
            maxWidth: "100vw",
            overflowX: "hidden"
        }}>
            <div style={{minWidth: 0}}>
                <h2>Singularis Demo</h2>
                <div style={{marginBottom: 8}}>
                    <label>Doc ID:&nbsp;</label>
                    <input value={docId} onChange={(e) => setDocId(e.target.value)} placeholder="auto after Parse"
                           style={{width: "220px"}}/>
                </div>
                <div className="flex items-center gap-2">
                    <label className="text-sm text-gray-600">Theme:</label>
                    <select
                        value={themeSel}
                        onChange={(e) => setThemeSel(e.target.value)}
                        className="border rounded px-2 py-1 text-sm"
                    >
                        <option value="auto">Auto (recommended)</option>
                        {themes.map(t => <option key={t} value={t}>{t}</option>)}
                    </select>
                </div>
                <div style={{marginBottom: 8}}>
                    <input type="file" accept="application/pdf" disabled={status?.state === "running"}
                           onChange={(e) => onPickFile(e.target.files?.[0] || null)}/>
                    <button onClick={uploadPdf} style={{marginLeft: 8}}>Parse (S0)</button>
                </div>
                <div style={{marginBottom: 8}}>
                    {status?.state === "queued" || status?.state === "running"
                        ? <div style={{fontSize: 12, opacity: .8}}>Pipeline is running asynchronously…</div>
                        : null
                    }
                    <button onClick={() => setPolling(true)} style={{marginLeft: 8}}>Refresh status</button>
                    <button
                        onClick={async () => {
                            if (!docId) {
                                alert("Set Doc ID");
                                return;
                            }
                            await axios.post(`${API}/extract?doc_id=${encodeURIComponent(docId)}&theme=${encodeURIComponent(themeSel)}`);
                            setPolling(true);
                        }}
                        style={{marginLeft: 8}}
                    >
                        Extract (S1+S2)
                    </button>
                </div>

                <div style={{margin: "12px 0"}}>
                    <div style={{fontSize: 12, marginBottom: 4}}>Progress: {progressValue()}%</div>
                    <div style={{height: 10, background: "#eee", borderRadius: 6, overflow: "hidden"}}>
                        <div style={{width: `${progressValue()}%`, height: "100%", background: "#3b82f6"}}/>
                    </div>
                </div>

                <div style={{
                    fontSize: 12,
                    lineHeight: 1.5,
                    background: "#fafafa",
                    border: "1px solid #eee",
                    borderRadius: 8,
                    padding: 8
                }}>
                    <div><b>State:</b> {status?.state || "—"}</div>
                    <div><b>Stage:</b> {status?.stage || "—"}</div>
                    <div><b>Total:</b> {status?.duration_ms ? `${(status.duration_ms / 1000).toFixed(2)} s` : "—"}</div>
                    <div style={{marginTop: 6}}>
                        <b>Stages:</b>
                        <ul style={{margin: 0, paddingLeft: 16}}>
                            {(status?.stages || []).map((s, i) => (
                                <li key={i}>
                                    {s.name}: {(s.duration_ms / 1000).toFixed(2)} s {s.notes ? `— ${s.notes}` : ""}
                                </li>
                            ))}
                        </ul>
                    </div>
                </div>

                <div style={{marginTop: 12}}>
                    <details>
                        <summary>S0 preview (JSON)</summary>
                        <pre style={{
                            whiteSpace: "pre-wrap",
                            fontSize: 12,
                            maxHeight: 220,
                            overflow: "auto",
                            background: "#111",
                            color: "#ddd",
                            padding: 8,
                            borderRadius: 6
                        }}>
{s0Preview || "—"}
            </pre>
                    </details>
                    <details style={{marginTop: 8}}>
                        <summary>S1 preview (JSON)</summary>
                        <pre style={{
                            whiteSpace: "pre-wrap",
                            fontSize: 12,
                            maxHeight: 220,
                            overflow: "auto",
                            background: "#111",
                            color: "#ddd",
                            padding: 8,
                            borderRadius: 6
                        }}>
{s1Preview || "—"}
  </pre>
                    </details>
                    <details style={{marginTop: 8}}>
                        <summary>S2 preview (JSON)</summary>
                        <pre style={{
                            whiteSpace: "pre-wrap", fontSize: 12, maxHeight: 220, overflow: "auto",
                            background: "#111", color: "#ddd", padding: 8, borderRadius: 6
                        }}>
{s2Preview || "—"}
  </pre>
                    </details>

                    <details style={{marginTop: 8}}>
                        <summary>Graph preview (JSON)</summary>
                        <pre style={{
                            whiteSpace: "pre-wrap",
                            fontSize: 12,
                            maxHeight: 220,
                            overflow: "auto",
                            background: "#111",
                            color: "#ddd",
                            padding: 8,
                            borderRadius: 6
                        }}>
{graphPreview || "—"}
            </pre>
                    </details>
                </div>
            </div>

            <div style={{minWidth: 0}}>
                <h3>Graph</h3>
                <div id="cy" style={{
                    width: "100%",
                    maxWidth: "100%",
                    height: "78vh",
                    border: "1px solid #ddd",
                    borderRadius: 8
                }}/>
            </div>
        </div>
    );
}
