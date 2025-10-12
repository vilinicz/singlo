import React, {useEffect, useRef, useState} from "react";
import axios from "axios";
import cytoscape from "cytoscape";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

const STAGES = ["S0", "S1", "S2", "finalize"];

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

        const {data} = await axios.post(`${API}/parse`, form, {
            headers: {"Content-Type": "multipart/form-data"},
        });

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

    async function startExtract() {
        const {data} = await axios.post(`${API}/extract?doc_id=${encodeURIComponent(docId)}`);
        setPolling(true);
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
            const elements = [];
            for (const n of data.nodes || []) {
                elements.push({data: {id: n.id, label: n.type, type: n.type, text: n.text}});
            }
            for (const e of data.edges || []) {
                elements.push({data: {id: `${e.from}->${e.to}:${e.type}`, source: e.from, target: e.to, type: e.type}});
            }
            const cy = cyRef.current;
            cy.elements().remove();
            cy.add(elements);
            cy.layout({name: "cose", animate: false}).run();
            cy.off("tap", "node");
            cy.on("tap", "node", (ev) => {
                const d = ev.target.data();
                alert(`${d.id}\n${d.type}\n\n${d.text || ""}`);
            });
        } catch (e) {
            console.error("loadGraph failed", e); // 👈 не молчим
        }
    }

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
        <div style={{fontFamily: "system-ui", padding: 12, display: "grid", gridTemplateColumns: "360px 1fr", gap: 16}}>
            <div>
                <h2>Singularis Demo</h2>
                <div style={{marginBottom: 8}}>
                    <label>Doc ID:&nbsp;</label>
                    <input value={docId} onChange={(e) => setDocId(e.target.value)} placeholder="auto after Parse"
                           style={{width: "220px"}}/>
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

            <div>
                <h3>Graph</h3>
                <div id="cy" style={{width: "100%", height: "78vh", border: "1px solid #ddd", borderRadius: 8}}/>
            </div>
        </div>
    );
}
