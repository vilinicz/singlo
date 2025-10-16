// TestBench.jsx
import React, {useEffect, useMemo, useRef, useState} from "react";

/** База API (учитывает VITE_API_URL); "" = same-origin */
const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

/** Безопасно склеивает `${base}/${path}?q=...` без двойных слешей */
function apiUrl(path) {
    const p = path.startsWith("/") ? path : `/${path}`;
    return `${API}${p}`;
}

export default function TestBench() {
    const [subdir, setSubdir] = useState("");
    const [limit, setLimit] = useState(50);
    const [theme, setTheme] = useState("");
    const [loadingList, setLoadingList] = useState(false);
    const [loadingRun, setLoadingRun] = useState(false);
    const [pdfs, setPdfs] = useState([]);
    const [statusLines, setStatusLines] = useState([]);
    const [reportKey, setReportKey] = useState(0);
    const reportRef = useRef(null);

    const api = useMemo(() => {
        return {
            list: async (subdir) => {
                const q = subdir ? `?subdir=${encodeURIComponent(subdir)}` : "";
                const r = await fetch(apiUrl(`/test/list${q}`));
                if (!r.ok) throw new Error(await r.text());
                return r.json();
            },
            run: async (subdir, limit, theme) => {
                const p = new URLSearchParams();
                if (subdir) p.set("subdir", subdir);
                if (limit) p.set("limit", String(limit));
                if (theme) p.set("theme", theme);
                const r = await fetch(apiUrl(`/test/run?${p.toString()}`), {
                    method: "POST"
                });
                if (!r.ok) throw new Error(await r.text());
                return r.json();
            },
            reportUrl: (subdir) =>
                apiUrl(`/test/report${subdir ? `?subdir=${encodeURIComponent(subdir)}` : ""}`),
        };
    }, []);

    const pushStatus = (msg, type = "info") => {
        setStatusLines((s) => [
            {t: new Date().toLocaleTimeString(), type, msg},
            ...s,
        ]);
    };

    const onList = async () => {
        setLoadingList(true);
        try {
            const data = await api.list(subdir.trim());
            setPdfs(data.items || []);
            pushStatus(`Найдено PDF: ${data.count}`, "ok");
        } catch (e) {
            pushStatus(`list: ${String(e.message || e)}`, "err");
        } finally {
            setLoadingList(false);
        }
    };

    const onRun = async () => {
        setLoadingRun(true);
        try {
            const data = await api.run(subdir.trim(), limit, theme.trim());
            pushStatus(
                `Запущено задач: ${Array.isArray(data.processed) ? data.processed.length : 0}`,
                "ok"
            );
            setTimeout(() => setReportKey((k) => k + 1), 800);
        } catch (e) {
            pushStatus(`run: ${String(e.message || e)}`, "err");
        } finally {
            setLoadingRun(false);
        }
    };

    const onRefreshReport = () => {
        setReportKey((k) => k + 1);
        pushStatus("Обновляю отчёт…", "info");
    };

    useEffect(() => {
        onList().catch(() => {
        });
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    return (
        <div style={st.page}>
            <header style={st.header}>
                <div style={st.container}>
                    <h1 style={st.h1}>Singularis — Test page</h1>
                    <div style={st.apiBadge}>
                        API: <code style={st.code}>{API || "same-origin"}</code>
                    </div>
                </div>
            </header>

            <main style={st.container}>
                <section style={st.card}>
                    <div style={st.grid3}>
                        <div>
                            <Label>Подпапка в <Mono>/app/dataset</Mono> (опц.)</Label>
                            <Input
                                placeholder="например: arxiv/2023/09"
                                value={subdir}
                                onChange={(e) => setSubdir(e.target.value)}
                            />
                        </div>
                        <div>
                            <Label>Limit (run)</Label>
                            <Input
                                type="number"
                                min={1}
                                max={2000}
                                value={limit}
                                onChange={(e) => setLimit(Math.max(1, Number(e.target.value) || 1))}
                            />
                        </div>
                        <div>
                            <Label>Тема override (через запятую)</Label>
                            <Input
                                placeholder="пример: biomed,ml"
                                value={theme}
                                onChange={(e) => setTheme(e.target.value)}
                            />
                        </div>
                    </div>

                    <div style={st.btnRow}>
                        <Button onClick={onList} disabled={loadingList}>
                            {loadingList ? "Сканирую…" : "Список PDF"}
                        </Button>
                        <Button primary onClick={onRun} disabled={loadingRun}>
                            {loadingRun ? "Запускаю…" : "Запустить обработку"}
                        </Button>
                        <Button onClick={onRefreshReport}>Обновить отчёт</Button>
                    </div>
                </section>

                <div style={st.split}>
                    <section style={st.card}>
                        <h3 style={st.h3}>PDF в датасете</h3>
                        <div style={{overflow: "auto", maxHeight: 420}}>
                            <Table>
                                <thead>
                                <tr>
                                    <Th width={56}>#</Th>
                                    <Th>Путь (от <Mono>/app/dataset</Mono>)</Th>
                                </tr>
                                </thead>
                                <tbody>
                                {pdfs.slice(0, 2000).map((p, i) => (
                                    <tr key={p + i}>
                                        <Td width={56}>{i + 1}</Td>
                                        <Td><Mono>{p}</Mono></Td>
                                    </tr>
                                ))}
                                {pdfs.length === 0 && (
                                    <tr>
                                        <Td colSpan={2} style={{opacity: 0.7}}>
                                            Пусто. Нажми «Список PDF».
                                        </Td>
                                    </tr>
                                )}
                                </tbody>
                            </Table>
                        </div>
                    </section>

                    <section style={st.card}>
                        <h3 style={st.h3}>Статус</h3>
                        <div style={st.statusBox}>
                            {statusLines.length === 0 && <Mono>—</Mono>}
                            {statusLines.map((s, idx) => (
                                <div key={idx} style={st.statusLine}>
                                    <Mono>[{s.t}]</Mono>{" "}
                                    <Pill type={s.type}/>
                                    <span>{s.msg}</span>
                                </div>
                            ))}
                        </div>
                    </section>
                </div>

                <section style={st.card}>
                    <h3 style={st.h3}>Сводный отчёт</h3>
                    <iframe
                        key={reportKey}
                        ref={reportRef}
                        title="dataset-report"
                        style={st.iframe}
                        src={api.reportUrl(subdir.trim())}
                    />
                </section>
            </main>
        </div>
    );
}

/* ——— Small UI helpers ——— */
const Label = ({children}) => (
    <label style={{display: "block", fontSize: 12, color: "#9aa3ae", marginBottom: 6}}>
        {children}
    </label>
);

const Input = (props) => (
    <input
        {...props}
        style={{
            width: "100%",
            background: "#0f1320",
            color: "#e7e9ee",
            border: "1px solid #22283a",
            borderRadius: 10,
            padding: "10px 12px",
            outline: "none",
        }}
    />
);

const Button = ({primary, ...props}) => (
    <button
        {...props}
        style={{
            background: primary ? "#5B8CFF" : "#151a2b",
            border: `1px solid ${primary ? "#4a78e0" : "#22283a"}`,
            color: "#eef1f7",
            padding: "10px 12px",
            borderRadius: 10,
            cursor: props.disabled ? "not-allowed" : "pointer",
            opacity: props.disabled ? 0.6 : 1,
            transition: "transform .04s ease",
        }}
        onMouseDown={(e) => (e.currentTarget.style.transform = "translateY(1px)")}
        onMouseUp={(e) => (e.currentTarget.style.transform = "translateY(0)")}
    />
);

const Table = (props) => (
    <table {...props} style={{width: "100%", borderCollapse: "collapse", fontSize: 13}}/>
);
const Th = ({children, width}) => (
    <th
        style={{
            textAlign: "left",
            borderBottom: "1px solid #232a40",
            padding: "8px 10px",
            color: "#9aa3ae",
            fontWeight: 600,
            width,
        }}
    >
        {children}
    </th>
);
const Td = ({children, width, ...rest}) => (
    <td
        {...rest}
        style={{
            borderBottom: "1px solid #1d2336",
            padding: "8px 10px",
            width,
        }}
    >
        {children}
    </td>
);
const Mono = ({children}) => (
    <code style={{fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace"}}>
        {children}
    </code>
);
const Pill = ({type}) => {
    const base = {
        padding: "2px 8px",
        borderRadius: 9999,
        fontSize: 12,
        border: "1px solid transparent",
        margin: "0 6px",
    };
    if (type === "ok") return <span
        style={{...base, background: "#0b2a17", color: "#3ddc84", borderColor: "#145d33"}}>ok</span>;
    if (type === "err") return <span
        style={{...base, background: "#2a1215", color: "#ff6b6b", borderColor: "#6b1f25"}}>err</span>;
    if (type === "warn") return <span
        style={{...base, background: "#2a240b", color: "#f7c948", borderColor: "#6b5a1f"}}>warn</span>;
    return <span style={{...base, background: "#1a1f33", color: "#9aa3ae", borderColor: "#2a3250"}}>info</span>;
};

/* ——— Layout / theme ——— */
const st = {
    page: {background: "#0b0f1a", minHeight: "100vh", color: "#e7e9ee"},
    header: {
        padding: "14px 20px",
        borderBottom: "1px solid #1a2140",
        background: "linear-gradient(180deg,#0d1222 0%, #0b0f1a 100%)",
        position: "sticky",
        top: 0,
        zIndex: 10,
    },
    container: {maxWidth: 1200, margin: "0 auto", padding: "16px 20px"},
    h1: {margin: 0, fontSize: 18, letterSpacing: ".2px"},
    apiBadge: {
        marginTop: 6,
        fontSize: 12,
        color: "#9aa3ae",
    },
    card: {
        background: "#111628",
        border: "1px solid #1c2442",
        borderRadius: 12,
        padding: 14,
        marginBottom: 14,
        overflow: "hidden",
    },
    grid3: {
        display: "grid",
        gridTemplateColumns: "1fr 160px 300px",
        gap: 12,
    },
    btnRow: {display: "flex", gap: 8, marginTop: 12, flexWrap: "wrap"},
    split: {display: "grid", gridTemplateColumns: "1.1fr .9fr", gap: 14},
    h3: {margin: "4px 0 10px", fontSize: 14, color: "#cbd3df"},
    statusBox: {
        background: "#0f1320",
        border: "1px solid #1c2442",
        borderRadius: 10,
        padding: "10px 12px",
        minHeight: 160,
        maxHeight: 420,
        overflow: "auto",
    },
    statusLine: {marginBottom: 6, display: "flex", alignItems: "center", gap: 6},
    iframe: {
        width: "100%",
        height: "72vh",
        border: "1px solid #1c2442",
        borderRadius: 10,
        background: "#0f1320",
        color: "#fff"
    },
};
