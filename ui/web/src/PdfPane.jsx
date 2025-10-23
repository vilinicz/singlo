// PdfPane.jsx — v5 clean, no flicker on highlight
import React, {useEffect, useRef, useState, forwardRef, useImperativeHandle} from "react";
import * as pdfjsLib from "pdfjs-dist/build/pdf";
import "pdfjs-dist/web/pdf_viewer.css";
import workerSrc from "pdfjs-dist/build/pdf.worker.mjs?url";

pdfjsLib.GlobalWorkerOptions.workerSrc = workerSrc;

const PdfPane = forwardRef(function PdfPane({pdfUrl, onClose}, ref) {
    const wrapRef = useRef(null);
    const canvasRef = useRef(null);
    const textLayerRef = useRef(null);
    const overlayRef = useRef(null);

    const [pdfDoc, setPdfDoc] = useState(null);
    const [pageNum, setPageNum] = useState(1);
    const [scale, setScale] = useState(1.3);
    const [numPages, setNumPages] = useState(0);

    const viewportRef = useRef(null);
    const textLayerReadyRef = useRef(Promise.resolve());
    const suppressEffectOnceRef = useRef(false); // ← не дать useEffect «стереть» подсветку

    // ------- utils -------
    const normalizeRects = (rects) => {
        if (!rects) return [];
        if (Array.isArray(rects) && rects.length && typeof rects[0] === "object" && !Array.isArray(rects[0])) {
            return rects.map(({x, y, w, h}) => [x, y, x + w, y + h]); // coords → bbox
        }
        if (Array.isArray(rects) && Array.isArray(rects[0])) return rects;
        if (Array.isArray(rects) && rects.length === 4) return [rects];
        if (rects && typeof rects === "object") {
            if ("x0" in rects) {
                const {x0, y0, x1, y1} = rects;
                return [[x0, y0, x1, y1]];
            }
            if ("x" in rects) {
                const {x, y, w, h} = rects;
                return [[x, y, x + w, y + h]];
            }
        }
        return [];
    };

    const renderTextLayerFallback = (viewport, textContent, container) => {
        container.innerHTML = "";
        for (const item of (textContent?.items || [])) {
            const tx = pdfjsLib.Util.transform(
                pdfjsLib.Util.transform(viewport.transform, item.transform),
                [1, 0, 0, -1, 0, 0]
            );
            const left = tx[4], top = tx[5];
            const fontSize = Math.hypot(tx[0], tx[1]);
            const span = document.createElement("span");
            span.textContent = item.str;
            Object.assign(span.style, {
                position: "absolute",
                left: `${left}px`,
                top: `${top - fontSize}px`,
                fontSize: `${fontSize}px`,
                whiteSpace: "pre",
                lineHeight: 1,
            });
            container.appendChild(span);
        }
    };

    // ------- load -------
    useEffect(() => {
        if (!pdfUrl) return;
        let cancelled = false;

        pdfjsLib.getDocument({
            url: pdfUrl,
            disableStream: true,
            disableRange: true,
            verbosity: pdfjsLib.VerbosityLevel.ERRORS,
        }).promise.then((doc) => {
            if (cancelled) return;
            setPdfDoc(doc);
            setNumPages(doc.numPages || 0);
        }).catch((e) => console.error("PDF load error:", e));

        return () => {
            cancelled = true;
        };
    }, [pdfUrl]);

    // ------- render page (single path) -------
    const renderPage = async (pageN, targetScale = scale) => {
        if (!pdfDoc) return;
        const page = await pdfDoc.getPage(pageN);
        const viewport = page.getViewport({scale: targetScale, rotation: page.rotate || 0});
        viewportRef.current = viewport;

        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d", {alpha: false});
        canvas.width = Math.floor(viewport.width);
        canvas.height = Math.floor(viewport.height);
        canvas.style.width = `${canvas.width}px`;
        canvas.style.height = `${canvas.height}px`;

        await page.render({canvasContext: ctx, viewport}).promise;

        const textLayerDiv = textLayerRef.current;
        const overlay = overlayRef.current;
        const textContent = await page.getTextContent();

        textLayerDiv.innerHTML = "";
        overlay.innerHTML = "";
        textLayerDiv.style.width = `${canvas.width}px`;
        textLayerDiv.style.height = `${canvas.height}px`;
        overlay.style.width = `${canvas.width}px`;
        overlay.style.height = `${canvas.height}px`;

        let textLayerPromise = Promise.resolve();
        if (typeof pdfjsLib.renderTextLayer === "function") {
            const task = pdfjsLib.renderTextLayer({
                textContent,
                container: textLayerDiv,
                viewport,
                textDivs: [],
            });
            textLayerPromise = task?.promise ?? task;
        }
        textLayerReadyRef.current = textLayerPromise;
        await textLayerPromise;

        if (!textLayerDiv.childElementCount) {
            renderTextLayerFallback(viewport, textContent, textLayerDiv);
        }
    };

    // ПРОСТОЕ РИСОВАНИЕ ПРЯМОУГОЛЬНИКОВ ПО coords (origin: top-left)
    const highlightByCoords = (coords) => {
        const viewport = viewportRef.current;
        const overlay = overlayRef.current;
        if (!viewport || !overlay || !Array.isArray(coords) || !coords.length) return;

        overlay.innerHTML = "";

        for (const {x, y, w, h} of coords) {
            // coords в "PDF единицах" от ЛЕВОГО ВЕРХНЕГО угла страницы
            const X = x * viewport.scale;
            const Y = y * viewport.scale;
            const W = w * viewport.scale;
            const H = h * viewport.scale;

            const r = document.createElement("div");
            Object.assign(r.style, {
                position: "absolute",
                left: `${X}px`,
                top: `${Y}px`,
                width: `${W}px`,
                height: `${H}px`,
                background: "rgba(250, 204, 21, 0.28)",
                outline: "1px solid rgba(245, 158, 11, 0.45)",
                borderRadius: "4px",
                mixBlendMode: "multiply",
                pointerEvents: "none",
                boxSizing: "border-box",
            });
            overlay.appendChild(r);
        }

        // авто-скролл к первому прямоугольнику
        const first = overlay.firstElementChild;
        if (first && wrapRef.current) {
            const box = first.getBoundingClientRect();
            const host = wrapRef.current.getBoundingClientRect();
            const dy = box.top - host.top - host.height * 0.25;
            wrapRef.current.scrollBy({top: dy, behavior: "smooth"});
        }
    };
    // ------- Public API -------
    useImperativeHandle(ref, () => ({
        openHighlight: async ({page, rects, zoom}) => {
            if (!pdfDoc) return;
            const needPage = Number.isFinite(+page) ? +page : null;
            if (typeof zoom === "number") setScale(zoom);

            if (needPage) {
                // быстрый прямой рендер нужной страницы
                await renderPage(needPage, scale);
                // не позволяем следующему эффекту мгновенно перерисовать и стереть подсветку
                suppressEffectOnceRef.current = true;
                setPageNum(needPage); // только для тулбара/состояния
            } else {
                try {
                    await textLayerReadyRef.current;
                } catch {
                }
            }

            highlightByCoords(rects);
        },
        setZoom: (z) => setScale(z),
        goToPage: (p) => setPageNum(p),
    }), [pdfDoc, pageNum, scale]);

    // ------- Effect-driven render (подавляем один раз после ручного рендера) -------
    useEffect(() => {
        if (!pdfDoc) return;
        if (suppressEffectOnceRef.current) {
            suppressEffectOnceRef.current = false;
            return; // пропускаем один cycle, чтобы не стереть свежее highlight
        }
        (async () => {
            await renderPage(pageNum, scale);
        })();
    }, [pdfDoc, pageNum, scale]);

    const zoomIn = () => setScale(s => Math.min(4, +(s + 0.2).toFixed(2)));
    const zoomOut = () => setScale(s => Math.max(0.4, +(s - 0.2).toFixed(2)));
    const nextPage = () => setPageNum(p => Math.min(numPages || p + 1, p + 1));
    const prevPage = () => setPageNum(p => Math.max(1, p - 1));

    return (
        <div className="w-full h-full flex flex-col">
            <div className="flex items-center justify-between border-b border-emerald-500/30 px-3 py-2">
                <div className="text-emerald-200/80 text-sm">
                    Page {pageNum}/{numPages || "…"} · Zoom {Math.round(scale * 100)}%
                </div>
                <div className="flex items-center gap-2">
                    <button onClick={prevPage}
                            className="px-2 py-1 rounded border border-emerald-500/40 text-emerald-200 hover:bg-emerald-500/10">Prev
                    </button>
                    <button onClick={nextPage}
                            className="px-2 py-1 rounded border border-emerald-500/40 text-emerald-200 hover:bg-emerald-500/10">Next
                    </button>
                    <div className="w-px h-6 bg-emerald-500/20 mx-1"/>
                    <button onClick={zoomOut}
                            className="px-2 py-1 rounded border border-emerald-500/40 text-emerald-200 hover:bg-emerald-500/10">–
                    </button>
                    <button onClick={zoomIn}
                            className="px-2 py-1 rounded border border-emerald-500/40 text-emerald-200 hover:bg-emerald-500/10">+
                    </button>
                    {onClose && (<>
                        <div className="w-px h-6 bg-emerald-500/20 mx-1"/>
                        <button onClick={onClose}
                                className="px-3 py-1 rounded border border-emerald-500/40 text-emerald-200 hover:bg-emerald-500/10">Close
                        </button>
                    </>)}
                </div>
            </div>

            <div ref={wrapRef} className="relative flex-1 overflow-auto overscroll-contain p-3">
                <div className="relative inline-block">
                    <canvas ref={canvasRef}/>
                    <div
                        ref={textLayerRef}
                        className="textLayer absolute left-0 top-0 pointer-events-auto select-text"
                        style={{transformOrigin: "0 0"}}
                    />
                    <div
                        ref={overlayRef}
                        className="absolute left-0 top-0 pointer-events-none"
                    />
                </div>
            </div>

            <style>{`
        .textLayer span { color: transparent; }
        .textLayer .pdf-span-hl {
          background: rgba(250, 204, 21, 0.35);
          border-radius: 2px;
          outline: 1px solid rgba(245, 158, 11, 0.35);
        }
      `}</style>
        </div>
    );
});

export default PdfPane;
