// PdfPane.jsx
import React, {useEffect, useMemo, useRef, useState, forwardRef, useImperativeHandle} from "react";
import * as pdfjsLib from "pdfjs-dist/legacy/build/pdf";

pdfjsLib.GlobalWorkerOptions.workerSrc = new URL(
    "pdf.worker.min.js",
    import.meta.url
).toString();

/**
 * Props:
 *  - pdfUrl: string
 *  - initialHighlight?: { page: number, rects: Array<[x0,y0,x1,y1]> | {x0,y0,x1,y1} }
 *  - onClose?: () => void
 *
 * Imperative API (via ref):
 *  - openHighlight({ page, rects })  // rects: bbox или массив bbox в PDF-координатах
 *  - setZoom(scale:number)
 *  - goToPage(page:number)
 */
const PdfPane = forwardRef(function PdfPane(
    {pdfUrl, initialHighlight, onClose},
    ref
) {
    const wrapRef = useRef(null);
    const canvasRef = useRef(null);
    const textLayerRef = useRef(null);
    const overlayRef = useRef(null);

    const [pdfDoc, setPdfDoc] = useState(null);
    const [pageNum, setPageNum] = useState(initialHighlight?.page || 1);
    const [scale, setScale] = useState(1.3);
    const [numPages, setNumPages] = useState(0);
    const [loading, setLoading] = useState(false);

    // текущий viewport страницы (после renderPage)
    const viewportRef = useRef(null);

    // helper: нормализуем rects к массиву
    const normalizeRects = (rects) => {
        if (!rects) return [];
        if (Array.isArray(rects) && Array.isArray(rects[0])) return rects;
        if (Array.isArray(rects) && rects.length === 4) return [rects];
        if (rects && typeof rects === "object") {
            const {x0, y0, x1, y1} = rects;
            return [[x0, y0, x1, y1]];
        }
        return [];
    };

    // загрузка документа
    useEffect(() => {
        let cancelled = false;
        if (!pdfUrl) return;
        setLoading(true);
        pdfjsLib.getDocument(pdfUrl).promise.then((doc) => {
            if (cancelled) return;
            setPdfDoc(doc);
            setNumPages(doc.numPages || 0);
            setLoading(false);
        }).catch((e) => {
            console.error("PDF load error", e);
            setLoading(false);
        });
        return () => {
            cancelled = true;
        };
    }, [pdfUrl]);

    // рендер выбранной страницы
    const renderPage = async (pageN = pageNum, targetScale = scale) => {
        if (!pdfDoc) return;
        const page = await pdfDoc.getPage(pageN);
        // учитываем поворот страницы
        const viewport = page.getViewport({scale: targetScale, rotation: page.rotate || 0});
        viewportRef.current = viewport;

        // canvas
        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d", {alpha: false});
        canvas.width = Math.floor(viewport.width);
        canvas.height = Math.floor(viewport.height);
        canvas.style.width = `${canvas.width}px`;
        canvas.style.height = `${canvas.height}px`;

        await page.render({canvasContext: ctx, viewport}).promise;

        // textLayer
        const textLayerDiv = textLayerRef.current;
        textLayerDiv.innerHTML = ""; // очистим прошлый слой
        textLayerDiv.style.width = `${canvas.width}px`;
        textLayerDiv.style.height = `${canvas.height}px`;

        const textContent = await page.getTextContent();
        // у legacy api есть renderer
        await pdfjsLib.renderTextLayer({
            textContent,
            container: textLayerDiv,
            viewport,
            textDivs: []
        }).promise;

        // overlay для геометрических прямоугольников (на всякий)
        const overlay = overlayRef.current;
        overlay.innerHTML = "";
        overlay.style.width = `${canvas.width}px`;
        overlay.style.height = `${canvas.height}px`;
    };

    // подсветка span'ов textLayer, чьи rect пересекаются с bbox'ами
    const highlightSpansByBBoxes = (rects, tolerance = 1.0) => {
        const viewport = viewportRef.current;
        if (!viewport) return;
        const textLayerDiv = textLayerRef.current;
        const overlay = overlayRef.current;
        if (!textLayerDiv) return;
        const list = normalizeRects(rects);
        if (!list.length) return;

        // Преобразуем PDF-координаты bbox → CSS координаты.
        // PDF origin: левый-нижний. CSS origin: левый-верхний.
        // viewport.height уже учтен с ротацией.
        const cssRects = list.map(([x0, y0, x1, y1]) => {
            const X0 = x0 * viewport.scale;
            const Y0 = (viewport.height - y1 * viewport.scale); // верхний
            const W = (x1 - x0) * viewport.scale;
            const H = (y1 - y0) * viewport.scale;
            return {x: X0, y: Y0, w: W, h: H};
        });

        // 1) Overlay прямоугольники (для стабильной визуальной подсветки)
        cssRects.forEach(({x, y, w, h}) => {
            const r = document.createElement("div");
            r.className = "pdf-hl-region";
            Object.assign(r.style, {
                position: "absolute",
                left: `${x}px`,
                top: `${y}px`,
                width: `${w}px`,
                height: `${h}px`,
                background: "rgba(250, 204, 21, 0.25)", // amber-300/25
                outline: "1px solid rgba(245, 158, 11, 0.4)",
                borderRadius: "4px",
                mixBlendMode: "multiply",
                pointerEvents: "none"
            });
            overlay.appendChild(r);
        });

        // 2) Подсветим текстовые дивы (span'ы) textLayer по пересечению
        const spans = Array.from(textLayerDiv.querySelectorAll("span"));
        const intersects = (a, b) => {
            return !(
                a.x + a.w < b.x + tolerance ||
                a.x > b.x + b.w - tolerance ||
                a.y + a.h < b.y + tolerance ||
                a.y > b.y + b.h - tolerance
            );
        };

        // для скорости возьмём boundingClientRect родителя и пересчитаем в локальные координаты
        const parentBox = textLayerDiv.getBoundingClientRect();
        spans.forEach((sp) => sp.classList.remove("pdf-span-hl"));
        spans.forEach((sp) => {
            const r = sp.getBoundingClientRect();
            const local = {
                x: r.left - parentBox.left,
                y: r.top - parentBox.top,
                w: r.width,
                h: r.height
            };
            for (const cr of cssRects) {
                if (intersects(local, cr)) {
                    sp.classList.add("pdf-span-hl");
                    break;
                }
            }
        });
    };

    // публичный API
    useImperativeHandle(ref, () => ({
        openHighlight: async ({page, rects, zoom}) => {
            if (typeof page === "number") setPageNum(page);
            if (typeof zoom === "number") setScale(zoom);
            // подождём рендера (ниже useEffect перерисует)
            // после перерисовки вызовем подсветку
            setTimeout(() => {
                highlightSpansByBBoxes(rects);
                // авто-скролл к первому прямоугольнику
                const overlay = overlayRef.current;
                const first = overlay?.firstElementChild;
                if (first && wrapRef.current) {
                    const box = first.getBoundingClientRect();
                    const host = wrapRef.current.getBoundingClientRect();
                    const dy = box.top - host.top - host.height * 0.25;
                    wrapRef.current.scrollBy({top: dy, behavior: "smooth"});
                }
            }, 30);
        },
        setZoom: (z) => setScale(z),
        goToPage: (p) => setPageNum(p),
    }), []);

    // рендер при смене doc/page/scale
    useEffect(() => {
        if (!pdfDoc) return;
        (async () => {
            await renderPage(pageNum, scale);
            // если пришёл initialHighlight для этой страницы — подсветим
            if (initialHighlight && initialHighlight.page === pageNum) {
                highlightSpansByBBoxes(initialHighlight.rects);
            }
        })();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [pdfDoc, pageNum, scale]);

    const zoomIn = () => setScale(s => Math.min(4, +(s + 0.2).toFixed(2)));
    const zoomOut = () => setScale(s => Math.max(0.4, +(s - 0.2).toFixed(2)));
    const nextPage = () => setPageNum(p => Math.min(numPages || p + 1, p + 1));
    const prevPage = () => setPageNum(p => Math.max(1, p - 1));

    return (
        <div className="w-full h-full flex flex-col">
            {/* toolbar */}
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
                    {onClose && (
                        <>
                            <div className="w-px h-6 bg-emerald-500/20 mx-1"/>
                            <button onClick={onClose}
                                    className="px-2 py-1 rounded border border-emerald-500/40 text-emerald-200 hover:bg-emerald-500/10">Close
                            </button>
                        </>
                    )}
                </div>
            </div>

            {/* viewport */}
            <div ref={wrapRef} className="relative flex-1 overflow-auto overscroll-contain p-3">
                <div className="relative inline-block">
                    <canvas ref={canvasRef}/>
                    {/* textLayer сверху канваса */}
                    <div
                        ref={textLayerRef}
                        className="textLayer absolute left-0 top-0 pointer-events-none select-text"
                        style={{transformOrigin: "0 0"}}
                    />
                    {/* overlay подсветки — ещё выше */}
                    <div
                        ref={overlayRef}
                        className="absolute left-0 top-0 pointer-events-none"
                    />
                </div>
            </div>

            {/* локальные стили для подсветок */}
            <style>{`
        .textLayer span {
          color: transparent; /* оставляем нативный вид PDF.js, но без паразитной окраски */
          mix-blend-mode: normal;
        }
        .textLayer .pdf-span-hl {
          background: rgba(250, 204, 21, 0.35); /* amber-300/35 */
          border-radius: 2px;
          outline: 1px solid rgba(245, 158, 11, 0.35);
        }
        .pdf-hl-region { /* overlay rectangles */
          box-sizing: border-box;
        }
      `}</style>
        </div>
    );
});

export default PdfPane;
