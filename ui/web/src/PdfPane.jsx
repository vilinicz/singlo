// PdfPane.jsx
import React, {useEffect, useMemo, useRef, useState, forwardRef, useImperativeHandle} from "react";
// 1) Корректные импорты v5 (ESM)
import * as pdfjsLib from "pdfjs-dist/build/pdf";
//import { EventBus, TextLayerBuilder } from "pdfjs-dist/web/pdf_viewer";
import "pdfjs-dist/web/pdf_viewer.css";

// 2) ВАЖНО: подтянуть worker как URL-строку (а не default-экспорт модуля)
import workerSrc from "pdfjs-dist/build/pdf.worker.mjs?url";

// 3) Назначить путь воркера
pdfjsLib.GlobalWorkerOptions.workerSrc = workerSrc;

/**
 * Props:
 *  - pdfUrl: string (обязателен)
 *  - initialHighlight?: { page: number, rects: Array<[x0,y0,x1,y1]> | {x0,y0,x1,y1} }
 *  - onClose?: () => void
 *
 * Imperative API (via ref):
 *  - openHighlight({ page, rects, zoom? })
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
        // активные задачи рендера
        const renderTaskRef = useRef(null);     // PDFRenderTask от page.render(...)
        const textLayerReadyRef = useRef(Promise.resolve()); // промис “text layer готов”
        const renderTokenRef = useRef(0);       // поколение рендера (для защиты от гонок)
        const lastRenderedRef = useRef({page: null, token: 0});
        const pageNumRef = useRef(1);

        const [pdfDoc, setPdfDoc] = useState(null);
        const [pageNum, setPageNum] = useState(initialHighlight?.page || 1);
        const [scale, setScale] = useState(1.3);
        const [numPages, setNumPages] = useState(0);
        const [loading, setLoading] = useState(false);

        const viewportRef = useRef(null);

        const normalizeRects = (rects) => {
            if (!rects) return [];

            // формат: [{x,y,w,h}, ...]
            if (Array.isArray(rects) && rects.length && typeof rects[0] === "object" && !Array.isArray(rects[0])) {
                return rects.map(({x, y, w, h}) => [x, y, x + w, y + h]);
            }

            // формат: [x0,y0,x1,y1] или массив таких массивов
            if (Array.isArray(rects) && Array.isArray(rects[0])) return rects;
            if (Array.isArray(rects) && rects.length === 4) return [rects];

            // формат: {x0,y0,x1,y1} или {x,y,w,h}
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

        // Загрузка документа
        // Загрузка документа один раз (pdfUrl постоянный)
        useEffect(() => {
            if (!pdfUrl) return;
            let cancelled = false;
            setLoading(true);

            const task = pdfjsLib.getDocument({
                url: pdfUrl,
                disableStream: true,
                disableRange: true,
                verbosity: pdfjsLib.VerbosityLevel.ERRORS,
            });

            task.promise
                .then((doc) => {
                    if (cancelled) return;
                    setPdfDoc(doc);
                    setNumPages(doc.numPages || 0);
                    setLoading(false);
                })
                .catch((err) => {
                    console.error("PDF load error:", err);
                    setLoading(false);
                });

            // ВАЖНО: не вызываем task.destroy() в cleanup — это и рвёт worker.
            return () => {
                cancelled = true;
            };
            // eslint-disable-next-line react-hooks/exhaustive-deps
        }, []); // ← пустой список зависимостей


        // Рендер выбранной страницы
        const renderPage = async (pageN = pageNum, targetScale = scale) => {
            if (!pdfDoc) return;

            // инкремент поколения — всё, что завершится не для этого поколения, игнорируем
            const myToken = ++renderTokenRef.current;

            // если шёл предыдущий рендер — отменим
            if (renderTaskRef.current) {
                try {
                    renderTaskRef.current.cancel();
                } catch {
                }
                renderTaskRef.current = null;
            }

            const page = await pdfDoc.getPage(pageN);
            const viewport = page.getViewport({scale: targetScale, rotation: page.rotate || 0});
            viewportRef.current = viewport;

            // canvas как у тебя:
            const canvas = canvasRef.current;
            const ctx = canvas.getContext("2d", {alpha: false});
            canvas.width = Math.floor(viewport.width);
            canvas.height = Math.floor(viewport.height);
            canvas.style.width = `${canvas.width}px`;
            canvas.style.height = `${canvas.height}px`;
            await page.render({canvasContext: ctx, viewport}).promise;

            const textContent = await page.getTextContent();
            console.log("[PDF] textContent items:", textContent?.items?.length);

// очистим слои
            const textLayerDiv = textLayerRef.current;
            const overlay = overlayRef.current;
            textLayerDiv.innerHTML = "";
            overlay.innerHTML = "";

            textLayerDiv.style.width = `${canvas.width}px`;
            textLayerDiv.style.height = `${canvas.height}px`;
            overlay.style.width = `${canvas.width}px`;
            overlay.style.height = `${canvas.height}px`;

            let textLayerPromise = Promise.resolve();
            if (typeof pdfjsLib.renderTextLayer === "function") {
                // v5-friendly способ: просто отрендерить span'ы
                const task = pdfjsLib.renderTextLayer({
                    textContent,
                    container: textLayerDiv,
                    viewport,
                    textDivs: [],
                });
                // у v5 задача возвращает объект с promise/или сам промис — подстрахуемся:
                textLayerPromise = task?.promise ?? task;
            } else {
                // fallback: нет textLayer — подсветим только overlay-прямоугольниками
                textLayerPromise = Promise.resolve();
            }

            // важно: сохраняем промис текущего textLayer
            textLayerReadyRef.current = textLayerPromise;

            await textLayerPromise;
            console.log("[PDF] textLayer children:", textLayerDiv?.childElementCount);

            // Fallback: если pdfjsLib.renderTextLayer не создал <span> — рисуем упрощённый слой сами,
            // чтобы и подсветка, и выделение мышью работали.
            if (!textLayerDiv?.childElementCount) {
                const items = textContent?.items || [];
                for (const item of items) {
                    const tx = pdfjsLib.Util.transform(
                        pdfjsLib.Util.transform(viewport.transform, item.transform),
                        [1, 0, 0, -1, 0, 0] // переворот по Y
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
                        lineHeight: 1.0,
                    });
                    textLayerDiv.appendChild(span);
                }
                console.log("[PDF] fallback textLayer children:", textLayerDiv.childElementCount);
            }

            // если за это время сменилось поколение — не трогаем дальше DOM
            if (renderTokenRef.current !== myToken) return;

            // страница успешно дорисована — фиксируем
            lastRenderedRef.current = {page: pageN, token: myToken};
        };

        // Подсветка span'ов по bbox
        const highlightSpansByBBoxes = (rects, tolerance = 1.0) => {
            console.log("[HL] highlight rects (raw):", rects);
            const viewport = viewportRef.current;
            if (!viewport) return;

            const textLayerDiv = textLayerRef.current;
            const overlay = overlayRef.current;
            if (!textLayerDiv || !overlay) return;

            const list = normalizeRects(rects);
            if (!list.length) return;

            // Конвертируем PDF-координаты в CSS
            const cssRects = list.map(([x0, y0, x1, y1]) => {
                const X0 = x0 * viewport.scale;
                const Y0 = viewport.height - y1 * viewport.scale; // PDF origin bottom-left → CSS top-left
                const W = (x1 - x0) * viewport.scale;
                const H = (y1 - y0) * viewport.scale;
                return {x: X0, y: Y0, w: W, h: H};
            });

            // Две интерпретации:
// A) PDF-origin снизу-слева (как в чистом PDF)
            const cssRectsBottom = list.map(([x0, y0, x1, y1]) => ({
                x: x0 * viewport.scale,
                y: viewport.height - y1 * viewport.scale,
                w: (x1 - x0) * viewport.scale,
                h: (y1 - y0) * viewport.scale,
            }));
// B) origin сверху-слева (как любят OCR/веб)
            const cssRectsTop = list.map(([x0, y0, x1, y1]) => ({
                x: x0 * viewport.scale,
                y: y0 * viewport.scale,
                w: (x1 - x0) * viewport.scale,
                h: (y1 - y0) * viewport.scale,
            }));

            // Overlay прямоугольники
            cssRects.forEach(({x, y, w, h}) => {
                const r = document.createElement("div");
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
                    pointerEvents: "none",
                    boxSizing: "border-box",
                });
                overlay.appendChild(r);
            });

            // Подсветка textLayer span'ов
            const spans = Array.from(textLayerDiv.querySelectorAll("span"));
            const parentBox = textLayerDiv.getBoundingClientRect();

            const intersects = (a, b) =>
                !(
                    a.x + a.w < b.x + tolerance ||
                    a.x > b.x + b.w - tolerance ||
                    a.y + a.h < b.y + tolerance ||
                    a.y > b.y + b.h - tolerance
                );

            spans.forEach((sp) => sp.classList.remove("pdf-span-hl"));
            spans.forEach((sp) => {
                const r = sp.getBoundingClientRect();
                const local = {
                    x: r.left - parentBox.left,
                    y: r.top - parentBox.top,
                    w: r.width,
                    h: r.height,
                };
                for (const cr of cssRects) {
                    if (intersects(local, cr)) {
                        sp.classList.add("pdf-span-hl");
                        break;
                    }
                }
            });

            const paintAndCount = (rectsCss) => {
                overlay.innerHTML = "";
                rectsCss.forEach(({x, y, w, h}) => {
                    const r = document.createElement("div");
                    Object.assign(r.style, {
                        position: "absolute", left: `${x}px`, top: `${y}px`,
                        width: `${w}px`, height: `${h}px`,
                        background: "rgba(250, 204, 21, 0.25)",
                        outline: "1px solid rgba(245, 158, 11, 0.4)",
                        borderRadius: "4px", mixBlendMode: "multiply",
                        pointerEvents: "none", boxSizing: "border-box",
                    });
                    overlay.appendChild(r);
                });
                const spans = Array.from(textLayerDiv.querySelectorAll("span"));
                const parentBox = textLayerDiv.getBoundingClientRect();
                const intersects = (a, b) => !(
                    a.x + a.w < b.x + tolerance ||
                    a.x > b.x + b.w - tolerance ||
                    a.y + a.h < b.y + tolerance ||
                    a.y > b.y + b.h - tolerance
                );
                let hits = 0;
                spans.forEach(sp => sp.classList.remove("pdf-span-hl"));
                spans.forEach(sp => {
                    const r = sp.getBoundingClientRect();
                    const local = {x: r.left - parentBox.left, y: r.top - parentBox.top, w: r.width, h: r.height};
                    for (const cr of rectsCss) {
                        if (intersects(local, cr)) {
                            sp.classList.add("pdf-span-hl");
                            hits++;
                            break;
                        }
                    }
                });
                return hits;
            };

            // Пытаемся сначала как "нижний-левый". Если нет попаданий — пробуем "верхний-левый".
            let hits = paintAndCount(cssRectsBottom);
            if (hits === 0) {
                hits = paintAndCount(cssRectsTop);
            }
            // авто-скролл к первому квадратику, если есть
            const first = overlay.firstElementChild;
            if (first && wrapRef.current) {
                const box = first.getBoundingClientRect();
                const host = wrapRef.current.getBoundingClientRect();
                const dy = box.top - host.top - host.height * 0.25;
                wrapRef.current.scrollBy({top: dy, behavior: "smooth"});
            }
        };

        // публичный API (Подсветка)
        useImperativeHandle(ref, () => ({
            openHighlight: async ({page, rects, zoom}) => {
                console.log("[HL] openHighlight args:", {page, rects, zoom});
                // 0) дождаться PDF (если вдруг ещё грузится)
                const waitUntil = async (pred, timeoutMs = 5000, step = 30) => {
                    const t0 = Date.now();
                    while (!pred()) {
                        if (Date.now() - t0 > timeoutMs) break;
                        await new Promise(r => setTimeout(r, step));
                    }
                };
                await waitUntil(() => !!pdfDoc);

                // 1) нормализуем номер страницы (строки → числа)
                const needPage = Number.isFinite(+page) ? +page : null;

                // 2) зум (если задан)
                if (typeof zoom === "number") setScale(zoom);

                // 3) если page не задан или совпадает с текущим — просто дождёмся textLayer и подсветим
                if (!needPage || needPage === pageNumRef.current) {
                    try {
                        await textLayerReadyRef.current;
                    } catch {
                    }
                    highlightSpansByBBoxes(rects);
                    return;
                }

                // 4) переключение на другую страницу
                const startToken = renderTokenRef.current;
                setPageNum(needPage);

                // дождёмся нового рендера (token вырос)
                await waitUntil(() => renderTokenRef.current > startToken);
                // дождёмся, когда именно нужная страница дорисуется
                await waitUntil(() => lastRenderedRef.current.page === needPage);

                // 5) подсветить
                highlightSpansByBBoxes(rects);
            },
            setZoom:
                (z) => setScale(z),
            goToPage:
                (p) => setPageNum(p),
        }), []);

        // Перерисовка при смене doc/page/scale
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
        useEffect(() => {
            pageNumRef.current = pageNum;
        }, [pageNum]);

        const zoomIn = () => setScale((s) => Math.min(4, +(s + 0.2).toFixed(2)));
        const zoomOut = () => setScale((s) => Math.max(0.4, +(s - 0.2).toFixed(2)));
        const nextPage = () => setPageNum((p) => Math.min(numPages || p + 1, p + 1));
        const prevPage = () => setPageNum((p) => Math.max(1, p - 1));

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
                                        className="px-3 py-1 rounded border border-emerald-500/40 text-emerald-200 hover:bg-emerald-500/10">Close
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
                            className="textLayer absolute left-0 top-0 pointer-events-auto select-text"
                            style={{transformOrigin: "0 0"}}
                        />
                        {/* overlay подсветки — ещё выше */}
                        <div
                            ref={overlayRef}
                            className="absolute left-0 top-0 pointer-events-none"
                        />
                    </div>
                </div>

                {/* локальные стили подсветок */}
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
    })
;

export default PdfPane;
