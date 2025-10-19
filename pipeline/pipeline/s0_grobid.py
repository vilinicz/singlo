#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GROBID → TEI (с предложениями и bbox) → spaCy rules (8 labels)
Labels: Input Fact | Hypothesis | Experiment | Technique | Result | Dataset | Analysis | Conclusion

Usage:
  poetry run python imrad_grobid_pipeline.py --pdf article.pdf --server http://localhost:8070 --out out.jsonl --md out.md
Deps:
  poetry add spacy requests lxml
  poetry run python -m spacy download en_core_web_sm
"""

import re, json, argparse
from pathlib import Path
from typing import List, Dict, Tuple
import requests
from lxml import etree
from collections import defaultdict

# ---------------- GROBID ----------------

def grobid_fulltext_tei(server: str, pdf_path: str, timeout: int = 120) -> str:
    url = server.rstrip("/") + "/api/processFulltextDocument"
    with open(pdf_path, "rb") as f:
        files = {"input": f}
        data = [
            ("segmentSentences", "1"),
            ("teiCoordinates", "s"),
            ("teiCoordinates", "p"),
            ("teiCoordinates", "head"),
            ("teiCoordinates", "figure"),
            ("teiCoordinates", "table"),
            ("teiCoordinates", "biblStruct"),
        ]
        r = requests.post(url, files=files, data=data, timeout=timeout)
    r.raise_for_status()
    return r.text

# ---------------- TEI parsing & coords ----------------

# допускаем два формата:
#  A) "1, x, y, w, h"   (то, что у тебя в этом TEI)
#  Б) "1: x, y, w, h"   (встречается в некоторых сборках)
COORD_ITEM_RE = re.compile(
    r"^\s*(?:p)?(?P<page>\d+)\s*[: ,]\s*(?P<x>[-\d.]+)\s*,\s*(?P<y>[-\d.]+)\s*,\s*(?P<w>[-\d.]+)\s*,\s*(?P<h>[-\d.]+)"
)

def parse_coords_attr(coords: str) -> List[Dict]:
    """
    coords="1,60.94,248.09,473.40,9.21;1,60.94,259.59,376.69,9.21;..."
    → [{"page":1,"x":60.94,"y":248.09,"w":473.40,"h":9.21}, ...]
    Примечание: если встречается формат с двоеточием — тоже парсим.
    Доп. числа после h (например кегль) игнорируем.
    """
    if not coords:
        return []
    boxes = []
    for chunk in coords.split(";"):
        m = COORD_ITEM_RE.match(chunk)
        if not m:
            continue
        page = int(m.group("page"))
        x = float(m.group("x")); y = float(m.group("y"))
        w = float(m.group("w")); h = float(m.group("h"))
        boxes.append({"page": page, "x": x, "y": y, "w": w, "h": h})
    return boxes

def union_bbox(boxes: List[Dict]) -> Tuple[int, List[float]]:
    """
    Объединение нескольких боксов одной страницы в один прямоугольник.
    Страницу берём из первого бокса; конверсию в 0-based делаем позже.
    """
    if not boxes:
        return 1, [0,0,0,0]
    page = boxes[0]["page"]
    xs0 = [b["x"] for b in boxes if b["page"] == page]
    ys0 = [b["y"] for b in boxes if b["page"] == page]
    xs1 = [b["x"] + b["w"] for b in boxes if b["page"] == page]
    ys1 = [b["y"] + b["h"] for b in boxes if b["page"] == page]
    return page, [min(xs0), min(ys0), max(xs1), max(ys1)]


# def map_head_to_hint(head_text: str) -> str:
#     t = (head_text or "").strip().lower()
#     if "abstract" in t: return "INTRO"
#     if any(k in t for k in ["introduction","background"]): return "INTRO"
#     if any(k in t for k in ["materials and methods","material and methods","methods","methodology","experimental"]): return "METHODS"
#     if any(k in t for k in ["results","findings","outcomes","evaluation"]): return "RESULTS"
#     if any(k in t for k in ["discussion","conclusion","conclusions","implications","limitations"]): return "DISCUSSION"
#     if any(k in t for k in ["references","bibliography","works cited"]): return "REFERENCES"
#     return "OTHER"

# Не работает!
# def nearest_div_head_text(elem, NS) -> str:
#     """
#     Идём вверх до ближайшего <div> и берём его <head> (если есть).
#     В твоём TEI <s> лежат внутри <p> внутри <div>, а <head> — сосед того же <div>.
#     """
#     node = elem.getparent()
#     div_tag = f"{{{NS['t']}}}div"
#     head_tag = f"{{{NS['t']}}}head"
#     while node is not None and node.tag != div_tag:
#         node = node.getparent()
#     if node is None:
#         return ""
#     head = node.find(head_tag)
#     return "".join(head.itertext()).strip() if head is not None else ""

# def topmost_div_head_text(elem, NS) -> str:
#     """
#     Возвращает текст САМОГО ВЕРХНЕГО <head> среди всех ancestor::<div> текущего узла.
#     XPath ancestor:: возвращает предков в порядке от КОРНЯ к РОДИТЕЛЮ,
#     т.е. divs[0] — самый верхний <div>, divs[-1] — ближайший <div>.
#     Мы берём первый предок, у которого реально есть <head>.
#     """
#     divs = elem.xpath("ancestor::t:div", namespaces=NS)
#     for d in divs:                         # идём сверху вниз
#         h = d.find("./t:head", NS)
#         if h is not None:
#             txt = "".join(h.itertext()).strip()
#             if txt:
#                 return txt
#     return ""

import re
_WORD = r"(?:^|[^a-z])"; _EOW = r"(?:$|[^a-z])"

def _clean_head_text(txt: str) -> str:
    t = (txt or "").strip()
    t = re.sub(r"^\s*(?:\d+|[IVXLCM]+)[\.)]?\s+", "", t, flags=re.I)
    return t.replace("&", "and").lower()

def map_head_to_hint(head_text: str) -> str:
    t = _clean_head_text(head_text)
    if not t: return "OTHER"
    if re.search(rf"{_WORD}(abstract|introduction|background|aims and scope){_EOW}", t): return "INTRO"
    if (re.search(rf"{_WORD}(materials? and methods?){_EOW}", t) or
        re.search(rf"{_WORD}(methods?|methodology){_EOW}", t) or
        re.search(rf"{_WORD}(experimental(?: section)?){_EOW}", t) or
        re.search(rf"{_WORD}(patients? and methods?|subjects? and methods?){_EOW}", t) or
        re.search(rf"{_WORD}(study design){_EOW}", t) or
        re.search(rf"{_WORD}(statistical (analysis|methods?)){_EOW}", t)):
        return "METHODS"
    if (re.search(rf"{_WORD}(results? and discussion){_EOW}", t) or
        re.search(rf"{_WORD}(general discussion|discussion|conclusions?|concluding remarks|implications|limitations){_EOW}", t)):
        return "DISCUSSION"
    if re.search(rf"{_WORD}(results?|findings|outcomes){_EOW}", t): return "RESULTS"
    if re.search(rf"{_WORD}(references|bibliography|works cited){_EOW}", t): return "REFERENCES"
    return "OTHER"

from lxml import etree

def _page0_from_boxes(boxes):
    page, _bbox = union_bbox(boxes)
    # union_bbox у тебя возвращает page как 1-based → переведём в 0-based
    return (page - 1) if page is not None else None, _bbox

def tei_iter_sentences(tei_xml: str):
    """
    Однопроходный итератор по TEI:
      - Держит current_imrad_section / current_head_text, обновляя их на IMRAD <head>.
      - Для каждого <s> и капшенов отдаёт {"text","page","bbox","section_hint","is_caption","caption_type"}.
      - Страницы — 0-based.
    """
    root = etree.fromstring(tei_xml.encode("utf-8")) if isinstance(tei_xml, str) else tei_xml
    NS = {"t": root.nsmap.get(None) or "http://www.tei-c.org/ns/1.0"}

    current_head_text = ""
    current_imrad_section = "OTHER"

    # В некоторых вёрстках встречаются повторяющиеся "running headers".
    # Простой фильтр: игнорировать <head>, которые маппятся в INTRO/REFERENCES и
    # повторяются десятки раз с очень маленьким bbox-высотой. Оставим флаг —
    # по умолчанию выключен. Если понадобится — можно включить.
    IGNORE_NOISY_HEADERS = False

    def _should_ignore_head(el, txt, label):
        if not IGNORE_NOISY_HEADERS:
            return False
        # пример простого правила: крошечные по высоте и попадающие в INTRO/REFERENCES
        boxes = parse_coords_attr(el.get("coords") or "")
        _page0, bbox = _page0_from_boxes(boxes)
        if not bbox:
            return False
        x, y, w, h = bbox
        return (h is not None and h < 12.0) and label in {"INTRO", "REFERENCES"}

    # Однопроходный стрим
    for el in root.iter():
        tag = etree.QName(el).localname

        if tag == "head":
            head_txt = "".join(el.itertext()).strip()
            label = map_head_to_hint(head_txt)
            # подзаголовки типа "Study population" / "Evaluation of …" дают OTHER
            # и НЕ меняют текущую IMRAD-секцию
            if label != "OTHER" and not _should_ignore_head(el, head_txt, label):
                current_head_text = head_txt
                current_imrad_section = label
            print(current_imrad_section, current_head_text)

        elif tag == "s":
            text = "".join(el.itertext()).strip()
            if not text:
                continue
            boxes = parse_coords_attr(el.get("coords") or "")
            page0, bbox = _page0_from_boxes(boxes)
            yield {
                "text": text,
                "page": page0,
                "bbox": bbox,
                "section_hint": current_imrad_section,
                "is_caption": False,
                "caption_type": ""
            }

        elif tag == "figDesc":
            # caption к figure
            text = "".join(el.itertext()).strip()
            if not text:
                continue
            # coords могут быть на figDesc, а могут на figure
            boxes = parse_coords_attr(el.get("coords") or "")
            if not boxes:
                # найти родителя <figure> и взять его coords
                fig = el.getparent() if el.getparent() is not None and etree.QName(el.getparent()).localname == "figure" else None
                if fig is not None:
                    boxes = parse_coords_attr(fig.get("coords") or "")
            page0, bbox = _page0_from_boxes(boxes)
            yield {
                "text": text,
                "page": page0,
                "bbox": bbox,
                "section_hint": current_imrad_section,
                "is_caption": True,
                "caption_type": "Figure"
            }

        elif tag == "table":
            # caption у таблицы обычно в <table><head>…</head>
            thead = el.find("./{http://www.tei-c.org/ns/1.0}head")
            if thead is None:
                continue
            text = "".join(thead.itertext()).strip()
            if not text:
                continue
            boxes = parse_coords_attr(thead.get("coords") or "") or parse_coords_attr(el.get("coords") or "")
            page0, bbox = _page0_from_boxes(boxes)
            yield {
                "text": text,
                "page": page0,
                "bbox": bbox,
                "section_hint": current_imrad_section,
                "is_caption": True,
                "caption_type": "Table"
            }

# ---------------- spaCy rules (как в прошлой версии) ----------------

import spacy
from spacy.matcher import Matcher, DependencyMatcher

LABELS = ["Input Fact","Hypothesis","Experiment","Technique","Result","Dataset","Analysis","Conclusion"]
SECTION_PRIORS = {
    "INTRO":      {"Input Fact":1, "Hypothesis":2},
    "METHODS":    {"Experiment":2, "Technique":2, "Dataset":1, "Analysis":1},
    "RESULTS":    {"Result":2, "Analysis":1},
    "DISCUSSION": {"Conclusion":2, "Result":1, "Hypothesis":1},
    "REFERENCES": {},
    "OTHER":      {}
}
PREFIX2LABEL = {
    "INF_":"Input Fact","HYP_":"Hypothesis","EXP_":"Experiment","TEC_":"Technique",
    "RES_":"Result","DATA_":"Dataset","ANA_":"Analysis","CONC_":"Conclusion"
}

def build_nlp(model="en_core_web_sm"):
    try:
        nlp = spacy.load(model)
    except Exception as e:
        raise SystemExit(
            "[error] spaCy model missing. In Poetry:\n"
            "  poetry add spacy\n"
            "  poetry run python -m spacy download en_core_web_sm\n"
        ) from e
    if "parser" not in nlp.pipe_names and "senter" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp

def build_matchers(nlp):
    m = Matcher(nlp.vocab)
    d = DependencyMatcher(nlp.vocab)
    # RESULT
    m.add("RES_VERB_CUES", [[{"LOWER":{"IN":["our","these"]}}, {"LOWER":{"IN":["results","findings"]}}, {"LEMMA":{"IN":["show","indicate","demonstrate","reveal","suggest"]}}], [{"LOWER":"we"}, {"LEMMA":{"IN":["show","find","observe","demonstrate","indicate","reveal","report"]}}]])
    m.add("RES_STATS", [[{"LOWER":{"IN":["p","p-value"]}}, {"IS_PUNCT":True,"OP":"?"}, {"LOWER":{"IN":["<","≤","<="]}}, {"LIKE_NUM":True}], [{"LIKE_NUM":True}, {"TEXT":{"REGEX":"%"}}, {"LOWER":{"IN":["increase","decrease","improvement","reduction"]}}], [{"LIKE_NUM":True}, {"TEXT":{"REGEX":"%"}}, {"LOWER":"ci"}], [{"LOWER":{"IN":["odds","hazard"]}}, {"LOWER":"ratio"}], [{"LOWER":{"IN":["rmse","auc","auroc","accuracy","precision","recall","sensitivity","specificity"]}}], [{"LOWER":"compared"}, {"LOWER":"to"}], [{"LOWER":"achieved"}], [{"LOWER":"yielded"}]])
    d.add("RES_WE_VERB", [[{"RIGHT_ID":"v","RIGHT_ATTRS":{"LEMMA":{"IN":["show","find","observe","demonstrate","indicate","reveal","suggest","report"]}}}, {"LEFT_ID":"v","REL_OP":">>","RIGHT_ID":"subj","RIGHT_ATTRS":{"DEP":{"IN":["nsubj","nsubjpass"]},"LOWER":"we"}}]])
    # EXPERIMENT
    m.add("EXP_SURFACE", [[{"LOWER":"we"}, {"LEMMA":{"IN":["conduct","perform","run","carry","implement"]}}], [{"LOWER":"we"}, {"LEMMA":{"IN":["measure","collect","recruit","randomize","enroll","administer"]}}], [{"LOWER":{"IN":["trial","experiment","study"]}}], [{"LOWER":{"IN":["placebo","control","controlled","double-blind","randomized"]}}], [{"TEXT":{"REGEX":"^n\\s*=\\s*\\d+"}}]])
    d.add("EXP_DOBJ", [[{"RIGHT_ID":"v","RIGHT_ATTRS":{"LEMMA":{"IN":["conduct","perform","run","carry","measure","use","utilize","apply","calibrate","assemble","administer"]}}}, {"LEFT_ID":"v","REL_OP":">>","RIGHT_ID":"obj","RIGHT_ATTRS":{"DEP":{"IN":["dobj","obj"]},"LOWER":{"IN":["experiment","experiments","study","trial","measurement","setup","apparatus","assay"]}}}]])
    # TECHNIQUE
    m.add("TEC_SURFACE", [[{"LOWER":{"IN":["using","with","via","through","by"]}}, {"POS":{"IN":["DET","ADJ"]},"OP":"*"}, {"LEMMA":{"IN":["method","technique","protocol","assay","algorithm","pipeline","architecture","classifier","model"]}}], [{"LOWER":{"IN":["pcr","rt-pcr","western","elisa","mass","spectrometry","mrna","rna-seq","immunohistochemistry","random-forest","svm","cox","kaplan-meier"]}}], [{"LOWER":{"IN":["assay","assays","protocol","protocols"]}}]])
    d.add("TEC_USING", [[{"RIGHT_ID":"v","RIGHT_ATTRS":{"POS":"VERB"}}, {"LEFT_ID":"v","REL_OP":">>","RIGHT_ID":"obl","RIGHT_ATTRS":{"DEP":{"IN":["prep","agent"]}, "LOWER":{"IN":["using","with","via","by","through"]}}}]])
    # DATASET
    m.add("DATA_SURFACE", [[{"LOWER":{"IN":["dataset","data","registry","cohort","biobank","database"]}}], [{"LOWER":{"IN":["mimic","mimic-iii","mimic-iv","uk","biobank","eicu","clinicaltrials.gov","tcga","physionet"]}}], [{"LOWER":{"IN":["patients","participants","subjects"]}}], [{"LOWER":"n"}, {"IS_PUNCT":True,"OP":"?"}, {"LOWER":"="}, {"LIKE_NUM":True}], [{"LOWER":"nct"}, {"IS_DIGIT":True,"OP":"+"}]])
    d.add("DATA_SOURCE", [[{"RIGHT_ID":"v","RIGHT_ATTRS":{"LEMMA":{"IN":["collect","use","utilize","obtain","source","recruit","enroll","include"]}}}, {"LEFT_ID":"v","REL_OP":">>","RIGHT_ID":"obj","RIGHT_ATTRS":{"DEP":{"IN":["obj","dobj"]}, "LOWER":{"IN":["data","dataset","datasets","patients","participants","subjects","records","cohort"]}}}]])
    # ANALYSIS
    m.add("ANA_SURFACE", [[{"LOWER":"we"}, {"LEMMA":{"IN":["analyze","analyse","assess","evaluate","model","fit","estimate","adjust","normalize","standardize"]}}], [{"LOWER":{"IN":["regression","logistic","linear","cox","anova","ancova","mixed-effects","multivariate","univariate"]}}], [{"LOWER":{"IN":["kaplan-meier","survival","hazard","odds"]}}], [{"LOWER":{"IN":["significance","multiple","testing","bonferroni","fdr"]}}]])
    d.add("ANA_DEP", [[{"RIGHT_ID":"v","RIGHT_ATTRS":{"LEMMA":{"IN":["analyze","analyse","evaluate","assess","model","fit","estimate","adjust"]}}}, {"LEFT_ID":"v","REL_OP":">>","RIGHT_ID":"obj","RIGHT_ATTRS":{"DEP":{"IN":["obj","dobj","obl"]}}}]])
    # HYPOTHESIS
    m.add("HYP_SURFACE", [[{"LOWER":{"IN":["we","our"]}}, {"LEMMA":{"IN":["hypothesize","hypothesise","postulate","posit","predict","propose"]}}], [{"LOWER":{"IN":["we","our"]}}, {"LOWER":{"IN":["hypothesis","hypotheses"]}}], [{"LOWER":{"IN":["we","this","the"]}}, {"LOWER":"study","OP":"?"}, {"LEMMA":{"IN":["aim","seek"]}}, {"LOWER":"to"}], [{"LOWER":{"IN":["we"]}}, {"LEMMA":{"IN":["expect"]}}, {"LOWER":"that"}]])
    d.add("HYP_THAT", [[{"RIGHT_ID":"v","RIGHT_ATTRS":{"LEMMA":{"IN":["hypothesize","postulate","posit","predict","propose","expect"]}}}, {"LEFT_ID":"v","REL_OP":">>","RIGHT_ID":"comp","RIGHT_ATTRS":{"DEP":{"IN":["ccomp","xcomp","advcl"]}}}]])
    # INPUT FACT
    m.add("INF_SURFACE", [[{"LOWER":{"IN":["according","given","based"]}}, {"LOWER":"on","OP":"?"}], [{"LOWER":{"IN":["it","this"]}}, {"LEMMA":{"IN":["be"]}}, {"LOWER":{"IN":["known","established","well-known"]}}], [{"LOWER":{"IN":["prior","previous","existing"]}}, {"LOWER":{"IN":["work","evidence","studies","literature"]}}], [{"LOWER":{"IN":["guidelines","consensus","recommendations"]}}], [{"LOWER":{"IN":["baseline","assumption","assumptions","inclusion","exclusion","criteria"]}}]])
    d.add("INF_CITATION", [[{"RIGHT_ID":"v","RIGHT_ATTRS":{"LEMMA":{"IN":["report","show","demonstrate"]}}}, {"LEFT_ID":"v","REL_OP":">>","RIGHT_ID":"obl","RIGHT_ATTRS":{"LOWER":{"IN":["previously","earlier"]}}}]])
    # CONCLUSION
    m.add("CONC_SURFACE", [[{"LOWER":{"IN":["in","overall"]}}, {"LOWER":"conclusion","OP":"?"}], [{"LOWER":{"IN":["in","overall"]}}, {"LOWER":"summary"}], [{"LOWER":{"IN":["we"]}}, {"LEMMA":{"IN":["conclude","confirm"]}}], [{"LOWER":{"IN":["these","our","the"]}}, {"LOWER":{"IN":["findings","results","data"]}}, {"LEMMA":{"IN":["support","suggest","highlight","underscore"]}}], [{"LOWER":{"IN":["implications","clinical","practice","translation","future"]}}]])
    d.add("CONC_DEP", [[{"RIGHT_ID":"v","RIGHT_ATTRS":{"LEMMA":{"IN":["conclude","suggest","support","confirm","highlight","underscore"]}}}, {"LEFT_ID":"v","REL_OP":">>","RIGHT_ID":"subj","RIGHT_ATTRS":{"DEP":{"IN":["nsubj","nsubjpass"]}}}]])
    return m, d

def score_sentence(doc_sent, section, matcher, depmatcher):
    scores = {lab:0 for lab in LABELS}
    hits = defaultdict(int)
    for mid, s, e in matcher(doc_sent):
        name = doc_sent.vocab.strings[mid]; hits[name]+=1
        for pref, lab in PREFIX2LABEL.items():
            if name.startswith(pref): scores[lab]+=1
    for mid, toks in depmatcher(doc_sent):
        name = doc_sent.vocab.strings[mid]; hits[name]+=1
        for pref, lab in PREFIX2LABEL.items():
            if name.startswith(pref): scores[lab]+=2
    for lab,w in SECTION_PRIORS.get(section,{}).items():
        scores[lab]+=w
    return scores, hits

def looks_like_reference(text: str) -> bool:
    if re.search(r"\b(vol\.?|no\.?|pp\.?|doi:?|issn|et al\.)\b", text, re.I): return True
    if re.search(r"\b(19|20)\d{2}\b", text) and re.search(r"\b\d{1,4}\s*[–-]\s*\d{1,4}\b", text): return True
    return False

def decide_label(scores: Dict[str,int], section: str, had_matches: bool) -> str:
    if section=="REFERENCES" or not had_matches:
        return "OTHER" if section != "RESULTS" else "Result"
    m = max(scores.values())
    if m <= 0: return "OTHER"
    cands = [lab for lab,v in scores.items() if v==m]
    order = ["Result","Experiment","Technique","Analysis","Dataset","Hypothesis","Conclusion","Input Fact"]
    for lab in order:
        if lab in cands: return lab
    return cands[0]

def merge_adjacent(records: List[Dict]) -> List[Dict]:
    if not records: return records
    merged = [records[0].copy()]
    for r in records[1:]:
        cur = merged[-1]
        if r["label"] == cur["label"] and r["page"] == cur["page"]:
            cur["text"] += " " + r["text"]
            x0 = min(cur["bbox"][0], r["bbox"][0]); y0 = min(cur["bbox"][1], r["bbox"][1])
            x1 = max(cur["bbox"][2], r["bbox"][2]); y1 = max(cur["bbox"][3], r["bbox"][3])
            cur["bbox"] = [x0,y0,x1,y1]
        else:
            merged.append(r.copy())
    return merged

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser(description="GROBID + spaCy rule labels (with bbox)")
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--server", default="http://localhost:8070")
    ap.add_argument("--out", default="out.jsonl")
    ap.add_argument("--md", default="")
    ap.add_argument("--model", default="en_core_web_sm")
    args = ap.parse_args()

    tei = grobid_fulltext_tei(args.server, args.pdf)

    tei_path = Path(args.pdf).with_suffix(".tei.xml")
    tei_path.write_text(tei, encoding="utf-8")
    print(f"[tei] saved to {tei_path}")

    items = list(tei_iter_sentences(tei))
    nlp = build_nlp(args.model)
    matcher, depmatcher = build_matchers(nlp)

    records = []
    for i, it in enumerate(items):
        text = it["text"]
        sec  = it["section_hint"]
        if looks_like_reference(text):
            label = "OTHER"; scores={lab:0 for lab in LABELS}; hits={}
        else:
            doc = nlp(text)  # single sentence from GROBID
            scores, hits = score_sentence(doc, sec, matcher, depmatcher)
            label = decide_label(scores, sec, bool(hits))
        rec = {
            "idx": i,
            "section": sec,
            "label": label,
            "text": text,
            "page": it["page"],            # 0-based
            "bbox": it["bbox"],            # [x0,y0,x1,y1] in PDF units
            "is_caption": it["is_caption"],
            "caption_type": it["caption_type"],
            "scores": scores,
            "matches": hits,
        }
        records.append(rec)

    # merge consecutive same-label spans
    # records = merge_adjacent(records)
    records = records

    with open(args.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if args.md:
        groups = defaultdict(list)
        for r in records: groups[r["label"]].append(r)
        lines = ["# Labels (merged spans)\n"]
        for lab in ["Result","Experiment","Technique","Analysis","Dataset","Hypothesis","Conclusion","Input Fact","OTHER"]:
            if not groups[lab]: continue
            lines.append(f"## {lab}  \n(count: {len(groups[lab])})")
            for r in groups[lab][:200]:
                lines.append(f"- p.{r['page']+1} {r['text']}")
            lines.append("")
        Path(args.md).write_text("\n".join(lines), encoding="utf-8")

    print(f"[done] items={len(records)} jsonl={args.out}" + (f" md={args.md}" if args.md else ""))

if __name__ == "__main__":
    main()