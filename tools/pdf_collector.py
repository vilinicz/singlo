#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скачивает случайную выборку OA-PDF по 42 категориям OECD FOS (уровень 2),
только EN (language:en), по одному документу на каждый из целевых годов.

Если в папке категории уже ровно 5 PDF — пропускаем эту категорию.
Если меньше 5 — докачиваем недостающие целевые годы.

Имена файлов: TARGETYEAR--PICKEDYEAR__Title.pdf
(старые файлы вида YYYY__Title.pdf также учитываются и засчитываются как закрывающие год YYYY)

Зависимости: requests
"""

import os
import re
import time
import json
import random
from typing import Dict, List, Optional, Iterable, Set, Tuple
import requests

# ------------------ НАСТРОЙКИ ------------------

OPENALEX = "https://api.openalex.org/works"

# ВАЖНО: укажите ваш рабочий email (polite pool)
MAILTO = "youremail@example.com"
HEADERS = {"User-Agent": f"fos42-yearly-sampler/1.5 ({MAILTO})"}

# Целевые годы
YEARS_TARGET = [2025, 2023, 2020, 2017, 2015]

# Радиус подбора ближайшего года, если по целевому пусто
YEAR_FALLBACK_RADIUS = 2

# Сколько статей на год (обычно 1 → всего 5 на категорию)
PER_YEAR = 1

# Куда сохраняем
OUT_DIR = "datasets/fos42_yearly_pdfs"

# Паузы
SLEEP_REQ = 0.2
SLEEP_DL = 0.15

# 42 FOS → эвристический запрос в OpenAlex
FOS_QUERIES: Dict[str, str] = {
    "mathematics": "mathematics",
    "computer_and_information_sciences": "computer science",
    "physical_sciences": "physics OR astronomy",
    "chemical_sciences": "chemistry",
    "earth_and_related_environmental_sciences": "earth science OR geology OR geophysics OR climatology",
    "biological_sciences": "biology OR life science",
    "other_natural_sciences": "natural sciences",
    "civil_engineering": "civil engineering",
    "electrical_electronic_and_information_engineering": "electrical engineering OR electronics OR information engineering",
    "mechanical_engineering": "mechanical engineering",
    "chemical_engineering": "chemical engineering",
    "materials_engineering": "materials science",
    "medical_engineering": "biomedical engineering",
    "environmental_engineering": "environmental engineering",
    "environmental_biotechnology": "environmental biotechnology",
    "industrial_biotechnology": "industrial biotechnology",
    "nano_technology": "nanotechnology",
    "other_engineering_and_technologies": "engineering technology",
    "basic_medicine": "basic medicine",
    "clinical_medicine": "clinical medicine",
    "health_sciences": "public health OR health sciences",
    "health_biotechnology": "health biotechnology",
    "other_medical_sciences": "medical sciences",
    "agriculture_forestry_and_fisheries": "agriculture OR forestry OR fisheries",
    "animal_and_dairy_science": "animal science OR dairy science",
    "veterinary_science": "veterinary",
    "agricultural_biotechnology": "agricultural biotechnology",
    "other_agricultural_sciences": "agricultural sciences",
    "psychology": "psychology",
    "economics_and_business": "economics OR business",
    "educational_sciences": "education science OR pedagogy",
    "sociology": "sociology",
    "law": "law",
    "political_science": "political science",
    "social_and_economic_geography": "economic geography OR human geography",
    "media_and_communications": "media studies OR communications",
    "other_social_sciences": "social sciences",
    "history_and_archaeology": "history OR archaeology",
    "languages_and_literature": "linguistics OR literature",
    "philosophy_ethics_and_religion": "philosophy OR ethics OR religion",
    "arts": "arts OR art history OR performing arts OR music",
    "other_humanities": "humanities",
}

# ------------------ УТИЛИТЫ ------------------

RE_NEW = re.compile(r"^(?P<target>\d{4})--(?P<picked>\d{4})__.+\.pdf$", re.IGNORECASE)
RE_OLD = re.compile(r"^(?P<year>\d{4})__.+\.pdf$", re.IGNORECASE)

def safe_filename(s: str) -> str:
    s = re.sub(r"[^\w\-.]+", "_", s)
    return s[:180]

def pick_pdf_url(work: dict) -> Optional[str]:
    boa = work.get("best_oa_location") or {}
    pdf = boa.get("pdf_url")
    if not pdf:
        prim = work.get("primary_location") or {}
        pdf = prim.get("pdf_url")
    return pdf

def _req_with_retries(params: dict, tries: int = 5, base_sleep: float = 1.0) -> dict:
    params = dict(params)
    params["mailto"] = MAILTO
    last_status = None
    for i in range(tries):
        try:
            r = requests.get(OPENALEX, params=params, headers=HEADERS, timeout=60)
            if r.status_code in (403, 429, 502, 503, 504):
                time.sleep(base_sleep * (2 ** i) + random.random())
                continue
            r.raise_for_status()
            return r.json()
        except requests.HTTPError:
            last_status = r.status_code if 'r' in locals() else None
            if last_status in (403, 429, 502, 503, 504):
                time.sleep(base_sleep * (2 ** i) + random.random())
                continue
            raise
    raise RuntimeError(f"OpenAlex failed after retries; last status={last_status}")

def _year_candidates(target: int, radius: int) -> Iterable[int]:
    yield target
    for d in range(1, radius + 1):
        yield target - d
        yield target + d

def fetch_random_oa_for_year(query: str, year: int, need: int, english_only: bool = True) -> List[dict]:
    """
    Случайная выборка OA-работ за год (без sort, только sample).
    Шаги: строгий (type:journal-article) → без type → has_oa_submitted_version:true.
    """
    def _try(sample_n: int, extra_filter: Optional[str] = None) -> List[dict]:
        seed = random.randint(1, 10_000_000)
        flt_parts = [f"is_oa:true", f"publication_year:{year}"]
        if english_only:
            flt_parts.append("language:en")
        if extra_filter:
            flt_parts.append(extra_filter)
        params = {
            "search": query,
            "filter": ",".join(flt_parts),
            "sample": sample_n,
            "select": "id,title,doi,language,open_access,best_oa_location,primary_location,publication_year,publication_date",
            "seed": seed,
        }
        data = _req_with_retries(params)
        return data.get("results", [])

    def is_en(w: dict) -> bool:
        return (w.get("language") == "en") if english_only else True

    got: List[dict] = []

    for w in _try(sample_n=max(need * 20, 50), extra_filter="type:journal-article"):
        if len(got) >= need: break
        if pick_pdf_url(w) and is_en(w): got.append(w)
    if len(got) >= need: return got[:need]

    for w in _try(sample_n=max(need * 40, 100)):
        if len(got) >= need: break
        if pick_pdf_url(w) and is_en(w): got.append(w)
    if len(got) >= need: return got[:need]

    for w in _try(sample_n=max(need * 60, 150), extra_filter="has_oa_submitted_version:true"):
        if len(got) >= need: break
        if pick_pdf_url(w) and is_en(w): got.append(w)

    return got[:need]

def download_pdf(url: str, dest: str):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with requests.get(url, headers=HEADERS, stream=True, timeout=90) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 15):
                if chunk: f.write(chunk)

def scan_existing_category_dir(cat_dir: str) -> Tuple[int, Set[int]]:
    """
    Возвращает (кол-во PDF, множество уже закрытых целевых годов).
    Поддерживает два формата имён:
      - NEW:  TARGET--PICKED__Title.pdf  → закрывает TARGET
      - OLD:  YYYY__Title.pdf            → считаем, что закрывает YEAR=YYYY
    """
    if not os.path.isdir(cat_dir):
        return 0, set()

    files = [f for f in os.listdir(cat_dir) if f.lower().endswith(".pdf")]
    covered: Set[int] = set()
    for fn in files:
        m = RE_NEW.match(fn)
        if m:
            try:
                t = int(m.group("target"))
                covered.add(t)
                continue
            except Exception:
                pass
        m = RE_OLD.match(fn)
        if m:
            try:
                y = int(m.group("year"))
                covered.add(y)
            except Exception:
                pass
    return len(files), covered

# ------------------ MAIN ------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    log = {}

    for fos, query in FOS_QUERIES.items():
        print(f"\n=== {fos} ===")
        cat_dir = os.path.join(OUT_DIR, fos)
        total_files, covered_years = scan_existing_category_dir(cat_dir)

        if total_files == len(YEARS_TARGET):
            print(f"  Уже есть ровно {total_files} файлов — пропускаю категорию.")
            continue
        elif total_files > len(YEARS_TARGET):
            print(f"  Найдено {total_files} файлов (>5). Оставляю как есть; новых не докачиваю.")
            continue
        else:
            print(f"  Найдено {total_files} файлов; покрытые годы: {sorted(covered_years) if covered_years else '—'}")

        # Определяем недостающие целевые годы
        missing_years = [y for y in YEARS_TARGET if y not in covered_years]
        print(f"  Недостающие годы: {missing_years if missing_years else 'нет'}")

        saved_total_new = 0
        log[fos] = {}

        # Для каждого недостающего целевого года — пытаемся докачать
        for target_year in missing_years:
            want = PER_YEAR
            saved = 0
            picked_year = None
            items: List[dict] = []

            # пробуем целевой и ближайшие годы
            for y in _year_candidates(target_year, YEAR_FALLBACK_RADIUS):
                items = fetch_random_oa_for_year(query, y, want, english_only=True)
                time.sleep(SLEEP_REQ)
                if items:
                    picked_year = y
                    break

            if not items:
                print(f"  {target_year}: 0/1 (нет OA-PDF на EN даже с ближайшими годами)")
                log[fos][str(target_year)] = {"saved": 0, "picked_year": None, "query": query}
                continue

            for w in items:
                pdf_url = pick_pdf_url(w)
                if not pdf_url:
                    continue
                title = w.get("title") or w.get("id", "work")
                fname = safe_filename(f"{target_year}--{picked_year}__{title}.pdf")
                dest = os.path.join(cat_dir, fname)
                try:
                    download_pdf(pdf_url, dest)
                    print(f"  {target_year}→{picked_year}: OK {title[:72]} -> {dest}")
                    saved += 1
                    saved_total_new += 1
                    time.sleep(SLEEP_DL)
                except Exception as e:
                    print(f"  {target_year}→{picked_year}: FAIL {pdf_url} ({e})")

            log[fos][str(target_year)] = {"saved": saved, "picked_year": picked_year, "query": query}
            if saved < want:
                print(f"  {target_year}: saved {saved}/{want} (скудно по OA на EN; перезапустите/увеличьте sample/расширьте радиус)")

        # Итог по категории (после докачки)
        total_files_after, covered_after = scan_existing_category_dir(cat_dir)
        print(f"  Итого в папке: {total_files_after} файлов; покрытые годы: {sorted(covered_after) if covered_after else '—'}")
        if total_files_after == len(YEARS_TARGET):
            print(f"  Категория заполнена (5/5).")

    # лог глобально
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print("\nГотово.")

if __name__ == "__main__":
    main()
