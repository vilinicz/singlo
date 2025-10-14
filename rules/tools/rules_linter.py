#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rules_linter.py — линтер для Singularis S1: rules/common.yaml + themes/*/{rules,triggers,lexicon}.yaml

Запуск:
  python rules_linter.py                      # автопоиск файлов от корня проекта
  python rules_linter.py --paths rules/common.yaml themes/biomed/rules.yaml
  python rules_linter.py --root . --json report.json --strict

Код выхода:
  0 — нет ошибок (могут быть предупреждения)
  2 — есть ошибки
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

try:
    import yaml
except Exception as e:
    print("ERROR: PyYAML is required. pip install pyyaml", file=sys.stderr)
    raise

NODE_TYPES = {
    "Input Fact","Hypothesis","Experiment","Technique",
    "Result","Dataset","Analysis","Conclusion"
}

# Разрешённые секции (нормализованные lower)
ALLOWED_SECTIONS = {
    "abstract","introduction","background","materials and methods","materials & methods",
    "methods","method","results","results and discussion","discussion","conclusions","conclusion",
    "body","unknown","figurecaption","tablecaption","theory","related work","limitations","appendix"
}

DANGEROUS_REGEX_HINTS = [
    (re.compile(r"\.\*[^?]"), "Unbounded '.*' — замените на .{0,N}?"),
    (re.compile(r"\(\.\*\)\+"), "Nested greedy '(.*)+' — перепишите шаблон"),
    (re.compile(r"\[[^\]]*\+\+[^\]]*\]"), "Suspicious character class with '++'"),
]

SECTION_NORMALIZE = {
    "materials & methods": "materials and methods",
    "results & discussion": "results and discussion",
    "figurecaption": "figurecaption",
    "tablecaption": "tablecaption",
}

RE_FLAGS = re.IGNORECASE | re.MULTILINE

@dataclass
class Issue:
    path: str
    kind: str   # "ERROR" | "WARN"
    where: str
    msg: str

@dataclass
class Report:
    errors: List[Issue] = field(default_factory=list)
    warns: List[Issue] = field(default_factory=list)

    def add(self, path: Path, kind: str, where: str, msg: str):
        issue = Issue(str(path), kind, where, msg)
        if kind == "ERROR":
            self.errors.append(issue)
        else:
            self.warns.append(issue)

    def has_errors(self) -> bool:
        return len(self.errors) > 0

    def to_json(self) -> Dict[str, Any]:
        def conv(is_list):
            return [
                {"path": i.path, "kind": i.kind, "where": i.where, "msg": i.msg}
                for i in is_list
            ]
        return {"errors": conv(self.errors), "warnings": conv(self.warns)}

def load_yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"YAML parse failed for {path}: {e}")

def norm_section(s: str) -> str:
    t = (s or "").strip().lower()
    return SECTION_NORMALIZE.get(t, t)

def check_regex(pattern: str, path: Path, where: str, rep: Report):
    # опасные конструкции
    for rx, hint in DANGEROUS_REGEX_HINTS:
        if rx.search(pattern):
            rep.add(path, "WARN", where, f"Regex hint: {hint} | pattern='{pattern}'")
    # мягкая проверка «длина окна» между якорями — рекомендовать .{0,N}
    if re.search(r"(?<!\{)\.(\*|\+)(?!\?)", pattern):
        rep.add(path, "WARN", where, "Regex has greedy dot-quantifier. Consider bounding with .{0,160}?")
    # компиляция
    try:
        re.compile(pattern, RE_FLAGS)
    except Exception as e:
        rep.add(path, "ERROR", where, f"Regex compile failed: {e} | pattern='{pattern}'")

def lint_rules_yaml(path: Path, rep: Report, id_registry: Dict[str, Path]):
    data = load_yaml(path)
    if not isinstance(data, dict) or "elements" not in data:
        rep.add(path, "ERROR", "file", "Expected mapping with 'elements: [...]'")
        return
    elems = data.get("elements") or []
    if not isinstance(elems, list):
        rep.add(path, "ERROR", "elements", "Must be a list")
        return

    for i, el in enumerate(elems):
        where = f"elements[{i}]"
        if not isinstance(el, dict):
            rep.add(path, "ERROR", where, "Element must be a mapping")
            continue

        # id
        eid = el.get("id")
        if not eid or not isinstance(eid, str):
            rep.add(path, "ERROR", where, "Missing/invalid 'id'")
        else:
            if eid in id_registry:
                rep.add(path, "ERROR", where, f"Duplicate id '{eid}' (first seen in {id_registry[eid]})")
            else:
                id_registry[eid] = path
            if len(eid) > 128:
                rep.add(path, "WARN", where, "Id is very long (>128 chars)")

        # type
        etype = el.get("type")
        if etype not in NODE_TYPES:
            rep.add(path, "ERROR", where, f"Invalid 'type': {etype} (allowed: {sorted(NODE_TYPES)})")

        # weight
        w = el.get("weight")
        if not isinstance(w, (int, float)):
            rep.add(path, "ERROR", where, "Missing/invalid 'weight' (0..1 recommended)")
        else:
            if not (0 < float(w) <= 1.5):  # допускаем >1 для доменных бустов (смешивание по theme_score)
                rep.add(path, "WARN", where, f"Suspicious weight {w}; expected (0, 1.5]")
            if float(w) > 1.0:
                rep.add(path, "WARN", where, "Weight > 1.0 — убедитесь, что это осознанный доменный буст")

        # sections
        secs = el.get("sections")
        if not isinstance(secs, list) or not secs:
            rep.add(path, "ERROR", where, "Missing/invalid 'sections' (non-empty list)")
        else:
            for s in secs:
                ns = norm_section(str(s))
                if ns not in ALLOWED_SECTIONS:
                    rep.add(path, "WARN", f"{where}.sections", f"Unknown/rare section '{s}'. Allowed set is curated; ensure this is intended.")

        # pattern
        pat = el.get("pattern")
        if not pat or not isinstance(pat, str):
            rep.add(path, "ERROR", where, "Missing/invalid 'pattern'")
        else:
            if len(pat) > 2000:
                rep.add(path, "WARN", where, "Pattern is very long (>2000 chars)")
            check_regex(pat, path, where, rep)

        # negatives (optional)
        negs = el.get("negatives", [])
        if negs is not None and not isinstance(negs, list):
            rep.add(path, "ERROR", where, "'negatives' must be a list of strings")
        elif isinstance(negs, list):
            for j, n in enumerate(negs):
                if not isinstance(n, str):
                    rep.add(path, "ERROR", f"{where}.negatives[{j}]", "Must be string")
                else:
                    check_regex(n, path, f"{where}.negatives[{j}]", rep)

        # captures (optional)
        caps = el.get("captures", [])
        if caps is not None and not isinstance(caps, list):
            rep.add(path, "ERROR", where, "'captures' must be a list[str]")
        elif isinstance(caps, list):
            if len(caps) != len(re.findall(r"\((?!\?:)(?!\?=)(?!\?!)(?!\?<=)(?!\?<!)(?!\?#)", pat or "")):
                rep.add(path, "WARN", where, "captures count doesn't match regex capture groups (non-named). Check ordering.")
            for j, c in enumerate(caps):
                if not isinstance(c, str) or not c:
                    rep.add(path, "ERROR", f"{where}.captures[{j}]", "Capture name must be non-empty string")
                elif len(c) > 64:
                    rep.add(path, "WARN", f"{where}.captures[{j}]", "Capture name is long (>64 chars)")

def lint_triggers_yaml(path: Path, rep: Report):
    data = load_yaml(path)
    if not isinstance(data, dict):
        rep.add(path, "ERROR", "file", "Expected mapping")
        return
    name = data.get("name")
    if not name:
        rep.add(path, "ERROR", "name", "Missing theme 'name'")
    version = data.get("version")
    if version is None:
        rep.add(path, "WARN", "version", "No 'version' field")

    def _chk_list(name: str, typ: str):
        v = data.get(name, [])
        if v is None:
            return []
        if not isinstance(v, list):
            rep.add(path, "ERROR", name, f"'{name}' must be a list")
            return []
        clean = []
        if typ == "str":
            for i, it in enumerate(v):
                if not isinstance(it, str):
                    rep.add(path, "ERROR", f"{name}[{i}]", "Must be string")
                else:
                    clean.append(it)
        elif typ == "pair":
            for i, it in enumerate(v):
                if (not isinstance(it, list)) or len(it) != 2 or not isinstance(it[0], str):
                    rep.add(path, "ERROR", f"{name}[{i}]", "Must be [token:str, weight:number]")
                else:
                    wt = it[1]
                    if not isinstance(wt, (int, float)):
                        rep.add(path, "ERROR", f"{name}[{i}]", "Weight must be number")
                    else:
                        clean.append((it[0], float(wt)))
        return clean

    must = _chk_list("must", "str")
    should = _chk_list("should", "pair")
    negative = _chk_list("negative", "pair")

    if (not must) and (not should):
        rep.add(path, "WARN", "triggers", "No 'must' and empty 'should' — тема может не активироваться")

    thr = data.get("threshold", 0.0)
    if not isinstance(thr, (int, float)):
        rep.add(path, "ERROR", "threshold", "Must be number")
    elif thr < 0:
        rep.add(path, "WARN", "threshold", "Threshold < 0 looks odd")

    topk = data.get("topk", None)
    if topk is not None and (not isinstance(topk, int) or topk <= 0):
        rep.add(path, "ERROR", "topk", "If provided, topk must be positive integer")

def lint_lexicon_yaml(path: Path, rep: Report):
    data = load_yaml(path)
    if not isinstance(data, dict):
        rep.add(path, "ERROR", "file", "Expected mapping")
        return
    # abbr: list of [short,long]
    abbr = data.get("abbr", [])
    if abbr is not None and not isinstance(abbr, list):
        rep.add(path, "ERROR", "abbr", "Must be a list")
    else:
        for i, pair in enumerate(abbr or []):
            if not (isinstance(pair, list) and len(pair) == 2 and all(isinstance(x, str) for x in pair)):
                rep.add(path, "ERROR", f"abbr[{i}]", "Must be [short:str, long:str]")
    # synonyms: list of [a,b]
    syn = data.get("synonyms", [])
    if syn is not None and not isinstance(syn, list):
        rep.add(path, "ERROR", "synonyms", "Must be a list")
    else:
        for i, pair in enumerate(syn or []):
            if not (isinstance(pair, list) and len(pair) == 2 and all(isinstance(x, str) for x in pair)):
                rep.add(path, "ERROR", f"synonyms[{i}]", "Must be [a:str, b:str]")
    # hedging_extra: list[str]
    hed = data.get("hedging_extra", [])
    if hed is not None and not isinstance(hed, list):
        rep.add(path, "ERROR", "hedging_extra", "Must be a list of strings")
    else:
        for i, it in enumerate(hed or []):
            if not isinstance(it, str):
                rep.add(path, "ERROR", f"hedging_extra[{i}]", "Must be string")

def discover_paths(root: Path) -> List[Path]:
    """Ищем известные файлы по структуре проекта."""
    paths: List[Path] = []
    # rules
    common = root / "rules" / "common.yaml"
    if common.exists(): paths.append(common)
    # themes
    themes_dir = root / "themes"
    if themes_dir.exists():
        for p in themes_dir.rglob("*.yaml"):
            # игнорируем временные/бэкап
            if any(s in p.name for s in (".swp", ".bak", ".tmp")):
                continue
            paths.append(p)
    return paths

def lint_path(path: Path, rep: Report, id_registry: Dict[str, Path]):
    name = path.name.lower()
    parent = path.parent.name.lower()

    try:
        if name == "triggers.yaml":
            lint_triggers_yaml(path, rep)
        elif name in ("lexicon.yaml", "shared-lexicon.yaml"):
            lint_lexicon_yaml(path, rep)
        else:
            # предполагаем rules.yaml / common.yaml
            lint_rules_yaml(path, rep, id_registry)
    except RuntimeError as e:
        rep.add(path, "ERROR", "file", str(e))

def main():
    ap = argparse.ArgumentParser(description="Linter for Singularis rules & themes")
    ap.add_argument("--root", type=str, default=".", help="project root (for autodiscovery)")
    ap.add_argument("--paths", type=str, nargs="*", help="explicit file paths (overrides autodiscovery)")
    ap.add_argument("--json", type=str, help="write JSON report to this file")
    ap.add_argument("--strict", action="store_true", help="treat warnings as errors")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if args.paths:
        paths = [Path(p).resolve() for p in args.paths]
    else:
        paths = discover_paths(root)

    if not paths:
        print("No YAML files found (rules/common.yaml or themes/*/*.yaml).", file=sys.stderr)
        return 2

    rep = Report()
    id_registry: Dict[str, Path] = {}

    for p in paths:
        lint_path(p, rep, id_registry)

    # Вывод
    for iss in rep.errors:
        print(f"[ERROR] {iss.path}: {iss.where}: {iss.msg}")
    for iss in rep.warns:
        print(f"[WARN ] {iss.path}: {iss.where}: {iss.msg}")

    if args.json:
        Path(args.json).write_text(json.dumps(rep.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON report saved to {args.json}")

    # Код выхода
    if rep.has_errors() or (args.strict and rep.warns):
        return 2
    return 0

if __name__ == "__main__":
    sys.exit(main())
