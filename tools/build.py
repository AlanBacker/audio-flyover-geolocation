#!/usr/bin/env python3
"""Check, sync and package the language editions of the skill.

  python tools/build.py check     validate every edition (CI runs this)
  python tools/build.py sync      copy the canonical scripts/ (en) into the other editions
  python tools/build.py package   write dist/audio-flyover-geolocation-<lang>.skill

The English edition holds the canonical scripts/; every other edition must carry a
byte-identical copy, so a fix is made once in skills/en and then synced.
"""
import argparse
import filecmp
import re
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL = "audio-flyover-geolocation"
CANON = "en"
LANGS = ["en", "zh-CN"]
SKIP = {"__pycache__", ".DS_Store"}
CJK = re.compile(r"[　-〿一-鿿＀-￯]")
FRONT_KEYS = {"name", "description", "license", "allowed-tools", "metadata", "compatibility"}


def skill_dir(lang):
    return ROOT / "skills" / lang / SKILL


def files(d):
    return sorted(p for p in d.rglob("*") if p.is_file() and not SKIP & set(p.parts) and p.suffix != ".pyc")


def frontmatter(text):
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not m:
        return None
    out, key = {}, None
    for line in m.group(1).splitlines():
        km = re.match(r"^([A-Za-z][\w-]*):\s*(.*)$", line)
        if km:
            key = km.group(1)
            out[key] = km.group(2).strip()
        elif key and line.startswith((" ", "\t")):
            out[key] = (out[key] + " " + line.strip()).strip()
    if out.get("description", "").startswith((">", "|")):
        out["description"] = out["description"].lstrip(">|-+ ").strip()
    return out


def check_edition(lang):
    errs = []
    d = skill_dir(lang)
    md = d / "SKILL.md"
    if not md.is_file():
        return [f"{lang}: SKILL.md missing"]
    text = md.read_text(encoding="utf-8")
    fm = frontmatter(text)
    if fm is None:
        return [f"{lang}: SKILL.md has no YAML frontmatter"]
    if fm.get("name") != SKILL:
        errs.append(f"{lang}: name is {fm.get('name')!r}, expected {SKILL!r}")
    desc = fm.get("description", "")
    if not desc:
        errs.append(f"{lang}: description missing")
    elif len(desc) > 1024:
        errs.append(f"{lang}: description is {len(desc)} chars (max 1024)")
    if set(fm) - FRONT_KEYS:
        errs.append(f"{lang}: unexpected frontmatter keys {sorted(set(fm) - FRONT_KEYS)}")
    if len(text.splitlines()) > 500:
        errs.append(f"{lang}: SKILL.md is over 500 lines")
    # every bundled path the docs point at must exist
    for p in files(d):
        if p.suffix != ".md":
            continue
        for ref in set(re.findall(r"\b((?:references|assets|scripts)/[\w./-]+\.\w+)", p.read_text(encoding="utf-8"))):
            if not (d / ref).exists():
                errs.append(f"{lang}: {p.relative_to(d)} points at missing {ref}")
    if lang == "en":
        for p in files(d):
            for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
                if CJK.search(line):
                    errs.append(f"en: CJK text in {p.relative_to(d)}:{i}")
                    break
    if lang != CANON:
        a, b = skill_dir(CANON) / "scripts", d / "scripts"
        ra = [p.relative_to(a) for p in files(a)]
        rb = [p.relative_to(b) for p in files(b)] if b.is_dir() else []
        if ra != rb:
            errs.append(f"{lang}: scripts/ file list differs from {CANON} (run: python tools/build.py sync)")
        else:
            bad = [str(r) for r in ra if not filecmp.cmp(a / r, b / r, shallow=False)]
            if bad:
                errs.append(f"{lang}: scripts differ from {CANON}: {bad} (run: python tools/build.py sync)")
    return errs


def cmd_check(_):
    errs = [e for lang in LANGS for e in check_edition(lang)]
    for e in errs:
        print("FAIL", e)
    if not errs:
        print(f"ok: {len(LANGS)} editions valid, scripts in sync")
    return 1 if errs else 0


def cmd_sync(_):
    src = skill_dir(CANON) / "scripts"
    for lang in LANGS:
        if lang == CANON:
            continue
        dst = skill_dir(lang) / "scripts"
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*SKIP, "*.pyc"))
        print(f"synced scripts -> skills/{lang}")
    return 0


def cmd_package(a):
    if cmd_check(a):
        return 1
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    for lang in LANGS:
        d = skill_dir(lang)
        z = out / f"{SKILL}-{lang}.skill"
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in files(d):
                zf.write(p, Path(SKILL) / p.relative_to(d))
        print(f"wrote {z.relative_to(ROOT)} ({z.stat().st_size // 1024} KB)")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check").set_defaults(fn=cmd_check)
    sub.add_parser("sync").set_defaults(fn=cmd_sync)
    s = sub.add_parser("package")
    s.add_argument("--out", default="dist")
    s.set_defaults(fn=cmd_package)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
