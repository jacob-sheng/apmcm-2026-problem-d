#!/usr/bin/env python3
"""
2026 APMCM Pre-Submission Hard-Gate Auditor & PDF Metadata Sanitizer

Usage:
  python scripts/audit_submission.py --problem D --team apmcm26302266
  python scripts/audit_submission.py --problem D --team apmcm26302266 --allow-missing-cns
"""

import argparse
import os
import re
import sys
import zipfile

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(BASE_DIR)
SUBMISSION_DIR = os.path.join(BASE_DIR, "submission")
PROBLEMS_DIR = os.path.join(BASE_DIR, "problems")
OFFICIAL_DATE_DIR = os.path.join(
    REPO_ROOT,
    "01_2026赛题与数据集",
    "Problem_D_量子计算钙钛矿稀有目标搜索",
    "date",
)

CANONICAL_RAW_FILES = {
    "02_castelli_train.csv",
    "03_castelli_val.csv",
    "04_castelli_test_holdout.csv",
    "06_castelli_no_physics.csv",
    "07_castelli_q2_pool.csv",
    "08_castelli_q3_pool.csv",
    "11_castelli_q2_pool_blind.csv",
    "12_castelli_q3_pool_blind.csv",
    "sha256_public.json",
}

MAX_ZIP_BYTES = 25 * 1024 * 1024
MAX_MAIN_BODY_PAGES = 25

# Generic leak patterns (local paths, common PII headers, Chinese residual characters)
PATH_LEAK_RE = re.compile(
    r"([A-Za-z]:[/\\]+Users[/\\]+|[A-Za-z]:[/\\]+Codex|/Users/|/home/|/mnt/[a-zA-Z]/|@(?:[\w.-]+\.)?(edu\.cn|qq\.com|163\.com|gmail\.com|outlook\.com|126\.com|foxmail\.com))",
    re.I,
)
CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def strip_pdf_metadata(pdf_path):
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        print("[WARN] pypdf not installed; cannot strip metadata.")
        return False

    reader = PdfReader(pdf_path)
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    writer.add_metadata({"/Title": "", "/Author": "", "/Subject": "", "/Creator": "", "/Producer": ""})
    tmp_path = pdf_path + ".clean.tmp"
    with open(tmp_path, "wb") as f:
        writer.write(f)
    os.replace(tmp_path, pdf_path)
    return True


def audit_pdf(pdf_path, team_id, extra_forbidden=None):
    errors = []
    warnings = []
    if not os.path.exists(pdf_path):
        return [f"Missing main paper PDF: {pdf_path}"], warnings

    try:
        import pymupdf as fitz
    except ImportError:
        return ["PyMuPDF (pymupdf) is required to audit PDF text & metadata."], warnings

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    print(f"[INFO] Main PDF: {os.path.basename(pdf_path)} ({total_pages} pages)")

    # 1. Metadata audit
    meta = doc.metadata or {}
    for k in ("author", "creator", "title", "subject", "keywords"):
        val = (meta.get(k) or "").strip()
        if val and (PATH_LEAK_RE.search(val) or CJK_RE.search(val) or any(t.lower() in val.lower() for t in (extra_forbidden or []))):
            errors.append(f"Suspicious PDF metadata '{k}': {val}")
        elif k == "author" and val:
            errors.append(f"PDF metadata 'author' MUST be empty for strict anonymity, got: {val}")

    # 2. Per-page header, CJK, and PII leak audit
    missing_header_pages = []
    cjk_pages = []
    appendices_page = None
    for idx, page in enumerate(doc, start=1):
        text = page.get_text("text")
        expected_team = team_id.lower()
        if expected_team not in text.lower():
            missing_header_pages.append(idx)
        if CJK_RE.search(text):
            cjk_pages.append(idx)
        for m in PATH_LEAK_RE.finditer(text):
            errors.append(f"Page {idx}: possible path/email leak '{m.group(0)}'")
        for word in extra_forbidden or []:
            if word and word.lower() in text.lower():
                errors.append(f"Page {idx}: forbidden identity keyword detected!")
        if idx >= 3 and appendices_page is None:
            lines = [ln.strip().lower() for ln in text.splitlines() if ln.strip()]
            if any(ln == "appendices" or ln.startswith("appendix a") for ln in lines[:10]):
                appendices_page = idx

    if missing_header_pages:
        errors.append(f"Team Control Number '{team_id}' missing on page(s): {missing_header_pages}")
    if cjk_pages:
        errors.append(f"Chinese characters detected in English paper on page(s): {cjk_pages}")

    # 3. Page 1 Summary & Page 2 Contents & <= 25 Main-Body Page Budget
    if total_pages >= 1:
        p1 = doc[0].get_text("text").lower()
        if "summary" not in p1 and "abstract" not in p1:
            warnings.append("Page 1 does not contain 'Summary' or 'Abstract' heading; verify Page 1 is the Summary Sheet.")
    if total_pages >= 2:
        p2 = doc[1].get_text("text").lower()
        if "contents" not in p2:
            errors.append("Page 2 does not contain 'Contents' table of contents.")

    main_body_pages = (appendices_page - 3) if appendices_page is not None else max(0, total_pages - 2)
    print(f"[INFO] Main Body + References pages (excl. P1/P2): {main_body_pages} (limit <= {MAX_MAIN_BODY_PAGES})")
    if main_body_pages > MAX_MAIN_BODY_PAGES:
        errors.append(f"Main body ({main_body_pages} pages) exceeds {MAX_MAIN_BODY_PAGES}-page limit!")

    doc.close()
    return errors, warnings


def audit_ai_docx(docx_path, problem, team_id, extra_forbidden=None):
    errors = []
    warnings = []
    if not os.path.exists(docx_path):
        return errors, warnings

    try:
        import docx
    except ImportError:
        warnings.append("python-docx not installed; skipped AI disclosure .docx audit.")
        return errors, warnings

    d = docx.Document(docx_path)
    cp = d.core_properties
    for prop_name in ("author", "last_modified_by", "title", "subject", "comments", "keywords"):
        val = (getattr(cp, prop_name, "") or "").strip()
        if val:
            errors.append(f"AI disclosure docx core_property '{prop_name}' is non-empty: {val!r}")

    chunks = [p.text for p in d.paragraphs]
    for tbl in d.tables:
        for row in tbl.rows:
            for cell in row.cells:
                chunks.append(cell.text)
    full_txt = "\n".join(chunks)

    if team_id.lower() not in full_txt.lower():
        errors.append(f"AI disclosure docx missing Team Control Number '{team_id}'.")
    if f"：{problem.upper()}" not in full_txt and f": {problem.upper()}" not in full_txt and problem.upper() not in d.paragraphs[1].text:
        errors.append(f"AI disclosure docx missing Problem Chosen '{problem.upper()}'.")
    if "待填" in full_txt or "[在此记录" in full_txt or "[示例" in full_txt:
        errors.append("AI disclosure docx still contains unfilled template placeholders.")
    for m in PATH_LEAK_RE.finditer(full_txt):
        errors.append(f"AI disclosure docx contains path/email leak '{m.group(0)}'")
    for word in extra_forbidden or []:
        if word and word.lower() in full_txt.lower():
            errors.append("AI disclosure docx contains forbidden identity keyword!")
    print(f"[INFO] Audited AI disclosure docx: {os.path.basename(docx_path)} (clean)")
    return errors, warnings


def audit_deliverables(problem, team_id, allow_missing_cns=False, extra_forbidden=None):
    errors = []
    warnings = []
    prefix = f"{problem.upper()}{team_id}"
    pdf_path = os.path.join(SUBMISSION_DIR, f"{prefix}.pdf")

    # Commitment sheet check
    cns_candidates = [os.path.join(SUBMISSION_DIR, f"{prefix}cns.{ext}") for ext in ("pdf", "jpg", "png")]
    cns_found = [p for p in cns_candidates if os.path.exists(p)]
    if not cns_found:
        msg = f"Missing Commitment Sheet: expected {prefix}cns.<pdf|jpg|png> in submission/"
        if allow_missing_cns:
            warnings.append(f"{msg} (allowed via --allow-missing-cns)")
        else:
            errors.append(msg)
    else:
        print(f"[INFO] Commitment Sheet found: {os.path.basename(cns_found[0])}")

    # Supporting materials check
    fj_candidates = [os.path.join(SUBMISSION_DIR, f"{prefix}fj.{ext}") for ext in ("zip", "rar")]
    fj_found = [p for p in fj_candidates if os.path.exists(p)]
    if not fj_found:
        warnings.append(f"No supporting archive ({prefix}fj.zip/.rar) found (REQUIRED if self-written code exists).")
    else:
        fj_path = fj_found[0]
        fj_size = os.path.getsize(fj_path)
        print(f"[INFO] Supporting Archive found: {os.path.basename(fj_path)} ({fj_size / (1024 * 1024):.2f} MB)")
        if fj_size >= MAX_ZIP_BYTES:
            errors.append(f"Supporting archive {os.path.basename(fj_path)} ({fj_size / (1024 * 1024):.2f} MB) exceeds 25 MB limit!")
        if fj_path.endswith(".zip"):
            raw_files = set(CANONICAL_RAW_FILES)
            for root_dir in (PROBLEMS_DIR, OFFICIAL_DATE_DIR):
                if os.path.isdir(root_dir):
                    for _, _, files in os.walk(root_dir):
                        for fn in files:
                            raw_files.add(fn.lower())
            with zipfile.ZipFile(fj_path, "r") as zf:
                names = zf.namelist()
                for n in names:
                    base = os.path.basename(n).lower()
                    if base and base in raw_files:
                        errors.append(f"Supporting archive contains official raw problem file '{n}' — MUST EXCLUDE raw data!")
                    if base.endswith(".npz"):
                        errors.append(f"Supporting archive contains forbidden .npz file '{n}'!")
                    if "dryrun" in base and base.endswith(".json"):
                        errors.append(f"Supporting archive contains dry-run file '{n}'!")
                    if base.startswith(".env") or base.endswith(".env"):
                        errors.append(f"Supporting archive contains .env file '{n}'!")
                    if n.lower().endswith((".py", ".m", ".r", ".txt", ".md", ".json", ".csv", ".typ")):
                        try:
                            content = zf.read(n).decode("utf-8", errors="ignore")
                            m = PATH_LEAK_RE.search(content)
                            if m:
                                errors.append(f"Possible path/email leak inside archive file '{n}': {m.group(0)}")
                            for word in extra_forbidden or []:
                                if word and word.lower() in content.lower():
                                    errors.append(f"Forbidden identity keyword inside archive file '{n}'")
                        except Exception:
                            pass

    # Optional AI disclosure draft check
    ai_docx = os.path.join(SUBMISSION_DIR, "AI工具使用详情说明_本队草稿.docx")
    docx_errors, docx_warnings = audit_ai_docx(ai_docx, problem, team_id, extra_forbidden=extra_forbidden)
    errors.extend(docx_errors)
    warnings.extend(docx_warnings)

    # On-disk documentation / source paper leak check (REPORT.md and paper/main.typ)
    for rel_parts in (("v2_optimized", "REPORT.md"), ("paper", "main.typ")):
        doc_path = os.path.join(BASE_DIR, *rel_parts)
        if os.path.isfile(doc_path):
            rel_label = "/".join(rel_parts)
            with open(doc_path, "r", encoding="utf-8", errors="ignore") as f:
                doc_txt = f.read()
            for m in PATH_LEAK_RE.finditer(doc_txt):
                errors.append(f"On-disk document '{rel_label}' contains path/email leak '{m.group(0)}'")
            for word in extra_forbidden or []:
                if word and word.lower() in doc_txt.lower():
                    errors.append(f"On-disk document '{rel_label}' contains forbidden identity keyword!")

    return pdf_path, errors, warnings


def main():
    parser = argparse.ArgumentParser(description="Audit 2026 APMCM submission deliverables")
    parser.add_argument("--problem", required=True, help="Selected problem letter (A/B/C/D)")
    parser.add_argument("--team", default="apmcm26302266", help="Team Control Number")
    parser.add_argument("--strip-metadata", action="store_true", help="Strip PDF metadata before auditing")
    parser.add_argument("--allow-missing-cns", action="store_true", help="Warn instead of failing if Commitment Sheet is kept offline")
    parser.add_argument("--forbid", nargs="*", default=[], help="Extra forbidden identity strings to check")
    args = parser.parse_args()

    pdf_path, errors, warnings = audit_deliverables(
        args.problem,
        args.team,
        allow_missing_cns=args.allow_missing_cns,
        extra_forbidden=args.forbid,
    )
    if args.strip_metadata and os.path.exists(pdf_path):
        if strip_pdf_metadata(pdf_path):
            print(f"[INFO] Stripped PDF metadata from {os.path.basename(pdf_path)}")

    pdf_errors, pdf_warnings = audit_pdf(pdf_path, args.team, extra_forbidden=args.forbid)
    errors.extend(pdf_errors)
    warnings.extend(pdf_warnings)

    print("\n=== AUDIT SUMMARY ===")
    for w in warnings:
        print(f"[WARN]  {w}")
    for e in errors:
        print(f"[ERROR] {e}")

    if errors:
        print(f"\nAUDIT FAILED with {len(errors)} error(s). DO NOT SUBMIT until resolved!")
        return 1
    print("\nAUDIT PASSED (0 errors).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
