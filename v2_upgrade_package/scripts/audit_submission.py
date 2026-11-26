#!/usr/bin/env python3
"""
2026 APMCM Pre-Submission Hard-Gate Auditor & PDF Metadata Sanitizer
Enforces docs/APMCM_2026_DEV_SPEC.md and docs/SUBMISSION_CHECKLIST.md.

Usage:
  python scripts/audit_submission.py --problem A --team apmcm26302266
  python scripts/audit_submission.py --problem A --team apmcm26302266 --strip-metadata
"""

import argparse
import os
import re
import sys
import zipfile

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBMISSION_DIR = os.path.join(BASE_DIR, "submission")
PROBLEMS_DIR = os.path.join(BASE_DIR, "problems")

# Generic leak patterns (local paths, common PII headers, Chinese residual characters)
PATH_LEAK_RE = re.compile(r"([A-Za-z]:\\Users\\|/Users/|/home/|@[\w.-]+\.(edu\.cn|qq\.com|163\.com|gmail\.com))", re.I)
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

    if missing_header_pages:
        errors.append(f"Team Control Number '{team_id}' missing on page(s): {missing_header_pages}")
    if cjk_pages:
        errors.append(f"Chinese characters detected in English paper on page(s): {cjk_pages}")

    # 3. Page 1 Summary check
    if total_pages >= 1:
        p1 = doc[0].get_text("text").lower()
        if "summary" not in p1 and "abstract" not in p1:
            warnings.append("Page 1 does not contain 'Summary' or 'Abstract' heading; verify Page 1 is the Summary Sheet.")

    doc.close()
    return errors, warnings


def audit_deliverables(problem, team_id):
    errors = []
    warnings = []
    prefix = f"{problem.upper()}{team_id}"
    pdf_path = os.path.join(SUBMISSION_DIR, f"{prefix}.pdf")

    # Commitment sheet check
    cns_candidates = [os.path.join(SUBMISSION_DIR, f"{prefix}cns.{ext}") for ext in ("pdf", "jpg", "png")]
    cns_found = [p for p in cns_candidates if os.path.exists(p)]
    if not cns_found:
        errors.append(f"Missing Commitment Sheet: expected {prefix}cns.<pdf|jpg|png> in submission/")
    else:
        print(f"[INFO] Commitment Sheet found: {os.path.basename(cns_found[0])}")

    # Supporting materials check
    fj_candidates = [os.path.join(SUBMISSION_DIR, f"{prefix}fj.{ext}") for ext in ("zip", "rar")]
    fj_found = [p for p in fj_candidates if os.path.exists(p)]
    if not fj_found:
        warnings.append(f"No supporting archive ({prefix}fj.zip/.rar) found (REQUIRED if self-written code exists).")
    else:
        fj_path = fj_found[0]
        print(f"[INFO] Supporting Archive found: {os.path.basename(fj_path)}")
        if fj_path.endswith(".zip"):
            raw_files = set()
            if os.path.isdir(PROBLEMS_DIR):
                for _, _, files in os.walk(PROBLEMS_DIR):
                    for fn in files:
                        raw_files.add(fn.lower())
            with zipfile.ZipFile(fj_path, "r") as zf:
                names = zf.namelist()
                for n in names:
                    base = os.path.basename(n).lower()
                    if base and base in raw_files:
                        errors.append(f"Supporting archive contains official raw problem file '{n}' — MUST EXCLUDE raw data!")
                    if n.endswith((".py", ".m", ".r", ".R", ".txt", ".md")):
                        try:
                            content = zf.read(n).decode("utf-8", errors="ignore")
                            if PATH_LEAK_RE.search(content):
                                errors.append(f"Possible path/email leak inside archive file '{n}'")
                        except Exception:
                            pass

    return pdf_path, errors, warnings


def main():
    parser = argparse.ArgumentParser(description="Audit 2026 APMCM submission deliverables")
    parser.add_argument("--problem", required=True, help="Selected problem letter (A/B/C/D)")
    parser.add_argument("--team", default="apmcm26302266", help="Team Control Number")
    parser.add_argument("--strip-metadata", action="store_true", help="Strip PDF metadata before auditing")
    parser.add_argument("--forbid", nargs="*", default=[], help="Extra forbidden identity strings to check")
    args = parser.parse_args()

    pdf_path, errors, warnings = audit_deliverables(args.problem, args.team)
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
