#!/usr/bin/env python3
"""
Compile paper/main.typ to PDF using Typst, scrub PDF metadata for strict anonymity,
and enforce the 25-page main-body limit (Summary on P1, Contents on P2, Main Body <= 25 pp).

Usage:
  python paper/build.py                         # outputs paper/draft_apmcm26302266.pdf
  python paper/build.py --out submission/Dapmcm26302266.pdf
"""

import argparse
import os
import re
import sys
import typst
import pymupdf as fitz
from pypdf import PdfReader, PdfWriter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SRC = os.path.join(BASE_DIR, "paper", "main.typ")
DEFAULT_OUT = os.path.join(BASE_DIR, "paper", "draft_apmcm26302266.pdf")


def scrub_metadata(pdf_path):
    reader = PdfReader(pdf_path)
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    writer.add_metadata({"/Title": "", "/Author": "", "/Subject": "", "/Creator": "", "/Producer": ""})
    tmp = pdf_path + ".clean.tmp"
    with open(tmp, "wb") as f:
        writer.write(f)
    os.replace(tmp, pdf_path)


def check_page_budget(pdf_path, max_main_pages=25):
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    appendices_page = None
    for idx in range(2, total_pages):  # physical page >= 3
        txt = doc[idx].get_text("text")
        lines = [ln.strip().lower() for ln in txt.splitlines() if ln.strip()]
        if any(ln == "appendices" or ln.startswith("appendix a") for ln in lines[:10]):
            appendices_page = idx + 1  # 1-indexed physical page
            break
    doc.close()

    if appendices_page is None:
        main_body_excl = max(0, total_pages - 2)
        main_body_incl = total_pages
        print(
            f"[WARN] Could not locate 'Appendices' heading; total={total_pages}, "
            f"main_excl_front={main_body_excl}"
        )
    else:
        # Page 1 = Summary, Page 2 = Contents, Pages 3..(appendices_page - 1) = Main Body + References
        main_body_excl = appendices_page - 3
        main_body_incl = appendices_page - 1
        print(
            f"[INFO] Pagination: total={total_pages} pp | P1=Summary | P2=Contents | "
            f"Main Body+Refs=P3..P{appendices_page - 1} ({main_body_excl} pp excl. front, "
            f"{main_body_incl} pp incl. front) | Appendices=P{appendices_page}..P{total_pages}"
        )

    if main_body_excl > max_main_pages:
        print(
            f"[ERROR] Main body ({main_body_excl} pages) exceeds {max_main_pages}-page limit!",
            file=sys.stderr,
        )
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Build 2026 APMCM paper PDF via Typst")
    parser.add_argument("--src", default=DEFAULT_SRC, help="Path to main .typ file")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output PDF path")
    parser.add_argument("--max-main-pages", type=int, default=25, help="Max allowed main-body pages")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    typst.compile(args.src, output=args.out, root=BASE_DIR)
    scrub_metadata(args.out)
    reader = PdfReader(args.out)
    print(f"[OK] Compiled & sanitized: {args.out} ({len(reader.pages)} pages)")
    if not check_page_budget(args.out, max_main_pages=args.max_main_pages):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
