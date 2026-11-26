#!/usr/bin/env python3
"""
Compile paper/main.typ to PDF using Typst and scrub PDF metadata for strict anonymity.

Usage:
  python paper/build.py                         # outputs paper/draft_apmcm26302266.pdf
  python paper/build.py --out submission/Aapmcm26302266.pdf
"""

import argparse
import os
import sys
import typst
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


def main():
    parser = argparse.ArgumentParser(description="Build 2026 APMCM paper PDF via Typst")
    parser.add_argument("--src", default=DEFAULT_SRC, help="Path to main .typ file")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output PDF path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    typst.compile(args.src, output=args.out, root=BASE_DIR)
    scrub_metadata(args.out)
    reader = PdfReader(args.out)
    print(f"[OK] Compiled & sanitized: {args.out} ({len(reader.pages)} pages)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
