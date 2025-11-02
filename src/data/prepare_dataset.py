"""Preprocess raw Sohu news XML into clean text splits.

This script converts the provided `.dat` XML export into UTF-8 plain text files
(one document per line). The cleaned text is stored under `artifacts/corpus`
with train/validation/test splits that downstream scripts consume.
"""

from __future__ import annotations

import argparse
import logging
import random
import re
from pathlib import Path
from typing import Iterable, List

LOGGER = logging.getLogger(__name__)


def iter_raw_docs(path: Path) -> Iterable[str]:
    """Yield raw XML snippets for each `<doc>` element in the file."""
    current: List[str] = []
    in_doc = False
    with path.open("r", encoding="gb18030", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped == "<doc>":
                current = [stripped]
                in_doc = True
                continue
            if not in_doc:
                continue
            current.append(stripped)
            if stripped == "</doc>":
                in_doc = False
                yield "\n".join(current)
                current = []


CONTENTTITLE_PATTERN = re.compile(r"<contenttitle>(.*?)</contenttitle>", re.IGNORECASE | re.DOTALL)
CONTENT_PATTERN = re.compile(r"<content>(.*?)</content>", re.IGNORECASE | re.DOTALL)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")


def extract_text(doc_str: str) -> str | None:
    """Extract title and body text out of an XML snippet."""
    title_match = CONTENTTITLE_PATTERN.search(doc_str)
    content_match = CONTENT_PATTERN.search(doc_str)
    title = title_match.group(1).strip() if title_match else ""
    content = content_match.group(1).strip() if content_match else ""
    merged = "\n".join(part for part in (title, content) if part)
    if not merged:
        return None
    # Drop residual markup and collapse whitespace
    merged = HTML_TAG_PATTERN.sub(" ", merged)
    merged = merged.replace("&nbsp;", " ")
    merged = merged.replace("&amp;", "&")
    merged = WHITESPACE_PATTERN.sub(" ", merged)
    cleaned = merged.strip()
    if len(cleaned) < 32:
        return None
    return cleaned


def preprocess_dataset(raw_path: Path, output_dir: Path, train_ratio: float, val_ratio: float, seed: int) -> None:
    docs: List[str] = []
    for idx, raw_doc in enumerate(iter_raw_docs(raw_path)):
        text = extract_text(raw_doc)
        if text:
            docs.append(text)
        if (idx + 1) % 5000 == 0:
            LOGGER.info("Processed %s documents", idx + 1)
    if not docs:
        raise RuntimeError("No documents extracted; please verify input file")

    random.Random(seed).shuffle(docs)
    total = len(docs)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        "train.txt": docs[:train_end],
        "val.txt": docs[train_end:val_end],
        "test.txt": docs[val_end:],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, lines in splits.items():
        target = output_dir / filename
        with target.open("w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
        LOGGER.info("Wrote %s with %s documents", target, len(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/news_sohusite_xml.dat"), help="Path to the raw .dat source")
    parser.add_argument("--output", type=Path, default=Path("artifacts/corpus"), help="Directory to store processed text splits")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Proportion of documents for training")
    parser.add_argument("--val-ratio", type=float, default=0.05, help="Proportion of documents for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splits")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO), format="[%(levelname)s] %(message)s")
    preprocess_dataset(args.input, args.output, args.train_ratio, args.val_ratio, args.seed)


if __name__ == "__main__":
    main()
