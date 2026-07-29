"""Standalone entry point for building and checking the local Chroma knowledge base."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.vector_store import VectorStoreService


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Chroma knowledge base from data/ documents.")
    parser.add_argument("--rebuild", action="store_true", help="Clear the existing collection before rebuilding.")
    parser.add_argument("--check", action="store_true", help="Only check whether the collection contains documents.")
    args = parser.parse_args()

    if args.rebuild and args.check:
        parser.error("--rebuild 和 --check 不能同时使用")

    store = VectorStoreService()
    if args.check:
        count = store.collection_count()
        print(f"Knowledge base document chunks: {count}")
        if count == 0:
            print("Knowledge base is empty. Run: python -m rag.build_knowledge_base")
            return 1
        print("Knowledge base is ready.")
        return 0

    summary = store.load_document(force_rebuild=args.rebuild)
    print("Knowledge base build completed.")
    print(f"Added files: {summary['added_files']}")
    print(f"Skipped files: {summary['skipped_files']}")
    print(f"Failed files: {summary['failed_files']}")
    print(f"Document chunks: {summary['document_count']}")
    return 0 if summary["document_count"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
