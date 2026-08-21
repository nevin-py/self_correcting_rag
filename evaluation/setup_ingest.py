"""
Ingest SQuAD passages into ChromaDB for evaluation.

Run once before evaluation:
    .venv/bin/python evaluation/setup_ingest.py
"""

import asyncio
import json
import uuid
from pathlib import Path

# Project root
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.documents.service import embed_n_store, chunking

PASSAGES_PATH = Path(__file__).parent / "squad_passages.json"

# Fixed IDs so evaluation can reference them
USER_ID = uuid.UUID("aabbccdd-1122-3344-5566-778899001122")
CHAT_ID = uuid.UUID("11223344-5566-7788-99aa-bbccddeeff00")


def main():
    passages = json.loads(PASSAGES_PATH.read_text(encoding="utf-8"))
    print(f"Ingesting {len(passages)} SQuAD passages into ChromaDB...")
    print(f"  user_id: {USER_ID}")
    print(f"  chat_id:  {CHAT_ID}")

    all_chunks = []
    for i, passage in enumerate(passages):
        metadata = {
            "source": f"squad_passage_{i}.txt",
            "filename": f"squad_passage_{i}.txt",
            "source_name": f"SQuAD Passage {i}",
            "file_type": ".txt",
            "file_size_kb": len(passage) // 1024,
        }
        chunks = chunking(text=passage, metadata=metadata)
        all_chunks.extend(chunks)
        print(f"  Passage {i}: {len(passage)} chars -> {len(chunks)} chunks")

    print(f"\nTotal chunks: {len(all_chunks)}")
    print("Embedding and storing...")

    collection = asyncio.run(
        embed_n_store(
            chunk_list=all_chunks,
            user_id=USER_ID,
            chat_id=CHAT_ID,
            scope="chat",
        )
    )
    print(f"Done! Collection: {collection}")
    print(f"\nUpdate test_rag.py USER_ID and CHAT_ID if needed:")
    print(f"  USER_ID = uuid.UUID(\"{USER_ID}\")")
    print(f"  CHAT_ID = uuid.UUID(\"{CHAT_ID}\")")


if __name__ == "__main__":
    main()
