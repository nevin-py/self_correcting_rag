"""
Migrate old chat_* ChromaDB collections to user_* format.

Old: chat_{user_id_hex[:12]}  (one collection per user, misnamed)
New: user_{user_id_hex[:16]}  (correct naming)

Run: python -m scripts.migrate_chroma_collections
"""

import chromadb


def migrate():
    client = chromadb.PersistentClient(path="./data/chroma")
    collections = client.list_collections()
    migrated = 0

    for col in collections:
        name = col.name
        if name.startswith("chat_"):
            # Extract user hex from old name: chat_{hex12}
            user_hex_12 = name[5:]
            new_name = f"user_{user_hex_12}{'0' * 4}"  # pad to 16 chars

            print(f"  Migrating: {name} -> {new_name}")
            try:
                client.rename_collection(name, new_name)
                migrated += 1
            except Exception as e:
                print(f"    Error: {e}")

    print(f"\nMigrated {migrated} collections.")


if __name__ == "__main__":
    migrate()
