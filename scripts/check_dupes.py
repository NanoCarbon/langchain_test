# scripts/check_dupes.py
import os, collections

import chromadb

# Adjust these paths if your script lives elsewhere
ROOT = os.path.dirname(os.path.abspath(__file__))         # .../scripts
PROJECT = os.path.dirname(ROOT)                           # project root
DB_DIR = os.path.join(PROJECT, "data", "index")
COLLECTION = "vacation_photos"

BATCH = 5000  # fetch in chunks to avoid large responses/timeouts

def main():
    client = chromadb.PersistentClient(path=DB_DIR)
    col = client.get_or_create_collection(COLLECTION)

    total = col.count()
    print(f"Total vectors: {total}")

    all_ids = []
    all_paths = []

    offset = 0
    while offset < total:
        res = col.get(include=["metadatas"], limit=min(BATCH, total - offset), offset=offset)
        ids = res.get("ids", []) or []
        metas = res.get("metadatas", []) or []
        all_ids.extend(ids)
        for m in metas:
            p = (m or {}).get("path", "")
            all_paths.append(p)
        offset += len(ids)

    # 1) Duplicate IDs (shouldn’t happen unless earlier code changed ID logic)
    id_counts = collections.Counter(all_ids)
    dup_id_list = [i for i, c in id_counts.items() if c > 1]

    # 2) Duplicate image paths in metadata (most common)
    path_counts = collections.Counter([p for p in all_paths if p])
    dup_path_list = [p for p, c in path_counts.items() if c > 1]

    print(f"Duplicate ids: {len(dup_id_list)} (showing up to 5): {dup_id_list[:5]}")
    print(f"Duplicate paths: {len(dup_path_list)} (showing up to 5): {dup_path_list[:5]}")

if __name__ == "__main__":
    main()
