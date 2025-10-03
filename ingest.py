import os
import glob
import json
from datetime import datetime

import chromadb
from PIL import Image, ExifTags
from sentence_transformers import SentenceTransformer

# --------- CONFIG ---------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PHOTOS_DIR = os.path.join(PROJECT_DIR, "photos", "S23 Camera Backup", "Camera")
THUMBS_DIR = os.path.join(PROJECT_DIR, "data", "thumbs")
DB_DIR = os.path.join(PROJECT_DIR, "data", "index")
MODEL_NAME = "clip-ViT-B-32"   # local CLIP
THUMB_SIZE = (384, 384)
COLLECTION_NAME = "vacation_photos"
# --------------------------

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def list_images(root: str):
    paths = []
    for ext in ALLOWED_EXTS:
        paths.extend(glob.glob(os.path.join(root, f"**/*{ext}"), recursive=True))
    return sorted(paths)

def ensure_dirs():
    os.makedirs(THUMBS_DIR, exist_ok=True)
    os.makedirs(DB_DIR, exist_ok=True)

# --- NEW: fix orientation ---
def load_with_exif_orientation(path: str) -> Image.Image:
    """Open an image and auto-rotate it based on EXIF orientation."""
    img = Image.open(path)
    try:
        exif = img._getexif()
        if exif is not None:
            orientation_key = next(
                (k for k, v in ExifTags.TAGS.items() if v == "Orientation"), None
            )
            if orientation_key and orientation_key in exif:
                orientation = exif[orientation_key]
                if orientation == 3:
                    img = img.rotate(180, expand=True)
                elif orientation == 6:
                    img = img.rotate(270, expand=True)
                elif orientation == 8:
                    img = img.rotate(90, expand=True)
    except Exception:
        pass
    return img
# ----------------------------

def make_thumb(src_path: str) -> str:
    base = os.path.basename(src_path)
    thumb_path = os.path.join(THUMBS_DIR, base)
    if not os.path.exists(thumb_path):
        try:
            img = load_with_exif_orientation(src_path)
            img.thumbnail(THUMB_SIZE)
            img.save(thumb_path)
        except Exception as e:
            print(f"[WARN] Could not thumbnail {src_path}: {e}")
            return ""
    return thumb_path

def extract_exif_datetime(path: str):
    try:
        img = Image.open(path)
        exif = img._getexif()
        if not exif:
            return None
        exif_decoded = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
        dt = exif_decoded.get("DateTimeOriginal") or exif_decoded.get("DateTime")
        if isinstance(dt, str):
            dt = dt.replace(":", "-", 2)  # convert YYYY:MM:DD → YYYY-MM-DD
            return dt
    except Exception:
        pass
    return None

def main():
    ensure_dirs()

    print(f"[*] Looking in {PHOTOS_DIR}")
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=DB_DIR)
    col = client.get_or_create_collection(COLLECTION_NAME)

    images = list_images(PHOTOS_DIR)
    if not images:
        print("[ERR] No images found in photos folder.")
        return

    print(f"[*] Found {len(images)} images")

    to_add_ids, to_add_embeds, to_add_metas, to_add_docs = [], [], [], []
    for idx, path in enumerate(images, 1):
        file_id = os.path.abspath(path)
        thumb_path = make_thumb(path)
        folder = os.path.basename(os.path.dirname(path))
        dt = extract_exif_datetime(path)
        metadata = {
            "path": path.replace("\\", "/"),
            "thumb": thumb_path.replace("\\", "/"),
            "folder": folder,
            "datetime": dt or "",
            "tags": "", 
        }

        try:
            img = load_with_exif_orientation(path).convert("RGB")
            emb = model.encode(img, convert_to_numpy=True, normalize_embeddings=True)
        except Exception as e:
            print(f"[WARN] Skipping {path}: {e}")
            continue

        to_add_ids.append(file_id)
        to_add_embeds.append(emb.tolist())
        to_add_metas.append(metadata)
        to_add_docs.append(json.dumps({"desc": folder}))

        if len(to_add_ids) >= 64:
            col.add(ids=to_add_ids, embeddings=to_add_embeds, metadatas=to_add_metas, documents=to_add_docs)
            to_add_ids, to_add_embeds, to_add_metas, to_add_docs = [], [], [], []
            print(f"    [+] Indexed {idx}/{len(images)}")

    if to_add_ids:
        col.add(ids=to_add_ids, embeddings=to_add_embeds, metadatas=to_add_metas, documents=to_add_docs)

    print(f"[*] Done. Total vectors now: {col.count()}")
    print(f"[*] Thumbnails in {THUMBS_DIR}")
    print(f"[*] DB at {DB_DIR}")

if __name__ == "__main__":
    main()
