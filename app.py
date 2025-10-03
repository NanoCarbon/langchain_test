import os
import json
import numpy as np
import streamlit as st
import chromadb
from PIL import Image, ExifTags
from sentence_transformers import SentenceTransformer

# --------- CONFIG ---------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(PROJECT_DIR, "data", "index")
COLLECTION_NAME = "vacation_photos"
MODEL_NAME = "clip-ViT-B-32"  # same as ingest
TOP_K_DEFAULT = 12

# Weighting for combined embedding
W_IMG = 0.7
W_TXT = 0.3
# --------------------------

def load_with_exif_orientation(path: str) -> Image.Image:
    img = Image.open(path)
    try:
        exif = img._getexif()
        if exif is not None:
            orientation_key = next((k for k, v in ExifTags.TAGS.items() if v == "Orientation"), None)
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

@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer(MODEL_NAME)

@st.cache_resource(show_spinner=False)
def load_collection():
    client = chromadb.PersistentClient(path=DB_DIR)
    return client.get_or_create_collection(COLLECTION_NAME)

def normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x)
    return x if n == 0 else x / n

def compute_combined_embedding(img_path: str, tags_text: str, model: SentenceTransformer) -> np.ndarray:
    """Produce a CLIP-space embedding mixing image + (optional) tags text."""
    img = load_with_exif_orientation(img_path).convert("RGB")
    e_img = model.encode(img, convert_to_numpy=True, normalize_embeddings=True)
    if tags_text.strip():
        e_txt = model.encode(tags_text, convert_to_numpy=True, normalize_embeddings=True)
        e = normalize(W_IMG * e_img + W_TXT * e_txt)
    else:
        e = e_img  # no tags yet
    return e

def search_images(query: str, k: int, model, collection):
    if not query.strip():
        return []
    qvec = model.encode(query, convert_to_numpy=True, normalize_embeddings=True).tolist()
    res = collection.query(query_embeddings=[qvec], n_results=k)
    hits = []
    for i in range(len(res["ids"][0])):
        meta = res["metadatas"][0][i] or {}
        # ensure fields exist
        meta.setdefault("path", "")
        meta.setdefault("thumb", "")
        meta.setdefault("folder", "")
        meta.setdefault("datetime", "")
        meta.setdefault("tags", "")  # NEW: string of comma-separated tags
        meta["_id"] = res["ids"][0][i]
        hits.append(meta)
    return hits

def update_tags_and_embedding(collection, model, meta: dict, new_tags: str):
    """Update Chroma metadata + vector for a single photo."""
    path = meta["path"]
    doc = json.dumps({"desc": meta.get("folder", ""), "tags": new_tags})
    emb = compute_combined_embedding(path, new_tags, model).tolist()
    # Update metadata too
    new_meta = dict(meta)
    new_meta["tags"] = new_tags
    # Chroma update
    collection.update(
        ids=[meta["_id"]],
        embeddings=[emb],
        metadatas=[{k: v for k, v in new_meta.items() if not k.startswith("_")}],
        documents=[doc],
    )

st.set_page_config(page_title="📸 Semantic Photo Search", layout="wide")
st.title("📸 Semantic Photo Search (Local)")

model = load_model()
collection = load_collection()

# Remember which card was clicked
if "selected" not in st.session_state:
    st.session_state.selected = None
if "last_results" not in st.session_state:
    st.session_state.last_results = []

with st.sidebar:
    st.markdown("### Settings")
    k = st.slider("Top-K results", min_value=4, max_value=48, value=TOP_K_DEFAULT, step=4)
    st.markdown("**Tip:** Try queries like `tokyo scenery`, `temple at night`, `beach sunrise`.")
    if st.session_state.selected:
        sel = st.session_state.selected
        st.markdown("---")
        st.subheader("Edit Tags")
        st.caption(os.path.basename(sel["path"]))
        tags_text = st.text_area("Tags (comma-separated)", value=sel.get("tags", ""), height=100)
        if st.button("Save tags"):
            with st.spinner("Updating vectors..."):
                update_tags_and_embedding(collection, model, sel, tags_text)
                # Update sidebar state + in-memory results
                sel["tags"] = tags_text
                # Also refresh the object in last_results list
                for r in st.session_state.last_results:
                    if r.get("_id") == sel.get("_id"):
                        r["tags"] = tags_text
                        break
            st.success("Saved & re-embedded ✅")

query = st.text_input("Search your photos", placeholder="e.g., tokyo scenery, temple, skyline, Mia")
go = st.button("Search")

if go and query:
    with st.spinner("Searching..."):
        results = search_images(query, k, model, collection)
        st.session_state.last_results = results
        st.session_state.selected = None  # reset selection on new search

results = st.session_state.last_results

if not results:
    st.info("Type a query and hit Search.")
else:
    cols_per_row = 4
    rows = (len(results) + cols_per_row - 1) // cols_per_row
    for r in range(rows):
        cols = st.columns(cols_per_row)
        for c in range(cols_per_row):
            i = r * cols_per_row + c
            if i >= len(results):
                break
            item = results[i]
            with cols[c]:
                thumb = item.get("thumb", "")
                if thumb and os.path.exists(thumb):
                    st.image(thumb, use_container_width=True)
                st.caption(f"{item.get('folder','')}  {item.get('datetime','')}")
                # Show tags preview
                if item.get("tags"):
                    st.code(item["tags"], language=None)
                # Select button to edit in sidebar
                if st.button("Edit tags", key=f"edit_{i}"):
                    st.session_state.selected = item
                # (Optional) Open file
                # if st.button("Open", key=f"open_{i}"):
                #     open_file(item["path"])
