import os
import json
import numpy as np
import streamlit as st
import chromadb
import hashlib

def _sid(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]

from PIL import Image, ExifTags
from sentence_transformers import SentenceTransformer

# --------- CONFIG ---------
# Resolve project-relative paths so renaming/moving the folder won't break things.
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(PROJECT_DIR, "data", "index")  # where Chroma stores the vector DB on disk
COLLECTION_NAME = "vacation_photos"                  # Chroma collection name (think: table/namespace)
MODEL_NAME = "clip-ViT-B-32"                         # same CLIP model used during ingest
TOP_K_DEFAULT = 12                                   # default number of results to return

# Weighting used when mixing image and tag (text) embeddings
W_IMG = 0.7
W_TXT = 0.3
# --------------------------

def load_with_exif_orientation(path: str) -> Image.Image:
    """
    Open an image and auto-rotate based on EXIF orientation.
    Many phone cameras store orientation as metadata instead of rotating pixels.
    This ensures thumbnails and embeddings are consistent with how photos should look.
    """
    img = Image.open(path)
    try:
        exif = img._getexif()
        if exif is not None:
            # Find the EXIF key for Orientation
            orientation_key = next((k for k, v in ExifTags.TAGS.items() if v == "Orientation"), None)
            if orientation_key and orientation_key in exif:
                orientation = exif[orientation_key]
                # Rotate to match intended display orientation
                if orientation == 3:
                    img = img.rotate(180, expand=True)
                elif orientation == 6:
                    img = img.rotate(270, expand=True)
                elif orientation == 8:
                    img = img.rotate(90, expand=True)
    except Exception:
        # If EXIF is missing/corrupt, just proceed with the original image
        pass
    return img

@st.cache_resource(show_spinner=False)
def load_model():
    """
    Load the CLIP model once and cache it for the Streamlit app lifetime.
    Using @st.cache_resource avoids reloading the model on every rerun.
    """
    return SentenceTransformer(MODEL_NAME)

@st.cache_resource(show_spinner=False)
def load_collection():
    """
    Connect to the on-disk Chroma DB and get/create the photo collection.
    Cached so we don’t reconnect on every UI interaction.
    """
    client = chromadb.PersistentClient(path=DB_DIR)
    return client.get_or_create_collection(COLLECTION_NAME)

def normalize(x: np.ndarray) -> np.ndarray:
    """
    L2-normalize a vector (defensive helper).
    SentenceTransformers already returns normalized vectors when requested,
    but we normalize the mixed vector again after combining image+text.
    """
    n = np.linalg.norm(x)
    return x if n == 0 else x / n

def compute_combined_embedding(img_path: str, tags_text: str, model: SentenceTransformer) -> np.ndarray:
    """
    Compute a single embedding that blends:
      - the image embedding, and
      - an optional text embedding built from user tags.
    This lets manual tags influence retrieval while preserving image semantics.
    """
    # Always embed the (orientation-corrected) image
    img = load_with_exif_orientation(img_path).convert("RGB")
    e_img = model.encode(img, convert_to_numpy=True, normalize_embeddings=True)

    # If the user provided tags, blend image + text embeddings
    if tags_text.strip():
        e_txt = model.encode(tags_text, convert_to_numpy=True, normalize_embeddings=True)
        e = normalize(W_IMG * e_img + W_TXT * e_txt)
    else:
        e = e_img  # no tags → pure image embedding
    return e

def search_images(query: str, k: int, model, collection):
    """
    Run a text → image search using CLIP:
      1) Encode the query with the text encoder,
      2) Query Chroma for the top-k nearest image vectors,
      3) Return each hit’s metadata (path, thumb, tags, etc.).
    """
    if not query.strip():
        return []
    # Encode text query into the same vector space as images
    qvec = model.encode(query, convert_to_numpy=True, normalize_embeddings=True).tolist()

    # Query Chroma by vector similarity
    res = collection.query(query_embeddings=[qvec], n_results=k)

    # Collect normalized metadata for each result
    hits = []
    for i in range(len(res["ids"][0])):
        meta = res["metadatas"][0][i] or {}
        meta.setdefault("path", "")
        meta.setdefault("thumb", "")
        meta.setdefault("folder", "")
        meta.setdefault("datetime", "")
        meta.setdefault("tags", "")   # comma-separated string of tags
        meta["_id"] = res["ids"][0][i]  # keep ID for later updates
        hits.append(meta)
    return hits

def update_tags_and_embedding(collection, model, meta: dict, new_tags: str):
    """
    Update a single photo’s tags + vector in Chroma.
    Steps:
      - Recompute the combined embedding (image + new tags),
      - Update the document metadata (store tags),
      - Write both back into Chroma with collection.update().
    """
    path = meta["path"]
    # Minimal “document” text payload; optional but nice to keep human-readable info
    doc = json.dumps({"desc": meta.get("folder", ""), "tags": new_tags})

    # Re-embed with new tags and normalize
    emb = compute_combined_embedding(path, new_tags, model).tolist()

    # Refresh metadata object (don’t persist ephemeral keys like "_id")
    new_meta = dict(meta)
    new_meta["tags"] = new_tags

    # Apply update in Chroma (in-place change of vector + metadata + doc)
    collection.update(
        ids=[meta["_id"]],
        embeddings=[emb],
        metadatas=[{k: v for k, v in new_meta.items() if not k.startswith("_")}],
        documents=[doc],
    )

# ---------- Streamlit UI ----------

# Basic Streamlit page settings and title
st.set_page_config(page_title="📸 Semantic Photo Search", layout="wide")
st.title("📸 Semantic Photo Search (Local)")

# Load heavy resources once (cached functions above)
model = load_model()
collection = load_collection()

# Keep transient UI state across interactions (which item is selected, etc.)
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

# ---- one-time defaults (outside the form) ----
if "query" not in st.session_state:
    st.session_state.query = st.session_state.get("last_query", "")
if "top_k" not in st.session_state:
    st.session_state.top_k = TOP_K_DEFAULT
if "run_search" not in st.session_state:
    st.session_state.run_search = False

# ---- the form (Enter submits) ----
with st.sidebar.form(key="search_form", clear_on_submit=False):
    st.header("Search")

    # NOTE: use `key=` and DO NOT pass `value=` here (prevents resets)
    st.text_input(
        "Describe the photo",
        key="query",
        placeholder="e.g., beach sunset, dog, Tokyo night",
    )
    st.slider(
        "Results",
        min_value=4, max_value=48, step=4,
        key="top_k",
    )
    submitted = st.form_submit_button("Search")  # Enter triggers this

# set a flag when submitted; handle search outside the form
if submitted:
    st.session_state.last_query = st.session_state.query
    st.session_state.run_search = True

# ---- run the search on rerun after submit ----
results = []
if st.session_state.run_search and st.session_state.query.strip():
    try:
        results = search_images(
            query=st.session_state.query,
            k=st.session_state.top_k,
            model=model,
            collection=collection,
        )
    finally:
        # reset the flag so subsequent Enter/backspace cycles work
        st.session_state.run_search = False

# ------------- Results area -------------
def path_exists(p: str) -> bool:
    # Be kind to Windows path differences
    if not p:
        return False
    p2 = p.replace("\\", "/")
    return os.path.exists(p) or os.path.exists(p2)

def _sid(s: str) -> str:
    # stable key across reruns
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]

def show_result_card(meta: dict):
    thumb  = meta.get("thumb", "")
    path   = meta.get("path", "")
    tags   = (meta.get("tags", "") or "").strip()
    folder = meta.get("folder", "")
    dt     = meta.get("datetime", "")

    # Pick the best display source: thumb → original → warn
    img_render = thumb if os.path.exists(thumb) else (path if os.path.exists(path) else None)
    if not img_render:
        st.warning("Missing file:\n" + (thumb or path))
        return

    # ✅ correct param
    st.image(img_render, use_container_width=True)
    st.caption(f"**{folder}**  \n{dt or '—'}")

    if tags:
        st.markdown(f"**Tags:** `{tags}`")
    else:
        st.caption("No custom tags yet.")

    sid = _sid(meta["_id"])
    input_key = f"tags_input_{sid}"
    saved_key = f"saved_notice_{sid}"

    # ensure session state has the current value once
    if input_key not in st.session_state:
        st.session_state[input_key] = tags

    # --- save handler used by Enter & the button ---
    def _save_tags(meta_local, input_key_local, saved_key_local):
        new_tags_val = (st.session_state.get(input_key_local, "") or "").strip()
        try:
            with st.spinner("Recomputing vector embedding…"):
                update_tags_and_embedding(collection, model, meta_local, new_tags_val)
            # reflect in current card immediately
            meta_local["tags"] = new_tags_val
            st.session_state[saved_key_local] = True
            # Non-blocking confirmation
            if hasattr(st, "toast"):
                st.toast("✅ Tags saved & embedding updated", icon="✅")
            st.success("Tags saved & embedding updated.")
        except Exception as e:
            st.error(f"Update failed: {e}")

    with st.expander("Edit tags"):
        # Hitting Enter in this text_input will call _save_tags()
        st.text_input(
            "Tags (comma-separated)",
            key=input_key,
            help="Press Enter to save, or use the Save button",
            on_change=_save_tags,
            args=(meta, input_key, saved_key),
        )
        st.button(
            "Save",
            key=f"savebtn_{sid}",
            on_click=_save_tags,
            args=(meta, input_key, saved_key),
        )

        # Persist a confirmation line on rerun if we just saved
        if st.session_state.get(saved_key):
            st.info("Tags are up to date for this photo.")

# -------- Render grid --------
if results:
    n_cols = 4
    rows = (len(results) + n_cols - 1) // n_cols
    for r in range(rows):
        cols = st.columns(n_cols)
        for c in range(n_cols):
            i = r * n_cols + c
            if i < len(results):
                with cols[c]:
                    show_result_card(results[i])
else:
    st.info("Type a query in the sidebar and press **Search**.")
