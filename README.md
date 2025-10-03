# Semantic Photo Search (Local)

This is a local photo search application built with Streamlit, Chroma, and CLIP embeddings.  
It allows you to search through your personal photo collection with natural language queries such as:

- "tokyo scenery at night"  
- "pictures of Mia at the temple"  
- "beach sunrise"  

You can also add or update tags for each photo. When tags are updated, the app recomputes the vector embeddings (mixing image and tag semantics) so that future searches immediately reflect your changes. Everything runs locally on your computer.

---

## Features

- Natural language search using CLIP embeddings (local, no cloud required).
- Inline tagging: click a photo, view and update tags in the sidebar.
- Auto re-embedding: tags and image embeddings are blended (default weights: image 70%, tags 30%).
- EXIF orientation fix for portrait/landscape photos.
- Lightweight database using Chroma (stored on disk).
- Thumbnail generation for faster browsing.

---

## Project Structure

```
langchain_test/
  photos/                       # your personal photos
    S23 Camera Backup/
      Camera/
  data/
    index/                      # Chroma DB (auto-created)
    thumbs/                     # thumbnails (auto-created)
  ingest.py                     # builds vectors and thumbnails
  app.py                        # Streamlit UI
  requirements.txt
  README.md
  .gitignore
```

---

## Setup

1. Place your photos under the `photos/` folder. Subfolders are allowed and will be indexed recursively.  
   Example: `photos/S23 Camera Backup/Camera/`

2. Create a virtual environment:

```bash
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
```

3. Install dependencies:

```bash
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

4. Ingest photos to build thumbnails, embeddings, and index:

```bash
.\.venv\Scripts\python.exe ingest.py
```

---

## Running the Application

Launch the Streamlit UI:

```bash
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the provided URL in your browser (usually http://localhost:8501).

---

## Usage

- Enter a query in the search box (for example, `tokyo skyline` or `beach sunrise`).  
- Browse results in a grid view.  
- Click **Edit tags** under a photo to view or update tags in the sidebar.  
- Click **Save tags** to update metadata and recompute embeddings.  
- Searches will immediately reflect any changes you make to tags.

---

## Configuration

- **Embedding model**: `clip-ViT-B-32` (via sentence-transformers).  
- **Embedding blend weights**: Image = 0.7, Tags = 0.3 (tweak in `app.py`).  
- **Thumbnail size**: 384x384 pixels (configurable in `ingest.py`).  
- **Top-K results**: adjustable in the Streamlit sidebar (default 12).  

---

## Roadmap

- Batch tagging for multiple photos.  
- "More like this" search from a specific photo.  
- Auto-suggest tags from folder names (for example, `Tokyo_2019`).  
- Export album as static HTML gallery.  
- Optional integration with LangChain/LangGraph for query parsing.  
- Face recognition index for finding specific people.  

---

## Privacy

- Photos never leave your machine.  
- All embeddings and metadata are stored locally in `data/index/`.  
- To avoid committing personal data, make sure `photos/` and `.venv/` are listed in `.gitignore`.  
