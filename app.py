"""
RAG Search App - যেকোনো ফাইল (zip, pdf, docx, xlsx, pptx, txt, csv, json, html) আপলোড করুন,
তারপর নাম/কীওয়ার্ড লিখে সার্চ করুন।

চালানোর নিয়ম:
    pip install -r requirements.txt
    streamlit run app.py
"""
import io
import re
import json
import zipfile

import numpy as np
import streamlit as st

# ---------------------------------------------------------------- ফাইল রিডার
TEXT_EXT = {".txt", ".md", ".csv", ".log", ".py", ".js", ".xml", ".yaml", ".yml", ".tsv"}


def read_pdf(data: bytes):
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    return [(f"পৃষ্ঠা {i + 1}", p.extract_text() or "") for i, p in enumerate(reader.pages)]


def read_docx(data: bytes):
    from docx import Document
    doc = Document(io.BytesIO(data))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for t in doc.tables:
        for row in t.rows:
            parts.append(" | ".join(c.text.strip() for c in row.cells))
    return [("ডকুমেন্ট", "\n".join(parts))]


def read_xlsx(data: bytes):
    from openpyxl import load_workbook
    wb = load_workbook(io.BytesIO(data), data_only=True, read_only=True)
    out = []
    for ws in wb.worksheets:
        rows = []
        for r in ws.iter_rows(values_only=True):
            if any(c is not None for c in r):
                rows.append(" | ".join("" if c is None else str(c) for c in r))
        out.append((f"শিট: {ws.title}", "\n".join(rows)))
    return out


def read_pptx(data: bytes):
    from pptx import Presentation
    prs = Presentation(io.BytesIO(data))
    out = []
    for i, slide in enumerate(prs.slides, 1):
        texts = [sh.text_frame.text for sh in slide.shapes if sh.has_text_frame]
        out.append((f"স্লাইড {i}", "\n".join(texts)))
    return out


def read_html(data: bytes):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(data.decode("utf-8", "ignore"), "html.parser")
    return [("HTML", soup.get_text("\n"))]


def decode_text(data: bytes) -> str:
    for enc in ("utf-8", "utf-16", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except Exception:
            continue
    return ""


def extract(name: str, data: bytes, depth: int = 0):
    """ফাইল থেকে [(ফাইলের নাম, লোকেশন, টেক্সট)] রিটার্ন করে। zip হলে ভেতরের ফাইলও পড়ে।"""
    ext = "." + name.lower().rsplit(".", 1)[-1] if "." in name else ""
    results = []
    try:
        if ext == ".zip" and depth < 3:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for info in z.infolist():
                    if info.is_dir() or info.filename.startswith("__MACOSX"):
                        continue
                    results += extract(f"{name}/{info.filename}", z.read(info), depth + 1)
            return results
        if ext == ".pdf":
            parts = read_pdf(data)
        elif ext == ".docx":
            parts = read_docx(data)
        elif ext in (".xlsx", ".xlsm"):
            parts = read_xlsx(data)
        elif ext == ".pptx":
            parts = read_pptx(data)
        elif ext in (".html", ".htm"):
            parts = read_html(data)
        elif ext == ".json":
            parts = [("JSON", json.dumps(json.loads(decode_text(data)), ensure_ascii=False, indent=1))]
        elif ext in TEXT_EXT:
            parts = [("টেক্সট", decode_text(data))]
        else:
            return results  # অজানা ফরম্যাট বাদ
        for loc, text in parts:
            if text.strip():
                results.append((name, loc, text))
    except Exception as e:
        st.warning(f"⚠️ {name} পড়া যায়নি: {e}")
    return results


# ---------------------------------------------------------------- চাঙ্কিং
def chunk_text(text: str, size: int = 800, overlap: int = 150):
    text = re.sub(r"[ \t]+", " ", text).strip()
    if len(text) <= size:
        return [text]
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i:i + size])
        i += size - overlap
    return chunks


# ---------------------------------------------------------------- এমবেডিং (ঐচ্ছিক, বাংলা সাপোর্টসহ)
@st.cache_resource(show_spinner="এমবেডিং মডেল লোড হচ্ছে (প্রথমবার একটু সময় লাগবে)...")
def load_embedder():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    except Exception:
        return None


def build_index(files):
    chunks = []
    for f in files:
        for src, loc, text in extract(f.name, f.getvalue()):
            for c in chunk_text(text):
                chunks.append({"source": src, "loc": loc, "text": c})
    emb = None
    model = load_embedder()
    if model and chunks:
        emb = model.encode([c["text"] for c in chunks], batch_size=32,
                           normalize_embeddings=True, show_progress_bar=False)
    return chunks, emb


# ---------------------------------------------------------------- সার্চ
def keyword_search(chunks, query):
    """নাম/কীওয়ার্ড যেসব চাঙ্কে আছে সবগুলো (সব শব্দ থাকলে)।"""
    words = [w for w in re.split(r"\s+", query.strip().lower()) if w]
    hits = []
    for c in chunks:
        t = c["text"].lower()
        if all(w in t for w in words):
            hits.append(c)
    return hits


def semantic_search(chunks, emb, query, k=8):
    model = load_embedder()
    if model is None or emb is None:
        return []
    q = model.encode([query], normalize_embeddings=True)[0]
    scores = emb @ q
    idx = np.argsort(-scores)[:k]
    return [dict(chunks[i], score=float(scores[i])) for i in idx if scores[i] > 0.25]


def highlight(text, query):
    for w in sorted({w for w in query.split() if w}, key=len, reverse=True):
        text = re.sub(f"({re.escape(w)})", r"<mark>\1</mark>", text, flags=re.I)
    return text.replace("\n", "<br>")


def ask_claude(api_key, query, ctx_chunks):
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    context = "\n\n".join(f"[{c['source']} - {c['loc']}]\n{c['text']}" for c in ctx_chunks[:15])
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1500,
        messages=[{"role": "user", "content":
                   f"নিচের ডকুমেন্ট অংশগুলো থেকে '{query}' সম্পর্কে সব তথ্য গুছিয়ে বাংলায় লিখো। "
                   f"শুধু ডকুমেন্টে যা আছে তাই বলো, নিজে বানিয়ে বলো না। কোন ফাইল থেকে তথ্য সেটাও উল্লেখ করো।\n\n{context}"}],
    )
    return msg.content[0].text


# ---------------------------------------------------------------- UI
st.set_page_config(page_title="RAG ফাইল সার্চ", page_icon="🔎", layout="wide")
st.title("🔎 RAG ফাইল সার্চ")

with st.sidebar:
    st.header("📁 ফাইল আপলোড")
    files = st.file_uploader("zip, pdf, docx, xlsx, pptx, txt, csv...",
                             accept_multiple_files=True)
    if st.button("ইনডেক্স তৈরি করো", type="primary", disabled=not files):
        with st.spinner("ফাইল পড়া হচ্ছে..."):
            chunks, emb = build_index(files)
        st.session_state.update(chunks=chunks, emb=emb)
    st.divider()
    api_key = st.text_input("Anthropic API Key (ঐচ্ছিক)", type="password",
                            help="দিলে AI সব তথ্য গুছিয়ে সারাংশ লিখে দেবে")

chunks = st.session_state.get("chunks")
if not chunks:
    st.info("বাম পাশ থেকে ফাইল আপলোড করে **ইনডেক্স তৈরি করো** চাপুন।")
    st.stop()

n_files = len({c["source"] for c in chunks})
st.success(f"✅ {n_files}টি ফাইল থেকে {len(chunks)}টি অংশ ইনডেক্স হয়েছে")

query = st.text_input("নাম বা কীওয়ার্ড লিখুন", placeholder="যেমন: রহিম উদ্দিন")
if query:
    kw = keyword_search(chunks, query)
    sem = semantic_search(chunks, st.session_state.get("emb"), query)

    if api_key:
        with st.spinner("AI সারাংশ তৈরি করছে..."):
            try:
                st.subheader("🤖 AI সারাংশ")
                st.write(ask_claude(api_key, query, kw if kw else sem))
            except Exception as e:
                st.error(f"AI এরর: {e}")

    tab1, tab2 = st.tabs([f"📌 সরাসরি মিল ({len(kw)})", f"🧠 অর্থগত মিল ({len(sem)})"])
    with tab1:
        if not kw:
            st.write("সরাসরি কোনো মিল পাওয়া যায়নি। অর্থগত মিল ট্যাব দেখুন।")
        by_file = {}
        for c in kw:
            by_file.setdefault(c["source"], []).append(c)
        for src, items in by_file.items():
            with st.expander(f"📄 {src}  ({len(items)}টি জায়গায়)", expanded=True):
                for c in items:
                    st.caption(c["loc"])
                    st.markdown(highlight(c["text"], query), unsafe_allow_html=True)
                    st.divider()
    with tab2:
        if st.session_state.get("emb") is None:
            st.write("sentence-transformers ইনস্টল নেই, তাই শুধু কীওয়ার্ড সার্চ চলছে।")
        for c in sem:
            with st.expander(f"📄 {c['source']} — {c['loc']}  (মিল: {c['score']:.2f})"):
                st.markdown(highlight(c["text"], query), unsafe_allow_html=True)
